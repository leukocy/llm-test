"""Bounded public Hub downloads, validated offline before atomic publication."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from collections.abc import Callable
from typing import Any

import httpx

from config.tokenizer_paths import MANIFEST_NAME
from core.tokenizer_utils import get_cached_tokenizer
from server.store import LeaseLost
from server.tokenizer_queue import (
    InstallCancelled,
    TokenizerInstallQueue,
    registered_repo,
    staging_path,
)

logger = logging.getLogger(__name__)
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_INSTALL_BYTES = 128 * 1024 * 1024
MAX_FILES = 32
MAX_DURATION_SECONDS = 1200
TOKENIZER_FILES = {
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "chat_template.jinja",
    "tekken.json",
}


class InstallFailure(ValueError):
    """Safe, actionable error text without response bodies or credentials."""


def _hub_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(15, connect=10),
        follow_redirects=True,
        headers={"Accept-Encoding": "identity", "User-Agent": "llm-test-tokenizer/1"},
    )


def _read_manifest(client: httpx.Client, repo: str, check: Callable[[], None]) -> dict[str, Any]:
    data = bytearray()
    with client.stream(
        "GET", f"https://huggingface.co/api/models/{repo}/revision/main?blobs=true"
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(65536):
            check()
            data.extend(chunk)
            if len(data) > MAX_METADATA_BYTES:
                raise InstallFailure("仓库文件清单超过 4 MiB 限制")
    try:
        metadata = json.loads(data)
    except ValueError as exc:
        raise InstallFailure("下载源返回了无效的文件清单") from exc
    if not isinstance(metadata, dict):
        raise InstallFailure("下载源返回了无效的文件清单")
    return metadata


def _selected_files(metadata: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    revision = metadata.get("sha", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise InstallFailure("下载源没有提供不可变的仓库版本")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise InstallFailure("下载源没有提供文件清单")
    files: list[dict[str, Any]] = []
    seen = set()
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = item.get("rfilename", "")
        if not isinstance(name, str) or (
            name not in TOKENIZER_FILES and not re.fullmatch(r"[A-Za-z0-9_-]+\.tiktoken", name)
        ):
            continue
        if name in seen:
            raise InstallFailure("下载源文件清单包含重复条目")
        size = item.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_FILE_BYTES:
            raise InstallFailure("Tokenizer 文件大小未知或超过单文件 64 MiB 限制")
        seen.add(name)
        lfs = item.get("lfs") or {}
        digest = lfs.get("sha256") or lfs.get("oid") if isinstance(lfs, dict) else None
        files.append({"name": name, "size": size, "expected_sha256": digest})
    if (
        not files
        or len(files) > MAX_FILES
        or sum(item["size"] for item in files) > MAX_INSTALL_BYTES
    ):
        raise InstallFailure("Tokenizer 文件集为空或超过 32 个文件 / 128 MiB 限制")
    return revision, sorted(files, key=lambda item: item["name"])


def execute_install(
    task: dict[str, Any],
    queue: TokenizerInstallQueue,
    worker_id: str,
    *,
    interrupted: Callable[[], bool] = lambda: False,
) -> None:
    identifier = task["install_id"]
    stage = staging_path(identifier)
    started = time.monotonic()
    downloaded = 0
    finished = 0
    total = 0
    revision = ""
    files: list[dict[str, Any]] = []

    def check() -> None:
        queue.check(identifier, worker_id)
        if interrupted():
            raise InstallFailure("worker 已停止；可重新下载")
        if time.monotonic() - started > MAX_DURATION_SECONDS:
            raise InstallFailure("下载超过 20 分钟限制；可重新下载")

    def progress(status: str, message: str) -> None:
        queue.progress(
            identifier,
            worker_id,
            status=status,
            revision=revision,
            downloaded_bytes=downloaded,
            total_bytes=total,
            completed_files=finished,
            total_files=len(files),
            message=message,
        )

    try:
        repo = registered_repo(task["name"])
        if repo != task["repo_id"]:
            raise InstallFailure("已登记的下载源发生变化，请重新创建下载")
        check()
        stage.mkdir(parents=True, exist_ok=False, mode=0o700)
        manifest_files = []
        with _hub_client() as client:
            metadata = _read_manifest(client, repo, check)
            revision, files = _selected_files(metadata)
            total = sum(item["size"] for item in files)
            progress("downloading", "正在下载 Tokenizer 文件")
            for item in files:
                check()
                name = item["name"]
                size = 0
                digest = hashlib.sha256()
                last_progress = time.monotonic()
                with client.stream(
                    "GET", f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
                ) as response:
                    response.raise_for_status()
                    with (stage / name).open("xb") as destination:
                        for chunk in response.iter_bytes(65536):
                            check()
                            size += len(chunk)
                            downloaded += len(chunk)
                            if (
                                size > item["size"]
                                or size > MAX_FILE_BYTES
                                or downloaded > MAX_INSTALL_BYTES
                            ):
                                raise InstallFailure("下载实际大小超过清单或 128 MiB 限制")
                            destination.write(chunk)
                            digest.update(chunk)
                            if time.monotonic() - last_progress >= 0.5:
                                progress("downloading", f"正在下载 {name}")
                                last_progress = time.monotonic()
                actual_hash = digest.hexdigest()
                expected_hash = item["expected_sha256"]
                if size != item["size"] or (
                    expected_hash is not None and actual_hash != expected_hash
                ):
                    raise InstallFailure("下载文件大小或 SHA-256 与源清单不符")
                manifest_files.append({"name": name, "size": size, "sha256": actual_hash})
                finished += 1
                progress("downloading", f"已下载 {finished} / {len(files)} 个文件")
        check()
        progress("validating", "正在离线校验 Tokenizer；禁止执行远程代码")
        tokenizer = get_cached_tokenizer(str(stage))
        if tokenizer is None:
            raise InstallFailure("Tokenizer 本地加载失败；可能需要自定义代码，未启用该安装")
        # Exercise the same public count operation before exposing the installation.
        tokenizer.encode("LLM 性能测量 validation 123", add_special_tokens=False)
        manifest = {
            "name": task["name"],
            "source": "huggingface",
            "repo_id": repo,
            "revision": revision,
            "files": manifest_files,
            "installed_at": time.time(),
            "trust_remote_code": False,
            "validation": "local_encode",
        }
        (stage / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        check()
        queue.publish(identifier, worker_id)
    except InstallCancelled:
        queue.fail(identifier, worker_id, "下载已取消；未启用临时文件", cancelled=True)
    except LeaseLost:
        logger.warning("Tokenizer installation lease lost: %s", identifier)
    except Exception as exc:
        if isinstance(exc, InstallFailure):
            message = str(exc)
        elif isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            message = (
                "下载源要求授权；此入口仅下载公开文件"
                if code in {401, 403}
                else f"下载源 HTTP {code}；请检查网络或稍后重试"
            )
        elif isinstance(exc, httpx.HTTPError):
            message = "无法连接下载源或连接超时；请检查 worker 出网配置"
        else:
            message = "本地安装失败；请检查持久化目录、剩余空间或 Tokenizer 格式"
        logger.warning("Tokenizer installation failed: %s (%s)", identifier, type(exc).__name__)
        try:
            queue.fail(identifier, worker_id, message)
        except LeaseLost:
            # The current owner records the outcome; this worker cannot overwrite it.
            pass
    finally:
        shutil.rmtree(stage, ignore_errors=True)
