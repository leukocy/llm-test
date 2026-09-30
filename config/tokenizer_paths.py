"""Shared paths for read-only tokenizer bundles and persistent installations."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from config.settings import TOKENIZER_SOURCES

MANIFEST_NAME = "llm-test-installation.json"


def describe_tokenizer_installation(raw: Any) -> dict[str, Any] | None:
    """Validate persisted provenance without consulting a mutable remote repository."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Invalid tokenizer provenance")
    for field, pattern, limit in (
        ("name", r"[A-Za-z0-9_.-]+", 120),
        ("repo_id", r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", 200),
        ("revision", r"[0-9a-f]{40}", 40),
    ):
        value = raw.get(field)
        if not isinstance(value, str) or len(value) > limit or not re.fullmatch(pattern, value):
            raise ValueError("Invalid tokenizer provenance")
    if (
        raw.get("source") != "huggingface"
        or raw.get("trust_remote_code") is not False
        or raw.get("validation") != "local_encode"
    ):
        raise ValueError("Invalid tokenizer provenance")
    installed_at = raw.get("installed_at")
    if (
        isinstance(installed_at, bool)
        or not isinstance(installed_at, (int, float))
        or installed_at <= 0
        or installed_at > 1e13
        or not math.isfinite(installed_at)
    ):
        raise ValueError("Invalid tokenizer installation time")
    files = raw.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 32:
        raise ValueError("Invalid tokenizer file manifest")
    clean_files: list[dict[str, Any]] = []
    names = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Invalid tokenizer file manifest")
        name, size, digest = item.get("name"), item.get("size"), item.get("sha256")
        if (
            not isinstance(name, str)
            or len(name) > 120
            or not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+", name)
            or name in names
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= 64 * 1024 * 1024
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("Invalid tokenizer file manifest")
        names.add(name)
        clean_files.append({"name": name, "size": size, "sha256": digest})
    if sum(item["size"] for item in clean_files) > 128 * 1024 * 1024:
        raise ValueError("Invalid tokenizer file manifest")
    return {
        **{
            key: raw[key]
            for key in (
                "name",
                "repo_id",
                "revision",
                "source",
                "trust_remote_code",
                "validation",
                "installed_at",
            )
        },
        "files": clean_files,
    }


def tokenizer_download_root() -> Path:
    configured = os.getenv("LLM_TEST_TOKENIZER_DOWNLOAD_ROOT")
    if configured:
        return Path(configured)
    cache = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache / "llm-test" / "tokenizers"


def _contained_path(root: Path, path: Path) -> Path | None:
    """Resolve symlinks before requiring a child of the trusted directory."""
    normalized = os.path.realpath(path)
    prefix = os.path.realpath(root).rstrip(os.sep) + os.sep
    if not normalized.startswith(prefix):
        return None
    return Path(normalized)


def installation_manifest(path: Path) -> dict[str, Any] | None:
    try:
        directory = next(
            (
                checked
                for root in (tokenizer_download_root(), Path("tokenizers"))
                if (checked := _contained_path(root, path)) is not None
            ),
            None,
        )
        if directory is None:
            return None
        manifest = _contained_path(directory, directory / MANIFEST_NAME)
        if manifest is None:
            return None
        with manifest.open("rb") as source:
            raw = source.read(65537)
        if len(raw) > 65536:
            return None
        data = json.loads(raw)
        return describe_tokenizer_installation(data)
    except (OSError, ValueError):
        return None


def registered_tokenizer_path(name: str) -> Path | None:
    if name not in TOKENIZER_SOURCES:
        return None
    try:
        root = tokenizer_download_root()
        installed = _contained_path(root, root / name)
        manifest = installation_manifest(installed) if installed else None
        if installed and installed.is_dir() and manifest and manifest["name"] == name:
            if all(
                (entry := _contained_path(installed, installed / item["name"])) is not None
                and entry.is_file()
                and entry.stat().st_size == item["size"]
                for item in manifest["files"]
            ):
                return installed
    except OSError:
        # An inaccessible installation may still have a usable read-only bundle.
        pass
    try:
        bundled = _contained_path(Path("tokenizers"), Path("tokenizers") / name)
        if bundled and bundled.is_dir() and any(bundled.iterdir()):
            return bundled
    except OSError:
        # Missing or inaccessible bundles are reported as unavailable.
        pass
    return None
