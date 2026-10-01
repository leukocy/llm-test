"""Worker-owned public dataset preparation with fenced atomic publication."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
from dataclasses import replace

from core.dataset_manager import DATASET_CONFIGS, DatasetManager, get_manager
from server.runner_adapter import RunOutput
from server.store import LeaseLost

MAX_BYTES = 2 * 1024**3


def downloadable_names() -> set[str]:
    from evaluators import list_available_datasets

    return {
        name
        for name in list_available_datasets()
        if name in DATASET_CONFIGS and DATASET_CONFIGS[name].hf_path
    }


def pinned_revision(repo: str) -> str:
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(repo, token=False, timeout=15).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset source did not provide an immutable revision")
    return revision


def execute_preparation(job: dict, settings, store, worker_id: str) -> RunOutput:
    name = job["parameters"]["name"]
    if name not in downloadable_names():
        raise ValueError("Select a registered public dataset")
    manager = get_manager()
    target = manager.get_local_path(name)
    stage = target.parent / ".staging" / job["job_id"] / f"attempt-{job['attempts']}" / name
    backup = target.parent / ".backup" / job["job_id"] / f"attempt-{job['attempts']}" / name

    def check(conn=None):
        if conn is None:
            with store._connection() as connection:
                return check(connection)
        current = dict(
            conn.execute("SELECT * FROM control_jobs WHERE job_id=?", (job["job_id"],)).fetchone()
        )
        if current["status"] == "cancelling":
            raise asyncio.CancelledError("Dataset preparation cancelled")
        owner, until = current.get("lease_owner"), current.get("lease_until") or 0
        if (
            current["status"] != "running"
            or current["attempts"] != job["attempts"]
            or owner != worker_id
            or until < time.time()
        ):
            raise LeaseLost(job["job_id"])

    def progress(fraction, message):
        with store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            check(conn)
            store._event(
                conn,
                job["job_id"],
                "running",
                "running",
                "DATASET_PREPARATION",
                worker_id,
                time.time(),
                {
                    "dataset": name,
                    "fraction": max(0, min(1, fraction)),
                    "message": str(message)[:300],
                },
            )
            conn.commit()
        if sum(path.stat().st_size for path in stage.rglob("*") if path.is_file()) > MAX_BYTES:
            raise ValueError("Prepared dataset exceeds 2 GiB")

    try:
        check()
        if manager.is_available(name):
            return RunOutput(completed=1, total=1)
        stage.mkdir(parents=True, exist_ok=False)
        config = manager.configs[name]
        assert config.hf_path is not None
        revision = pinned_revision(config.hf_path)
        downloader = DatasetManager(cache_dir=str(stage.parent), auto_download=False)
        downloader.configs[name] = replace(config, local_path=str(stage), version=revision)
        if not downloader.download(name, force=True, progress_callback=progress):
            # Cancellation/lease errors may be swallowed by third-party download helpers.
            progress(0, "下载失败，未发布")
            raise ValueError("Dataset download failed; inspect the preparation events")
        split = config.split_mapping.get("test", "test")
        samples = downloader.load(name, split=split)
        if not samples:
            raise ValueError("Downloaded dataset has no evaluation samples")
        if name == "mbpp":
            from evaluators.mbpp_evaluator import MBPPEvaluator

            # Published data must support the platform's default five exemplars.
            prepared = MBPPEvaluator(dataset_path=str(stage), num_shots=5)
            prepared.load_dataset()
        if name in {"ceval", "cmmlu"}:
            from evaluators.ceval_evaluator import CEvalEvaluator
            from evaluators.cmmlu_evaluator import CMMLUEvaluator

            evaluator_class = CEvalEvaluator if name == "ceval" else CMMLUEvaluator
            samples = evaluator_class.normalize(samples, split)
            dev = evaluator_class.normalize(downloader.load(name, split="dev"), "dev")
            if any(
                sum(example["subject"] == sample["subject"] for example in dev) < 5
                for sample in samples
            ):
                raise ValueError(
                    f"{name} preparation requires five development exemplars per scoring subject"
                )
        if sum(path.stat().st_size for path in stage.iterdir() if path.is_file()) > MAX_BYTES:
            raise ValueError("Prepared dataset exceeds 2 GiB")
        files = []
        for path in stage.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("Downloaded dataset contains unsupported entries")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(65536), b""):
                    check()
                    digest.update(chunk)
            files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
            )
        size = sum(item["bytes"] for item in files)
        if size > MAX_BYTES:
            raise ValueError("Prepared dataset exceeds 2 GiB")
        receipt = {
            "dataset": name,
            "repo": config.hf_path,
            "revision": revision,
            "sample_count": len(samples),
            "evaluation_split": split,
            "bytes": size,
            "files": files,
        }
        (stage / "preparation.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            check(conn)
            if target.is_symlink():
                raise ValueError("Dataset target may not be a symbolic link")
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                target.rename(backup)
            try:
                stage.rename(target)
            except Exception:
                if backup.exists():
                    backup.rename(target)
                raise
            store._event(
                conn,
                job["job_id"],
                "running",
                "running",
                "DATASET_PREPARED",
                worker_id,
                time.time(),
                receipt,
            )
            conn.commit()
        artifact = settings.artifact_root / job["job_id"] / "report.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps({"dataset_preparation": receipt}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return RunOutput(
            result_artifact=str(artifact.relative_to(settings.artifact_root)), completed=1, total=1
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
