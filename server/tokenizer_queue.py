"""Durable installation queue, mutually exclusive with benchmark execution."""

from __future__ import annotations

import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from config.settings import TOKENIZER_SOURCES
from config.tokenizer_paths import registered_tokenizer_path, tokenizer_download_root
from server.store import JobStore, LeaseLost

ACTIVE_INSTALL_STATES = {"queued", "downloading", "validating", "cancelling"}


class InstallNotFound(LookupError):
    """Unknown installation ID."""


class InstallCancelled(RuntimeError):
    """Cancellation requested before publication."""


def registered_repo(name: str) -> str:
    source = TOKENIZER_SOURCES.get(name)
    repo = source if isinstance(source, str) else source.get("hf") if source else None
    if not repo or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Select a registered tokenizer")
    return repo


def staging_path(install_id: str) -> Path:
    if str(uuid.UUID(install_id)) != install_id:
        raise ValueError("Invalid installation ID")
    return tokenizer_download_root() / ".staging" / install_id


class TokenizerInstallQueue:
    def __init__(self, store: JobStore) -> None:
        self.store = store

    @staticmethod
    def _public(row: Any) -> dict[str, Any]:
        return {
            key: value
            for key, value in dict(row).items()
            if key not in {"id", "lease_owner", "lease_until"}
        }

    def latest(self) -> dict[str, dict[str, Any]]:
        with self.store._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM tokenizer_installs WHERE id IN
                   (SELECT MAX(id) FROM tokenizer_installs GROUP BY name)"""
            ).fetchall()
        return {row["name"]: self._public(row) for row in rows}

    def get(self, install_id: str) -> dict[str, Any]:
        with self.store._connection() as conn:
            row = conn.execute(
                "SELECT * FROM tokenizer_installs WHERE install_id = ?", (install_id,)
            ).fetchone()
        if row is None:
            raise InstallNotFound(install_id)
        return self._public(row)

    def enqueue(self, names: list[str]) -> dict[str, Any]:
        if not names or len(names) > len(TOKENIZER_SOURCES) or len(set(names)) != len(names):
            raise ValueError("Select unique registered tokenizer names")
        repos = {name: registered_repo(name) for name in names}
        now = time.time()
        identifiers, skipped = [], []
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for name in names:
                if registered_tokenizer_path(name) is not None:
                    skipped.append(name)
                    continue
                active = conn.execute(
                    "SELECT install_id FROM tokenizer_installs WHERE name = ? "
                    "AND status IN ('queued', 'downloading', 'validating', 'cancelling')",
                    (name,),
                ).fetchone()
                if active:
                    identifiers.append(active["install_id"])
                    continue
                identifier = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO tokenizer_installs
                       (install_id, name, repo_id, status, message, created_at, updated_at)
                       VALUES (?, ?, ?, 'queued', ?, ?, ?)""",
                    (identifier, name, repos[name], "等待 worker；测量结束后开始下载", now, now),
                )
                identifiers.append(identifier)
            conn.commit()
        return {"items": [self.get(item) for item in identifiers], "skipped_names": skipped}

    def claim(self, worker_id: str, *, lease_seconds: int = 60) -> dict[str, Any] | None:
        if not worker_id or lease_seconds < 5:
            raise ValueError("Invalid worker lease")
        now = time.time()
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            measurements = conn.execute(
                "SELECT 1 FROM control_jobs WHERE status IN "
                "('queued', 'running', 'pausing', 'paused', 'cancelling') LIMIT 1"
            ).fetchone()
            installing = conn.execute(
                "SELECT 1 FROM tokenizer_installs WHERE status IN "
                "('downloading', 'validating', 'cancelling') LIMIT 1"
            ).fetchone()
            row = conn.execute(
                "SELECT * FROM tokenizer_installs WHERE status = 'queued' ORDER BY id LIMIT 1"
            ).fetchone()
            if measurements or installing or row is None:
                conn.commit()
                return None
            conn.execute(
                """UPDATE tokenizer_installs SET status = 'downloading', lease_owner = ?,
                   lease_until = ?, updated_at = ?, message = ? WHERE install_id = ?""",
                (
                    worker_id,
                    now + lease_seconds,
                    now,
                    "正在读取固定版本的文件清单",
                    row["install_id"],
                ),
            )
            conn.commit()
        return self.get(row["install_id"])

    def check(self, install_id: str, worker_id: str) -> None:
        with self.store._connection() as conn:
            row = conn.execute(
                "SELECT status, lease_owner, lease_until FROM tokenizer_installs WHERE install_id = ?",
                (install_id,),
            ).fetchone()
        if not row or row["lease_owner"] != worker_id or (row["lease_until"] or 0) < time.time():
            raise LeaseLost(install_id)
        if row["status"] == "cancelling":
            raise InstallCancelled(install_id)
        if row["status"] not in {"downloading", "validating"}:
            raise LeaseLost(install_id)

    def heartbeat(self, install_id: str, worker_id: str) -> bool:
        with self.store._connection() as conn:
            cursor = conn.execute(
                """UPDATE tokenizer_installs SET lease_until = ?, updated_at = ?
                   WHERE install_id = ? AND lease_owner = ?
                   AND status IN ('downloading', 'validating', 'cancelling')""",
                (time.time() + 60, time.time(), install_id, worker_id),
            )
        return cursor.rowcount == 1

    def progress(
        self,
        install_id: str,
        worker_id: str,
        *,
        status: str,
        revision: str,
        downloaded_bytes: int,
        total_bytes: int,
        completed_files: int,
        total_files: int,
        message: str,
    ) -> None:
        self.check(install_id, worker_id)
        if status not in {"downloading", "validating"}:
            raise ValueError("Invalid progress status")
        with self.store._connection() as conn:
            cursor = conn.execute(
                """UPDATE tokenizer_installs SET status = ?, revision = ?, downloaded_bytes = ?,
                   total_bytes = ?, completed_files = ?, total_files = ?, message = ?, updated_at = ?
                   WHERE install_id = ? AND lease_owner = ? AND status IN ('downloading', 'validating')""",
                (
                    status,
                    revision,
                    downloaded_bytes,
                    total_bytes,
                    completed_files,
                    total_files,
                    message,
                    time.time(),
                    install_id,
                    worker_id,
                ),
            )
        if cursor.rowcount != 1:
            self.check(install_id, worker_id)
            raise LeaseLost(install_id)

    def cancel(self, install_id: str) -> dict[str, Any]:
        with self.store._connection() as conn:
            conn.execute(
                """UPDATE tokenizer_installs SET status = CASE WHEN status = 'queued'
                   THEN 'cancelled' ELSE 'cancelling' END, message = '取消下载', updated_at = ?
                   WHERE install_id = ? AND status IN ('queued', 'downloading', 'validating')""",
                (time.time(), install_id),
            )
        return self.get(install_id)

    def fail(
        self, install_id: str, worker_id: str, message: str, *, cancelled: bool = False
    ) -> None:
        with self.store._connection() as conn:
            cursor = conn.execute(
                """UPDATE tokenizer_installs SET status = CASE WHEN status = 'cancelling'
                   THEN 'cancelled' ELSE ? END, message = ?, updated_at = ?,
                   lease_owner = NULL, lease_until = NULL WHERE install_id = ? AND lease_owner = ?
                   AND status IN ('downloading', 'validating', 'cancelling')""",
                (
                    "cancelled" if cancelled else "failed",
                    message,
                    time.time(),
                    install_id,
                    worker_id,
                ),
            )
        if cursor.rowcount != 1:
            raise LeaseLost(install_id)

    def publish(self, install_id: str, worker_id: str) -> None:
        """Lease/cancellation checks and rename share the scheduler write lock."""
        stage = staging_path(install_id)
        now = time.time()
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM tokenizer_installs WHERE install_id = ?", (install_id,)
            ).fetchone()
            if not row or row["lease_owner"] != worker_id or (row["lease_until"] or 0) < now:
                raise LeaseLost(install_id)
            if row["status"] == "cancelling":
                raise InstallCancelled(install_id)
            if row["status"] != "validating":
                raise LeaseLost(install_id)
            registered_repo(row["name"])
            destination = tokenizer_download_root() / row["name"]
            if destination.exists() or destination.is_symlink():
                raise ValueError("Installation already exists; existing files were preserved")
            stage.rename(destination)
            conn.execute(
                """UPDATE tokenizer_installs SET status = 'completed', message = '安装完成 · 本地校验通过',
                   updated_at = ?, lease_owner = NULL, lease_until = NULL WHERE install_id = ?""",
                (now, install_id),
            )
            conn.commit()

    def reap_expired(self) -> int:
        now = time.time()
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT install_id FROM tokenizer_installs WHERE status IN "
                "('downloading', 'validating', 'cancelling') AND lease_until < ?",
                (now,),
            ).fetchall()
            conn.execute(
                """UPDATE tokenizer_installs SET status = CASE WHEN status = 'cancelling'
                   THEN 'cancelled' ELSE 'failed' END, message = '下载中断；可重新下载',
                   updated_at = ?, lease_owner = NULL, lease_until = NULL
                   WHERE status IN ('downloading', 'validating', 'cancelling') AND lease_until < ?""",
                (now, now),
            )
            conn.commit()
        for row in rows:
            shutil.rmtree(staging_path(row["install_id"]), ignore_errors=True)
        return len(rows)
