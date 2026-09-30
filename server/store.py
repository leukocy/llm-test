"""SQLite-backed, process-safe benchmark job queue and lifecycle audit log."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.database.migrations import run_migrations
from core.database.schema import create_tables
from core.run_lifecycle import RunEvent, RunStatus, advance_run

logger = logging.getLogger(__name__)


class IdempotencyConflict(ValueError):
    """An idempotency key was reused with different job content."""


class BatchNotFound(LookupError):
    """The requested batch does not exist."""


class JobNotFound(LookupError):
    """The requested job does not exist."""


class LeaseLost(RuntimeError):
    """The worker no longer owns this running job."""


class PresetNotFound(LookupError):
    """The requested preset does not exist."""


class PresetConflict(ValueError):
    """A preset already uses this name."""


JobList = list[dict[str, Any]]


class JobStore:
    """A short transaction per operation, safe across API and worker processes."""

    def __init__(self, db_path: str | Path = "data/benchmark.db") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            create_tables(conn)
            run_migrations(conn)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _as_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        job = dict(row)
        job["parameters"] = json.loads(job.pop("parameters_json"))
        return job

    @staticmethod
    def _as_preset(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        preset = dict(row)
        preset["parameters"] = json.loads(preset.pop("parameters_json"))
        preset["run_config"] = json.loads(preset.pop("run_config_json") or "{}")
        return preset

    def list_presets(self, *, limit: int = 200) -> JobList:
        if not 1 <= limit <= 200:
            raise ValueError("Invalid preset limit")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM control_presets ORDER BY updated_at DESC, name LIMIT ?", (limit,)
            ).fetchall()
        return [self._as_preset(row) for row in rows]  # type: ignore[misc]

    def get_preset(self, preset_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM control_presets WHERE preset_id = ?", (preset_id,)
            ).fetchone()
        preset = self._as_preset(row)
        if preset is None:
            raise PresetNotFound(preset_id)
        return preset

    def save_preset(
        self,
        *,
        name: str,
        description: str = "",
        endpoint_id: str,
        test_type: str,
        parameters: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        preset_id: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        identifier = preset_id or str(uuid.uuid4())
        parameters_json = json.dumps(
            parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        run_config_json = json.dumps(
            run_config or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if (
                preset_id
                and not conn.execute(
                    "SELECT 1 FROM control_presets WHERE preset_id = ?", (preset_id,)
                ).fetchone()
            ):
                conn.rollback()
                raise PresetNotFound(preset_id)
            try:
                if preset_id:
                    conn.execute(
                        """UPDATE control_presets SET name = ?, description = ?, endpoint_id = ?, test_type = ?,
                           parameters_json = ?, run_config_json = ?, updated_at = ? WHERE preset_id = ?""",
                        (
                            name,
                            description,
                            endpoint_id,
                            test_type,
                            parameters_json,
                            run_config_json,
                            now,
                            identifier,
                        ),
                    )
                else:
                    conn.execute(
                        """INSERT INTO control_presets
                           (preset_id, name, description, endpoint_id, test_type, parameters_json,
                            run_config_json, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            identifier,
                            name,
                            description,
                            endpoint_id,
                            test_type,
                            parameters_json,
                            run_config_json,
                            now,
                            now,
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise PresetConflict("Preset name is already in use") from exc
            conn.commit()
        return self.get_preset(identifier)

    def delete_preset(self, preset_id: str) -> None:
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM control_presets WHERE preset_id = ?", (preset_id,))
        if cursor.rowcount == 0:
            raise PresetNotFound(preset_id)

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        job_id: str,
        before: str | None,
        after: str,
        event: str,
        actor: str,
        now: float,
        detail: dict[str, Any] | None = None,
    ) -> None:
        conn.execute(
            """INSERT INTO job_events
               (job_id, from_status, to_status, event, actor, detail_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                before,
                after,
                event,
                actor,
                json.dumps(detail, sort_keys=True) if detail else None,
                now,
            ),
        )

    def submit(
        self,
        *,
        test_type: str,
        endpoint_id: str,
        model_id: str,
        parameters: dict[str, Any],
        progress_total: int = 0,
        idempotency_key: str | None = None,
        parent_job_id: str | None = None,
    ) -> dict[str, Any]:
        if progress_total < 0:
            raise ValueError("Progress total cannot be negative")
        parameters_json = json.dumps(
            parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        now = time.time()
        job_id = str(uuid.uuid4())
        status = advance_run(RunStatus.CREATED, RunEvent.ENQUEUE).value
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = conn.execute(
                    "SELECT * FROM control_jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if existing is not None:
                    if (
                        existing["test_type"] != test_type
                        or existing["endpoint_id"] != endpoint_id
                        or existing["model_id"] != model_id
                        or existing["parameters_json"] != parameters_json
                    ):
                        conn.rollback()
                        raise IdempotencyConflict("Idempotency key is already used for another job")
                    conn.commit()
                    return self._as_job(existing)  # type: ignore[return-value]
            conn.execute(
                """INSERT INTO control_jobs
                   (job_id, idempotency_key, parent_job_id, status, test_type, endpoint_id,
                    model_id, parameters_json, progress_total, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    idempotency_key,
                    parent_job_id,
                    status,
                    test_type,
                    endpoint_id,
                    model_id,
                    parameters_json,
                    progress_total,
                    now,
                    now,
                ),
            )
            self._event(conn, job_id, RunStatus.CREATED.value, status, "enqueue", "api", now)
            conn.commit()
        return self.get(job_id)

    def submit_batch(
        self,
        *,
        batch_id: str,
        name: str,
        description: str,
        default_endpoint_id: str,
        request_hash: str,
        requested_items: int,
        items: list[dict[str, Any]],
        max_parallel: int = 1,
        stop_on_error: bool = False,
    ) -> list[dict[str, Any]]:
        """Create a named batch and every enabled child in one SQLite transaction.

        The hash covers the validated request, including disabled items. Replaying an
        idempotency key returns the original children; changing its meaning fails.
        """
        if not items or requested_items < len(items):
            raise ValueError("A batch needs at least one enabled item")
        if not 1 <= max_parallel <= 8:
            raise ValueError("Batch parallel limit must be between 1 and 8")
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT request_hash FROM control_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    conn.rollback()
                    raise IdempotencyConflict("Idempotency key is already used for another batch")
                rows = conn.execute(
                    "SELECT * FROM control_jobs WHERE parent_job_id = ? ORDER BY created_at, rowid",
                    (batch_id,),
                ).fetchall()
                conn.commit()
                return [self._as_job(row) for row in rows if row is not None]  # type: ignore[misc]
            if conn.execute(
                "SELECT 1 FROM control_jobs WHERE parent_job_id = ? LIMIT 1", (batch_id,)
            ).fetchone():
                conn.rollback()
                raise IdempotencyConflict("Batch ID is already used by an older batch")
            try:
                conn.execute(
                    """INSERT INTO control_batches
                       (batch_id, name, description, default_endpoint_id, requested_items,
                        submitted_items, max_parallel, stop_on_error, request_hash, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        batch_id,
                        name,
                        description,
                        default_endpoint_id,
                        requested_items,
                        len(items),
                        max_parallel,
                        int(stop_on_error),
                        request_hash,
                        now,
                    ),
                )
                status = advance_run(RunStatus.CREATED, RunEvent.ENQUEUE).value
                for index, item in enumerate(items):
                    job_id = str(uuid.uuid4())
                    parameters_json = json.dumps(
                        item["parameters"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    conn.execute(
                        """INSERT INTO control_jobs
                           (job_id, parent_job_id, status, test_type, endpoint_id, model_id,
                            parameters_json, progress_total, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            job_id,
                            batch_id,
                            status,
                            item["test_type"],
                            item["endpoint_id"],
                            item["model_id"],
                            parameters_json,
                            item["progress_total"],
                            now + index * 0.000001,
                            now,
                        ),
                    )
                    self._event(
                        conn, job_id, RunStatus.CREATED.value, status, "enqueue", "api", now
                    )
                rows = conn.execute(
                    "SELECT * FROM control_jobs WHERE parent_job_id = ? ORDER BY created_at, rowid",
                    (batch_id,),
                ).fetchall()
            except (sqlite3.IntegrityError, ValueError) as exc:
                conn.rollback()
                raise IdempotencyConflict("Batch could not be submitted") from exc
            conn.commit()
        return [self._as_job(row) for row in rows if row is not None]  # type: ignore[misc]

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM control_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise BatchNotFound(batch_id)
            counts = conn.execute(
                """SELECT status, COUNT(*) AS count FROM control_jobs
                   WHERE parent_job_id = ? GROUP BY status""",
                (batch_id,),
            ).fetchall()
        result = dict(row)
        result["stop_on_error"] = bool(result["stop_on_error"])
        result.pop("request_hash")
        result["status_counts"] = {item["status"]: item["count"] for item in counts}
        return result

    def list_batches(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("Invalid batch limit")
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT batch_id, name, description, default_endpoint_id,
                          requested_items, submitted_items, max_parallel, stop_on_error, created_at
                   FROM control_batches ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [{**dict(row), "stop_on_error": bool(row["stop_on_error"])} for row in rows]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)).fetchone()
        job = self._as_job(row)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def list(
        self,
        *,
        status: RunStatus | None = None,
        recoverable: bool = False,
        saved_progress: bool = False,
        limit: int = 50,
        offset: int = 0,
        parent_job_id: str | None = None,
    ) -> tuple[JobList, int]:
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("Invalid pagination")
        clauses: list[str] = []
        params_list: list[Any] = []
        if status:
            clauses.append("status = ?")
            params_list.append(status.value)
        if recoverable:
            from server.checkpoints import recovery_filter

            clause, values = recovery_filter()
            clauses.append(clause)
            params_list.extend(values)
        if saved_progress:
            clauses.append(
                "EXISTS (SELECT 1 FROM job_checkpoints h WHERE h.job_id = control_jobs.job_id)"
            )
        if parent_job_id:
            clauses.append("parent_job_id = ?")
            params_list.append(parent_job_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params = tuple(params_list)
        projection = "control_jobs.*"
        if saved_progress:
            projection += """,
                (SELECT COUNT(*) FROM checkpoint_units u WHERE u.job_id=control_jobs.job_id) AS saved_progress_planned,
                (SELECT COUNT(*) FROM checkpoint_units u WHERE u.job_id=control_jobs.job_id AND result_json IS NOT NULL) AS saved_progress_committed,
                (SELECT updated_at FROM job_checkpoints h WHERE h.job_id=control_jobs.job_id) AS saved_progress_at"""
        with self._connection() as conn:
            conn.execute("BEGIN")
            count = int(
                conn.execute(f"SELECT COUNT(*) FROM control_jobs {where}", params).fetchone()[0]
            )
            rows = conn.execute(
                f"SELECT {projection} FROM control_jobs {where} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                params + (limit, offset),
            ).fetchall()
        items = [self._as_job(row) for row in rows if row is not None]
        if saved_progress:
            from server.checkpoints import MEASUREMENT_TYPES

            for item in items:
                if item is not None:
                    item["saved_progress_unit"] = (
                        "测量组" if item["test_type"] in MEASUREMENT_TYPES else "样本"
                    )
        return items, count  # type: ignore[return-value]

    def events(self, job_id: str, *, limit: int = 100) -> JobList:
        if not 1 <= limit <= 500:
            raise ValueError("Invalid event limit")
        self.get(job_id)
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY id DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def claim(self, worker_id: str, *, lease_seconds: int = 60) -> dict[str, Any] | None:
        if not worker_id or lease_seconds < 5:
            raise ValueError("Invalid worker lease")
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM tokenizer_installs "
                "WHERE status IN ('downloading', 'validating', 'cancelling') LIMIT 1"
            ).fetchone():
                conn.commit()
                return None
            queued = conn.execute(
                """SELECT job.job_id FROM control_jobs AS job
                   LEFT JOIN control_batches AS batch ON batch.batch_id = job.parent_job_id
                   WHERE job.status = ? AND (
                       job.parent_job_id IS NULL OR (
                           SELECT COUNT(*) FROM control_jobs AS active
                           WHERE active.parent_job_id = job.parent_job_id
                           AND active.status IN ('running', 'pausing', 'paused', 'cancelling')
                       ) < COALESCE(batch.max_parallel, 1)
                   ) ORDER BY job.created_at, job.rowid LIMIT 1""",
                (RunStatus.QUEUED.value,),
            ).fetchone()
            if queued is None:
                conn.commit()
                return None
            job_id = str(queued["job_id"])
            running = advance_run(RunStatus.QUEUED, RunEvent.START).value
            conn.execute(
                """UPDATE control_jobs SET status = ?, lease_owner = ?, lease_until = ?,
                   attempts = attempts + 1, started_at = COALESCE(started_at, ?), updated_at = ?
                   WHERE job_id = ? AND status = ?""",
                (running, worker_id, now + lease_seconds, now, now, job_id, RunStatus.QUEUED.value),
            )
            self._event(conn, job_id, RunStatus.QUEUED.value, running, "start", worker_id, now)
            row = conn.execute("SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)).fetchone()
            conn.commit()
        return self._as_job(row)

    def heartbeat(self, job_id: str, worker_id: str, *, lease_seconds: int = 60) -> bool:
        now = time.time()
        with self._connection() as conn:
            cursor = conn.execute(
                """UPDATE control_jobs SET lease_until = ?, updated_at = ?
                   WHERE job_id = ? AND lease_owner = ? AND status IN (?, ?, ?, ?)""",
                (
                    now + lease_seconds,
                    now,
                    job_id,
                    worker_id,
                    RunStatus.RUNNING.value,
                    RunStatus.PAUSING.value,
                    RunStatus.PAUSED.value,
                    RunStatus.CANCELLING.value,
                ),
            )
        return cursor.rowcount == 1

    def update_progress(self, job_id: str, worker_id: str, *, completed: int, total: int) -> bool:
        if completed < 0 or total < 0:
            raise ValueError("Progress cannot be negative")
        with self._connection() as conn:
            cursor = conn.execute(
                """UPDATE control_jobs SET
                   progress_completed = MAX(progress_completed, ?),
                   progress_total = MAX(progress_total, ?), updated_at = ?
                   WHERE job_id = ? AND lease_owner = ? AND status IN (?, ?, ?, ?)""",
                (
                    completed,
                    total,
                    time.time(),
                    job_id,
                    worker_id,
                    RunStatus.RUNNING.value,
                    RunStatus.PAUSING.value,
                    RunStatus.PAUSED.value,
                    RunStatus.CANCELLING.value,
                ),
            )
        return cursor.rowcount == 1

    def sync_result_run(self, job_id: str, worker_id: str) -> int | None:
        """Expose durable observations while the measurement is still running."""
        with self._connection() as conn:
            row = conn.execute(
                """SELECT id, total_requests,
                   (SELECT COUNT(*) FROM test_results WHERE run_id = test_runs.id) AS recorded_requests
                   FROM test_runs WHERE test_id = ?""",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """UPDATE control_jobs SET result_run_id = ?,
                   progress_completed = MAX(progress_completed, ?),
                   progress_total = MAX(progress_total, ?), updated_at = ?
                   WHERE job_id = ? AND lease_owner = ? AND status IN (?, ?, ?, ?)""",
                (
                    row["id"],
                    row["recorded_requests"] or 0,
                    row["total_requests"] or 0,
                    time.time(),
                    job_id,
                    worker_id,
                    RunStatus.RUNNING.value,
                    RunStatus.PAUSING.value,
                    RunStatus.PAUSED.value,
                    RunStatus.CANCELLING.value,
                ),
            )
        return int(row["id"])

    def verify_persisted_run(self, job_id: str, run_id: int, expected_rows: int) -> None:
        """Reject success if the measured requests were not durably saved."""
        if expected_rows <= 0:
            raise RuntimeError("Measurement has no completed requests")
        with self._connection() as conn:
            row = conn.execute(
                """SELECT test_id, status,
                   (SELECT COUNT(*) FROM test_results WHERE run_id = test_runs.id) AS recorded
                   FROM test_runs WHERE id = ?""",
                (run_id,),
            ).fetchone()
        if row is None or row["test_id"] != job_id:
            raise RuntimeError("Measurement run is not linked to the claimed job")
        if row["status"] != RunStatus.COMPLETED.value:
            raise RuntimeError("Measurement run did not complete successfully")
        if row["recorded"] != expected_rows:
            raise RuntimeError(
                f"Measurement persistence mismatch: {row['recorded']} of {expected_rows} rows saved"
            )

    def request_pause(self, job_id: str, *, actor: str = "api") -> dict[str, Any]:
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            before = RunStatus(row["status"])
            if before in (RunStatus.PAUSING, RunStatus.PAUSED):
                conn.commit()
                return self._as_job(row)  # type: ignore[return-value]
            after = advance_run(before, RunEvent.REQUEST_PAUSE)
            if not row["lease_owner"] or row["lease_until"] is None or row["lease_until"] < now:
                raise LeaseLost(job_id)
            conn.execute(
                "UPDATE control_jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (after.value, now, job_id),
            )
            self._event(conn, job_id, before.value, after.value, "request_pause", actor, now)
            conn.commit()
        return self.get(job_id)

    def acknowledge_pause(self, job_id: str, worker_id: str) -> None:
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if (
                row is None
                or row["lease_owner"] != worker_id
                or row["lease_until"] is None
                or row["lease_until"] < now
            ):
                raise LeaseLost(job_id)
            # Cancellation may arrive while the worker is reaching this boundary.
            if row["status"] == RunStatus.CANCELLING.value:
                conn.commit()
                return
            if row["status"] == RunStatus.PAUSED.value:
                conn.commit()
                return
            after = advance_run(row["status"], RunEvent.PAUSE)
            conn.execute(
                """UPDATE control_jobs SET status = ?, pause_count = pause_count + 1,
                   pause_started_at = ?, updated_at = ? WHERE job_id = ?""",
                (after.value, now, now, job_id),
            )
            self._event(
                conn,
                job_id,
                row["status"],
                after.value,
                "pause",
                worker_id,
                now,
                {"boundary": "drained_request_group"},
            )
            conn.commit()

    def resume(self, job_id: str, *, actor: str = "api") -> dict[str, Any]:
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            after = advance_run(row["status"], RunEvent.RESUME)
            if not row["lease_owner"] or row["lease_until"] is None or row["lease_until"] < now:
                raise LeaseLost(job_id)
            paused = max(0.0, now - (row["pause_started_at"] or now))
            conn.execute(
                """UPDATE control_jobs SET status = ?, paused_seconds = paused_seconds + ?,
                   pause_started_at = NULL, updated_at = ? WHERE job_id = ?""",
                (after.value, paused, now, job_id),
            )
            self._event(
                conn,
                job_id,
                row["status"],
                after.value,
                "resume",
                actor,
                now,
                {"paused_seconds": paused},
            )
            conn.commit()
        return self.get(job_id)

    def request_cancel(self, job_id: str, *, actor: str = "api") -> dict[str, Any]:
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                conn.rollback()
                raise JobNotFound(job_id)
            before = RunStatus(row["status"])
            if before in (RunStatus.CANCELLING, RunStatus.CANCELLED):
                conn.commit()
                return self._as_job(row)  # type: ignore[return-value]
            event = RunEvent.CANCEL if before == RunStatus.QUEUED else RunEvent.REQUEST_CANCEL
            after = advance_run(before, event)
            terminal = after == RunStatus.CANCELLED
            conn.execute(
                """UPDATE control_jobs SET status = ?, updated_at = ?,
                   finished_at = CASE WHEN ? THEN ? ELSE finished_at END,
                   lease_owner = CASE WHEN ? THEN NULL ELSE lease_owner END,
                   lease_until = CASE WHEN ? THEN NULL ELSE lease_until END
                   WHERE job_id = ? AND status = ?""",
                (after.value, now, terminal, now, terminal, terminal, job_id, before.value),
            )
            self._event(conn, job_id, before.value, after.value, event.value, actor, now)
            updated = conn.execute(
                "SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            self._persist_execution_context(conn, updated)
            conn.commit()
        return self._as_job(updated)  # type: ignore[return-value]

    @staticmethod
    def _persist_execution_context(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
        run = conn.execute(
            "SELECT config_json FROM test_runs WHERE test_id = ?", (job["job_id"],)
        ).fetchone()
        if run is None:
            return
        try:
            config = json.loads(run["config_json"] or "{}")
        except (TypeError, ValueError):
            return
        if not isinstance(config, dict):
            return
        batch = conn.execute(
            "SELECT max_parallel, stop_on_error FROM control_batches WHERE batch_id = ?",
            (job["parent_job_id"],),
        ).fetchone()
        config["execution_control"] = {
            "pause_count": job["pause_count"],
            "paused_seconds": job["paused_seconds"],
            "pause_policy": config.get("pause_policy"),
            "batch_id": job["parent_job_id"],
            "max_parallel": batch["max_parallel"] if batch else 1,
            "stop_on_error": bool(batch["stop_on_error"]) if batch else False,
        }
        try:
            serialized = json.dumps(config, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            logger.warning(
                "Could not attach execution conditions to invalid run config: %s", job["job_id"]
            )
            return
        conn.execute(
            "UPDATE test_runs SET config_json = ? WHERE test_id = ?",
            (serialized, job["job_id"]),
        )

    def finish(
        self,
        job_id: str,
        worker_id: str,
        *,
        outcome: RunStatus,
        result_run_id: int | None = None,
        result_artifact: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                conn.rollback()
                raise JobNotFound(job_id)
            if (
                row["lease_owner"] != worker_id
                or row["lease_until"] is None
                or row["lease_until"] < now
            ):
                conn.rollback()
                raise LeaseLost(job_id)
            before = RunStatus(row["status"])
            if before == RunStatus.CANCELLING:
                event = RunEvent.CANCEL
            elif outcome == RunStatus.COMPLETED:
                event = RunEvent.COMPLETE
            elif outcome == RunStatus.CANCELLED:
                event = RunEvent.CANCEL
            elif outcome == RunStatus.FAILED:
                event = RunEvent.FAIL
            else:
                conn.rollback()
                raise ValueError("Invalid terminal outcome")
            after = advance_run(before, event)
            if after == RunStatus.CANCELLED and row["error_code"] == "BATCH_STOP_ON_ERROR":
                error_code = row["error_code"]
                error_message = row["error_message"]
            paused = max(0.0, now - row["pause_started_at"]) if row["pause_started_at"] else 0.0
            conn.execute(
                """UPDATE control_jobs SET status = ?,
                   result_run_id = COALESCE(?, result_run_id),
                   result_artifact = COALESCE(?, result_artifact),
                   error_code = ?, error_message = ?, lease_owner = NULL, lease_until = NULL,
                   paused_seconds = paused_seconds + ?, pause_started_at = NULL,
                   finished_at = ?, updated_at = ? WHERE job_id = ? AND status = ? AND lease_owner = ?""",
                (
                    after.value,
                    result_run_id,
                    result_artifact,
                    error_code,
                    (error_message or "")[:1000] or None,
                    paused,
                    now,
                    now,
                    job_id,
                    before.value,
                    worker_id,
                ),
            )
            self._event(conn, job_id, before.value, after.value, event.value, worker_id, now)
            failed_observation = (
                after == RunStatus.COMPLETED
                and conn.execute(
                    """SELECT 1 FROM test_results WHERE run_id = ?
                       AND error IS NOT NULL AND error != '' LIMIT 1""",
                    (result_run_id or row["result_run_id"],),
                ).fetchone()
                is not None
            )
            if after == RunStatus.FAILED or failed_observation:
                self._stop_batch_after_error(conn, row, now, worker_id)
            updated = conn.execute(
                "SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            self._persist_execution_context(conn, updated)
            conn.commit()
        return self._as_job(updated)  # type: ignore[return-value]

    def _stop_batch_after_error(
        self, conn: sqlite3.Connection, failed_job: sqlite3.Row, now: float, actor: str
    ) -> None:
        batch_id = failed_job["parent_job_id"]
        if (
            not batch_id
            or not conn.execute(
                "SELECT 1 FROM control_batches WHERE batch_id = ? AND stop_on_error = 1",
                (batch_id,),
            ).fetchone()
        ):
            return
        siblings = conn.execute(
            """SELECT job_id, status FROM control_jobs WHERE parent_job_id = ? AND job_id != ?
               AND status IN ('queued', 'running', 'pausing', 'paused')""",
            (batch_id, failed_job["job_id"]),
        ).fetchall()
        for sibling in siblings:
            before = RunStatus(sibling["status"])
            event = RunEvent.CANCEL if before == RunStatus.QUEUED else RunEvent.REQUEST_CANCEL
            after = advance_run(before, event)
            conn.execute(
                """UPDATE control_jobs SET status = ?, error_code = 'BATCH_STOP_ON_ERROR',
                   error_message = ?, updated_at = ?,
                   finished_at = CASE WHEN ? THEN ? ELSE finished_at END WHERE job_id = ?""",
                (
                    after.value,
                    "Stopped after another batch item failed",
                    now,
                    after == RunStatus.CANCELLED,
                    now,
                    sibling["job_id"],
                ),
            )
            self._event(
                conn,
                sibling["job_id"],
                before.value,
                after.value,
                event.value,
                actor,
                now,
                {"reason": "batch_stop_on_error", "failed_job_id": failed_job["job_id"]},
            )

    def reap_expired(self) -> int:
        """Mark abandoned measurements failed; never blend retries into a score."""
        now = time.time()
        recovered = 0
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT * FROM control_jobs WHERE status IN (?, ?, ?, ?)
                   AND lease_until < ?""",
                (
                    RunStatus.RUNNING.value,
                    RunStatus.PAUSING.value,
                    RunStatus.PAUSED.value,
                    RunStatus.CANCELLING.value,
                    now,
                ),
            ).fetchall()
            for row in rows:
                before = RunStatus(row["status"])
                event = RunEvent.CANCEL if before == RunStatus.CANCELLING else RunEvent.FAIL
                after = advance_run(before, event)
                conn.execute(
                    """UPDATE control_jobs SET status = ?, error_code = ?, error_message = ?,
                       lease_owner = NULL, lease_until = NULL, finished_at = ?, updated_at = ?
                       WHERE job_id = ? AND status = ?""",
                    (
                        after.value,
                        row["error_code"] if event == RunEvent.CANCEL else "WORKER_LOST",
                        row["error_message"]
                        if event == RunEvent.CANCEL
                        else "Worker lease expired; measurement is invalid",
                        now,
                        now,
                        row["job_id"],
                        before.value,
                    ),
                )
                self._event(
                    conn, row["job_id"], before.value, after.value, event.value, "reaper", now
                )
                if row["pause_started_at"]:
                    conn.execute(
                        """UPDATE control_jobs SET paused_seconds = paused_seconds + ?,
                           pause_started_at = NULL WHERE job_id = ?""",
                        (max(0.0, now - row["pause_started_at"]), row["job_id"]),
                    )
                recovered += 1
            for row in rows:
                if row["status"] != RunStatus.CANCELLING.value:
                    self._stop_batch_after_error(conn, row, now, "reaper")
                updated = conn.execute(
                    "SELECT * FROM control_jobs WHERE job_id = ?", (row["job_id"],)
                ).fetchone()
                self._persist_execution_context(conn, updated)
            conn.commit()
        return recovered
