"""Lease-fenced evaluation plans and committed samples; JSON only, no executable state."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.run_lifecycle import RunEvent, RunStatus, advance_run
from server.settings import Endpoint
from server.store import JobNotFound, JobStore, LeaseLost

CONTRACT = "evaluation-checkpoint-v1"
RECOVERABLE_ERRORS = {"WORKER_LOST", "INTERRUPTED"}
MEASUREMENT_TYPES = {"concurrency", "matrix", "custom_text", "dataset"}
MEASUREMENT_CONTRACT = "measurement-group-checkpoint-v1"
SUPPORTED_TYPES = {"quality", "robustness"} | MEASUREMENT_TYPES


def checkpoint_contract(test_type: str) -> str:
    return MEASUREMENT_CONTRACT if test_type in MEASUREMENT_TYPES else CONTRACT


MAX_BYTES = 16 * 1024 * 1024


def recovery_filter() -> tuple[str, tuple[str, ...]]:
    """Candidate eligibility only; recovery performs full provenance verification."""
    types = sorted(SUPPORTED_TYPES)
    errors = sorted(RECOVERABLE_ERRORS)
    measurement_types = sorted(MEASUREMENT_TYPES)
    return (
        "test_type IN ("
        + ",".join("?" for _ in types)
        + ")"
        + " AND ((status = 'failed' AND error_code IN ("
        + ",".join("?" for _ in errors)
        + ")) OR (status = 'cancelled' AND error_code IS NULL))"
        + " AND EXISTS (SELECT 1 FROM job_checkpoints h WHERE h.job_id = control_jobs.job_id AND h.contract = CASE WHEN control_jobs.test_type IN ("
        + ",".join("?" for _ in measurement_types)
        + ") THEN ? ELSE ? END)"
        + " AND EXISTS (SELECT 1 FROM checkpoint_units u WHERE u.job_id = control_jobs.job_id)",
        (*types, *errors, *measurement_types, MEASUREMENT_CONTRACT, CONTRACT),
    )


def resumable_state(job: Any) -> bool:
    return (
        job["status"] == "failed"
        and job["error_code"] in RECOVERABLE_ERRORS
        or job["status"] == "cancelled"
        and job["error_code"] is None
    )


class CheckpointConflict(ValueError):
    pass


def _encode(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(encoded.encode("utf-8")) > MAX_BYTES:
        raise CheckpointConflict("Checkpoint unit exceeds the supported size")
    return encoded


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decode(raw: str, digest: str) -> Any:
    if _hash(raw) != digest:
        raise CheckpointConflict("Checkpoint contents failed integrity verification")
    if len(raw.encode("utf-8")) > MAX_BYTES:
        raise CheckpointConflict("Checkpoint unit exceeds the supported size")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise CheckpointConflict("Checkpoint JSON is invalid") from exc
    if not isinstance(value, dict):
        raise CheckpointConflict("Checkpoint unit must be a JSON object")
    return value


def execution_signature(job: dict, endpoint: Endpoint) -> str:
    """Pin model identity, parameters and the code that creates/evaluates responses.

    Credentials are excluded so rotation can preserve a plan. Worker identity and
    restart timing remain audit events, rather than being presented as continuous load.
    """
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = [
        *root.joinpath("core").rglob("*.py"),
        *root.joinpath("evaluators").rglob("*.py"),
        *root.joinpath("config").rglob("*.py"),
        *root.joinpath("utils").rglob("*.py"),
        *root.joinpath("task_configs").rglob("*.yaml"),
        *root.joinpath("task_configs").rglob("*.yml"),
        Path(__file__),
        root / "server/runner_adapter.py",
        root / "server/measurement_checkpoints.py",
    ]
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return _hash(
        _encode(
            {
                "contract": checkpoint_contract(job["test_type"]),
                "code_sha256": digest.hexdigest(),
                "test_type": job["test_type"],
                "parameters": job["parameters"],
                "endpoint_id": endpoint.id,
                "provider": endpoint.provider,
                "url": endpoint.api_base_url,
                "model_id": endpoint.model_id,
                "tokenizer": endpoint.tokenizer_option,
            }
        )
    )


def _read_checkpoint(conn, job_id: str):
    job = conn.execute("SELECT * FROM control_jobs WHERE job_id = ?", (job_id,)).fetchone()
    if job is None:
        raise JobNotFound(job_id)
    header = conn.execute("SELECT * FROM job_checkpoints WHERE job_id = ?", (job_id,)).fetchone()
    counts = conn.execute(
        """SELECT COUNT(*) AS planned, COALESCE(SUM(result_json IS NOT NULL),0) AS committed,
        COALESCE(SUM(issued_attempts),0) AS issued,
        COALESCE(SUM(MAX(0,issued_attempts-1)),0) AS repeated
        FROM checkpoint_units WHERE job_id = ?""",
        (job_id,),
    ).fetchone()
    revision = (
        _hash(
            _encode(
                {
                    "header": dict(header),
                    "counts": dict(counts),
                    "job_updated_at": job["updated_at"],
                    "status": job["status"],
                }
            )
        )
        if header
        else None
    )
    return job, header, counts, revision


def checkpoint_info(store: JobStore, job: dict) -> dict[str, Any]:
    with store._connection() as conn:
        conn.execute("BEGIN")
        current, header, counts, revision = _read_checkpoint(conn, job["job_id"])
    return {
        "contract": checkpoint_contract(current["test_type"]),
        "unit_label": "测量组" if current["test_type"] in MEASUREMENT_TYPES else "样本",
        "supported": current["test_type"] in SUPPORTED_TYPES,
        "available": bool(header),
        "revision": revision,
        "saved_at": header["updated_at"] if header else None,
        "planned_units": counts["planned"],
        "committed_units": counts["committed"],
        "issued_unit_attempts": counts["issued"],
        "repeated_unit_attempts": counts["repeated"],
        "recoveries": header["recoveries"] if header else 0,
        "can_delete": bool(header and current["status"] in {"failed", "cancelled", "completed"}),
        "can_recover": bool(
            header
            and header["contract"] == checkpoint_contract(current["test_type"])
            and counts["planned"]
            and resumable_state(current)
            and current["test_type"] in SUPPORTED_TYPES
        ),
        "notes": [
            "测量恢复按完整请求组复用原计时和观测；未提交组可能整组重发，预热复用不证明服务端缓存仍然有效。",
            "每组实际提示词在请求前冻结；尚未生成的后续组将按原配置生成。",
            "跨中断组仅可分别解释吞吐，合并样本不能证明连续负载能力。",
        ]
        if current["test_type"] in MEASUREMENT_TYPES
        else [
            "恢复会复用已提交样本，保留原始样本顺序、few-shot 和扰动计划，并核验模型、参数与执行代码。",
            "中断时尚未提交的样本可能再次调用接口；已发起次数不是接口已返回响应数，也不能证明远端只执行一次。",
            "含恢复样本的结果跨越执行中断；当前执行时长只描述本次尝试，不能作为连续吞吐或不中断稳定性的证明。",
        ],
    }


def delete_checkpoint(store: JobStore, job: dict, revision: str) -> dict[str, Any]:
    """Delete terminal saved progress, retaining observations, reports and audit events."""
    with store._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current, header, counts, expected = _read_checkpoint(conn, job["job_id"])
        if current["status"] not in {"failed", "cancelled", "completed"}:
            raise CheckpointConflict("正在排队或执行的任务不能删除保存进度")
        if header:
            if revision != expected:
                raise CheckpointConflict("保存进度已变更，请重新读取后删除")
            now = time.time()
            conn.execute("DELETE FROM job_checkpoints WHERE job_id = ?", (job["job_id"],))
            conn.execute(
                "UPDATE control_jobs SET updated_at = ? WHERE job_id = ?", (now, job["job_id"])
            )
            store._event(
                conn,
                job["job_id"],
                current["status"],
                current["status"],
                "delete_checkpoint",
                "api",
                now,
                {
                    "planned_units": counts["planned"],
                    "committed_units": counts["committed"],
                    "revision": revision,
                },
            )
        conn.commit()
    return checkpoint_info(store, job)


class JobJournal:
    def __init__(self, store: JobStore, job: dict, worker_id: str, endpoint: Endpoint) -> None:
        self.store, self.job, self.worker_id = store, job, worker_id
        self.signature = execution_signature(job, endpoint)
        with store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._lease(conn)
            previous = conn.execute(
                "SELECT * FROM job_checkpoints WHERE job_id = ?", (job["job_id"],)
            ).fetchone()
            if previous and (
                previous["contract"] != checkpoint_contract(job["test_type"])
                or previous["signature"] != self.signature
            ):
                raise CheckpointConflict(
                    "Model, parameters or execution code changed; checkpoint cannot be reused"
                )
            now = time.time()
            conn.execute(
                "INSERT OR IGNORE INTO job_checkpoints(job_id,contract,signature,created_at,updated_at) VALUES (?,?,?,?,?)",
                (job["job_id"], checkpoint_contract(job["test_type"]), self.signature, now, now),
            )
            self._touch(conn)
            conn.commit()

    def _touch(self, conn) -> None:
        conn.execute(
            "UPDATE job_checkpoints SET updated_at=? WHERE job_id=?",
            (time.time(), self.job["job_id"]),
        )

    def _lease(self, conn) -> None:
        if not conn.execute(
            """SELECT 1 FROM control_jobs WHERE job_id = ? AND lease_owner = ?
            AND lease_until >= ? AND attempts = ? AND status IN ('running','pausing','paused','cancelling')""",
            (self.job["job_id"], self.worker_id, time.time(), self.job["attempts"]),
        ).fetchone():
            raise LeaseLost(self.job["job_id"])

    def prepare_scope(
        self, key: str, factory: Callable[[], tuple[dict, list[dict]]], *, allow_empty: bool = False
    ) -> tuple[dict, list[dict]]:
        with self.store._connection() as conn:
            conn.execute("BEGIN")
            self._lease(conn)
            existing = conn.execute(
                "SELECT * FROM checkpoint_scopes WHERE job_id=? AND scope_key=?",
                (self.job["job_id"], key),
            ).fetchone()
            if existing:
                metadata = _decode(existing["metadata_json"], existing["metadata_sha256"])
                rows = conn.execute(
                    "SELECT unit_index,input_json,input_sha256 FROM checkpoint_units WHERE job_id=? AND scope_key=? ORDER BY unit_index",
                    (self.job["job_id"], key),
                ).fetchall()
                if [r["unit_index"] for r in rows] != list(range(existing["units"])):
                    raise CheckpointConflict("Frozen checkpoint sample membership changed")
                return metadata, [_decode(r["input_json"], r["input_sha256"]) for r in rows]
        metadata, inputs = factory()
        raw_metadata = _encode(metadata)
        encoded = [_encode(unit) for unit in inputs]
        if not encoded and not allow_empty:
            raise CheckpointConflict("Cannot checkpoint an empty sample plan")
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._lease(conn)
            conn.execute(
                "INSERT INTO checkpoint_scopes VALUES (?,?,?,?,?)",
                (self.job["job_id"], key, raw_metadata, _hash(raw_metadata), len(encoded)),
            )
            conn.executemany(
                "INSERT INTO checkpoint_units(job_id,scope_key,unit_index,input_json,input_sha256) VALUES (?,?,?,?,?)",
                [(self.job["job_id"], key, i, raw, _hash(raw)) for i, raw in enumerate(encoded)],
            )
            self._touch(conn)
            conn.commit()
        return metadata, inputs

    def results(self, key: str) -> dict[int, dict[str, Any]]:
        with self.store._connection() as conn:
            self._lease(conn)
            rows = conn.execute(
                "SELECT unit_index,result_json,result_sha256 FROM checkpoint_units WHERE job_id=? AND scope_key=? AND result_json IS NOT NULL",
                (self.job["job_id"], key),
            ).fetchall()
        return {r["unit_index"]: _decode(r["result_json"], r["result_sha256"]) for r in rows}

    def start(self, key: str, index: int) -> None:
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._lease(conn)
            updated = conn.execute(
                """UPDATE checkpoint_units SET issued_attempts=issued_attempts+1, issued_by_attempt=?
                WHERE job_id=? AND scope_key=? AND unit_index=? AND result_json IS NULL
                AND (issued_by_attempt IS NULL OR issued_by_attempt != ?)""",
                (self.job["attempts"], self.job["job_id"], key, index, self.job["attempts"]),
            )
            if updated.rowcount != 1:
                raise CheckpointConflict("Sample is absent or already committed")
            self._touch(conn)
            conn.commit()

    def _commit(self, conn, key: str, index: int, result: dict[str, Any]) -> None:
        raw = _encode(result)
        self._lease(conn)
        updated = conn.execute(
            """UPDATE checkpoint_units SET result_json=?,result_sha256=?,committed_at=?
            WHERE job_id=? AND scope_key=? AND unit_index=? AND result_json IS NULL AND issued_by_attempt=?""",
            (raw, _hash(raw), time.time(), self.job["job_id"], key, index, self.job["attempts"]),
        )
        if updated.rowcount != 1:
            raise CheckpointConflict(
                "Sample was not issued by this attempt or is already committed"
            )
        self._touch(conn)

    def commit(self, key: str, index: int, result: dict[str, Any]) -> None:
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._commit(conn, key, index, result)
            conn.commit()

    def describe(self) -> dict[str, Any]:
        return checkpoint_info(self.store, self.store.get(self.job["job_id"]))


def recover_job(store: JobStore, job: dict, endpoint: Endpoint) -> dict[str, Any]:
    signature = execution_signature(job, endpoint)
    with store._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            "SELECT * FROM control_jobs WHERE job_id=?", (job["job_id"],)
        ).fetchone()
        saved = conn.execute(
            "SELECT * FROM job_checkpoints WHERE job_id=?", (job["job_id"],)
        ).fetchone()
        if (
            not saved
            or saved["contract"] != checkpoint_contract(job["test_type"])
            or saved["signature"] != signature
        ):
            raise CheckpointConflict(
                "Model, parameters or execution code changed; checkpoint cannot be recovered"
            )
        if current["status"] == "queued" and saved["recoveries"]:
            conn.commit()
            return store.get(job["job_id"])
        if not resumable_state(current) or current["test_type"] not in SUPPORTED_TYPES:
            raise CheckpointConflict(
                "Only interrupted or explicitly stopped evaluations with a saved plan can be recovered"
            )
        scopes = conn.execute(
            "SELECT * FROM checkpoint_scopes WHERE job_id=?", (job["job_id"],)
        ).fetchall()
        if not scopes:
            raise CheckpointConflict("No frozen sample plan is available")
        for scope in scopes:
            _decode(scope["metadata_json"], scope["metadata_sha256"])
            rows = conn.execute(
                "SELECT * FROM checkpoint_units WHERE job_id=? AND scope_key=? ORDER BY unit_index",
                (job["job_id"], scope["scope_key"]),
            ).fetchall()
            if [r["unit_index"] for r in rows] != list(range(scope["units"])):
                raise CheckpointConflict("Frozen checkpoint sample membership changed")
            for row in rows:
                _decode(row["input_json"], row["input_sha256"])
                if row["result_json"] is not None:
                    _decode(row["result_json"], row["result_sha256"])
        if current["test_type"] in MEASUREMENT_TYPES:
            from server.measurement_checkpoints import verify_measurement_observations

            verify_measurement_observations(conn, current)
        now = time.time()
        before = RunStatus(current["status"])
        after = advance_run(before, RunEvent.RECOVER)
        conn.execute(
            """UPDATE control_jobs SET status=?,finished_at=NULL,error_code=NULL,error_message=NULL,
            lease_owner=NULL,lease_until=NULL,updated_at=? WHERE job_id=?""",
            (after.value, now, job["job_id"]),
        )
        conn.execute(
            "UPDATE job_checkpoints SET recoveries=recoveries+1,updated_at=? WHERE job_id=?",
            (now, job["job_id"]),
        )
        store._event(
            conn,
            job["job_id"],
            before.value,
            after.value,
            "recover",
            "api",
            now,
            {"contract": checkpoint_contract(current["test_type"])},
        )
        conn.commit()
    return store.get(job["job_id"])
