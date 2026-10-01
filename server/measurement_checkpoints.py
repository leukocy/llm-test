"""Replay complete measured groups with their original clocks and atomic observations."""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime

from core.models.test_result import TestResult
from core.models.test_run import TestRun
from server.checkpoints import MEASUREMENT_CONTRACT, CheckpointConflict, JobJournal


class MeasurementJournal(JobJournal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ordinal = 0
        self.pending: list[dict] = []
        self.originals: dict[int, dict] = {}
        self.sources: dict[int, dict] = {}

    def bind_run(self, db, *, test_type, model_id, provider, config, system_info):
        """Create/reopen the original run, fenced by both lease and execution attempt."""
        if db.db.path.resolve() != self.store.path.resolve():
            raise CheckpointConflict("Measurement and checkpoint databases differ")
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._lease(conn)
            row = conn.execute(
                "SELECT * FROM test_runs WHERE test_id=?", (self.job["job_id"],)
            ).fetchone()
            if row:
                if row["test_type"] != test_type or row["model_id"] != model_id:
                    raise CheckpointConflict("Original measurement identity changed")
                run = TestRun.from_row(dict(row))
                if run.config.get("tokenizer_installation") != config.get("tokenizer_installation"):
                    raise CheckpointConflict("Installed tokenizer provenance changed")
                # This path is reached only after explicit control-plane recovery and claim.
                run.status = "running"
                run.completed_at = None
                conn.execute(
                    "UPDATE test_runs SET status='running', completed_at=NULL WHERE id=?", (run.id,)
                )
            else:
                run = TestRun.create(
                    test_type=test_type,
                    model_id=model_id,
                    provider=provider,
                    test_id=self.job["job_id"],
                )
                run.config, run.system_info = config, system_info
                data = run.to_dict()
                fields = [key for key, value in data.items() if value is not None and key != "id"]
                run.id = conn.execute(
                    f"INSERT INTO test_runs ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                    [data[key] for key in fields],
                ).lastrowid
            conn.execute(
                "UPDATE control_jobs SET result_run_id=? WHERE job_id=?",
                (run.id, self.job["job_id"]),
            )
            conn.commit()
        return run

    def restore_capacity(self, runner):
        metadata, _ = self.prepare_scope(
            "measurement:capacity",
            lambda: (
                {"kv_budget": runner._kv_budget, "kv_budget_source": runner._kv_budget_source},
                [],
            ),
            allow_empty=True,
        )
        runner._kv_budget, runner._kv_budget_source = (
            metadata["kv_budget"],
            metadata["kv_budget_source"],
        )

    async def measure(
        self,
        runner,
        kind: str,
        prompts: list[str],
        sources: list[str],
        *,
        concurrency: int,
        max_tokens: int,
        session_start: int,
        warmup: bool,
        observe,
        capture_state=None,
        restore_state=None,
    ):
        key = f"measurement:{self.ordinal}"
        self.ordinal += 1
        metadata, units = self.prepare_scope(
            key,
            lambda: (
                {},
                [
                    {
                        "kind": kind,
                        "prompts": prompts,
                        "sources": sources,
                        "concurrency": concurrency,
                        "max_tokens": max_tokens,
                        "session_start": session_start,
                        "warmup": warmup,
                    }
                ],
            ),
        )
        plan = units[0]
        for name, expected in {
            "kind": kind,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "session_start": session_start,
            "warmup": warmup,
        }.items():
            if plan[name] != expected:
                raise CheckpointConflict("Measurement group order or workload changed")
        if len(plan["prompts"]) != len(prompts):
            raise CheckpointConflict("Measurement group membership changed")
        saved = self.results(key).get(0)
        if saved is not None:
            results = saved["results"]
            if len(results) != len(prompts) or saved["warmup"] != warmup:
                raise CheckpointConflict("Saved measurement group is incomplete")
            if restore_state is not None:
                restore_state(saved.get("state") or {})
            for result in results:
                if not warmup:
                    self.originals[id(result)] = copy.deepcopy(result)
                    runner._persisted_result_ids.add(id(result))
            if kind == "continuous" and not warmup:
                runner.completed_requests += len(results)
            return results
        self.start(key, 0)
        results = await observe(plan["prompts"])
        if any(isinstance(row, dict) and row.get("error") == "UserCancelled" for row in results):
            raise asyncio.CancelledError("Stopped measurement group remains uncommitted")
        if len(results) != len(prompts) or any(not isinstance(row, dict) for row in results):
            raise CheckpointConflict("Incomplete measurement group remains uncommitted")
        by_session = {row["session_id"]: row for row in results}
        if set(by_session) != set(range(session_start, session_start + len(prompts))):
            raise CheckpointConflict("Measurement response membership changed")
        # Completion order must not change source attribution or round assignment.
        results = [by_session[session_start + i] for i in range(len(prompts))]
        for index, result in enumerate(results):
            if not warmup:
                result["concurrency"] = plan["concurrency"]
                result["request_index"] = session_start + index
            self.sources[id(result)] = {
                "prompt_text": plan["prompts"][index],
                "prompt_source": plan["sources"][index],
            }
            result.setdefault("extra_metrics", {})["measurement_checkpoint"] = {
                "contract": MEASUREMENT_CONTRACT,
                "group": key,
                "attempt": self.job["attempts"],
                "request_index": index,
                "warmup": warmup,
            }
        self.pending.append(
            {
                "key": key,
                "results": results,
                "warmup": warmup,
                "state": copy.deepcopy(capture_state()) if capture_state is not None else None,
            }
        )
        return results

    def admit(self):
        with self.store._connection() as conn:
            self._lease(conn)
            status = conn.execute(
                "SELECT status FROM control_jobs WHERE job_id=?", (self.job["job_id"],)
            ).fetchone()[0]
            if status == "cancelling":
                raise asyncio.CancelledError
            if status != "running" and status != "pausing":
                raise CheckpointConflict("Cannot admit a measurement batch in this state")

    def restore_fields(self, result):
        original = self.originals.get(id(result))
        if original is not None:
            result.clear()
            result.update(copy.deepcopy(original))
        elif id(result) in self.sources:
            result.update(self.sources[id(result)])

    def flush(self, runner):
        if runner._db_run is None:
            raise CheckpointConflict("Measurement has no original run")
        db, run = runner._get_db_manager(), runner._db_run
        visible = {id(row) for row in runner.results_list}
        while self.pending:
            group = self.pending[0]
            results = group["results"]
            if not group["warmup"] and any(id(row) not in visible for row in results):
                raise CheckpointConflict("Measurement group was not fully incorporated")
            for row in results:
                self.restore_fields(row)
            with self.store._connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._lease(conn)
                self._commit(
                    conn,
                    group["key"],
                    0,
                    {
                        "results": results,
                        "warmup": group["warmup"],
                        "run_id": run.id,
                        "state": group["state"],
                    },
                )
                if not group["warmup"]:
                    models = [TestResult.from_api_result(run.id, row) for row in results]
                    if db.results.insert_batch(models, connection=conn) != len(results):
                        raise CheckpointConflict("Incomplete atomic observation persistence")
                conn.commit()
            runner._persisted_result_ids.update(id(row) for row in results if not group["warmup"])
            if group["warmup"]:
                # Warmup objects are discarded after this commit; their ids may be reused.
                for row in results:
                    self.originals.pop(id(row), None)
                    self.sources.pop(id(row), None)
            self.pending.pop(0)
        # Adaptive skip rows have no API call; retain them transactionally as local evidence.
        extras = [row for row in runner.results_list if id(row) not in runner._persisted_result_ids]
        if extras:
            if any(not str(row.get("error", "")).startswith("over_kv_budget") for row in extras):
                raise CheckpointConflict("Untracked measurement observation")
            with self.store._connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._lease(conn)
                # Verify the entire fixed prefix, rather than assuming a count proves identity.
                stored_skips = conn.execute(
                    "SELECT * FROM test_results WHERE run_id=? AND error LIKE 'over_kv_budget%' ORDER BY id",
                    (run.id,),
                ).fetchall()
                all_skips = [
                    row
                    for row in runner.results_list
                    if str(row.get("error", "")).startswith("over_kv_budget")
                ]
                for stored, replayed in zip(stored_skips, all_skips, strict=False):
                    expected = TestResult.from_api_result(run.id, replayed).to_dict()
                    for name in (
                        "concurrency_level",
                        "context_length_target",
                        "input_tokens_target",
                        "round",
                        "error",
                    ):
                        if stored[name] != expected[name]:
                            raise CheckpointConflict("Stored adaptive skip workload changed")
                fresh = all_skips[len(stored_skips) :]
                db.results.insert_batch(
                    [TestResult.from_api_result(run.id, row) for row in fresh], connection=conn
                )
                conn.commit()
            runner._persisted_result_ids.update(id(row) for row in extras)
        runner.completed_requests = len(runner.results_list)
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._lease(conn)
            count = conn.execute(
                "SELECT COUNT(*) FROM test_results WHERE run_id=?", (run.id,)
            ).fetchone()[0]
            conn.execute(
                "UPDATE control_jobs SET progress_completed=?,progress_total=? WHERE job_id=?",
                (count, runner.total_requests, self.job["job_id"]),
            )
            conn.execute(
                "UPDATE test_runs SET completed_requests=?,total_requests=? WHERE id=?",
                (count, runner.total_requests, run.id),
            )
            conn.commit()

    def finish_run(self, runner, success, extra_fields):
        self.flush(runner)
        db, run = runner._get_db_manager(), runner._db_run
        stats = db.results.get_aggregate_metrics(run.id)
        percentiles = db.results.get_percentiles(run.id, "ttft")
        fields = {
            key: stats[key]
            for key in (
                "avg_ttft",
                "avg_tps",
                "avg_tpot",
                "total_tokens",
                "total_requests",
                "completed_requests",
                "failed_requests",
                "success_rate",
            )
            if key in stats
        }
        fields.update({f"{name}_ttft": percentiles.get(name) for name in ("p50", "p95", "p99")})
        fields.update(extra_fields)
        checkpoint = self.describe()
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._lease(conn)
            stored = conn.execute(
                "SELECT config_json FROM test_runs WHERE id=?", (run.id,)
            ).fetchone()
            config = json.loads(stored[0] or "{}")
            protocol = config.get("measurement_protocol")
            if isinstance(protocol, dict):
                protocol.update(
                    warmup_recorded=runner._warmup_recorded, warmup_failures=runner._warmup_failures
                )
            config["measurement_recovery"] = {
                "contract": MEASUREMENT_CONTRACT,
                "checkpoint": checkpoint,
                "attempts": self.job["attempts"],
                "duration_scope": "independent_continuous_windows"
                if self.job["test_type"] == "stability"
                else "complete_groups_across_attempts",
                "plan_scope": "admitted_requests"
                if self.job["test_type"] == "stability"
                else "materialized_groups",
                "resource_scope": "current_attempt",
            }
            fields.update(
                status="completed" if success else "failed",
                completed_at=datetime.now().isoformat(),
                config_json=json.dumps(config, ensure_ascii=False, allow_nan=False),
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(test_runs)")}
            fields = {
                key: value
                for key, value in fields.items()
                if key in columns and key not in {"id", "test_id"} and value is not None
            }
            conn.execute(
                f"UPDATE test_runs SET {','.join(key + '=?' for key in fields)} WHERE id=?",
                [*fields.values(), run.id],
            )
            conn.commit()
        run.config = config


def verify_measurement_observations(conn, job):
    """Verify committed group membership and immutable DB observations before requeue."""
    run = conn.execute("SELECT * FROM test_runs WHERE test_id=?", (job["job_id"],)).fetchone()
    if run is None:
        raise CheckpointConflict("Original measurement run is missing")
    expected_rows = {}
    for unit in conn.execute(
        "SELECT * FROM checkpoint_units WHERE job_id=? AND result_json IS NOT NULL",
        (job["job_id"],),
    ):
        payload = json.loads(unit["result_json"])
        plan = json.loads(unit["input_json"])
        if (
            payload.get("run_id") != run["id"]
            or payload.get("warmup") != plan.get("warmup")
            or not isinstance(payload.get("results"), list)
            or len(payload["results"]) != len(plan.get("prompts", []))
        ):
            raise CheckpointConflict("Saved measurement group is incomplete or misbound")
        if payload["warmup"]:
            continue
        for index, result in enumerate(payload["results"]):
            expected_rows[(unit["scope_key"], index)] = TestResult.from_api_result(
                run["id"], result
            ).to_dict()
    seen = set()
    for row in conn.execute("SELECT * FROM test_results WHERE run_id=?", (run["id"],)):
        if str(row["error"] or "").startswith("over_kv_budget"):
            continue
        try:
            marker = json.loads(row["extra_metrics"] or "{}")["measurement_checkpoint"]
            key = (marker["group"], marker["request_index"])
            expected = expected_rows[key]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointConflict("Persisted measurement membership changed") from exc
        if key in seen:
            raise CheckpointConflict("Persisted measurement observation was duplicated")
        seen.add(key)
        for name, value in expected.items():
            if name in {"id", "created_at"}:
                continue
            actual = row[name]
            if name == "extra_metrics":
                actual = json.loads(actual or "{}")
                value = json.loads(value or "{}")
            if actual != value:
                raise CheckpointConflict("Persisted measurement observation changed")
    if seen != set(expected_rows):
        raise CheckpointConflict("Committed measurement observations are missing")
