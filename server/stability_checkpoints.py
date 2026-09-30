"""Durable continuous admission with independent clocks for every execution attempt."""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict

from core.models.test_result import TestResult
from server.checkpoints import MEASUREMENT_CONTRACT, CheckpointConflict, _decode, _encode, _hash
from server.measurement_checkpoints import MeasurementJournal
from server.time_series import _valid_window

REQUEST_PREFIX = "stability:request:"
WINDOW_PREFIX = "stability:window:"


def stored_windows(conn, job_id):
    rows = conn.execute(
        "SELECT metadata_json,metadata_sha256 FROM checkpoint_scopes WHERE job_id=? AND scope_key LIKE ?",
        (job_id, WINDOW_PREFIX + "%"),
    )
    windows = [_decode(row["metadata_json"], row["metadata_sha256"]) for row in rows]
    if any(type(window.get("attempt")) is not int or window["attempt"] < 1 for window in windows):
        raise CheckpointConflict("Continuous window attempt is invalid")
    if len({window["attempt"] for window in windows}) != len(windows):
        raise CheckpointConflict("Continuous window attempt was duplicated")
    return sorted(windows, key=lambda window: window["attempt"])


def budget(windows, planned):
    consumed = sum(min(w["admission_budget_seconds"], w["scheduling_seconds"]) for w in windows)
    remaining = max(0.0, planned - consumed)
    return {
        "planned_seconds": planned,
        "saved_scheduling_seconds": consumed,
        "remaining_seconds": 0.0 if remaining < 1e-9 else remaining,
        "scope": "saved_admission_time_across_attempts",
    }


class StabilityJournal(MeasurementJournal):
    def begin_stability(self, runner):
        self.planned_seconds = float(runner._db_run.config["duration_seconds"])
        self.prepare_scope(
            "stability:schedule",
            lambda: ({"planned_seconds": self.planned_seconds}, []),
            allow_empty=True,
        )
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._lease(conn)
            windows = stored_windows(conn, self.job["job_id"])
            for window in windows:
                if window["state"] in {"running", "paused"}:
                    window["state"] = "interrupted"
                    raw = _encode(window)
                    conn.execute(
                        "UPDATE checkpoint_scopes SET metadata_json=?,metadata_sha256=? WHERE job_id=? AND scope_key=?",
                        (
                            raw,
                            _hash(raw),
                            self.job["job_id"],
                            WINDOW_PREFIX + str(window["attempt"]),
                        ),
                    )
            units = conn.execute(
                "SELECT input_json,input_sha256,result_json,result_sha256 FROM checkpoint_units WHERE job_id=? AND scope_key LIKE ?",
                (self.job["job_id"], REQUEST_PREFIX + "%"),
            ).fetchall()
            plans = [(row, _decode(row["input_json"], row["input_sha256"])) for row in units]
            self.unknown = sorted(
                plan["session_start"] for row, plan in plans if row["result_json"] is None
            )
            self.next_session = max((plan["session_start"] for _, plan in plans), default=-1) + 1
            restored = []
            for row, _ in plans:
                if row["result_json"] is not None:
                    restored.extend(_decode(row["result_json"], row["result_sha256"])["results"])
            restored.sort(key=lambda row: row["session_id"])
            config = json.loads(
                conn.execute(
                    "SELECT config_json FROM test_runs WHERE id=?", (runner._db_run.id,)
                ).fetchone()[0]
            )
            config["stability_windows"] = windows
            config["stability_budget"] = budget(windows, self.planned_seconds)
            conn.execute(
                "UPDATE test_runs SET config_json=? WHERE id=?",
                (_encode(config), runner._db_run.id),
            )
            self._touch(conn)
            conn.commit()
        runner._db_run.config = config
        runner.total_requests = len(plans)
        for row in restored:
            self.originals[id(row)] = copy.deepcopy(row)
            runner._persisted_result_ids.add(id(row))
        self.minimum_requests = len(self.unknown)
        self.remaining = config["stability_budget"]["remaining_seconds"]
        return restored, self.remaining

    def session_id(self, index):
        return (
            self.unknown[index]
            if index < len(self.unknown)
            else self.next_session + index - len(self.unknown)
        )

    def freeze_request(self, runner, identifier, generate):
        self.admit()

        def plan():
            prompt, source = generate(identifier)
            return {}, [
                {
                    "kind": "stability_request",
                    "prompts": [prompt],
                    "sources": [source],
                    "session_start": identifier,
                    "concurrency": runner._db_run.config["concurrency"],
                    "max_tokens": runner._db_run.config["max_tokens"],
                    "warmup": False,
                }
            ]

        key = REQUEST_PREFIX + str(identifier)
        _, units = self.prepare_scope(key, plan)
        self.start(key, 0)
        return units[0]["prompts"][0], units[0]["sources"][0]

    def tag_request(self, row, identifier, prompt, source):
        if row.get("session_id") != identifier:
            raise CheckpointConflict("Continuous response membership changed")
        row["prompt_text"], row["prompt_source"] = prompt, source
        row.setdefault("extra_metrics", {})["measurement_checkpoint"] = {
            "contract": MEASUREMENT_CONTRACT,
            "group": REQUEST_PREFIX + str(identifier),
            "attempt": self.job["attempts"],
            "request_index": 0,
            "warmup": False,
        }

    def save_snapshot(self, runner, rows, snapshot):
        window = {
            **snapshot,
            "attempt": self.job["attempts"],
            "admission_budget_seconds": self.remaining,
        }
        if not _valid_window(window):
            raise CheckpointConflict("Continuous clock snapshot is invalid")
        key = WINDOW_PREFIX + str(self.job["attempts"])
        raw = _encode(window)
        db, run = runner._get_db_manager(), runner._db_run
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._lease(conn)
            previous = conn.execute(
                "SELECT metadata_json,metadata_sha256 FROM checkpoint_scopes WHERE job_id=? AND scope_key=?",
                (self.job["job_id"], key),
            ).fetchone()
            if previous:
                before = _decode(previous["metadata_json"], previous["metadata_sha256"])
                if (
                    before["id"] != window["id"]
                    or before["window_seconds"] > window["window_seconds"]
                ):
                    raise CheckpointConflict("Continuous clock origin or order changed")
                conn.execute(
                    "UPDATE checkpoint_scopes SET metadata_json=?,metadata_sha256=? WHERE job_id=? AND scope_key=?",
                    (raw, _hash(raw), self.job["job_id"], key),
                )
            else:
                conn.execute(
                    "INSERT INTO checkpoint_scopes VALUES (?,?,?,?,0)",
                    (self.job["job_id"], key, raw, _hash(raw)),
                )
            for row in rows:
                self._commit(
                    conn,
                    REQUEST_PREFIX + str(row["session_id"]),
                    0,
                    {
                        "results": [row],
                        "warmup": False,
                        "run_id": run.id,
                        "state": None,
                    },
                )
            if rows and db.results.insert_batch(
                [TestResult.from_api_result(run.id, row) for row in rows], connection=conn
            ) != len(rows):
                raise CheckpointConflict("Incomplete continuous observation persistence")
            windows = stored_windows(conn, self.job["job_id"])
            config = json.loads(
                conn.execute("SELECT config_json FROM test_runs WHERE id=?", (run.id,)).fetchone()[
                    0
                ]
            )
            config.update(
                stability_windows=windows, stability_budget=budget(windows, self.planned_seconds)
            )
            count = conn.execute(
                "SELECT COUNT(*) FROM test_results WHERE run_id=?", (run.id,)
            ).fetchone()[0]
            total = conn.execute(
                "SELECT COUNT(*) FROM checkpoint_units WHERE job_id=? AND scope_key LIKE ?",
                (self.job["job_id"], REQUEST_PREFIX + "%"),
            ).fetchone()[0]
            conn.execute(
                """UPDATE test_runs SET config_json=?,total_requests=?,
                completed_requests=(SELECT COUNT(*) FROM test_results WHERE run_id=? AND (error IS NULL OR error='')),
                failed_requests=(SELECT COUNT(*) FROM test_results WHERE run_id=? AND error IS NOT NULL AND error!='') WHERE id=?""",
                (_encode(config), total, run.id, run.id, run.id),
            )
            conn.execute(
                "UPDATE control_jobs SET progress_completed=?,progress_total=? WHERE job_id=?",
                (count, total, self.job["job_id"]),
            )
            self._touch(conn)
            conn.commit()
        run.config = config
        runner.total_requests = total
        runner._persisted_result_ids.update(id(row) for row in rows)


def verify_stability_state(conn, job):
    windows = stored_windows(conn, job["job_id"])
    run = conn.execute("SELECT * FROM test_runs WHERE test_id=?", (job["job_id"],)).fetchone()
    config = json.loads(run["config_json"])
    if not windows or windows != config.get("stability_windows"):
        raise CheckpointConflict("Continuous window provenance changed")
    planned = config.get("duration_seconds")
    if (
        isinstance(planned, bool)
        or not isinstance(planned, (int, float))
        or not 0 < planned <= 3600
    ):
        raise CheckpointConflict("Continuous duration is invalid")
    submitted = json.loads(job["parameters_json"])["duration_seconds"]
    if planned != submitted:
        raise CheckpointConflict("Continuous duration differs from the submitted plan")
    by_window = defaultdict(list)
    for row in conn.execute("SELECT extra_metrics FROM test_results WHERE run_id=?", (run["id"],)):
        timing = json.loads(row["extra_metrics"])["timing_observation"]
        by_window[timing["id"]].append(timing)
    ids = set()
    remaining = planned
    for window in windows:
        if (
            not _valid_window(window)
            or window["id"] in ids
            or window["planned_seconds"] != planned
            or not math.isclose(window.get("admission_budget_seconds", -1), remaining, abs_tol=1e-9)
        ):
            raise CheckpointConflict("Continuous clock window is invalid or duplicated")
        ids.add(window["id"])
        remaining = max(
            0, remaining - min(window["admission_budget_seconds"], window["scheduling_seconds"])
        )
        members = by_window[window["id"]]
        if (
            len(members) != window["recorded_requests"]
            or len({m["index"] for m in members}) != len(members)
            or any(
                type(m["index"]) is not int
                or not 0 <= m["index"] < window["expected_requests"]
                or not 0 <= m["start_seconds"] <= m["end_seconds"] <= window["window_seconds"]
                for m in members
            )
        ):
            raise CheckpointConflict("Continuous clock membership changed")
    if set(by_window) - ids:
        raise CheckpointConflict("Continuous observation has no original clock window")
    if config.get("stability_budget") != budget(windows, planned):
        raise CheckpointConflict("Continuous scheduling budget changed")
    if budget(windows, planned)["saved_scheduling_seconds"] > planned + 1e-9:
        raise CheckpointConflict("Continuous scheduling budget exceeds the submitted duration")
