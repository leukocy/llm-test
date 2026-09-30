"""Real engine/SQLite recovery preserves complete groups, clocks, and atomicity."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.benchmark_runner import BenchmarkRunner
from core.cancel_state import reset_all
from core.database.connection import Database
from core.repositories.test_result import TestResultRepository as ResultRepository
from core.repositories.test_run import TestRunRepository as RunRepository
from core.run_lifecycle import RunStatus
from server.api import create_app
from server.checkpoints import MEASUREMENT_CONTRACT, checkpoint_info, recover_job
from server.measurement_checkpoints import MeasurementJournal
from server.runner_adapter import execute_job
from server.settings import Endpoint, Settings
from server.store import JobStore, LeaseLost


@pytest.fixture
def lab(tmp_path, monkeypatch):
    reset_all()
    store = JobStore(tmp_path / "measurements.db")
    isolated = object.__new__(Database)
    isolated._init(str(store.path))
    db = SimpleNamespace(
        db=isolated, results=ResultRepository(isolated), runs=RunRepository(isolated)
    )
    endpoint = Endpoint(
        "lab",
        "Synthetic",
        "OpenAI",
        "http://127.0.0.1:9010/v1",
        "m",
        "TEST_KEY",
        api_key_value="synthetic",  # pragma: allowlist secret (test fixture)
    )  # pragma: allowlist secret
    settings = Settings(
        api_token="measurement-synthetic-" + "a" * 32,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={"lab": endpoint},
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BenchmarkRunner, "_get_db_manager", lambda _: db)
    monkeypatch.setattr(BenchmarkRunner, "_apply_seed", lambda _: None)
    monkeypatch.setattr(BenchmarkRunner, "_get_tokenizer", lambda _: None)
    monkeypatch.setattr(BenchmarkRunner, "_start_resource_monitor", lambda _: None)
    monkeypatch.setattr(BenchmarkRunner, "_start_engine_poller", lambda _: None)
    monkeypatch.setattr(BenchmarkRunner, "_probe_kv_budget", lambda _: None)
    monkeypatch.setattr(BenchmarkRunner, "_finalize_system_info", lambda _: None)
    monkeypatch.setattr(BenchmarkRunner, "_build_warehouse_extra_fields", lambda *_: {})
    monkeypatch.setattr("core.system_info.get_cached_system_info", lambda **_: {})
    counter = iter(range(1000))
    monkeypatch.setattr(
        BenchmarkRunner,
        "_calibrate_prompt_with_source",
        lambda *_: (f"original-{next(counter)}", "synthetic"),
    )
    yield store, endpoint, settings, db
    isolated.close()
    reset_all()


CASES = [
    (
        "concurrency",
        {
            "selected_concurrencies": [2],
            "rounds_per_level": 2,
            "max_tokens": 4,
            "input_tokens_target": 64,
            "warmup_rounds_per_level": 1,
        },
        2,
        4,
        2,
    ),
    (
        "matrix",
        {
            "concurrencies": [2],
            "context_lengths": [64, 128],
            "rounds": 1,
            "max_tokens": 4,
            "enable_warmup": True,
        },
        2,
        4,
        3,
    ),
    (
        "custom_text",
        {
            "selected_concurrencies": [2, 1],
            "rounds_per_level": 2,
            "max_tokens": 4,
            "base_prompt": "fixed",
            "suffix_instruction": "",
            "avoid_cache": True,
        },
        4,
        6,
        1,
    ),
    (
        "dataset",
        {
            "rows": [{"prompt": f"dataset-{i}"} for i in range(4)],
            "concurrency": 2,
            "max_tokens": 4,
            "rounds": 1,
        },
        2,
        4,
        1,
    ),
]


def result(session_id, prompt):
    now = time.time()
    return {
        "session_id": session_id,
        "prompt_text": prompt,
        "output_text": "response",
        "ttft": 0.01,
        "tps": 100.0,
        "decode_time": 0.01,
        "total_time": 0.02,
        "prefill_tokens": 64,
        "decode_tokens": 2,
        "cache_hit_tokens": 0,
        "token_source": "API",
        "token_calc_method": "api_usage",
        "start_time": now,
        "end_time": now + 0.02,
        "first_token_time": now + 0.01,
        "error": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("test_type,parameters,blocked_session,expected,committed", CASES)
@pytest.mark.parametrize("termination", ["interruption", "stop"])
async def test_adapter_recovery_preserves_completed_groups_and_repeats_only_unknown_group(
    lab, monkeypatch, test_type, parameters, blocked_session, expected, committed, termination
):
    store, endpoint, settings, db = lab
    entered = asyncio.Event()
    calls = []

    async def response(self, client, session_id, prompt, max_tokens, barrier=None):
        if barrier:
            await barrier.wait()
        calls.append((session_id, prompt))
        if session_id == blocked_session:
            entered.set()
            await asyncio.Event().wait()
        return result(session_id, prompt)

    monkeypatch.setattr(BenchmarkRunner, "get_completion", response)
    submitted = store.submit(
        test_type=test_type,
        endpoint_id="lab",
        model_id="m",
        parameters=parameters,
        progress_total=expected,
    )
    job = store.claim("w")
    task = asyncio.create_task(execute_job(job, endpoint, settings, store, "w"))
    await asyncio.wait_for(entered.wait(), 5)
    if termination == "stop":
        store.request_cancel(job["job_id"])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    info = checkpoint_info(store, store.get(job["job_id"]))
    assert info["committed_units"] == committed and info["unit_label"] == "测量组"
    run_id = store.get(job["job_id"])["result_run_id"]
    with store._connection() as conn:
        original_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM test_results WHERE run_id=? ORDER BY id", (run_id,)
            )
        ]
        frozen = json.loads(
            conn.execute(
                "SELECT input_json FROM checkpoint_units WHERE job_id=? AND result_json IS NULL",
                (job["job_id"],),
            ).fetchone()[0]
        )["prompts"]
    assert original_rows and len(original_rows) < expected
    first_csv = settings.artifact_root / job["job_id"] / "attempt-1" / "requests.csv"
    original_csv = first_csv.read_bytes()
    store.finish(
        job["job_id"],
        "w",
        outcome=RunStatus.CANCELLED if termination == "stop" else RunStatus.FAILED,
        error_code=None if termination == "stop" else "WORKER_LOST",
    )
    headers = {"Authorization": f"Bearer {settings.api_token}"}
    client = TestClient(create_app(settings, store))
    assert client.get("/api/v1/jobs?recoverable=true", headers=headers).json()["total"] == 1
    assert client.post(f"/api/v1/jobs/{job['job_id']}/recover", headers=headers).status_code == 200
    restored_calls = []

    async def restored(self, client, session_id, prompt, max_tokens, barrier=None):
        if barrier:
            await barrier.wait()
        restored_calls.append((session_id, prompt))
        return result(session_id, prompt)

    monkeypatch.setattr(BenchmarkRunner, "get_completion", restored)
    monkeypatch.setattr(
        BenchmarkRunner,
        "_calibrate_prompt_with_source",
        lambda *_: ("changed-generator-output", "changed-source"),
    )
    resumed = store.claim("w2")
    output = await execute_job(resumed, endpoint, settings, store, "w2")
    assert output.result_run_id == run_id and output.completed == expected
    assert first_csv.read_bytes() == original_csv
    assert sorted(restored_calls) == [
        (blocked_session + i, prompt) for i, prompt in enumerate(frozen)
    ]
    assert all(session >= blocked_session for session, _ in restored_calls)
    store.finish(
        job["job_id"], "w2", outcome=RunStatus.COMPLETED, result_run_id=output.result_run_id
    )
    with store._connection() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM test_results WHERE run_id=? ORDER BY id", (run_id,)
            )
        ]
        assert conn.execute("SELECT COUNT(*) FROM test_runs").fetchone()[0] == 1
        config = json.loads(
            conn.execute("SELECT config_json FROM test_runs WHERE id=?", (run_id,)).fetchone()[0]
        )
    assert rows[: len(original_rows)] == original_rows and len(rows) == expected
    assert len({row["session_id"] for row in rows}) == expected
    assert config["measurement_recovery"]["checkpoint"]["recoveries"] == 1
    assert config["measurement_recovery"]["checkpoint"]["repeated_unit_attempts"] == 1
    assert config["measurement_recovery"]["duration_scope"] == "complete_groups_across_attempts"
    report = client.get(f"/api/v1/jobs/{job['job_id']}/report", headers=headers).json()["summary"]
    assert report["integrity"]["verified"], report["integrity"]
    assert report["checkpoint"]["recoveries"] == 1
    assert report["checkpoint"]["unit_label"] == "测量组"
    assert report["measurement_recovery"]["resource_scope"] == "current_attempt"
    for format in ("html", "markdown"):
        content = client.get(
            f"/api/v1/jobs/{job['job_id']}/report?format={format}", headers=headers
        )
        assert (
            content.status_code == 200
            and "测量组检查点" in content.text
            and "跨中断" in content.text
        )
    if test_type in {"matrix", "concurrency"}:
        protocol = config["measurement_protocol"]
        assert protocol["warmup_recorded"] == protocol["warmup_requests"]
        import csv

        with (settings.artifact_root / job["job_id"] / "attempt-2" / "warmup.csv").open() as handle:
            assert len(list(csv.DictReader(handle))) == protocol["warmup_requests"]
    assert all(
        row["concurrency_level"] is not None and row["request_index"] == row["session_id"]
        for row in rows
    )
    for row in rows:
        checkpoint = json.loads(row["extra_metrics"])["measurement_checkpoint"]
        assert checkpoint["contract"] == MEASUREMENT_CONTRACT


@pytest.mark.asyncio
async def test_group_commit_rolls_back_observations_and_checkpoint_together(lab, monkeypatch):
    store, endpoint, settings, db = lab
    job = store.submit(
        test_type="concurrency", endpoint_id="lab", model_id="m", parameters=CASES[0][1]
    )
    job = store.claim("w")

    async def response(self, client, session_id, prompt, max_tokens, barrier=None):
        return result(session_id, prompt)

    monkeypatch.setattr(BenchmarkRunner, "get_completion", response)
    original = db.results.insert_batch

    def fail_after_insert(rows, *, connection=None):
        original(rows[:1], connection=connection)
        raise RuntimeError("synthetic disk failure")

    monkeypatch.setattr(db.results, "insert_batch", fail_after_insert)
    with pytest.raises(RuntimeError, match="synthetic disk failure"):
        await execute_job(job, endpoint, settings, store, "w")
    assert checkpoint_info(store, store.get(job["job_id"]))["committed_units"] == 1  # warmup only
    with store._connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM test_results").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM checkpoint_units WHERE result_json IS NOT NULL"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.asyncio
async def test_old_attempt_with_same_worker_name_cannot_bind_or_commit(lab):
    store, endpoint, _, db = lab
    store.submit(test_type="concurrency", endpoint_id="lab", model_id="m", parameters={})
    job = store.claim("same")
    journal = MeasurementJournal(store, job, "same", endpoint)
    run = journal.bind_run(
        db, test_type="concurrency", model_id="m", provider="OpenAI", config={}, system_info={}
    )
    runner = SimpleNamespace(_persisted_result_ids=set(), completed_requests=0)

    async def observe(prompts):
        return [result(0, prompts[0])]

    rows = await journal.measure(
        runner,
        "barrier",
        ["fixed"],
        ["synthetic"],
        concurrency=1,
        max_tokens=4,
        session_start=0,
        warmup=False,
        observe=observe,
    )
    runner._db_run = run
    runner.results_list = rows
    runner._get_db_manager = lambda: db
    runner.total_requests = 1
    store.finish(job["job_id"], "same", outcome=RunStatus.FAILED, error_code="INTERRUPTED")
    recover_job(store, store.get(job["job_id"]), endpoint)
    newer = store.claim("same")
    assert newer["attempts"] == 2
    with pytest.raises(LeaseLost):
        journal.flush(runner)
    with pytest.raises(LeaseLost):
        journal.bind_run(
            db, test_type="concurrency", model_id="m", provider="OpenAI", config={}, system_info={}
        )
    with store._connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM test_results").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["change", "missing", "duplicate"])
async def test_recovery_rejects_changed_persisted_observations(lab, damage):
    store, endpoint, _, db = lab
    store.submit(test_type="concurrency", endpoint_id="lab", model_id="m", parameters={})
    job = store.claim("w")
    journal = MeasurementJournal(store, job, "w", endpoint)
    run = journal.bind_run(
        db, test_type="concurrency", model_id="m", provider="OpenAI", config={}, system_info={}
    )
    runner = SimpleNamespace(
        _persisted_result_ids=set(),
        completed_requests=0,
        _db_run=run,
        _get_db_manager=lambda: db,
        total_requests=1,
        results_list=[],
    )

    async def observe(prompts):
        return [result(0, prompts[0])]

    rows = await journal.measure(
        runner,
        "barrier",
        ["fixed"],
        ["synthetic"],
        concurrency=1,
        max_tokens=4,
        session_start=0,
        warmup=False,
        observe=observe,
    )
    runner.results_list = rows
    journal.flush(runner)
    store.finish(job["job_id"], "w", outcome=RunStatus.FAILED, error_code="INTERRUPTED")
    with store._connection() as conn:
        if damage == "change":
            conn.execute("UPDATE test_results SET ttft=ttft+1")
        elif damage == "missing":
            conn.execute("DELETE FROM test_results")
        else:
            row = conn.execute("SELECT * FROM test_results").fetchone()
            fields = [name for name in row.keys() if name != "id"]  # noqa: SIM118 (sqlite3.Row)
            conn.execute(
                f"INSERT INTO test_results ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                [row[name] for name in fields],
            )
    from server.checkpoints import CheckpointConflict

    with pytest.raises(CheckpointConflict):
        recover_job(store, store.get(job["job_id"]), endpoint)
    assert store.get(job["job_id"])["status"] == "failed"


@pytest.mark.asyncio
async def test_multiple_adaptive_skip_groups_keep_original_capacity_and_observations(
    lab, monkeypatch
):
    store, endpoint, settings, db = lab

    def capacity(runner):
        runner._kv_budget = 180
        runner._kv_budget_source = "synthetic"

    monkeypatch.setattr(BenchmarkRunner, "_probe_kv_budget", capacity)
    params = {
        "selected_concurrencies": [4, 3, 2, 1],
        "rounds_per_level": 1,
        "max_tokens": 4,
        "input_tokens_target": 64,
        "warmup_rounds_per_level": 0,
    }
    store.submit(
        test_type="concurrency",
        endpoint_id="lab",
        model_id="m",
        parameters=params,
        progress_total=10,
    )
    job = store.claim("w")
    entered = asyncio.Event()

    async def response(self, client, session_id, prompt, max_tokens, barrier=None):
        if session_id == 9:
            entered.set()
            await asyncio.Event().wait()
        return result(session_id, prompt)

    monkeypatch.setattr(BenchmarkRunner, "get_completion", response)
    task = asyncio.create_task(execute_job(job, endpoint, settings, store, "w"))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with store._connection() as conn:
        original = [dict(row) for row in conn.execute("SELECT * FROM test_results ORDER BY id")]
    assert (
        len(original) == 9
        and sum(str(row["error"] or "").startswith("over_kv_budget") for row in original) == 7
    )
    store.finish(job["job_id"], "w", outcome=RunStatus.FAILED, error_code="INTERRUPTED")
    recover_job(store, store.get(job["job_id"]), endpoint)

    def changed_capacity(runner):
        runner._kv_budget = 99999
        runner._kv_budget_source = "changed"

    monkeypatch.setattr(BenchmarkRunner, "_probe_kv_budget", changed_capacity)
    calls = []

    async def restored(self, client, session_id, prompt, max_tokens, barrier=None):
        calls.append(session_id)
        return result(session_id, prompt)

    monkeypatch.setattr(BenchmarkRunner, "get_completion", restored)
    output = await execute_job(store.claim("w2"), endpoint, settings, store, "w2")
    assert output.completed == 10 and calls == [9]
    with store._connection() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM test_results ORDER BY id")]
    assert rows[:9] == original and len(rows) == 10


@pytest.mark.asyncio
async def test_missing_response_is_failure_and_not_user_cancellation(lab):
    store, endpoint, _, db = lab
    store.submit(test_type="concurrency", endpoint_id="lab", model_id="m", parameters={})
    job = store.claim("w")
    journal = MeasurementJournal(store, job, "w", endpoint)
    runner = SimpleNamespace(_persisted_result_ids=set(), completed_requests=0)

    async def missing(prompts):
        return []

    from server.checkpoints import CheckpointConflict

    with pytest.raises(CheckpointConflict, match="Incomplete"):
        await journal.measure(
            runner,
            "barrier",
            ["fixed"],
            ["synthetic"],
            concurrency=1,
            max_tokens=4,
            session_start=0,
            warmup=False,
            observe=missing,
        )
    assert checkpoint_info(store, store.get(job["job_id"]))["committed_units"] == 0
