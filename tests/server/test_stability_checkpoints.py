"""Continuous recovery preserves request identity, time budgets and independent clocks."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.benchmark.metrics import METRIC_CONTRACT_VERSION
from core.benchmark_runner import BenchmarkRunner
from core.run_lifecycle import RunStatus
from server.api import create_app
from server.checkpoints import checkpoint_info, recover_job
from server.runner_adapter import execute_job
from server.stability_checkpoints import StabilityJournal
from server.store import LeaseLost
from tests.server.test_live_stability import window
from tests.server.test_measurement_checkpoints import lab as lab
from tests.server.test_measurement_checkpoints import result, wait_during_execution


def submit(store):
    return store.submit(
        test_type="stability",
        endpoint_id="lab",
        model_id="m",
        parameters={
            "concurrency": 2,
            "duration_seconds": 5,
            "max_tokens": 4,
            "input_tokens_target": 64,
        },
    )


def state(store, run_id):
    with store._connection() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM test_results WHERE run_id=? ORDER BY id", (run_id,)
            )
        ]
        config = json.loads(
            conn.execute("SELECT config_json FROM test_runs WHERE id=?", (run_id,)).fetchone()[0]
        )
    return rows, config


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["interruption", "stop", "lease_loss"])
async def test_real_adapter_recovers_requests_and_remaining_time_without_joining_clocks(
    lab, monkeypatch, termination
):
    store, endpoint, settings, _ = lab
    job_id = submit(store)["job_id"]
    blocked = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def response(self, client, session_id, prompt, max_tokens, barrier=None):
        calls.append((session_id, prompt))
        if session_id >= 16:
            blocked.set()
            await release.wait()
        await asyncio.sleep(0.08)
        return result(session_id, prompt)

    monkeypatch.setattr(BenchmarkRunner, "get_completion", response)
    task = asyncio.create_task(execute_job(store.claim("w"), endpoint, settings, store, "w"))
    await wait_during_execution(task, blocked.wait())
    run_id = store.get(job_id)["result_run_id"]
    before, config = state(store, run_id)
    assert before and not task.done()
    if termination == "lease_loss":
        with store._connection() as conn:
            conn.execute("UPDATE control_jobs SET lease_until=1 WHERE job_id=?", (job_id,))
        assert store.reap_expired() == 1
        release.set()
        with pytest.raises(LeaseLost):
            await task
        assert state(store, run_id)[0] == before
    else:
        if termination == "stop":
            store.request_cancel(job_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        store.finish(
            job_id,
            "w",
            outcome=RunStatus.CANCELLED if termination == "stop" else RunStatus.FAILED,
            error_code=None if termination == "stop" else "INTERRUPTED",
        )
    original_rows, original_config = state(store, run_id)
    original_csv = settings.artifact_root / job_id / "attempt-1" / "requests.csv"
    original_bytes = original_csv.read_bytes()
    with store._connection() as conn:
        unknown = [
            json.loads(row[0])
            for row in conn.execute(
                "SELECT input_json FROM checkpoint_units WHERE job_id=? AND result_json IS NULL",
                (job_id,),
            )
        ]
    unknown.sort(key=lambda plan: plan["session_start"])
    assert unknown
    saved_time = original_config["stability_budget"]["saved_scheduling_seconds"]
    assert 0 < saved_time < 5
    client = TestClient(create_app(settings, store))
    headers = {"Authorization": f"Bearer {settings.api_token}"}
    candidate = client.get(
        "/api/v1/jobs?recoverable=true&saved_progress=true", headers=headers
    ).json()
    assert candidate["total"] == 1 and candidate["items"][0]["saved_progress_unit"] == "请求"
    assert client.post(f"/api/v1/jobs/{job_id}/recover", headers=headers).status_code == 200
    restored_calls = []

    async def restored(self, client, session_id, prompt, max_tokens, barrier=None):
        restored_calls.append((session_id, prompt))
        await asyncio.sleep(0.08)
        return result(session_id, prompt)

    monkeypatch.setattr(BenchmarkRunner, "get_completion", restored)
    monkeypatch.setattr(
        BenchmarkRunner,
        "_calibrate_prompt_with_source",
        lambda *_, **__: ("new-generator", "new-source"),
    )
    output = await execute_job(store.claim("w2"), endpoint, settings, store, "w2")
    store.finish(job_id, "w2", outcome=RunStatus.COMPLETED, result_run_id=output.result_run_id)
    assert output.result_run_id == run_id and output.completed == output.total
    rows, config = state(store, run_id)
    assert (
        rows[: len(original_rows)] == original_rows and original_csv.read_bytes() == original_bytes
    )
    committed = {row["session_id"] for row in original_rows}
    assert all(sid not in committed for sid, _ in restored_calls)
    assert restored_calls[: len(unknown)] == [
        (plan["session_start"], plan["prompts"][0]) for plan in unknown
    ]
    assert len({row["session_id"] for row in rows}) == len(rows) == output.completed
    assert config["stability_budget"]["remaining_seconds"] == 0
    assert config["stability_budget"]["saved_scheduling_seconds"] == pytest.approx(5)
    assert len(config["stability_windows"]) == 2
    assert config["stability_windows"][0]["state"] == "interrupted"
    assert config["stability_windows"][1]["admission_budget_seconds"] == pytest.approx(
        5 - saved_time
    )
    assert config["stability_windows"][0]["id"] != config["stability_windows"][1]["id"]
    report = client.get(f"/api/v1/jobs/{job_id}/report", headers=headers).json()["summary"]
    assert report["integrity"]["verified"], report["integrity"]
    assert report["extended_observations"]["system"]["valid_batches"] == 1
    assert report["overall"]["metrics"]["system_qpm"]["count"] == 1
    timeline = report["time_series"]
    assert timeline["cross_interruption"] and not timeline["complete"] and not timeline["bins"]
    assert not timeline["conflict"] and timeline["timed_requests"] == len(rows)
    assert len(timeline["segments"]) == 2 and timeline["segments"][1]["complete"]
    assert timeline["segments"][0]["attempt"] == 1 and timeline["segments"][1]["attempt"] == 2
    for attempt in (1, 2):
        figure = client.get(
            f"/api/v1/jobs/{job_id}/figure?view=timeline&attempt={attempt}", headers=headers
        )
        assert figure.status_code == 200
        assert f"执行尝试 #{attempt}" in figure.json()["figure"]["layout"]["title"]["text"]
    assert (
        client.get(
            f"/api/v1/jobs/{job_id}/figure?view=timeline&attempt=99", headers=headers
        ).status_code
        == 409
    )
    for fmt in ("html", "markdown"):
        text = client.get(f"/api/v1/jobs/{job_id}/report?format={fmt}", headers=headers).text
        assert (
            "执行尝试 #1" in text
            and "执行尝试 #2" in text
            and "独立时钟" in text
            and "剩余 0.000" in text
        )
    assert checkpoint_info(store, store.get(job_id))["repeated_unit_attempts"] == len(unknown)


def journal_runner(lab):
    store, endpoint, _, db = lab
    submit(store)
    job = store.claim("w")
    journal = StabilityJournal(store, job, "w", endpoint)
    run = journal.bind_run(
        db,
        test_type="stability",
        model_id="m",
        provider="OpenAI",
        config={**job["parameters"], "metric_contract_version": METRIC_CONTRACT_VERSION},
        system_info={},
    )
    runner = SimpleNamespace(
        _db_run=run,
        _get_db_manager=lambda: db,
        _persisted_result_ids=set(),
        results_list=[],
        total_requests=0,
        completed_requests=0,
    )
    journal.begin_stability(runner)
    journal.save_snapshot(
        runner,
        [],
        window(
            planned_seconds=5,
            window_seconds=0.01,
            scheduling_seconds=0.01,
            expected_requests=0,
            recorded_requests=0,
            state="running",
        ),
    )
    return journal, runner


def measured_row(journal, sid=0):
    row = result(sid, "frozen")
    journal.tag_request(row, sid, "frozen", "synthetic")
    row["extra_metrics"]["metric_contract_version"] = METRIC_CONTRACT_VERSION
    row["extra_metrics"]["timing_observation"] = {
        "version": "stability-clock-v1",
        "clock": "monotonic",
        "anchor": "scheduler_start",
        "id": "a" * 32,
        "index": 0,
        "start_seconds": 0.1,
        "end_seconds": 0.2,
    }
    return row


def test_live_checkpoint_rows_and_budget_roll_back_in_one_transaction(lab, monkeypatch):
    store, _, _, db = lab
    journal, runner = journal_runner(lab)
    journal.freeze_request(runner, 0, lambda _: ("frozen", "synthetic"))
    _, original = state(store, runner._db_run.id)

    def fail(rows, *, connection):
        connection.execute(
            "INSERT INTO test_results(run_id,ttft) VALUES(?,.1)", (runner._db_run.id,)
        )
        raise RuntimeError("synthetic write failure")

    monkeypatch.setattr(db.results, "insert_batch", fail)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        journal.save_snapshot(runner, [measured_row(journal)], window(planned_seconds=5))
    assert state(store, runner._db_run.id) == ([], original)
    assert checkpoint_info(store, store.get(journal.job["job_id"]))["committed_units"] == 0


def test_same_worker_name_from_old_attempt_cannot_admit_or_flush(lab):
    store, endpoint, _, _ = lab
    journal, runner = journal_runner(lab)
    journal.freeze_request(runner, 0, lambda _: ("frozen", "synthetic"))
    store.finish(journal.job["job_id"], "w", outcome=RunStatus.FAILED, error_code="WORKER_LOST")
    recover_job(store, store.get(journal.job["job_id"]), endpoint)
    assert store.claim("w")["attempts"] == 2
    with pytest.raises(LeaseLost):
        journal.freeze_request(runner, 1, lambda _: ("wrong", "synthetic"))
    with pytest.raises(LeaseLost):
        journal.save_snapshot(runner, [measured_row(journal)], window(planned_seconds=5))
    assert state(store, runner._db_run.id)[0] == []


@pytest.mark.parametrize("field", ["stability_windows", "stability_budget"])
def test_recovery_rejects_changed_clock_or_budget(lab, field):
    store, endpoint, settings, _ = lab
    journal, runner = journal_runner(lab)
    journal.freeze_request(runner, 0, lambda _: ("frozen", "synthetic"))
    _, config = state(store, runner._db_run.id)
    config[field] = [] if field == "stability_windows" else {"remaining_seconds": 999}
    with store._connection() as conn:
        conn.execute(
            "UPDATE test_runs SET config_json=? WHERE id=?", (json.dumps(config), runner._db_run.id)
        )
    job_id = journal.job["job_id"]
    store.finish(job_id, "w", outcome=RunStatus.FAILED, error_code="WORKER_LOST")
    response = TestClient(create_app(settings, store)).post(
        f"/api/v1/jobs/{job_id}/recover", headers={"Authorization": f"Bearer {settings.api_token}"}
    )
    assert response.status_code == 409
    assert store.get(job_id)["status"] == "failed"


@pytest.mark.asyncio
async def test_phase_estimates_use_provider_intervals_independent_of_epoch(lab, monkeypatch):
    store, endpoint, settings, _ = lab
    captured = []

    async def response(self, client, session_id, prompt, max_tokens, barrier=None):
        await asyncio.sleep(0.05)
        row = result(session_id, prompt)
        row.update(
            start_time=1_900_000_000.0, first_token_time=1_900_000_000.01, end_time=1_900_000_000.05
        )
        captured.append(row)
        return row

    monkeypatch.setattr(BenchmarkRunner, "get_completion", response)
    store.submit(
        test_type="stability",
        endpoint_id="lab",
        model_id="m",
        parameters={
            "concurrency": 1,
            "duration_seconds": 0.15,
            "max_tokens": 4,
            "input_tokens_target": 64,
        },
    )
    output = await execute_job(store.claim("w"), endpoint, settings, store, "w")
    assert output.completed >= 2
    assert all(0 < row["system_output_throughput"] < 150 for row in captured)


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_zero_remaining_budget_drains_only_unknown_requests_and_preserves_completed_window(
    lab, monkeypatch, committed
):
    store, endpoint, settings, _ = lab
    journal, runner = journal_runner(lab)
    journal.freeze_request(runner, 0, lambda _: ("frozen", "synthetic"))
    journal.save_snapshot(
        runner,
        [measured_row(journal)] if committed else [],
        window(
            planned_seconds=5,
            window_seconds=5,
            scheduling_seconds=5,
            expected_requests=1,
            recorded_requests=int(committed),
            state="completed" if committed else "interrupted",
        ),
    )
    job_id = journal.job["job_id"]
    store.finish(job_id, "w", outcome=RunStatus.FAILED, error_code="WORKER_LOST")
    recover_job(store, store.get(job_id), endpoint)
    calls = []

    async def response(self, client, session_id, prompt, max_tokens, barrier=None):
        calls.append((session_id, prompt))
        await asyncio.sleep(0.01)
        return result(session_id, prompt)

    monkeypatch.setattr(BenchmarkRunner, "get_completion", response)
    output = await execute_job(store.claim("w2"), endpoint, settings, store, "w2")
    assert output.completed == output.total == 1
    assert calls == ([] if committed else [(0, "frozen")])
    rows, config = state(store, runner._db_run.id)
    assert len(rows) == 1
    assert config["stability_budget"]["remaining_seconds"] == 0
    assert len(config["stability_windows"]) == (1 if committed else 2)
    if not committed:
        assert config["stability_windows"][1]["admission_budget_seconds"] == 0
    store.finish(job_id, "w2", outcome=RunStatus.COMPLETED, result_run_id=output.result_run_id)
    if not committed:
        report = (
            TestClient(create_app(settings, store))
            .get(
                f"/api/v1/jobs/{job_id}/report",
                headers={"Authorization": f"Bearer {settings.api_token}"},
            )
            .json()["summary"]
        )
        assert report["overall"]["metrics"]["system_qpm"]["count"] == 0


@pytest.mark.asyncio
async def test_paused_worker_loss_preserves_saved_admission_budget(lab, monkeypatch):
    store, endpoint, settings, _ = lab
    job_id = submit(store)["job_id"]
    calls = []

    async def response(self, client, session_id, prompt, max_tokens, barrier=None):
        calls.append(session_id)
        await asyncio.sleep(0.02)
        return result(session_id, prompt)

    monkeypatch.setattr(BenchmarkRunner, "get_completion", response)
    task = asyncio.create_task(execute_job(store.claim("w"), endpoint, settings, store, "w"))

    async def visible():
        while store.get(job_id)["progress_completed"] == 0:
            await asyncio.sleep(0.01)

    await wait_during_execution(task, visible())
    store.request_pause(job_id)

    async def paused():
        while store.get(job_id)["status"] != "paused":
            await asyncio.sleep(0.01)

    await wait_during_execution(task, paused())
    run_id = store.get(job_id)["result_run_id"]
    count = len(calls)
    rows, config = state(store, run_id)
    saved = config["stability_budget"]["saved_scheduling_seconds"]
    assert config["stability_windows"][0]["state"] == "paused"
    await asyncio.sleep(0.25)
    assert len(calls) == count and state(store, run_id)[0] == rows
    with store._connection() as conn:
        conn.execute("UPDATE control_jobs SET lease_until=1 WHERE job_id=?", (job_id,))
    store.reap_expired()
    with pytest.raises(LeaseLost):
        await task
    recover_job(store, store.get(job_id), endpoint)
    recovered = StabilityJournal(store, store.claim("w2"), "w2", endpoint)
    assert recovered.describe()["can_recover"] is False
    _, after = state(store, run_id)
    assert after["stability_budget"]["remaining_seconds"] == pytest.approx(5 - saved)
