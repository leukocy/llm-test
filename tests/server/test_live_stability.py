"""Continuous load survives live reads, drained pauses, cancellation and failed flushes."""

import asyncio
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from core.benchmark.metrics import METRIC_CONTRACT_VERSION
from core.benchmark.stability import continuous_load
from core.benchmark_runner import BenchmarkRunner
from core.database.connection import Database
from core.database.manager import DatabaseManager
from core.run_lifecycle import RunStatus
from server.analytics import run_results_csv, run_summary
from server.api import create_app
from server.control import JobControl
from server.runner_adapter import HeadlessOutput
from server.settings import Settings
from server.store import JobStore
from server.time_series import resolve_stability_row, stability_time_series
from tests.server.test_time_series import timed


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(Database, "_instance", None)
    monkeypatch.setattr(DatabaseManager, "_instance", None)
    path = tmp_path / "lab.db"
    store = JobStore(path)
    db = DatabaseManager(str(path))
    job = store.submit(test_type="stability", endpoint_id="synthetic", model_id="m", parameters={})
    store.claim("w")
    run = db.start_test_run(
        "stability",
        "m",
        test_id=job["job_id"],
        config={"metric_contract_version": METRIC_CONTRACT_VERSION, "preserve": "value"},
    )
    yield store, db, job, run
    with db.db.get_connection() as conn:
        conn.close()


def observer(db, run, snapshots):
    def flush(rows, window):
        db.save_stability_snapshot(run, rows, window, worker_id="w")
        snapshots.append(dict(window))

    return flush


def success(session_id):
    return {
        "session_id": session_id,
        "concurrency": 2,
        "ttft": 0.2,
        "prefill_tokens": 100,
        "decode_tokens": 20,
        "token_source": "API",
        "extra_metrics": {"metric_contract_version": METRIC_CONTRACT_VERSION},
    }


async def wait_state(store, job, state):
    async def wait():
        while store.get(job["job_id"])["status"] != state:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 3)


@pytest.mark.asyncio
async def test_live_rows_and_window_are_visible_before_test_completes(lab):
    store, db, job, run = lab
    snapshots = []
    visible = asyncio.Event()
    save = observer(db, run, snapshots)

    async def request(sid):
        await asyncio.sleep(0.02)
        return success(sid)

    def observe(rows, window):
        save(rows, window)
        if rows and window["state"] == "running":
            visible.set()

    task = asyncio.create_task(
        continuous_load(
            request, concurrency=2, duration=0.8, stopped=lambda: False, observe=observe
        )
    )
    try:
        await asyncio.wait_for(visible.wait(), 3)
        assert not task.done()
        store.sync_result_run(job["job_id"], "w")
        summary = run_summary(str(store.path), run.id, job=store.get(job["job_id"]))
        assert summary["time_series"]["live"] and not summary["time_series"]["complete"]
        assert summary["overall"]["requests"] > 0
        assert summary["time_series"]["timed_requests"] == summary["overall"]["requests"]
        assert summary["overall"]["metrics"]["system_qpm"]["count"] == 0
        assert not summary["integrity"]["verified"]
        rows = await asyncio.wait_for(task, 3)
        db.complete_test_run(run)
        store.sync_result_run(job["job_id"], "w")
        final_job = store.finish(
            job["job_id"], "w", outcome=RunStatus.COMPLETED, result_run_id=run.id
        )
        final = run_summary(str(store.path), run.id, job=final_job)
        assert final["time_series"]["complete"] and not final["time_series"]["live"]
        assert final["integrity"]["verified"] and len(rows) == final["overall"]["requests"]
        assert final["overall"]["metrics"]["system_qpm"]["count"] == 1
        assert "stability-clock-v1" in run_results_csv(str(store.path), run.id)
        assert run.config["preserve"] == "value"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_pause_stops_admission_then_drains_and_preserves_remaining_duration(lab):
    store, db, job, run = lab
    control = JobControl(store, job["job_id"], "w")
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    in_flight = [0]
    snapshots = []

    async def request(sid):
        calls.append(sid)
        in_flight[0] += 1
        try:
            entered.set()
            await release.wait()
            await asyncio.sleep(0.02)
            return success(sid)
        finally:
            in_flight[0] -= 1

    task = asyncio.create_task(
        continuous_load(
            request,
            concurrency=2,
            duration=0.5,
            stopped=lambda: False,
            pause_requested=control.pause_requested,
            checkpoint=control.checkpoint,
            observe=observer(db, run, snapshots),
        )
    )
    try:
        await entered.wait()
        store.request_pause(job["job_id"])
        await asyncio.sleep(0.23)
        assert len(calls) == 2 and store.get(job["job_id"])["status"] == "pausing"
        release.set()
        await wait_state(store, job, "paused")
        assert in_flight[0] == 0 and len(calls) == 2
        await asyncio.sleep(0.12)
        assert len(calls) == 2 and not task.done()
        assert snapshots[-1]["state"] == "paused"
        assert store.heartbeat(job["job_id"], "w")
        store.resume(job["job_id"])
        rows = await asyncio.wait_for(task, 3)
        assert len(rows) > 2
        assert snapshots[-1]["state"] == "completed"
        assert snapshots[-1]["paused_seconds"] >= 0.12
        assert snapshots[-1]["scheduling_seconds"] >= 0.5
        assert snapshots[-1]["window_seconds"] >= 0.62
        assert sorted(r["request_index"] for r in rows) == list(range(len(rows)))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_keeps_completed_rows_and_marks_window_interrupted(lab):
    store, db, job, run = lab
    snapshots = []
    visible = asyncio.Event()
    save = observer(db, run, snapshots)
    stopped = [False]

    async def request(sid):
        await asyncio.sleep(0.02)
        return success(sid)

    def observe(rows, window):
        save(rows, window)
        if rows:
            visible.set()

    task = asyncio.create_task(
        continuous_load(
            request, concurrency=2, duration=10, stopped=lambda: stopped[0], observe=observe
        )
    )
    await asyncio.wait_for(visible.wait(), 3)
    stopped[0] = True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert snapshots[-1]["state"] == "interrupted"
    summary = run_summary(str(store.path), run.id)
    assert summary["overall"]["requests"] > 0
    assert not summary["time_series"]["complete"] and not summary["time_series"]["live"]
    assert summary["overall"]["metrics"]["system_qpm"]["count"] == 0


def window(**changes):
    return {
        "version": "stability-clock-v1",
        "clock": "monotonic",
        "anchor": "scheduler_start",
        "id": "a" * 32,
        "window_seconds": 1.0,
        "planned_seconds": 1,
        "expected_requests": 1,
        "recorded_requests": 1,
        "paused_seconds": 0.0,
        "scheduling_seconds": 1.0,
        "state": "completed",
        **changes,
    }


def test_flush_rolls_back_rows_and_window_together(lab, monkeypatch):
    _, db, _, run = lab

    def broken(results, *, connection):
        connection.execute("INSERT INTO test_results (run_id,ttft) VALUES (?,.2)", (run.id,))
        raise RuntimeError("synthetic write failure")

    monkeypatch.setattr(db.results, "insert_batch", broken)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        db.save_stability_snapshot(run, [success(0)], window(), worker_id="w")
    with db.db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM test_results").fetchone()[0] == 0
        assert "stability_window" not in json.loads(
            conn.execute("SELECT config_json FROM test_runs").fetchone()[0]
        )


def test_expired_or_replaced_worker_cannot_append_observations(lab):
    store, db, _, run = lab
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE control_jobs SET lease_until=1")
    with pytest.raises(RuntimeError, match="lease was lost"):
        db.save_stability_snapshot(run, [success(0)], window(), worker_id="w")
    assert db.results.count_by_run(run.id)["total"] == 0


def test_abandoned_job_never_displays_a_stale_window_as_live(lab):
    store, db, job, run = lab
    db.save_stability_snapshot(
        run, [], window(state="running", expected_requests=0, recorded_requests=0), worker_id="w"
    )
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE control_jobs SET lease_until=1")
    store.reap_expired()
    summary = run_summary(str(store.path), run.id, job=store.get(job["job_id"]))
    assert not summary["time_series"]["live"]
    assert summary["time_series"]["window_state"] == "interrupted"
    assert not summary["time_series"]["complete"]


def test_live_stability_has_pause_and_resume_api(lab, tmp_path):
    store, _, job, _ = lab
    token = "a" * 40
    client = TestClient(
        create_app(
            Settings(
                api_token=token,
                db_path=store.path,
                artifact_root=tmp_path / "artifacts",
                endpoints={},
            ),
            store,
        )
    )
    path = f"/api/v1/jobs/{job['job_id']}"
    headers = {"Authorization": "Bearer " + token}
    assert client.post(path + "/pause", headers=headers).status_code == 200
    store.acknowledge_pause(job["job_id"], "w")
    assert client.post(path + "/resume", headers=headers).status_code == 200


@pytest.mark.asyncio
async def test_benchmark_entry_persists_live_and_final_flush_does_not_duplicate(
    lab, tmp_path, monkeypatch
):
    store, db, job, run = lab
    output = HeadlessOutput()
    runner = BenchmarkRunner(
        placeholder=output,
        progress_bar=output,
        status_text=output,
        log_placeholder=None,
        api_base_url="http://127.0.0.1:9999/v1",
        model_id="m",
        tokenizer_option="字符数 (Fallback)",
        csv_filename=str(tmp_path / "requests.csv"),
        api_key="synthetic",  # pragma: allowlist secret -- isolated fake provider
        provider="TestProvider",
        enable_live_log_server=False,
        external_test_id=job["job_id"],
        persistence_owner="w",
    )
    monkeypatch.setattr(runner, "_db_manager", db)

    def start(*args):
        runner._db_run = run

    monkeypatch.setattr(runner, "_start_db_run", start)
    monkeypatch.setattr(
        runner, "_complete_db_run", lambda success: db.complete_test_run(run, success)
    )
    monkeypatch.setattr(
        runner,
        "_calibrate_prompt_with_source",
        lambda *args, **kwargs: ("synthetic", "synthetic-source"),
    )

    async def completion(_, sid, *args):
        await asyncio.sleep(0.025)
        return success(sid)

    monkeypatch.setattr(runner, "get_completion", completion)
    task = asyncio.create_task(runner.run_stability_test(2, 0.8, 20, input_tokens_target=64))
    try:

        async def visible():
            while db.results.count_by_run(run.id)["total"] == 0:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(visible(), 3)
        assert not task.done()
        assert db.results.count_by_run(run.id)["total"] > 0
        frame = await asyncio.wait_for(task, 3)
        assert len(frame) == db.results.count_by_run(run.id)["total"] == len(runner.results_list)
        assert len(set(frame.session_id)) == len(frame)
        summary = run_summary(str(store.path), run.id)
        assert summary["time_series"]["complete"]
        assert summary["overall"]["metrics"]["system_qpm"]["count"] == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("state", ["running", "paused", "interrupted", "completed"])
def test_shared_window_only_marks_closed_complete_rows_as_complete(state):
    row = success(0)
    row["request_index"] = 0
    row["batch_id"] = "a" * 32
    row["extra_metrics"]["timing_observation"] = {
        "version": "stability-clock-v1",
        "clock": "monotonic",
        "anchor": "scheduler_start",
        "id": "a" * 32,
        "index": 0,
        "start_seconds": 0.0,
        "end_seconds": 0.5,
    }
    row["extra_metrics"] = json.dumps(row["extra_metrics"])
    timeline = stability_time_series([row], window(state=state))
    assert timeline["complete"] == (state == "completed")
    assert timeline["live"] == (state in {"running", "paused"})
    resolved = resolve_stability_row(row, window(state=state))
    assert ("system_measurement" in json.loads(resolved["extra_metrics"])) == (state == "completed")
    assert not stability_time_series([row], window(recorded_requests=0))["complete"]


@pytest.mark.parametrize("bad_window", [window(state=[]), window(clock="wall"), {}, []])
def test_invalid_shared_window_cannot_fall_back_to_legacy_completeness(bad_window):
    result = stability_time_series([timed(0, 0.5, expected=1)], bad_window)
    assert result["conflict"] and not result["complete"] and not result["bins"]


def test_raw_csv_remains_exportable_when_legacy_run_config_is_malformed(lab):
    store, db, _, run = lab
    db.save_result(run, success(0))
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE test_runs SET config_json='invalid legacy config'")
    assert "timing_observation_json" in run_results_csv(str(store.path), run.id)
