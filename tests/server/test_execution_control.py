"""Pause leases and batch policies survive independent queue connections."""

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from core.cancel_state import reset_all
from core.run_lifecycle import InvalidRunTransition, RunStatus
from server.control import JobControl
from server.store import JobStore, LeaseLost


@pytest.fixture
def store(tmp_path: Path):
    reset_all()
    yield JobStore(tmp_path / "jobs.db")
    reset_all()


def submit(store: JobStore):
    return store.submit(test_type="concurrency", endpoint_id="lab", model_id="m", parameters={})


def batch(store: JobStore, *, batch_id="campaign", max_parallel=1, stop_on_error=False, size=4):
    return store.submit_batch(
        batch_id=batch_id,
        name="Campaign",
        description="",
        default_endpoint_id="lab",
        request_hash=batch_id,
        requested_items=size,
        max_parallel=max_parallel,
        stop_on_error=stop_on_error,
        items=[
            {
                "test_type": "concurrency",
                "endpoint_id": "lab",
                "model_id": "m",
                "parameters": {},
                "progress_total": 1,
            }
            for _ in range(size)
        ],
    )


async def wait_for_status(store: JobStore, job_id: str, status: str):
    async def wait():
        while store.get(job_id)["status"] != status:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=3)


@pytest.mark.asyncio
async def test_pause_keeps_lease_and_cancel_wakes_waiting_worker(store: JobStore):
    job_id = submit(store)["job_id"]
    store.claim("w")
    store.request_pause(job_id)
    waiting = asyncio.create_task(JobControl(store, job_id, "w").checkpoint())
    try:
        await wait_for_status(store, job_id, "paused")
        assert not waiting.done()
        assert store.heartbeat(job_id, "w")
        assert store.update_progress(job_id, "w", completed=1, total=3)
        paused = store.get(job_id)
        assert paused["pause_count"] == 1
        assert paused["lease_owner"] == "w"
        assert store.request_cancel(job_id)["status"] == "cancelling"
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiting, timeout=3)
        finished = store.finish(job_id, "w", outcome=RunStatus.CANCELLED)
        assert finished["status"] == "cancelled"
        assert finished["pause_started_at"] is None
        assert finished["paused_seconds"] > 0
    finally:
        waiting.cancel()


def test_resume_accounts_pause_time_and_rejects_expired_worker(store: JobStore, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("server.store.time.time", lambda: now[0])
    job_id = submit(store)["job_id"]
    store.claim("w", lease_seconds=100)
    store.request_pause(job_id)
    store.request_pause(job_id)  # Repeated API request must not add another pause.
    store.acknowledge_pause(job_id, "w")
    store.acknowledge_pause(job_id, "w")
    now[0] = 112.5
    resumed = store.resume(job_id)
    assert resumed["status"] == "running"
    assert resumed["paused_seconds"] == 12.5
    assert resumed["pause_count"] == 1
    with pytest.raises(InvalidRunTransition):
        store.resume(job_id)
    store.request_pause(job_id)
    store.acknowledge_pause(job_id, "w")
    now[0] = 201.0
    with pytest.raises(LeaseLost):
        store.resume(job_id)
    assert store.reap_expired() == 1
    expired = store.get(job_id)
    assert expired["status"] == "failed"
    assert expired["error_code"] == "WORKER_LOST"
    assert expired["paused_seconds"] == 101.0
    assert expired["pause_started_at"] is None


@pytest.mark.parametrize("max_parallel", [1, 2])
def test_batch_claim_cap_is_atomic_across_worker_connections(store: JobStore, max_parallel):
    batch(store, max_parallel=max_parallel)
    barrier = threading.Barrier(5)
    claimed = []

    def claim(worker):
        other = JobStore(store.path)
        barrier.wait(timeout=5)
        claimed.append(other.claim(worker))

    threads = [threading.Thread(target=claim, args=(f"w-{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(claimed) == 4
    assert sum(job is not None for job in claimed) == max_parallel


def test_paused_batch_reserves_slot_and_unrelated_work_can_run(store: JobStore):
    items = batch(store, size=2)
    first = store.claim("w")
    assert first["job_id"] == items[0]["job_id"]
    store.request_pause(first["job_id"])
    store.acknowledge_pause(first["job_id"], "w")
    unrelated = submit(store)
    assert store.claim("other")["job_id"] == unrelated["job_id"]
    assert store.claim("third") is None
    store.resume(first["job_id"])
    store.finish(first["job_id"], "w", outcome=RunStatus.COMPLETED)
    assert store.claim("third")["job_id"] == items[1]["job_id"]


@pytest.mark.parametrize("sibling_status", ["running", "pausing", "paused"])
def test_batch_failure_stops_siblings_and_preserves_reason(store: JobStore, sibling_status):
    items = batch(store, max_parallel=2, stop_on_error=True)
    first = store.claim("one")
    second = store.claim("two")
    if sibling_status in {"pausing", "paused"}:
        store.request_pause(second["job_id"])
    if sibling_status == "paused":
        store.acknowledge_pause(second["job_id"], "two")
    unrelated = batch(store, batch_id="unrelated", size=1)[0]
    store.finish(first["job_id"], "one", outcome=RunStatus.FAILED, error_code="EXECUTION_FAILED")
    stopped = store.get(second["job_id"])
    assert stopped["status"] == "cancelling"
    assert stopped["lease_owner"] == "two"
    assert stopped["error_code"] == "BATCH_STOP_ON_ERROR"
    assert all(store.get(job["job_id"])["status"] == "cancelled" for job in items[2:])
    assert store.get(unrelated["job_id"])["status"] == "queued"
    finished = store.finish(second["job_id"], "two", outcome=RunStatus.CANCELLED)
    assert finished["error_code"] == "BATCH_STOP_ON_ERROR"
    assert store.claim("three")["job_id"] == unrelated["job_id"]


@pytest.mark.parametrize("stop_on_error", [False, True])
def test_completed_run_with_formal_request_error_obeys_batch_policy(store: JobStore, stop_on_error):
    items = batch(store, stop_on_error=stop_on_error, size=2)
    first = store.claim("w")
    with sqlite3.connect(store.path) as conn:
        run_id = conn.execute(
            "INSERT INTO test_runs(test_id,test_type,model_id) VALUES (?, 'concurrency', 'm')",
            (first["job_id"],),
        ).lastrowid
        conn.execute("INSERT INTO test_results(run_id,error) VALUES (?, 'HTTP 429')", (run_id,))
    store.finish(first["job_id"], "w", outcome=RunStatus.COMPLETED, result_run_id=run_id)
    assert store.get(first["job_id"])["status"] == "completed"
    assert store.get(items[1]["job_id"])["status"] == ("cancelled" if stop_on_error else "queued")


def test_expired_parallel_workers_fail_before_queued_siblings_are_stopped(store: JobStore):
    items = batch(store, max_parallel=2, stop_on_error=True, size=3)
    store.claim("one")
    store.claim("two")
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE control_jobs SET lease_until = 1 WHERE status = 'running'")
    assert store.reap_expired() == 2
    assert [store.get(job["job_id"])["status"] for job in items] == [
        "failed",
        "failed",
        "cancelled",
    ]
    for job in items[:2]:
        assert store.get(job["job_id"])["error_code"] == "WORKER_LOST"
        assert store.events(job["job_id"])[-1]["from_status"] == "running"


def test_terminal_run_retains_pause_and_batch_conditions_in_provenance(
    store: JobStore, monkeypatch
):
    now = [100.0]
    monkeypatch.setattr("server.store.time.time", lambda: now[0])
    item = batch(store, max_parallel=2, stop_on_error=True, size=1)[0]
    job_id = item["job_id"]
    store.claim("w")
    config = {"pause_policy": "drained_request_group", "parameters": {"seed": 42}}
    with sqlite3.connect(store.path) as conn:
        run_id = conn.execute(
            "INSERT INTO test_runs(test_id,test_type,model_id,config_json) VALUES (?, 'concurrency', 'm', ?)",
            (job_id, json.dumps(config)),
        ).lastrowid
    store.request_pause(job_id)
    store.acknowledge_pause(job_id, "w")
    now[0] = 105.0
    store.resume(job_id)
    store.finish(job_id, "w", outcome=RunStatus.COMPLETED, result_run_id=run_id)
    with sqlite3.connect(store.path) as conn:
        persisted = json.loads(conn.execute("SELECT config_json FROM test_runs").fetchone()[0])
    assert persisted["parameters"] == {"seed": 42}
    assert persisted["execution_control"] == {
        "pause_count": 1,
        "paused_seconds": 5.0,
        "pause_policy": "drained_request_group",
        "batch_id": "campaign",
        "max_parallel": 2,
        "stop_on_error": True,
    }


@pytest.mark.parametrize("config", ["broken JSON", "[1]", '{"old_value": NaN}'])
def test_invalid_historical_config_does_not_block_terminal_transition(store, config):
    job_id = submit(store)["job_id"]
    store.claim("w")
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "INSERT INTO test_runs(test_id,test_type,model_id,config_json) VALUES (?,'concurrency','m',?)",
            (job_id, config),
        )
    assert store.finish(job_id, "w", outcome=RunStatus.FAILED)["status"] == "failed"
