"""Cancellation crosses the API/worker process boundary through the queue."""

import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest

from core.cancel_state import is_stop_requested
from server.control import JobControl
from server.runner_adapter import RunOutput
from server.settings import Endpoint, Settings
from server.store import JobStore
from server.tokenizer_queue import TokenizerInstallQueue
from server.worker import run_claimed_install, run_claimed_job


@pytest.mark.asyncio
async def test_install_worker_maintains_lease_until_thread_finishes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_TEST_TOKENIZER_DOWNLOAD_ROOT", str(tmp_path / "cache"))
    store = JobStore(tmp_path / "install.db")
    queue = TokenizerInstallQueue(store)
    queue.enqueue(["DeepSeek-V3.2"])
    task = queue.claim("installer")
    with store._connection() as conn:
        original = conn.execute("SELECT lease_until FROM tokenizer_installs").fetchone()[0]
    finished = threading.Event()

    def fake_execute(task, queue, worker_id, **_kwargs):
        finished.wait(5)
        queue.fail(task["install_id"], worker_id, "Finished by test", cancelled=True)

    monkeypatch.setattr("server.worker.execute_install", fake_execute)
    running = asyncio.create_task(run_claimed_install(task, queue, "installer", asyncio.Event()))
    try:

        async def wait_for_heartbeat():
            while True:
                with store._connection() as conn:
                    lease = conn.execute("SELECT lease_until FROM tokenizer_installs").fetchone()[0]
                if lease > original:
                    return
                await asyncio.sleep(0.02)

        await asyncio.wait_for(wait_for_heartbeat(), timeout=4)
        assert queue.get(task["install_id"])["status"] == "downloading"
        finished.set()
        await asyncio.wait_for(running, timeout=2)
        assert queue.get(task["install_id"])["status"] == "cancelled"
    finally:
        finished.set()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_paused_job_keeps_monitor_heartbeat_then_completes_after_resume(
    tmp_path, monkeypatch
):
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.submit(
        test_type="concurrency", endpoint_id="lab", model_id="m", parameters={}, progress_total=1
    )["job_id"]
    claimed = store.claim("worker")
    original_lease = claimed["lease_until"]
    entered = asyncio.Event()
    checkpoint_ready = asyncio.Event()

    async def fake_execute(*_args):
        entered.set()
        await checkpoint_ready.wait()
        await JobControl(store, job_id, "worker").checkpoint()
        with sqlite3.connect(store.path) as conn:
            run_id = conn.execute(
                """INSERT INTO test_runs(test_id,test_type,status,model_id)
                   VALUES (?, 'concurrency', 'completed', 'm')""",
                (job_id,),
            ).lastrowid
            conn.execute("INSERT INTO test_results(run_id,ttft) VALUES (?,0.2)", (run_id,))
        return RunOutput(result_run_id=run_id, completed=1, total=1)

    monkeypatch.setattr("server.worker.execute_job", fake_execute)
    settings = Settings(
        api_token="x" * 40,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={
            "lab": Endpoint("lab", "Lab", "OpenAI", "http://127.0.0.1:9010/v1", "m", "LAB_KEY")
        },
    )
    running = asyncio.create_task(run_claimed_job(claimed, settings, store, "worker"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        store.request_pause(job_id)
        checkpoint_ready.set()

        async def wait_for_heartbeat():
            while store.get(job_id)["lease_until"] <= original_lease:
                await asyncio.sleep(0.02)

        await asyncio.wait_for(wait_for_heartbeat(), timeout=5)
        assert store.get(job_id)["status"] == "paused"
        assert not running.done()
        store.resume(job_id)
        await asyncio.wait_for(running, timeout=3)
        assert store.get(job_id)["status"] == "completed"
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_running_job_observes_persisted_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = JobStore(tmp_path / "jobs.db")
    job = store.submit(
        test_type="stability",
        endpoint_id="lab",
        model_id="m",
        parameters={"concurrency": 1, "duration_seconds": 60, "max_tokens": 10},
    )
    claimed = store.claim("worker")
    assert claimed is not None

    async def fake_execute(*_args):
        while not is_stop_requested():
            await asyncio.sleep(0.02)
        raise asyncio.CancelledError

    monkeypatch.setattr("server.worker.execute_job", fake_execute)
    settings = Settings(
        api_token="x" * 40,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={
            "lab": Endpoint("lab", "Lab", "OpenAI", "http://127.0.0.1:9010/v1", "m", "LAB_KEY")
        },
    )
    running = asyncio.create_task(run_claimed_job(claimed, settings, store, "worker"))
    store.request_cancel(job["job_id"])
    await asyncio.wait_for(running, timeout=5)
    assert store.get(job["job_id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_missing_persisted_observation_fails_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = JobStore(tmp_path / "jobs.db")
    job = store.submit(
        test_type="concurrency",
        endpoint_id="lab",
        model_id="m",
        parameters={},
        progress_total=2,
    )
    claimed = store.claim("worker")
    assert claimed is not None

    async def fake_execute(*_args):
        with sqlite3.connect(store.path) as conn:
            conn.execute(
                """INSERT INTO test_runs(test_id,test_type,status,model_id)
                   VALUES (?, 'concurrency', 'completed', 'm')""",
                (job["job_id"],),
            )
            conn.execute("INSERT INTO test_results(run_id,ttft) VALUES (1,0.2)")
        return RunOutput(result_run_id=1, completed=2, total=2)

    monkeypatch.setattr("server.worker.execute_job", fake_execute)
    settings = Settings(
        api_token="x" * 40,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={
            "lab": Endpoint("lab", "Lab", "OpenAI", "http://127.0.0.1:9010/v1", "m", "LAB_KEY")
        },
    )
    await run_claimed_job(claimed, settings, store, "worker")
    finished = store.get(job["job_id"])
    assert finished["status"] == "failed"
    assert finished["result_run_id"] == 1
    assert finished["error_code"] == "EXECUTION_FAILED"
