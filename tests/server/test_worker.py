"""Cancellation crosses the API/worker process boundary through the queue."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from core.cancel_state import is_stop_requested
from server.runner_adapter import RunOutput
from server.settings import Endpoint, Settings
from server.store import JobStore
from server.worker import run_claimed_job


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
