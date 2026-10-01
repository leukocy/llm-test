"""Process boundary, authorization and lifecycle behavior of the new control plane."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.run_lifecycle import RunStatus
from server.api import create_app
from server.settings import Endpoint, Settings
from server.store import IdempotencyConflict, JobStore, LeaseLost

TOKEN = "a" * 48
PARAMS = {"selected_concurrencies": [1, 4], "rounds_per_level": 2, "max_tokens": 64}


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


def submit(store: JobStore, key: str | None = None):
    return store.submit(
        test_type="concurrency",
        endpoint_id="lab",
        model_id="test-model",
        parameters=PARAMS,
        idempotency_key=key,
    )


def test_idempotency_and_audit(store: JobStore):
    first = submit(store, "campaign-001")
    second = submit(store, "campaign-001")
    assert first["job_id"] == second["job_id"]
    assert len(store.events(first["job_id"])) == 1
    with pytest.raises(IdempotencyConflict):
        store.submit(
            test_type="concurrency",
            endpoint_id="lab",
            model_id="another-model",
            parameters=PARAMS,
            idempotency_key="campaign-001",
        )
    assert store.list()[1] == 1


def test_claim_is_exclusive_across_connections(store: JobStore):
    submit(store)
    barrier = threading.Barrier(3)
    claimed = []

    def worker(name: str):
        other_store = JobStore(store.path)
        barrier.wait()
        claimed.append(other_store.claim(name))

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("one", "two")]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
    assert sum(job is not None for job in claimed) == 1
    assert store.list(status=RunStatus.RUNNING)[1] == 1


def test_cancel_and_lease_expiry(store: JobStore):
    queued = submit(store)
    cancelled = store.request_cancel(queued["job_id"])
    assert cancelled["status"] == "cancelled"
    assert store.claim("w") is None

    running = submit(store)
    claimed = store.claim("w")
    assert claimed["job_id"] == running["job_id"]
    assert store.heartbeat(running["job_id"], "other") is False
    assert store.update_progress(running["job_id"], "w", completed=3, total=10)
    assert store.get(running["job_id"])["progress_completed"] == 3
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """INSERT INTO test_runs(test_id, test_type, model_id, completed_requests, total_requests)
               VALUES (?, 'concurrency', 'test-model', 4, 10)""",
            (running["job_id"],),
        )
        conn.executemany(
            "INSERT INTO test_results(run_id,ttft) VALUES (1,?)",
            [(0.1,), (0.2,), (0.3,), (0.4,)],
        )
    persisted_run_id = store.sync_result_run(running["job_id"], "w")
    assert persisted_run_id is not None
    assert store.get(running["job_id"])["progress_completed"] == 4
    assert store.request_cancel(running["job_id"])["status"] == "cancelling"
    finished = store.finish(running["job_id"], "w", outcome=RunStatus.COMPLETED)
    assert finished["status"] == "cancelled"
    assert finished["result_run_id"] == persisted_run_id
    assert [event["to_status"] for event in store.events(running["job_id"])] == [
        "queued",
        "running",
        "cancelling",
        "cancelled",
    ]

    abandoned = submit(store)
    store.claim("lost")
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE control_jobs SET lease_until = 1 WHERE job_id = ?", (abandoned["job_id"],)
        )
    assert store.reap_expired() == 1
    assert store.get(abandoned["job_id"])["error_code"] == "WORKER_LOST"
    with pytest.raises(LeaseLost):
        store.finish(abandoned["job_id"], "lost", outcome=RunStatus.COMPLETED)


@pytest.fixture
def client(store: JobStore, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LAB_API_KEY", "placeholder")
    settings = Settings(
        api_token=TOKEN,
        db_path=store.path,
        artifact_root=store.path.parent / "artifacts",
        endpoints={
            "lab": Endpoint(
                id="lab",
                label="Lab endpoint",
                provider="OpenAI",
                api_base_url="http://127.0.0.1:9010/v1",
                model_id="test-model",
                api_key_env="LAB_API_KEY",  # pragma: allowlist secret
            )
        },
    )
    return TestClient(create_app(settings, store))


def test_api_auth_validation_and_report_boundary(client: TestClient):
    path = "/api/v1/jobs"
    body = {"endpoint_id": "lab", "test_type": "concurrency", "parameters": PARAMS}
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    headers = {"Authorization": f"Bearer {TOKEN}", "Idempotency-Key": "same-request"}
    first = client.post(path, json=body, headers=headers)
    assert first.status_code == 201
    assert client.post(path, json=body, headers=headers).json()["job_id"] == first.json()["job_id"]
    assert client.get(path, headers=headers).json()["total"] == 1
    assert (
        client.get(f"{path}/{first.json()['job_id']}/summary", headers=headers).status_code == 409
    )
    assert (
        client.get(f"{path}/{first.json()['job_id']}/events", headers=headers).json()["items"][0][
            "event"
        ]
        == "enqueue"
    )
    assert (
        client.post(f"{path}/{first.json()['job_id']}/cancel", headers=headers).json()["status"]
        == "cancelled"
    )

    invalid = {**body, "parameters": {**PARAMS, "selected_concurrencies": [1025]}}
    assert client.post(path, json=invalid, headers=headers).status_code == 422
    assert (
        client.post(path, json={**body, "endpoint_id": "unlisted"}, headers=headers).status_code
        == 422
    )
    assert client.get("/health/ready").status_code == 200


def test_pause_resume_api_auth_and_lifecycle(client: TestClient, store: JobStore):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    job_id = submit(store)["job_id"]
    path = f"/api/v1/jobs/{job_id}"
    assert client.post(f"{path}/pause").status_code == 401
    assert client.post(f"{path}/resume").status_code == 401
    assert client.post(f"{path}/pause", headers=headers).status_code == 409  # Still queued.
    store.claim("w")
    assert client.post(f"{path}/pause", headers=headers).json()["status"] == "pausing"
    assert client.post(f"{path}/resume", headers=headers).status_code == 409  # Not drained yet.
    store.acknowledge_pause(job_id, "w")
    assert client.post(f"{path}/resume", headers=headers).json()["status"] == "running"
    store.finish(job_id, "w", outcome=RunStatus.COMPLETED)
    assert client.post(f"{path}/pause", headers=headers).status_code == 409
    assert client.post("/api/v1/jobs/missing/resume", headers=headers).status_code == 404


@pytest.mark.parametrize("test_type", ["quality", "robustness"])
def test_pause_api_accepts_quality_workloads(client, store, test_type):
    job = store.submit(test_type=test_type, endpoint_id="lab", model_id="m", parameters={})
    store.claim("w")
    response = client.post(
        f"/api/v1/jobs/{job['job_id']}/pause", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200
    assert store.get(job["job_id"])["status"] == "pausing"


def test_plan_preview_validates_and_does_not_enqueue(client: TestClient):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = client.post(
        "/api/v1/jobs/plan",
        json={
            "endpoint_id": "lab",
            "test_type": "prefill",
            "parameters": {
                "token_levels": [512, 2048],
                "requests_per_level": 3,
                "warmup_requests_per_level": 1,
                "max_tokens": 64,
            },
        },
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["measured_requests"] == 6
    assert response.json()["warmup_requests"] == 2
    assert client.get("/api/v1/jobs", headers=headers).json()["total"] == 0


def test_warmup_export_requires_auth_and_job_scope(client: TestClient, store: JobStore):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    job = client.post(
        "/api/v1/jobs",
        json={"endpoint_id": "lab", "test_type": "concurrency", "parameters": PARAMS},
        headers=headers,
    ).json()
    folder = store.path.parent / "artifacts" / job["job_id"]
    folder.mkdir(parents=True)
    (folder / "warmup.csv").write_text("condition,ttft\n1 并发,0.1\n", encoding="utf-8")
    path = f"/api/v1/jobs/{job['job_id']}/warmup.csv"
    assert client.get(path).status_code == 401
    assert client.get(path, headers=headers).text.startswith("condition,ttft")
    assert client.get("/api/v1/jobs/missing/warmup.csv", headers=headers).status_code == 404
