"""Managed measurement endpoints are usable across API and worker processes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.api import create_app
from server.endpoints import EndpointCredentialUnavailable, EndpointRegistry
from server.settings import Settings
from server.store import JobStore
from server.worker import run_claimed_job

TOKEN = "control-token-with-enough-entropy-for-tests-123456"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
CONFIG = {
    "label": "Local model",
    "provider": "OpenAI",
    "api_base_url": "http://127.0.0.1:9010/v1",
    "model_id": "lab-model",
    "tokenizer_option": "auto",
    "api_key": "test-credential-only-in-request",  # pragma: allowlist secret
}
PARAMS = {"selected_concurrencies": [1], "rounds_per_level": 1, "max_tokens": 8}


def _platform(tmp_path: Path) -> tuple[TestClient, EndpointRegistry, JobStore]:
    store = JobStore(tmp_path / "jobs.db")
    settings = Settings(
        api_token=TOKEN,
        db_path=store.path,
        artifact_root=tmp_path / "results",
        endpoints={},
    )
    return TestClient(create_app(settings, store)), EndpointRegistry(settings, store), store


def test_managed_endpoint_persists_secret_and_runs_without_static_config(tmp_path: Path):
    client, registry, store = _platform(tmp_path)
    assert client.get("/api/v1/endpoints").status_code == 401
    created = client.post("/api/v1/endpoints", json=CONFIG, headers=HEADERS)
    assert created.status_code == 201
    endpoint_id = created.json()["id"]
    assert created.json()["source"] == "managed"
    assert created.json()["credential_configured"] is True
    assert CONFIG["api_key"] not in created.text
    assert CONFIG["api_key"] not in client.get("/api/v1/endpoints", headers=HEADERS).text
    assert registry.get(endpoint_id).api_key() == CONFIG["api_key"]

    with sqlite3.connect(store.path) as conn:
        ciphertext = conn.execute(
            "SELECT credential_ciphertext FROM control_endpoints WHERE endpoint_id = ?",
            (endpoint_id,),
        ).fetchone()[0]
    assert CONFIG["api_key"] not in ciphertext

    submitted = client.post(
        "/api/v1/jobs",
        json={"endpoint_id": endpoint_id, "test_type": "concurrency", "parameters": PARAMS},
        headers=HEADERS,
    )
    assert submitted.status_code == 201
    assert submitted.json()["model_id"] == CONFIG["model_id"]
    assert CONFIG["api_key"] not in submitted.text

    blocked = client.put(
        f"/api/v1/endpoints/{endpoint_id}",
        json={**CONFIG, "api_key": None, "model_id": "changed-model"},
        headers=HEADERS,
    )
    assert blocked.status_code == 409
    assert client.delete(f"/api/v1/endpoints/{endpoint_id}", headers=HEADERS).status_code == 409
    assert (
        client.post(
            f"/api/v1/jobs/{submitted.json()['job_id']}/cancel", headers=HEADERS
        ).status_code
        == 200
    )

    updated = client.put(
        f"/api/v1/endpoints/{endpoint_id}",
        json={**CONFIG, "api_key": None, "model_id": "changed-model"},
        headers=HEADERS,
    )
    assert updated.status_code == 200
    assert registry.get(endpoint_id).api_key() == CONFIG["api_key"]
    assert registry.get(endpoint_id).model_id == "changed-model"
    assert client.delete(f"/api/v1/endpoints/{endpoint_id}", headers=HEADERS).status_code == 204
    assert registry.list_public() == []


def test_endpoint_rejects_private_special_address_and_missing_key(tmp_path: Path):
    client, _, _ = _platform(tmp_path)
    assert (
        client.post(
            "/api/v1/endpoints",
            json={**CONFIG, "api_base_url": "http://169.254.169.254/latest"},
            headers=HEADERS,
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/endpoints", json={**CONFIG, "api_key": None}, headers=HEADERS
        ).status_code
        == 422
    )
    too_long = "never-echo-this-key" * 300
    response = client.post(
        "/api/v1/endpoints", json={**CONFIG, "api_key": too_long}, headers=HEADERS
    )
    assert response.status_code == 422
    assert "never-echo-this-key" not in response.text


def test_separate_encryption_key_survives_control_token_rotation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "LLM_TEST_ENDPOINT_ENCRYPTION_KEY", "stable-encryption-key-for-platform-tests-12345"
    )
    client, _, store = _platform(tmp_path)
    endpoint_id = client.post("/api/v1/endpoints", json=CONFIG, headers=HEADERS).json()["id"]
    rotated = Settings("new-control-token-with-enough-entropy-67890", store.path, tmp_path, {})
    assert EndpointRegistry(rotated, store).get(endpoint_id).api_key() == CONFIG["api_key"]
    monkeypatch.setenv(
        "LLM_TEST_ENDPOINT_ENCRYPTION_KEY", "different-encryption-key-for-tests-123456"
    )
    with pytest.raises(EndpointCredentialUnavailable):
        EndpointRegistry(rotated, store).get(endpoint_id)


def test_probe_returns_only_connection_result(tmp_path: Path, monkeypatch):
    client, _, _ = _platform(tmp_path)
    endpoint_id = client.post("/api/v1/endpoints", json=CONFIG, headers=HEADERS).json()["id"]

    class FakeProvider:
        async def get_completion(self, **kwargs):
            assert kwargs["max_tokens"] == 8
            return {"error": None, "full_response_content": "private model output"}

    def fake_provider(_provider, _url, key, _model):
        assert key == CONFIG["api_key"]
        return FakeProvider()

    monkeypatch.setattr("server.api.get_provider", fake_provider)
    response = client.post(f"/api/v1/endpoints/{endpoint_id}/probe", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert "private model output" not in response.text
    assert CONFIG["api_key"] not in response.text


@pytest.mark.asyncio
async def test_worker_reads_endpoint_added_after_settings_loaded(tmp_path: Path, monkeypatch):
    client, _, store = _platform(tmp_path)
    endpoint_id = client.post("/api/v1/endpoints", json=CONFIG, headers=HEADERS).json()["id"]
    job = store.submit(
        test_type="stability",
        endpoint_id=endpoint_id,
        model_id=CONFIG["model_id"],
        parameters={"concurrency": 1, "duration_seconds": 5, "max_tokens": 8},
    )
    claimed = store.claim("test-worker")
    assert claimed is not None
    seen = {}

    async def fake_execute(_job, endpoint, *_args):
        seen["model_id"] = endpoint.model_id
        seen["api_key"] = endpoint.api_key()
        raise RuntimeError("Stop before a real request")

    monkeypatch.setattr("server.worker.execute_job", fake_execute)
    settings = Settings(TOKEN, store.path, tmp_path / "results", {})
    await run_claimed_job(claimed, settings, store, "test-worker")
    assert seen == {"model_id": CONFIG["model_id"], "api_key": CONFIG["api_key"]}
    assert store.get(job["job_id"])["status"] == "failed"
