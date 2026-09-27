"""Presets and warehouse reads preserve the control plane's safety boundaries."""

import csv
import io
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.run_lifecycle import RunStatus
from server.api import create_app
from server.settings import Endpoint, Settings
from server.store import JobStore


@pytest.fixture
def migrated_client(tmp_path: Path) -> tuple[TestClient, Path, dict[str, str]]:
    database = tmp_path / "platform.db"
    store = JobStore(database)
    settings = Settings(
        api_token="a" * 40,
        db_path=database,
        artifact_root=tmp_path / "artifacts",
        endpoints={
            "lab": Endpoint(
                id="lab",
                label="Lab",
                provider="OpenAI",
                api_base_url="http://127.0.0.1:9010/v1",
                model_id="model-a",
                api_key_env="LAB_KEY",  # pragma: allowlist secret
            )
        },
    )
    return (
        TestClient(create_app(settings, store)),
        database,
        {"Authorization": "Bearer " + "a" * 40},
    )


def test_preset_roundtrip_and_strict_admission(migrated_client):
    client, _, headers = migrated_client
    payload = {
        "name": "标准并发测量",
        "schema_version": 1,
        "endpoint_id": "lab",
        "test_type": "concurrency",
        "parameters": {
            "selected_concurrencies": [1, 4],
            "rounds_per_level": 2,
            "max_tokens": 64,
        },
        "run_config": {"temperature": 0.3, "random_seed": 42},
    }
    assert client.get("/api/v1/presets").status_code == 401
    created = client.post("/api/v1/presets", json=payload, headers=headers)
    assert created.status_code == 201
    preset_id = created.json()["preset_id"]
    assert created.json()["parameters"]["input_tokens_target"] == 0
    assert created.json()["run_config"]["temperature"] == 0.3
    assert created.json()["run_config"]["random_seed"] == 42
    assert client.post("/api/v1/presets", json=payload, headers=headers).status_code == 409
    assert len(client.get("/api/v1/presets", headers=headers).json()["items"]) == 1

    updated = client.put(
        f"/api/v1/presets/{preset_id}",
        json={**payload, "name": "回归方案"},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "回归方案"
    invalid = {
        **payload,
        "parameters": {
            **payload["parameters"],
            "api_key": "placeholder",  # pragma: allowlist secret
        },
    }
    assert client.post("/api/v1/presets", json=invalid, headers=headers).status_code == 422
    assert client.delete(f"/api/v1/presets/{preset_id}", headers=headers).status_code == 204
    assert client.get("/api/v1/presets", headers=headers).json()["items"] == []


def _insert_run(
    path: Path,
    test_id: str,
    *,
    model: str,
    machine: str,
    tps: float,
    supersedes: str | None = None,
    config_json: str = "{}",
) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """INSERT INTO test_runs
               (test_id, test_type, status, model_id, machine_id, external_level,
                avg_tps, avg_ttft, created_at, supersedes_test_id, config_json)
               VALUES (?, 'concurrency', 'completed', ?, ?, 'publishable', ?, 0.2,
                       '2026-09-26 10:00:00', ?, ?)""",
            (test_id, model, machine, tps, supersedes, config_json),
        )


def test_warehouse_filters_superseded_runs_and_exports(migrated_client):
    client, path, headers = migrated_client
    _insert_run(path, "old", model="model-a", machine="node-a", tps=30)
    _insert_run(path, "new", model="model-a", machine="node-a", tps=40, supersedes="old")
    _insert_run(path, "other", model="model-b", machine="node-b", tps=90)
    assert client.get("/api/v1/warehouse").status_code == 401
    response = client.get("/api/v1/warehouse?model_id=model-a", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["scope"]["matched_total"] == 2
    assert data["scope"]["shown"] == 1
    assert data["rows"][0]["test_id"] == "new"
    assert data["matrix"]["cells"]["node-a"]["model-a"] == 40
    assert data["kpis"]["publishable"] == 1
    assert data["inventory"][0]["machine_id"] == "node-a"
    detail = client.get("/api/v1/warehouse/runs/new", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["fields"]["decode_tps"] == 40
    assert client.get("/api/v1/warehouse/runs/missing", headers=headers).status_code == 404
    exported = client.get(
        "/api/v1/warehouse/export?model_id=model-a&template=hmTest&format=csv",
        headers=headers,
    )
    assert exported.status_code == 200
    rows = list(csv.DictReader(io.StringIO(exported.text.lstrip("\ufeff"))))
    assert [row["test_id"] for row in rows] == ["new"]
    assert rows[0]["model_name"] == "model-a"


def test_warehouse_refuses_partial_or_malformed_export(migrated_client, monkeypatch):
    client, path, headers = migrated_client
    _insert_run(path, "one", model="a", machine="x", tps=1)
    _insert_run(path, "two", model="b", machine="y", tps=2)
    monkeypatch.setattr("server.warehouse.SCAN_LIMIT", 1)
    response = client.get("/api/v1/warehouse", headers=headers)
    assert response.json()["scope"]["truncated"] is True
    assert (
        client.get("/api/v1/warehouse/export?template=hmTest", headers=headers).status_code == 409
    )

    monkeypatch.setattr("server.warehouse.SCAN_LIMIT", 1000)
    _insert_run(path, "bad", model="c", machine="z", tps=3, config_json="{")
    response = client.get("/api/v1/warehouse", headers=headers)
    assert response.json()["scope"]["invalid_rows"] == 1
    assert (
        client.get("/api/v1/warehouse/export?template=hmTest", headers=headers).status_code == 409
    )


def test_warehouse_nonfinite_values_are_missing(migrated_client):
    client, path, headers = migrated_client
    _insert_run(path, "nonfinite", model="model-a", machine="node-a", tps=float("inf"))
    response = client.get("/api/v1/warehouse", headers=headers)
    assert response.status_code == 200
    assert response.json()["rows"][0]["decode_tps"] is None
    assert response.json()["matrix"]["cells"]["node-a"]["model-a"] is None
    exported = client.get("/api/v1/warehouse/export?template=hmTest&format=json", headers=headers)
    assert exported.status_code == 200
    assert json.loads(exported.text)[0]["decode_tps"] is None


def test_quality_failure_csv_is_complete_and_spreadsheet_safe(migrated_client):
    client, path, headers = migrated_client
    store = JobStore(path)
    job = store.submit(test_type="quality", endpoint_id="lab", model_id="model-a", parameters={})
    assert store.claim("worker") is not None
    artifact = path.parent / "artifacts" / "report.json"
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "datasets": {
                    "GSM8K": {
                        "details": [
                            {
                                "sample_id": "1",
                                "is_correct": False,
                                "question": '=HYPERLINK("https://example.invalid")',
                                "prompt": "question",
                                "model_response": "wrong",
                            },
                            {"sample_id": "2", "is_correct": True},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    store.finish(
        job["job_id"], "worker", outcome=RunStatus.COMPLETED, result_artifact="report.json"
    )
    response = client.get(f"/api/v1/jobs/{job['job_id']}/report/errors.csv", headers=headers)
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text.lstrip("\ufeff"))))
    assert len(rows) == 1
    assert rows[0]["dataset"] == "GSM8K"
    assert rows[0]["question"].startswith("'=HYPERLINK")
