"""Original preset ranges through authenticated plan/admission, with no worker."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.api import create_app
from server.settings import Endpoint, Settings
from server.specs import JobSubmission, measurement_plan, spec_catalog
from server.store import JobStore

CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CONTEXT = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 130000, 260000, 520000, 1000000]
MATRIX_CONCURRENCY = [1, 2, 4, 8, 16, 64, 128, 256, 512, 1024]
MATRIX_CONTEXT = [length for length in CONTEXT if length != 130000]


def parameters(kind):
    return {
        "concurrency": {
            "selected_concurrencies": CONCURRENCY,
            "rounds_per_level": 1,
            "input_tokens_target": 1000000,
            "max_tokens": 1,
        },
        "prefill": {"token_levels": CONTEXT, "requests_per_level": 1, "max_tokens": 1},
        "long_context": {"context_lengths": CONTEXT, "rounds_per_level": 1, "max_tokens": 1},
        "matrix": {
            "concurrencies": MATRIX_CONCURRENCY,
            "context_lengths": MATRIX_CONTEXT,
            "rounds": 1,
            "max_tokens": 1,
            "enable_warmup": True,
        },
    }[kind]


@pytest.mark.parametrize("kind", ["concurrency", "prefill", "long_context", "matrix"])
def test_every_original_preset_is_accepted_together_and_preserved(kind):
    body = JobSubmission(endpoint_id="lab", test_type=kind, parameters=parameters(kind))
    for field, value in parameters(kind).items():
        assert body.parameters[field] == value
    assert measurement_plan(kind, body.parameters)["total_requests"] > 0


def test_schema_supplies_exact_per_scenario_presets():
    catalog = spec_catalog()

    def choices(kind, field):
        return catalog[kind]["schema"]["properties"][field]["x-presets"]

    assert choices("concurrency", "selected_concurrencies") == CONCURRENCY
    assert choices("prefill", "token_levels") == CONTEXT
    assert choices("long_context", "context_lengths") == CONTEXT
    assert choices("matrix", "concurrencies") == MATRIX_CONCURRENCY
    assert choices("matrix", "context_lengths") == MATRIX_CONTEXT
    assert choices("custom_text", "selected_concurrencies") == [1, 2, 4, 6, 8, 16]


def test_volume_uses_each_cell_target_and_includes_warmup():
    p = {
        "concurrencies": [1, 1024],
        "context_lengths": [260000, 1000000],
        "rounds": 1,
        "max_tokens": 32,
        "enable_warmup": True,
    }
    body = JobSubmission(endpoint_id="lab", test_type="matrix", parameters=p)
    plan = measurement_plan("matrix", body.parameters)
    assert plan["measured_requests"] == plan["warmup_requests"] == 2050
    assert plan["configured_input_token_volume"] == 1260000 * 1025 * 2
    assert plan["maximum_output_token_volume"] == 4100 * 32


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("concurrency", "selected_concurrencies", [1025]),
        ("concurrency", "input_tokens_target", 1000001),
        ("prefill", "token_levels", [1000001]),
        ("long_context", "context_lengths", [1000001]),
        ("matrix", "context_lengths", [1000001]),
    ],
)
def test_values_beyond_declared_limits_are_rejected(kind, field, value):
    p = parameters(kind)
    p[field] = value
    with pytest.raises(ValueError):
        JobSubmission(endpoint_id="lab", test_type=kind, parameters=p)


def test_api_preview_submission_cancel_and_quota(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PARITY_SYNTHETIC", "synthetic")
    store = JobStore(tmp_path / "jobs.db")
    token = "x" * 40
    endpoint = Endpoint(
        "lab", "Synthetic", "OpenAI", "http://127.0.0.1:9/v1", "synthetic", "PARITY_SYNTHETIC"
    )
    settings = Settings(
        api_token=token,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={"lab": endpoint},
    )
    client = TestClient(create_app(settings, store))
    headers = {"Authorization": "Bearer " + token}
    body = {"endpoint_id": "lab", "test_type": "matrix", "parameters": parameters("matrix")}
    plan = client.post("/api/v1/jobs/plan", json=body, headers=headers)
    assert plan.status_code == 200 and plan.json()["total_requests"] == 40300
    queued = client.post("/api/v1/jobs", json=body, headers=headers)
    assert queued.status_code == 201 and queued.json()["status"] == "queued"
    assert queued.json()["parameters"]["context_lengths"] == MATRIX_CONTEXT
    job_id = queued.json()["job_id"]
    assert (
        client.post(f"/api/v1/jobs/{job_id}/cancel", headers=headers).json()["status"]
        == "cancelled"
    )
    body["parameters"]["rounds"] = 20
    assert client.post("/api/v1/jobs/plan", json=body, headers=headers).status_code == 422
