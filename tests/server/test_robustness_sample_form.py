"""Sample editing admission preserves text metadata and rejects invalid workloads."""

import pytest
from fastapi.testclient import TestClient

from server.api import create_app
from server.specs import RobustnessSpec
from tests.server.test_measurement_checkpoints import lab  # noqa: F401


@pytest.mark.parametrize(
    "samples",
    [
        [{"question": " "}],
        [{"question": "x" * 50001}],
        [{"question": "Q", "correct_answer": "x" * 5001}],
        [{"question": "Q", "correct_answer": 2}],
        [],
    ],
)
def test_bad_sample_rows_rejected(samples):
    with pytest.raises(ValueError):
        RobustnessSpec(samples=samples)


def test_duplicate_perturbations_rejected_instead_of_overwriting_results():
    with pytest.raises(ValueError, match="不能重复"):
        RobustnessSpec(samples=[{"question": "Q"}], perturbation_types=["case", "case"])


def test_api_preset_and_queued_job_preserve_row_metadata(lab):  # noqa: F811
    store, endpoint, settings, _ = lab
    client = TestClient(create_app(settings, store))
    headers = {"Authorization": "Bearer " + settings.api_token}
    samples = [
        {
            "question": "line one\nline two",
            "correct_answer": "A",
            "sample_id": "q1",
            "category": "science",
        },
        {"question": "Q2", "correct_answer": "B"},
    ]
    body = {
        "endpoint_id": endpoint.id,
        "test_type": "robustness",
        "parameters": {"samples": samples, "perturbation_types": ["case"], "max_tokens": 8},
    }
    assert client.post("/api/v1/jobs/plan", json=body).status_code == 401
    assert (
        client.post("/api/v1/jobs/plan", json=body, headers=headers).json()["total_requests"] == 4
    )
    saved = client.post("/api/v1/presets", json={**body, "name": "Sample form"}, headers=headers)
    assert saved.status_code == 201 and saved.json()["parameters"]["samples"] == samples
    queued = client.post("/api/v1/jobs", json=body, headers=headers)
    assert queued.status_code == 201 and queued.json()["parameters"]["samples"] == samples
    assert store.get(queued.json()["job_id"])["status"] == "queued"
