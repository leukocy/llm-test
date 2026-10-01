"""Multi-run display/export semantics through the actual authenticated API."""

import csv
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.run_lifecycle import RunStatus
from server.api import create_app
from server.quality_matrix import quality_matrix
from server.settings import Settings
from server.store import JobStore
from tests.server.test_paired_quality import dataset


def result(outcomes):
    return dataset(outcomes, correct_samples=sum(outcomes))


def entries():
    return [
        (
            {"job_id": "a", "model_id": "same-model", "endpoint_id": "ep-a"},
            {"datasets": {"mmlu": result([True, False]), "gsm8k": result([True] * 4)}},
        ),
        (
            {"job_id": "b", "model_id": "same-model", "endpoint_id": "ep-b"},
            {"datasets": {"mmlu": result([True] * 3)}},
        ),
        (
            {"job_id": "c", "model_id": "other-model", "endpoint_id": "ep-c"},
            {"datasets": {"gsm8k": result([False] * 4)}},
        ),
    ]


def test_matrix_preserves_each_denominator_missing_cells_and_same_named_runs():
    matrix = quality_matrix(entries(), "final")
    assert len(matrix["rows"]) == 3 and len(matrix["figure"]["data"]) == 3
    assert matrix["rows"][0]["cells"]["mmlu"]["accuracy"] == 0.5
    assert matrix["rows"][1]["cells"]["mmlu"]["total"] == 3
    assert matrix["rows"][1]["cells"]["gsm8k"] is None
    assert matrix["figure"]["data"][1]["y"] == [None, 100]
    assert "<svg" in matrix["exports"]["html"]
    assert "未记录" in matrix["exports"]["markdown"]


def test_standard_score_does_not_fall_back_to_final_and_judge_is_excluded():
    data = entries()
    data[0][1]["datasets"]["mmlu"]["details"][0]["is_judge_corrected"] = True
    matrix = quality_matrix(data, "standard")
    assert matrix["rows"][0]["cells"]["mmlu"]["correct"] == 0
    data[1][1]["datasets"]["mmlu"]["details"] = []
    cell = quality_matrix(data, "standard")["rows"][1]["cells"]["mmlu"]
    assert cell["accuracy"] is None and cell["warnings"]


def test_source_and_condition_mismatch_warn_and_spreadsheet_html_are_safe():
    data = entries()
    data[0][0]["model_id"] = "=HYPERLINK(1)"
    data[1][1]["datasets"]["mmlu"]["config"]["temperature"] = 0.5
    matrix = quality_matrix(data, "final")
    assert "条件" in matrix["warnings"]["mmlu"]
    rows = list(csv.DictReader(io.StringIO(matrix["exports"]["csv"].lstrip("\ufeff"))))
    assert rows[0]["model_id"].startswith("'=HYPERLINK")
    data[0][0]["model_id"] = "<script>bad</script>"
    assert "<script>bad" not in quality_matrix(data, "final")["exports"]["html"]


def test_authenticated_api_checks_terminal_quality_unique_ids_and_bound(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    token = "x" * 40
    settings = Settings(
        api_token=token, db_path=store.path, artifact_root=tmp_path / "artifacts", endpoints={}
    )
    identities = []
    for _, payload in entries():
        job = store.submit(
            test_type="quality", endpoint_id="synthetic", model_id="same-model", parameters={}
        )
        store.claim("fixture")
        path = settings.artifact_root / job["job_id"]
        path.mkdir(parents=True)
        (path / "report.json").write_text(json.dumps(payload))
        store.finish(
            job["job_id"],
            "fixture",
            outcome=RunStatus.COMPLETED,
            result_artifact=f"{job['job_id']}/report.json",
        )
        identities.append(job["job_id"])
    client = TestClient(create_app(settings, store))
    route = "/api/v1/compare/matrix"
    headers = {"Authorization": "Bearer " + token}
    body = {"job_ids": identities}
    assert client.post(route, json=body).status_code == 401
    response = client.post(route, json=body, headers=headers)
    assert response.status_code == 200
    assert len(response.json()["rows"]) == 3
    assert (
        client.post(route, json={"job_ids": [identities[0]] * 2}, headers=headers).status_code
        == 422
    )
    assert client.post(route, json={"job_ids": identities * 3}, headers=headers).status_code == 422
    pending = store.submit(
        test_type="quality", endpoint_id="synthetic", model_id="m", parameters={}
    )
    assert (
        client.post(
            route, json={"job_ids": [identities[0], pending["job_id"]]}, headers=headers
        ).status_code
        == 409
    )
