"""Reference coverage is distinct from incorrect answers and text changes."""

from dataclasses import asdict

import pytest

from core.robustness_tester import PerturbationType, RobustnessTester
from server.reports import render_robustness_markdown
from tests.server.test_data_api import env  # noqa: F401


@pytest.mark.asyncio
async def test_reference_coverage_and_noop_counts():
    tester = RobustnessTester([PerturbationType.NUMBER_FORMAT])
    responses = iter(["42", "wrong", "wrong", "wrong"])

    async def respond(_):
        return next(responses)

    report = await tester.test_batch(
        [{"question": "hello", "correct_answer": "42"}, {"question": "hello"}], respond
    )
    assert report.original_accuracy == 1
    assert report.perturbed_accuracy == report.overall_robustness == 0
    assert report.accuracy_drop == 1
    assert report.scored_samples == report.unscored_samples == report.scored_perturbations == 1
    assert report.unchanged_perturbations == 2
    assert report.overall_consistency == 0.5
    assert report.results[1].original_correct is None
    assert report.results[1].perturbed_results[0]["is_correct"] is None
    assert report.results[1].robustness_score is None
    assert report.sensitivity_by_type == {"number_format": 1}


@pytest.mark.asyncio
async def test_no_reference_is_unscored_even_for_empty_response():
    tester = RobustnessTester([PerturbationType.NUMBER_FORMAT])

    async def respond(_):
        return ""

    report = await tester.test_batch([{"question": "hello", "correct_answer": "  "}], respond)
    assert report.original_accuracy is report.perturbed_accuracy is report.accuracy_drop is None
    assert report.overall_robustness is None
    assert report.overall_consistency == 1
    assert report.scored_samples == report.scored_perturbations == 0
    assert report.unscored_samples == report.unchanged_perturbations == 1
    assert report.sensitivity_by_type == {}
    md = render_robustness_markdown(
        {"job_id": "lab", "model_id": "synthetic"}, {"robustness": asdict(report)}
    )
    assert "原始准确率：未评分" in md and "评分样本：0" in md
    assert "并非独立观测" in md


@pytest.mark.asyncio
async def test_unscored_report_exports_preserve_null_and_coverage(env):  # noqa: F811
    import json

    from core.database.connection import Database
    from tests.server.test_data_api import auth

    client, _, settings = env
    job = client.post(
        "/api/v1/jobs",
        json={
            "endpoint_id": "lab",
            "test_type": "robustness",
            "parameters": {"samples": [{"question": "hello"}]},
        },
        headers=auth(),
    ).json()
    tester = RobustnessTester([PerturbationType.NUMBER_FORMAT])

    async def respond(_):
        return "42"

    report = await tester.test_batch([{"question": "hello"}], respond)
    payload = {"job_id": job["job_id"], "model_id": "synthetic", "robustness": asdict(report)}
    directory = settings.artifact_root / job["job_id"]
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(json.dumps(payload))
    Database().execute(
        "UPDATE control_jobs SET result_artifact = ? WHERE job_id = ?",
        (f"{job['job_id']}/report.json", job["job_id"]),
    )
    base = f"/api/v1/jobs/{job['job_id']}/report"
    assert client.get(base, headers=auth()).json()["robustness"]["original_accuracy"] is None
    for fmt in ("html", "markdown"):
        response = client.get(base + "?format=" + fmt, headers=auth())
        assert response.status_code == 200
        assert "未评分" in response.text and "评分样本" in response.text
        assert "并非独立观测" in response.text
