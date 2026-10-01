"""Usage partitions must be disjoint, complete and tied to persisted timing."""

import json

import pytest

from core.provider_usage import openai_usage_counts
from server.sample_visuals import sample_visuals, sample_visuals_html
from tests.server.test_data_api import auth, env  # noqa: F401


def native(**usage):
    return {
        "sample_id": "<script>",
        "ttft_ms": 300,
        "total_time_ms": 1200,
        "ttut_ms": 0,
        "measurement_provenance": {
            "output_token_scope": "response_candidates_excluding_thoughts",
            "ttft_scope": "first_answer_text",
            "provider_usage": usage,
        },
    }


def test_native_usage_and_missing_ttut_are_not_fabricated():
    result = sample_visuals(
        native(
            promptTokenCount=100,
            thoughtsTokenCount=20,
            candidatesTokenCount=30,
            cachedContentTokenCount=80,
            totalTokenCount=150,
        )
    )
    assert result["token_figure"]["data"][0]["values"] == [100, 20, 30]
    assert result["latencies"][0]["value"] == 300
    assert result["latencies"][1]["figure"] is None
    assert "首个" in result["latencies"][1]["scope"]
    html = sample_visuals_html(result)
    assert html.count("<svg") == 3 and "&lt;script&gt;" in html and "<script>" not in html


@pytest.mark.parametrize(
    "usage",
    [
        {"promptTokenCount": 100, "candidatesTokenCount": 30},
        {"promptTokenCount": 0, "thoughtsTokenCount": 0, "candidatesTokenCount": 0},
        {
            "promptTokenCount": 100,
            "thoughtsTokenCount": 20,
            "candidatesTokenCount": 30,
            "totalTokenCount": 10,
        },
    ],
)
def test_missing_zero_or_conflicting_native_counts_do_not_create_percentages(usage):
    assert sample_visuals(native(**usage))["token_figure"] is None


def test_openai_reasoning_is_subtracted_from_total_output_and_counts_are_bounded():
    usage = openai_usage_counts(
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 20},
            "arbitrary": "never saved",
            "total_tokens": True,
        }
    )
    assert "arbitrary" not in usage and "total_tokens" not in usage
    result = sample_visuals({"measurement_provenance": {"provider_usage": usage}})
    assert result["token_figure"]["data"][0]["values"] == [100, 20, 30]
    assert result["tokens"][2]["label"] == "非思考输出"
    usage["completion_tokens_details"]["reasoning_tokens"] = 60
    assert (
        sample_visuals({"measurement_provenance": {"provider_usage": usage}})["token_figure"]
        is None
    )
    assert openai_usage_counts({"prompt_tokens": -1, "completion_tokens": 10**13}) is None


def test_cache_and_error_timings_do_not_become_live_measurements():
    for field in ("from_cache", "error"):
        row = native()
        if field == "from_cache":
            row["measurement_provenance"][field] = True
        else:
            row[field] = "request failed"
        assert all(item["figure"] is None for item in sample_visuals(row)["latencies"])


def test_authenticated_sample_endpoint_is_indexed_and_exports_same_observations(env):  # noqa: F811
    from core.database.connection import Database

    client, _, settings = env
    job = client.post(
        "/api/v1/jobs",
        json={
            "endpoint_id": "lab",
            "test_type": "quality",
            "parameters": {"datasets": ["gsm8k"], "max_samples": 1},
        },
        headers=auth(),
    ).json()
    directory = settings.artifact_root / job["job_id"]
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "gsm8k": {
                        "details": [
                            native(
                                promptTokenCount=100, thoughtsTokenCount=0, candidatesTokenCount=30
                            )
                        ]
                    }
                }
            }
        )
    )
    Database().execute(
        "UPDATE control_jobs SET result_artifact=? WHERE job_id=?",
        (f"{job['job_id']}/report.json", job["job_id"]),
    )
    url = f"/api/v1/jobs/{job['job_id']}/report/sample-visuals?dataset=gsm8k&index=0"
    assert client.get(url).status_code == 401
    response = client.get(url, headers=auth())
    assert response.status_code == 200
    assert response.json()["tokens"][1]["value"] == 0 and response.json()["index"] == 0
    html = client.get(url + "&format=html", headers=auth())
    assert html.status_code == 200 and "<svg" in html.text
    assert client.get(url.replace("index=0", "index=1"), headers=auth()).status_code == 404
    assert client.get(url.replace("index=0", "index=-1"), headers=auth()).status_code == 422


@pytest.mark.asyncio
async def test_quality_collection_retains_openai_reasoning_usage_without_model_calls(
    tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock

    from core.quality_evaluator import QualityEvaluator

    monkeypatch.chdir(tmp_path)
    evaluator = QualityEvaluator(
        api_base_url="http://127.0.0.1:9/v1",
        model_id="synthetic",
        api_key="synthetic",  # pragma: allowlist secret
        enable_cache=False,
        output_dir=str(tmp_path),
    )
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 20},
        "untrusted_text": "not retained",
    }
    complete = AsyncMock(
        return_value={
            "error": None,
            "full_response_content": "A",
            "usage_info": usage,
            "start_time": 0,
            "end_time": 1,
            "first_token_time": 0.1,
        }
    )
    monkeypatch.setattr(evaluator.provider, "get_completion", complete)
    monkeypatch.setattr(evaluator, "count_tokens", lambda text: 1)
    result = await evaluator._get_response_with_metrics("Q", use_cache=False, retries=0)
    assert complete.await_count == 1
    retained = result["measurement_provenance"]["provider_usage"]
    assert retained["completion_tokens_details"]["reasoning_tokens"] == 20
    assert "untrusted_text" not in retained
    assert sample_visuals(result)["token_figure"]["data"][0]["values"] == [100, 20, 30]
