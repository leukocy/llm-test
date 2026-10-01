"""Scientific report statistics, export fidelity and actual control API wiring."""

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.quality_evaluator import QualityEvaluator
from core.run_lifecycle import RunStatus
from evaluators.base_evaluator import EvaluationResult, SampleResult
from evaluators.mbpp_evaluator import MBPPEvaluator
from server.api import create_app
from server.quality_analysis import (
    quality_analysis,
    quality_analysis_html,
    quality_analysis_markdown,
)
from server.quality_export import quality_samples_csv, quality_summary_csv
from server.settings import Settings
from server.store import JobStore


def fixture_report():
    rows = []
    for index, ttft in enumerate([1, 2, 100, 900, 800]):
        rows.append(
            SampleResult(
                str(index),
                f"q{index}",
                "A",
                "B",
                "B",
                index in (1, 3),
                ttft_ms=ttft,
                tps=10 * (index + 1),
                total_time_ms=200 + index,
                latency_ms=210 + index,
                input_tokens=index * 2,
                output_tokens=20,
                category="cat",
                error="request failed" if index == 4 else None,
                measurement_provenance={
                    "version": "quality-response-v1",
                    "from_cache": index == 3,
                    "input_token_source": "api",
                    "output_token_source": "tokenizer",
                },
                evaluation_method="regex",
                answer_parse_method="letter",
                answer_parse_confidence=0.9,
            )
        )
    result = EvaluationResult("gsm8k", "synthetic", 0.4, 5, 2, details=rows)
    return {"job_id": "fixture", "model_id": "synthetic", "datasets": {"gsm8k": result.to_dict()}}


def test_statistics_exclude_cache_and_request_errors_but_keep_wrong_answers():
    analysis = quality_analysis(fixture_report())
    group = analysis["datasets"]["gsm8k"]
    stat = group["metrics"]["ttft_ms"]
    assert stat["count"] == stat["eligible"] == 3
    assert stat["mean"] == pytest.approx(103 / 3)
    assert stat["median"] == 2 and stat["p95"] == pytest.approx(90.2)
    assert stat["p99"] == pytest.approx(98.04)
    assert group["cache_count"] == 1 and group["request_errors"] == 1
    assert group["metrics"]["input_tokens"]["total"] == 6
    assert group["metrics"]["input_tokens"]["count"] == 3  # explicit API zero is observed
    assert group["metrics"]["output_tokens"]["sources"] == {"tokenizer": 3}
    assert group["accuracy"] == 0.4 and group["standard_correct"] == 2
    assert group["ci95"] is None  # a request error is not a wrong model answer
    report = fixture_report()
    dataset = report["datasets"]["gsm8k"]
    dataset["details"] = dataset["details"][:-1]
    dataset["total_samples"] = 4
    clean = quality_analysis(report)["datasets"]["gsm8k"]
    assert clean["ci95"][0] < 0.5 < clean["ci95"][1]
    assert analysis["figures"]["ttft_ms"]["data"][0]["y"] == [stat["mean"]]
    assert "radar" not in analysis["figures"]


def test_missing_legacy_observations_are_not_fabricated_zeros():
    report = fixture_report()
    report["datasets"]["gsm8k"]["details"] = [{"sample_id": "a", "is_correct": False}]
    group = quality_analysis(report)["datasets"]["gsm8k"]
    assert group["metrics"]["ttft_ms"]["mean"] is None
    assert group["metrics"]["input_tokens"]["total"] is None
    assert group["confidence"]["mean"] is None and not group["complete"]
    assert len(group["warnings"]) == 2 and group["judge_corrected"] is None


@pytest.mark.parametrize("flag", ["false", None])
def test_invalid_judge_flags_do_not_fabricate_rule_counts_or_intervals(flag):
    report = fixture_report()
    report["datasets"]["gsm8k"]["details"][0]["is_judge_corrected"] = flag
    group = quality_analysis(report)["datasets"]["gsm8k"]
    assert group["standard_correct"] is None and group["ci95"] is None


def test_radar_uses_each_dataset_denominator_and_safe_labels():
    report = fixture_report()
    group = report["datasets"]["gsm8k"]
    report["datasets"].update(
        {
            "second": {**group, "correct_samples": 1},
            "<script>bad</script>": {**group, "correct_samples": 3},
        }
    )
    analysis = quality_analysis(report)
    assert analysis["figures"]["radar"]["data"][0]["r"] == [40, 20, 60, 40]
    html = quality_analysis_html(analysis)
    assert "<script>bad" not in html and "&lt;script&gt;bad" in html
    assert "<svg" in html and "90.2" in html
    markdown = quality_analysis_markdown(analysis)
    assert "正值解析置信度" in markdown
    assert "| second |" in markdown.split("###")[0]


def test_csv_exports_all_samples_summary_counts_and_formula_safe_text():
    report = fixture_report()
    report["datasets"]["gsm8k"]["details"][0]["question"] = "=HYPERLINK(1)"
    rows = list(csv.DictReader(io.StringIO(quality_samples_csv(report).lstrip("\ufeff"))))
    assert len(rows) == 5 and rows[0]["question"].startswith("'=HYPERLINK")
    assert json.loads(rows[0]["measurement_provenance"])["input_token_source"] == "api"
    summary = list(csv.DictReader(io.StringIO(quality_summary_csv(report).lstrip("\ufeff"))))[0]
    assert summary["ttft_ms_count"] == "3" and float(summary["ttft_ms_mean"]) == pytest.approx(
        103 / 3
    )
    assert summary["model"] == "synthetic" and summary["total"] == "5"


def test_report_api_enriches_json_and_all_exports_from_one_source(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    token = "x" * 40
    settings = Settings(
        api_token=token, db_path=store.path, artifact_root=tmp_path / "artifacts", endpoints={}
    )
    job = store.submit(
        test_type="quality", endpoint_id="synthetic", model_id="synthetic", parameters={}
    )
    store.claim("fixture")
    path = settings.artifact_root / job["job_id"]
    path.mkdir(parents=True)
    (path / "report.json").write_text(json.dumps(fixture_report()))
    store.finish(
        job["job_id"],
        "fixture",
        outcome=RunStatus.COMPLETED,
        result_artifact=f"{job['job_id']}/report.json",
    )
    client = TestClient(create_app(settings, store))
    base = f"/api/v1/jobs/{job['job_id']}/report"
    headers = {"Authorization": "Bearer " + token}
    assert client.get(base + "/summary.csv").status_code == 401
    payload = client.get(base, headers=headers).json()
    assert payload["analysis"]["datasets"]["gsm8k"]["metrics"]["ttft_ms"]["p95"] == pytest.approx(
        90.2
    )
    for suffix in ("?format=html", "?format=markdown", "/summary.csv", "/samples.csv"):
        response = client.get(base + suffix, headers=headers)
        assert response.status_code == 200 and "gsm8k" in response.text
    assert "<svg" in client.get(base + "?format=html", headers=headers).text


@pytest.mark.asyncio
async def test_primary_response_provenance_survives_secondary_calls():
    evaluator = MBPPEvaluator(num_shots=0)

    async def evaluate(sample, get_response_func, index):
        await get_response_func("primary")
        await get_response_func("secondary")
        return SampleResult(str(index), "q", "a", "a", "a", True)

    evaluator.evaluate_single = evaluate

    async def respond(prompt):
        return {"content": "A", "measurement_provenance": {"from_cache": prompt == "primary"}}

    results = await evaluator.evaluate_batch([{}], respond, concurrency=1)
    assert results[0].measurement_provenance == {"from_cache": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,expected", [({"prompt_tokens": 0, "completion_tokens": 0}, "api"), ({}, "tokenizer")]
)
async def test_provider_explicit_zero_is_preserved_and_estimates_are_labelled(
    monkeypatch, tmp_path, usage, expected
):
    monkeypatch.setattr(QualityEvaluator, "_init_tokenizer", lambda *a: None)
    evaluator = QualityEvaluator(
        api_base_url="http://127.0.0.1:9/v1",
        model_id="synthetic",
        provider="OpenAI",
        api_key="synthetic",  # pragma: allowlist secret
        enable_cache=False,
        output_dir=str(tmp_path / "outputs"),
    )

    async def complete(**kwargs):
        return {
            "full_response_content": "A",
            "usage_info": usage,
            "start_time": 1,
            "first_token_time": 2,
            "end_time": 3,
            "timing_clock": "client_monotonic",
        }

    evaluator.provider = SimpleNamespace(get_completion=complete)
    evaluator.count_tokens = lambda text: 7
    result = await evaluator._get_response_with_metrics("Q", use_cache=False)
    assert result["input_tokens"] == (0 if expected == "api" else 7)
    assert result["measurement_provenance"]["input_token_source"] == expected
