"""Stored diagnostics, report exports and real evaluator annotation flow."""

import csv
import io

import pytest

from core.failure_analyzer import FailureAnalyzer, FailureCategory
from server.failure_reports import failure_html, failure_markdown, failure_summary
from server.quality_analysis import quality_analysis
from server.quality_export import quality_errors_csv
from server.worker import run_claimed_job
from tests.server.test_online_comparison import lab as lab
from tests.server.test_online_comparison import report, submit


def records():
    return {
        "total_samples": 4,
        "details": [
            {"is_correct": False, "error": None},
            {"is_correct": False, "error": None},
            {"is_correct": True},
            {"is_correct": False, "error": "request failed"},
        ],
        "extended_metrics": {
            "failure_analysis": {
                "version": "heuristic-failure-v1",
                "category_distribution": {"logic_error": 2},
                "top_issues": ["logic_error: 2"],
                "suggestions": ["Review boundary conditions"],
            }
        },
    }


def test_rate_excludes_request_failures_and_preserves_stored_suggestions():
    summary = failure_summary(records())
    assert summary["failed_samples"] == 2 and summary["response_samples"] == 3
    assert summary["failure_rate"] == pytest.approx(2 / 3)
    assert summary["request_errors"] == 1 and summary["status"] == "recorded"
    assert summary["figure"]["data"][0]["values"] == [2]
    assert "Review boundary conditions" in failure_html("dataset", summary)
    assert "Review boundary conditions" in failure_markdown("dataset", summary)


@pytest.mark.parametrize("distribution", [{"bad": -1}, {"bad": True}, {"bad": 3}])
def test_invalid_or_mismatched_distribution_does_not_become_verified_chart(distribution):
    result = records()
    result["extended_metrics"]["failure_analysis"]["category_distribution"] = distribution
    summary = failure_summary(result)
    assert summary["figure"] is None and summary["warnings"]


def test_no_summary_and_incomplete_legacy_records_are_not_reanalysed():
    result = records()
    result["extended_metrics"] = {}
    summary = failure_summary(result)
    assert summary["status"] == "unavailable" and not summary["suggestions"]
    result["total_samples"] = 99
    assert failure_summary(result)["failure_rate"] is None
    result = {"total_samples": 1, "details": [{"is_correct": True}]}
    assert failure_summary(result)["status"] == "none"


def test_html_and_markdown_escape_stored_suggestions():
    result = records()
    result["extended_metrics"]["failure_analysis"]["suggestions"] = ["<script>bad</script>"]
    summary = failure_summary(result)
    for output in (failure_html("<img>", summary), failure_markdown("dataset", summary)):
        assert "<script>" not in output and "&lt;script&gt;" in output


def test_code_failure_is_observed_test_outcome_not_network_timeout_guess():
    analyzer = FailureAnalyzer()
    case = analyzer.analyze_single(
        "x", "q", "answer", "code", "code", execution_error="TimeoutError: code exceeded 10s"
    )
    assert case.category is FailureCategory.CODE_TEST_FAILURE
    assert "具体错误根因尚未核验" in case.root_cause
    assert "Check网络Connect" not in case.suggestions


@pytest.mark.asyncio
async def test_real_worker_persists_sample_diagnostics_and_same_export_summary(lab):
    store, settings, _, _, _, _, _ = lab
    group = submit(lab)
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    job = store.get(group["jobs"][1]["job_id"])
    assert job["status"] == "completed"
    payload = report(settings, job)
    result = payload["datasets"]["ceval"]
    stored = result["extended_metrics"]["failure_analysis"]
    assert stored["version"] == "heuristic-failure-v1"
    assert sum(stored["category_distribution"].values()) == 6
    assert all(
        row["failure_category"] and row["failure_confidence"] is not None
        for row in result["details"]
    )
    projected = quality_analysis(payload)["datasets"]["ceval"]["failure_analysis"]
    assert projected["status"] == "recorded" and projected["failed_samples"] == 6
    assert projected["suggestions"] == stored["suggestions"]
    rows = list(csv.DictReader(io.StringIO(quality_errors_csv(payload).lstrip("\ufeff"))))
    assert (
        len(rows) == 6 and rows[0]["failure_category"] == result["details"][0]["failure_category"]
    )
