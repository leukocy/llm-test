"""Quality-report uncertainty and sample-coverage checks."""

import pytest

from evaluators.base_evaluator import EvaluationResult, SampleResult
from ui.quality_reports import (
    build_quality_performance_figures,
    build_quality_performance_summary,
    generate_quality_summary,
    render_accuracy_chart,
    render_radar_chart,
)


def _result(name: str, correct: int, total: int, stored: float = 0.0) -> EvaluationResult:
    return EvaluationResult(
        dataset_name=name,
        model_id="m",
        accuracy=stored,
        correct_samples=correct,
        total_samples=total,
    )


def test_quality_summary_uses_counts_and_reports_wilson_bounds():
    summary = generate_quality_summary({"A": _result("A", 8, 10, stored=0.9)})

    assert summary.loc[0, "Accuracy"] == pytest.approx(0.8)
    assert summary.loc[0, "Accuracy CI low (%)"] == pytest.approx(49.0162, abs=0.001)
    assert summary.loc[0, "Accuracy CI high (%)"] == pytest.approx(94.3318, abs=0.001)
    chart = render_accuracy_chart({"A": _result("A", 8, 10)})
    assert chart.data[0].type == "scatter"
    assert chart.data[0].error_x.array[0] > 0


def test_empty_quality_counts_are_unavailable_and_coverage_is_not_radar():
    results = {"A": _result("A", 0, 0), "B": _result("B", 8, 10)}
    summary = generate_quality_summary(results)

    assert (
        summary.loc[0, "Accuracy"] is None
        or summary.loc[0, "Accuracy"] != summary.loc[0, "Accuracy"]
    )
    assert list(render_accuracy_chart(results).data[0].y) == ["B"]
    assert render_radar_chart(results).data[0].type == "bar"


def test_quality_performance_excludes_failed_and_missing_sample_measurements():
    result = _result("A", 2, 4)
    result.details = [
        SampleResult("1", "q", "a", "a", "a", True, ttft_ms=100, tps=20),
        SampleResult("2", "q", "a", "a", "a", True, ttft_ms=300, tps=40),
        SampleResult("3", "q", "a", "a", "a", False, ttft_ms=0, tps=0),
        SampleResult("4", "q", "a", "a", "a", False, error="timeout", ttft_ms=999, tps=999),
    ]

    summary = build_quality_performance_summary({"A": result})

    assert summary.loc[0, "TTFT valid n"] == 2
    assert summary.loc[0, "TTFT P50 (ms)"] == 200
    assert summary.loc[0, "TPS P10"] == pytest.approx(22)
    assert len(build_quality_performance_figures(summary)) == 2
