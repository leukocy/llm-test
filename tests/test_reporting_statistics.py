import math

import pandas as pd
import pytest

from ui.reporting.presentation import build_quantile_figure
from ui.reporting.statistics import (
    analysis_to_markdown,
    build_scientific_summary,
    compact_summary_table,
    wilson_interval,
)


def test_wilson_interval_covers_observed_rate_and_handles_edges():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.490162, abs=1e-5)
    assert high == pytest.approx(0.943318, abs=1e-5)
    assert wilson_interval(0, 0) is None
    assert wilson_interval(11, 10) is None


def test_scientific_summary_counts_failures_and_excludes_invalid_measurements():
    rows = pd.DataFrame(
        {
            "concurrency": [1, 1, 1, 2],
            "error": [None, "timeout", "", None],
            "ttft": [0.2, 90.0, 0.4, 0.0],
            "tps": [50.0, 500.0, 0.0, 20.0],
            "tpot": [0.02, 0.001, 0.04, 0.05],
            "token_calc_method": ["API"] * 4,
        }
    )

    report = build_scientific_summary(rows, "concurrency")

    assert report.rows == 4
    assert report.success_count == 3
    assert report.success_rate == 75
    level_one = report.groups.iloc[0]
    assert level_one["Requests"] == 3
    assert level_one["Succeeded"] == 2
    assert level_one["TTFT (s) valid n"] == 2
    assert level_one["TTFT (s) P50"] == pytest.approx(0.3)
    assert level_one["Decode TPS (tokens/s) valid n"] == 1
    assert level_one["Decode TPS (tokens/s) P50"] == 50
    assert level_one["TPOT (ms) P50"] == 30
    compact = compact_summary_table(report)
    assert compact.loc[0, "Success 95% CI (%)"]
    assert "TTFT (s) P50" in compact
    assert "TPOT (ms) P50" not in compact
    assert any("failed request" in note for note in report.notes)
    assert "95% Wilson interval" in analysis_to_markdown(report)


def test_scientific_summary_marks_unknown_status_and_missing_metrics():
    report = build_scientific_summary(
        pd.DataFrame({"input_tokens_target": [1024], "ttft": [0]}), "prefill"
    )

    assert report.success_count is None
    assert report.success_ci is None
    assert math.isnan(report.groups.loc[0, "TTFT (s) P50"])
    assert any("status is missing" in note for note in report.notes)


def test_scientific_summary_rejects_mixed_metric_contracts():
    rows = pd.DataFrame(
        {
            "concurrency": [1, 1],
            "metric_contract_version": ["decode-interval-v2", None],
        }
    )

    with pytest.raises(ValueError, match="different metric contract"):
        build_scientific_summary(rows, "concurrency")


def test_scientific_summary_reports_tail_pattern_only_with_enough_measurements():
    rows = pd.DataFrame(
        {
            "concurrency": [1] * 20,
            "error": [None] * 20,
            "ttft": [0.1] * 18 + [1.0, 1.2],
        }
    )

    report = build_scientific_summary(rows, "concurrency")

    assert any("Tail latency" in finding for finding in report.findings)
    assert "Observed patterns" in analysis_to_markdown(report)
    assert not build_scientific_summary(rows.head(10), "concurrency").findings


def test_quantile_chart_uses_numeric_spacing_and_grid_for_two_dimensional_sweeps():
    one_dimensional = pd.DataFrame(
        {"concurrency": [1, 8], "error": [None, None], "ttft": [0.2, 0.8]}
    )
    one_report = build_scientific_summary(one_dimensional, "concurrency")
    line = build_quantile_figure(one_report, "TTFT (s)")
    assert list(line.data[0].x) == [1, 8]
    assert line.data[0].type == "scatter"

    grid_rows = pd.DataFrame(
        {
            "context_length_target": [1024, 1024, 4096, 4096],
            "concurrency": [1, 4, 1, 4],
            "error": [None, None, None, "timeout"],
            "ttft": [0.2, 0.3, 0.5, 0.0],
        }
    )
    grid_report = build_scientific_summary(grid_rows, "matrix")
    heatmap = build_quantile_figure(grid_report, "TTFT (s)")
    assert heatmap.data[0].type == "heatmap"
    assert pd.isna(heatmap.data[0].z[1][1])
