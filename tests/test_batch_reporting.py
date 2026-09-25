"""Batch comparison keeps failed requests and valid measurement counts visible."""

import pandas as pd
import pytest

from core.batch_test import BatchTestItem, BatchTestResult, _extract_metrics_from_dataframe
from ui.batch_test import add_batch_success_intervals, build_batch_success_figure


def test_batch_metrics_include_partial_failures_without_invented_accuracy():
    item = BatchTestItem("test", "http://example.com", "model", "secret")
    rows = pd.DataFrame(
        {
            "error": [None, "timeout", None],
            "ttft": [0.2, 9.0, 0.0],
            "tps": [40.0, 900.0, 20.0],
            "prefill_tokens": [100, 0, 50],
            "decode_tokens": [20, 0, 10],
        }
    )

    measured = _extract_metrics_from_dataframe(rows, item)

    assert measured["status"] == "partial_failure"
    assert measured["request_count"] == 3
    assert measured["failed_count"] == 1
    assert measured["success_rate"] == pytest.approx(2 / 3)
    assert measured["ttft_valid_n"] == 1
    assert measured["ttft_p50_ms"] == 200
    assert measured["tps_valid_n"] == 2
    assert measured["tps_p50"] == 30
    assert "accuracy" not in measured

    batch = BatchTestResult("batch", "start", "end", 1.0, item_results=[measured])
    comparison = batch.get_comparison_df()
    assert len(comparison) == 1
    assert comparison.loc[0, "Request Success (%)"] == pytest.approx(200 / 3)
    assert "Accuracy" not in comparison
    with_intervals = add_batch_success_intervals(comparison)
    assert with_intervals.loc[0, "Success CI low (%)"] < 200 / 3
    assert build_batch_success_figure(with_intervals) is not None


def test_batch_missing_status_does_not_become_full_success():
    item = BatchTestItem("test", "http://example.com", "model", "secret")
    measured = _extract_metrics_from_dataframe(pd.DataFrame({"ttft": [0.3]}), item)

    assert measured["success_rate"] is None
    assert measured["failed_count"] is None
    assert measured["ttft_p50_ms"] == 300
    assert (
        add_batch_success_intervals(
            BatchTestResult(
                "batch", "start", "end", 1.0, item_results=[measured]
            ).get_comparison_df()
        )["Success CI low (%)"]
        .isna()
        .all()
    )


def test_empty_batch_result_is_failed_and_has_no_success_estimate():
    item = BatchTestItem("test", "http://example.com", "model", "secret")
    measured = _extract_metrics_from_dataframe(pd.DataFrame(), item)

    assert measured["status"] == "failed"
    assert measured["request_count"] == 0
    assert measured["success_rate"] is None

    batch = BatchTestResult(
        "batch",
        "start",
        "end",
        1.0,
        item_results=[measured, {"name": "skipped", "model_id": "other", "status": "skipped"}],
    )
    comparison = batch.get_comparison_df()
    assert list(comparison["Status"]) == ["failed", "skipped"]
    assert comparison["Request Success (%)"].isna().all()
