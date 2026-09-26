"""Regression tests for the UI defect-fix pass (fix/ui-logic-defects).

Covers the pure-logic defects that do not require a running Streamlit
context: test-type inference, radar-chart guards, value formatting,
realtime dashboard counters, and safe column extraction.
"""

import math

import pandas as pd

from ui.charts import plot_performance_summary, smart_format_value
from ui.export import _safe_col_list
from ui.page_layout import _detect_test_type_from_df
from ui.realtime_dashboard import RealtimeDashboard


class TestDetectTestTypeFromDf:
    """_detect_test_type_from_df must not confuse test types whose CSV
    column sets overlap (stability rows always carry input_tokens_target)."""

    def test_stability_not_misdetected_as_prefill(self):
        df = pd.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "concurrency": [4, 4],
                "input_tokens_target": [512, 512],
                "ttft": [0.1, 0.2],
            }
        )
        assert _detect_test_type_from_df(df) == "Stability Test"

    def test_prefill_detected_when_no_concurrency_column(self):
        df = pd.DataFrame(
            {
                "input_tokens_target": [512, 1024],
                "ttft": [0.1, 0.2],
            }
        )
        assert _detect_test_type_from_df(df) == "Prefill Stress Test"

    def test_concurrency_with_kv_skip_rows_not_misdetected(self):
        # Concurrency runs include input_tokens_target in their fixed
        # csv_columns, and KV-budget skip rows add the same column.
        df = pd.DataFrame(
            {
                "concurrency": [1, 2],
                "round": [1, 1],
                "input_tokens_target": [0, 0],
                "ttft": [0.1, 0.2],
            }
        )
        assert _detect_test_type_from_df(df) == "Concurrency Test"

    def test_explicit_test_type_column_wins(self):
        df = pd.DataFrame(
            {
                "test_type": ["stability", "stability"],
                "concurrency": [1, 1],
                "input_tokens_target": [0, 0],
            }
        )
        assert _detect_test_type_from_df(df) == "Stability Test"

    def test_empty_df_returns_none(self):
        assert _detect_test_type_from_df(pd.DataFrame()) is None
        assert _detect_test_type_from_df(None) is None


class TestSmartFormatValue:
    def test_none_returns_na(self):
        assert smart_format_value(None) == "N/A"

    def test_nan_returns_na(self):
        assert smart_format_value(float("nan")) == "N/A"

    def test_normal_values(self):
        assert smart_format_value(123.456) == "123"
        assert smart_format_value(12.3456) == "12.3"
        assert smart_format_value(1.23456) == "1.23"
        assert smart_format_value(0) == "0.00"


class TestRadarChartGuards:
    def test_empty_metrics_returns_none(self):
        df = pd.DataFrame({"tps": [1.0, 2.0]})
        assert plot_performance_summary(df, []) is None

    def test_all_nan_metric_column_does_not_crash(self):
        df = pd.DataFrame(
            {
                "tps": [float("nan"), float("nan")],
                "ttft": [0.1, 0.2],
            }
        )
        fig = plot_performance_summary(df, ["tps", "ttft"])
        assert fig is not None


class TestSafeColList:
    def test_na_values_become_zero(self):
        df = pd.DataFrame({"x": [1.0, None, float("nan"), pd.NA, 3.5]})
        result = _safe_col_list(df, "x")
        assert result == [1.0, 0.0, 0.0, 0.0, 3.5]
        assert all(isinstance(v, float) and math.isfinite(v) for v in result)

    def test_missing_column_returns_empty(self):
        assert _safe_col_list(pd.DataFrame({"a": [1]}), "b") == []
        assert _safe_col_list(pd.DataFrame({"a": [1]}), None) == []


class TestRealtimeDashboardCounters:
    def test_total_requests_increments(self):
        dash = RealtimeDashboard()
        dash.update(timestamp=1.0, ttft=0.1, tps=10.0, status="success")
        dash.update(timestamp=2.0, ttft=0.2, tps=8.0, status="failed")
        metrics = dash.get_metrics()
        assert metrics["total"] == 2
        assert metrics["completed"] == 1
        assert metrics["failed"] == 1


class TestOnboardingSaveResilience:
    def test_save_onboarding_file_survives_oserror(self, tmp_path, monkeypatch):
        from ui import onboarding

        monkeypatch.setattr(onboarding, "_ONBOARDING_FILE", "/nonexistent-dir/state.json")
        # Must not raise even when the target directory does not exist
        onboarding._save_onboarding_file({"dismissed": True})
