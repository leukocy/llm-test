from unittest.mock import MagicMock

import pandas as pd
import pytest

from ui.charts import plot_plotly_line
from ui.formatters import format_results_for_display
from ui.metric_sanitizer import summarize_metric_extreme


def test_result_display_hides_zero_performance_metrics_but_keeps_count_zeros():
    df = pd.DataFrame(
        {
            "session_id": ["bad-zero", "good"],
            "concurrency": [1, 1],
            "round": [1, 2],
            "prefill_tokens": [0, 2048],
            "decode_tokens": [0, 128],
            "cache_hit_tokens": [0, 0],
            "ttft": [0.0, 0.8421],
            "prefill_speed": [0.0, 3512.6],
            "tps": [0.0, 83.3],
            "tpot": [0.0, 0.012],
            "system_output_throughput": [0.0, 920.0],
            "error": [None, None],
        }
    )

    display_df = format_results_for_display(df, "Concurrency Test")

    assert pd.isna(display_df.loc[0, "TTFT (s)"])
    assert pd.isna(display_df.loc[0, "Prefill Speed (t/s)"])
    assert pd.isna(display_df.loc[0, "TPS (t/s)"])
    assert pd.isna(display_df.loc[0, "TPOT (s)"])
    assert pd.isna(display_df.loc[0, "Output Throughput (t/s)"])
    assert display_df.loc[0, "Prefill Tokens"] == 0
    assert display_df.loc[0, "Decode Tokens"] == 0
    assert display_df.loc[0, "Cache Hit (tokens)"] == 0


def test_result_display_labels_tpot_tail_metrics_as_stream_chunk_latency():
    df = pd.DataFrame(
        {
            "session_id": ["good"],
            "concurrency": [1],
            "round": [1],
            "tpot_p95": [0.014],
            "tpot_p99": [0.018],
        }
    )

    display_df = format_results_for_display(df, "Concurrency Test")

    assert "TPOT Chunk P95 (s)" in display_df.columns
    assert "TPOT Chunk P99 (s)" in display_df.columns


def test_chart_data_marks_zero_performance_points_as_missing():
    df = pd.DataFrame(
        {
            "level": ["bad-zero", "good"],
            "Best_TTFT (s)": [0.0, 0.931],
        }
    )

    fig = plot_plotly_line(
        df,
        "level",
        "Best_TTFT (s)",
        "TTFT",
        "Level",
        "Time (s)",
        "model",
        2,
        force_linear_scale=True,
    )

    y_values = list(fig.data[0].y)
    assert pd.isna(y_values[0])
    assert y_values[1] == 0.931


def test_chart_uses_numeric_workload_distance_for_token_sweeps():
    df = pd.DataFrame(
        {
            "x_label": ["4k", "1k"],
            "input_tokens_target": [4096, 1024],
            "TTFT": [0.5, 0.2],
        }
    )

    fig = plot_plotly_line(
        df,
        "x_label",
        "TTFT",
        "TTFT",
        "Input tokens",
        "s",
        "model",
        2,
        force_linear_scale=True,
    )

    assert list(fig.data[0].x) == [1024, 4096]
    assert fig.layout.xaxis.type == "linear"


def test_log_scale_chart_uses_logarithmic_axis_range_and_labels_it():
    df = pd.DataFrame({"size": [1, 2], "TTFT": [0.0001, 100.0]})

    fig = plot_plotly_line(df, "size", "TTFT", "TTFT", "Size", "s", "model", 2)

    assert fig.layout.yaxis.type == "log"
    assert fig.layout.yaxis.range[0] < 0 < fig.layout.yaxis.range[1]
    assert "Log Scale" in fig.layout.title.text


def test_grouped_extreme_ignores_zero_performance_values():
    df = pd.DataFrame(
        {
            "concurrency": [1, 1, 2],
            "ttft": [0.0, 1.25, 0.0],
            "system_output_throughput": [0.0, 550.0, 0.0],
        }
    )

    best_ttft = summarize_metric_extreme(
        df,
        group_cols=["concurrency"],
        metric_col="ttft",
        output_col="Best_TTFT",
        how="min",
    )
    max_output = summarize_metric_extreme(
        df,
        group_cols=["concurrency"],
        metric_col="system_output_throughput",
        output_col="Max_System_Output_Throughput",
        how="max",
    )

    assert best_ttft.loc[best_ttft["concurrency"] == 1, "Best_TTFT"].item() == 1.25
    assert pd.isna(best_ttft.loc[best_ttft["concurrency"] == 2, "Best_TTFT"].item())
    assert (
        max_output.loc[max_output["concurrency"] == 1, "Max_System_Output_Throughput"].item()
        == 550.0
    )
    assert pd.isna(
        max_output.loc[max_output["concurrency"] == 2, "Max_System_Output_Throughput"].item()
    )


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_concurrency_report_table_uses_valid_values_instead_of_zero_extremes(
    monkeypatch,
):
    import streamlit as st

    from ui import reports

    def fake_columns(spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [_Context() for _ in range(count)]

    st.columns = fake_columns
    st.expander = lambda *args, **kwargs: _Context()
    st.dataframe = MagicMock()
    st.plotly_chart = MagicMock()
    st.markdown = MagicMock()
    st.metric = MagicMock()
    st.subheader = MagicMock()

    monkeypatch.setattr(reports, "export_benchmark_summary_chart", lambda *args, **kwargs: None)
    monkeypatch.setattr(reports, "generate_performance_insights", lambda *args, **kwargs: ([], []))

    df = pd.DataFrame(
        {
            "concurrency": [1, 1],
            "round": [1, 2],
            "session_id": ["zero-row", "valid-row"],
            "error": [None, None],
            "token_calc_method": ["api", "api"],
            "prefill_tokens": [1024, 1024],
            "decode_tokens": [128, 128],
            "ttft": [0.0, 1.25],
            "tpot": [0.0, 0.012],
            "tpot_p95": [0.0, 0.014],
            "tpot_p99": [0.0, 0.018],
            "system_throughput": [0.0, 120.0],
            "system_output_throughput": [0.0, 240.0],
            "system_input_throughput": [0.0, 360.0],
            "rps": [0.0, 3.0],
        }
    )

    reports.generate_concurrency_report(df, model_id="model", provider="provider")

    styled_table = st.dataframe.call_args_list[0].args[0]
    table_df = styled_table.data
    assert table_df.loc[0, "Best_TTFT (s)"] == 1.25
    assert table_df.loc[0, "Max_System_Output_Throughput (tokens/s)"] == 240.0
    assert table_df.loc[0, "Max_System_Input_Throughput (tokens/s)"] == 360.0
    assert table_df.loc[0, "Max_RPS (req/s)"] == 3.0


def test_concurrency_report_success_rate_uses_percent_and_blank_errors(monkeypatch):
    import streamlit as st

    from ui import reports

    def fake_columns(spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [_Context() for _ in range(count)]

    st.columns = fake_columns
    st.expander = lambda *args, **kwargs: _Context()
    st.dataframe = MagicMock()
    st.plotly_chart = MagicMock()
    st.markdown = MagicMock()
    st.metric = MagicMock()
    st.subheader = MagicMock()

    monkeypatch.setattr(reports, "export_benchmark_summary_chart", lambda *args, **kwargs: None)
    monkeypatch.setattr(reports, "generate_performance_insights", lambda *args, **kwargs: ([], []))

    df = pd.DataFrame(
        {
            "concurrency": [1, 1, 1],
            "round": [1, 2, 3],
            "session_id": ["success-none", "success-blank", "failed"],
            "error": [None, "", "timeout"],
            "token_calc_method": ["api", "api", "api"],
            "prefill_tokens": [1024, 1024, 1024],
            "decode_tokens": [128, 128, 128],
            "ttft": [1.0, 1.1, 1.2],
            "tpot": [0.012, 0.013, 0.014],
            "tpot_p95": [0.014, 0.015, 0.016],
            "tpot_p99": [0.018, 0.019, 0.020],
            "system_throughput": [120.0, 121.0, 122.0],
            "system_output_throughput": [240.0, 241.0, 242.0],
            "system_input_throughput": [360.0, 361.0, 362.0],
            "rps": [3.0, 3.1, 3.2],
        }
    )

    reports.generate_concurrency_report(df, model_id="model", provider="provider")

    styled_table = st.dataframe.call_args_list[0].args[0]
    table_df = styled_table.data
    assert table_df.loc[0, "Success_Rate (%)"] == pytest.approx(2 / 3 * 100)
