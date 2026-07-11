"""Pure pandas builders for benchmark report summaries."""

from __future__ import annotations

import pandas as pd

from core.result_metrics import (
    calculate_success_rate_percent,
    fill_non_performance_na,
    positive_max,
    positive_mean,
    positive_min,
    positive_quantile,
    sanitize_performance_metrics,
    summarize_metric_extreme,
    summarize_metric_row,
)


def build_concurrency_summary(df_group: pd.DataFrame) -> pd.DataFrame:
    """Build the concurrency report summary without Streamlit dependencies."""
    if "concurrency" not in df_group.columns:
        raise ValueError("Concurrency summary failed: missing 'concurrency' column in data.")

    work = df_group.copy()
    work = work[pd.to_numeric(work["concurrency"], errors="coerce").notna()]
    work = work[work["concurrency"] > 0].copy()
    if work.empty:
        raise ValueError("Concurrency summary failed: no valid concurrency data.")

    rounds_per_level = work.groupby("concurrency")["round"].max().max()

    if "tpot" not in work.columns:
        work["tpot"] = 0
    if "tpot_p95" not in work.columns:
        work["tpot_p95"] = work["tpot"]
    if "tpot_p99" not in work.columns:
        work["tpot_p99"] = work["tpot"]

    if "system_throughput" not in work.columns:
        work["system_throughput"] = 0
    if "system_output_throughput" not in work.columns:
        work["system_output_throughput"] = work["system_throughput"]
    if "system_input_throughput" not in work.columns:
        work["system_input_throughput"] = 0
    if "rps" not in work.columns:
        work["rps"] = 0

    work = sanitize_performance_metrics(work)

    round_stats = (
        work.groupby(["concurrency", "round"])
        .agg(
            {
                "ttft": positive_mean,
                "tpot": positive_mean,
                "tpot_p95": positive_mean,
                "tpot_p99": positive_mean,
                "system_throughput": positive_mean,
                "session_id": "count",
            }
        )
        .reset_index()
    )

    summary_ttft = summarize_metric_extreme(
        round_stats,
        "concurrency",
        "ttft",
        "Best_TTFT",
        how="min",
    )
    summary_tpot = (
        round_stats.groupby("concurrency")
        .agg(
            TPOT_Mean=("tpot", positive_mean),
            TPOT_P95=("tpot_p95", positive_mean),
            TPOT_P99=("tpot_p99", positive_mean),
        )
        .reset_index()
    )
    summary_tpot["TPOT_Mean"] = summary_tpot["TPOT_Mean"] * 1000
    summary_tpot["TPOT_P95"] = summary_tpot["TPOT_P95"] * 1000
    summary_tpot["TPOT_P99"] = summary_tpot["TPOT_P99"] * 1000

    summary_output = summarize_metric_extreme(
        work,
        "concurrency",
        "system_output_throughput",
        "Max_System_Output_Throughput",
    )
    summary_input = summarize_metric_extreme(
        work,
        "concurrency",
        "system_input_throughput",
        "Max_System_Input_Throughput",
    )
    summary_rps = summarize_metric_extreme(work, "concurrency", "rps", "Max_RPS")

    summary_sys_tps = pd.merge(summary_output, summary_input, on="concurrency", how="left")
    summary_sys_tps = pd.merge(summary_sys_tps, summary_rps, on="concurrency", how="left")
    summary_sys_tps["Max_QPM"] = summary_sys_tps["Max_RPS"] * 60

    req_stats = (
        work.groupby("concurrency")
        .agg(
            Success_Rate=("error", calculate_success_rate_percent),
            Token_Calc_Method=("token_calc_method", "first"),
            Num_Requests=("session_id", "size"),
            Actual_Tokens_Mean=("prefill_tokens", "mean"),
            Actual_Decode_Max=("decode_tokens", "max"),
        )
        .reset_index()
    )
    req_stats = fill_non_performance_na(req_stats)

    summary = pd.merge(summary_ttft, summary_tpot, on="concurrency", how="left")
    summary = pd.merge(summary, summary_sys_tps, on="concurrency", how="left")
    summary = pd.merge(summary, req_stats, on="concurrency", how="left")
    summary.attrs["rounds_per_level"] = int(rounds_per_level)
    return summary


def build_prefill_summary(df_group: pd.DataFrame) -> pd.DataFrame:
    """Build the prefill stress report summary without Streamlit dependencies."""
    if "input_tokens_target" not in df_group.columns:
        raise ValueError("Prefill summary failed: missing 'input_tokens_target' column in data.")

    work = df_group.copy()
    if work.empty:
        raise ValueError("Prefill summary failed: no valid prefill data.")

    requests_per_level = work.groupby("input_tokens_target").size().max()

    if "prefill_speed" not in work.columns:
        work["prefill_speed"] = 0
    if "ttft" not in work.columns:
        work["ttft"] = 0
    if "tpot" not in work.columns:
        work["tpot"] = 0
    if "tpot_p95" not in work.columns:
        work["tpot_p95"] = work["tpot"]
    if "tpot_p99" not in work.columns:
        work["tpot_p99"] = work["tpot"]

    work = sanitize_performance_metrics(work)

    summary_best = summarize_metric_row(
        work,
        "input_tokens_target",
        "prefill_speed",
        {"prefill_speed": "Best_Prefill_Speed", "ttft": "TTFT_at_Best_Speed"},
        how="max",
    )

    stats = (
        work.groupby("input_tokens_target")
        .agg(
            Num_Requests=("session_id", "size"),
            Success_Rate=("error", calculate_success_rate_percent),
            Token_Calc_Method=("token_calc_method", "first"),
            Actual_Tokens_Mean=("prefill_tokens", "mean"),
            Best_TTFT=("ttft", positive_min),
            Max_Prefill_Speed=("prefill_speed", positive_max),
            TPOT_Mean=("tpot", lambda x: positive_mean(x) * 1000),
            TPOT_P95=("tpot_p95", lambda x: positive_quantile(x, 0.95) * 1000),
            TPOT_P99=("tpot_p99", lambda x: positive_quantile(x, 0.99) * 1000),
        )
        .reset_index()
    )
    stats = fill_non_performance_na(stats)

    summary = pd.merge(
        stats,
        summary_best[["input_tokens_target", "TTFT_at_Best_Speed"]],
        on="input_tokens_target",
        how="left",
    )
    summary["x_label"] = (summary["input_tokens_target"] / 1024).round(1).astype(str) + "k"
    summary = summary.sort_values(by="input_tokens_target").reset_index(drop=True)
    summary.attrs["requests_per_level"] = int(requests_per_level)
    return summary


def build_long_context_summary(df_group: pd.DataFrame) -> pd.DataFrame:
    """Build the long-context report summary without Streamlit dependencies."""
    if "context_length_target" not in df_group.columns:
        raise ValueError(
            "Long context summary failed: missing 'context_length_target' column in data."
        )

    work = df_group.copy()
    if work.empty:
        raise ValueError("Long context summary failed: no valid long context data.")

    requests_per_level = work.groupby("context_length_target").size().max()

    if "ttft" not in work.columns:
        work["ttft"] = 0
    if "prefill_speed" not in work.columns:
        work["prefill_speed"] = 0
    if "tps" not in work.columns:
        work["tps"] = 0
    if "tpot" not in work.columns:
        work["tpot"] = 0
    if "tpot_p95" not in work.columns:
        work["tpot_p95"] = work["tpot"]
    if "tpot_p99" not in work.columns:
        work["tpot_p99"] = work["tpot"]
    if "system_input_throughput" not in work.columns:
        work["system_input_throughput"] = 0
    if "system_output_throughput" not in work.columns:
        work["system_output_throughput"] = 0
    if "system_throughput" not in work.columns:
        work["system_throughput"] = 0

    # Long-context runs are single-request decode measurements per context level.
    # Loaded historical results can have stale pre-skip system_output_throughput,
    # while tps/tpot already use the corrected skip-first-token decode window.
    tps_values = pd.to_numeric(work["tps"], errors="coerce")
    output_values = pd.to_numeric(work["system_output_throughput"], errors="coerce")
    work["system_output_throughput"] = output_values.where(~(tps_values > 0), tps_values)

    work = sanitize_performance_metrics(work)

    stats = (
        work.groupby("context_length_target")
        .agg(
            Num_Requests=("session_id", "size"),
            Success_Rate=("error", calculate_success_rate_percent),
            Token_Calc_Method=("token_calc_method", "first"),
            Actual_Tokens_Mean=("prefill_tokens", "mean"),
            Actual_Decode_Max=("decode_tokens", "max"),
            TPOT_Mean=("tpot", lambda x: positive_mean(x) * 1000),
            TPOT_P95=("tpot_p95", lambda x: positive_quantile(x, 0.95) * 1000),
            TPOT_P99=("tpot_p99", lambda x: positive_quantile(x, 0.99) * 1000),
        )
        .reset_index()
    )
    stats = fill_non_performance_na(stats)

    summary_ttft = summarize_metric_extreme(
        work,
        "context_length_target",
        "ttft",
        "Best_TTFT",
        how="min",
    )
    summary_prefill = summarize_metric_extreme(
        work,
        "context_length_target",
        "prefill_speed",
        "Max_Prefill_Speed",
    )
    summary_tps = summarize_metric_extreme(work, "context_length_target", "tps", "Max_TPS")

    summary = pd.merge(stats, summary_ttft, on="context_length_target", how="left")
    summary = pd.merge(summary, summary_prefill, on="context_length_target", how="left")
    summary = pd.merge(summary, summary_tps, on="context_length_target", how="left")

    summary_sys_input = summarize_metric_extreme(
        work,
        "context_length_target",
        "system_input_throughput",
        "Max_System_Input_Throughput",
    )
    summary_sys_output = summarize_metric_extreme(
        work,
        "context_length_target",
        "system_output_throughput",
        "Max_System_Output_Throughput",
    )
    summary_sys_total = summarize_metric_extreme(
        work,
        "context_length_target",
        "system_throughput",
        "Max_System_Throughput",
    )
    summary = pd.merge(summary, summary_sys_input, on="context_length_target", how="left")
    summary = pd.merge(summary, summary_sys_output, on="context_length_target", how="left")
    summary = pd.merge(summary, summary_sys_total, on="context_length_target", how="left")
    summary["x_label"] = (summary["context_length_target"] / 1024).round(1).astype(str) + "k"
    summary.attrs["requests_per_level"] = int(requests_per_level)
    return summary
