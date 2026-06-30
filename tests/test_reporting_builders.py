import pandas as pd
import pytest

from ui.reporting.builders import (
    build_concurrency_summary,
    build_long_context_summary,
    build_prefill_summary,
)


def test_build_concurrency_summary_masks_invalid_zeros_and_returns_percent_success_rate():
    df = pd.DataFrame(
        {
            "concurrency": [1, 1, 1],
            "round": [1, 2, 3],
            "session_id": ["zero-row", "success-blank", "failed"],
            "error": [None, "", "timeout"],
            "token_calc_method": ["api", "api", "api"],
            "prefill_tokens": [1024, 1024, 1024],
            "decode_tokens": [128, 128, 128],
            "ttft": [0.0, 1.25, 1.5],
            "tpot": [0.0, 0.012, 0.014],
            "tpot_p95": [0.0, 0.014, 0.016],
            "tpot_p99": [0.0, 0.018, 0.020],
            "system_throughput": [0.0, 120.0, 122.0],
            "system_output_throughput": [0.0, 240.0, 242.0],
            "system_input_throughput": [0.0, 360.0, 362.0],
            "rps": [0.0, 3.0, 3.2],
        }
    )

    summary = build_concurrency_summary(df)

    assert summary.attrs["rounds_per_level"] == 3
    assert summary.loc[0, "Best_TTFT"] == 1.25
    assert summary.loc[0, "Max_System_Output_Throughput"] == 242.0
    assert summary.loc[0, "Max_System_Input_Throughput"] == 362.0
    assert summary.loc[0, "Max_RPS"] == 3.2
    assert summary.loc[0, "Max_QPM"] == pytest.approx(192.0)
    assert summary.loc[0, "Success_Rate"] == pytest.approx(2 / 3 * 100)


def test_build_concurrency_summary_rejects_missing_concurrency_column():
    with pytest.raises(ValueError, match="missing 'concurrency'"):
        build_concurrency_summary(pd.DataFrame({"session_id": ["a"]}))


def test_build_prefill_summary_masks_invalid_zeros_and_sorts_by_input_tokens():
    df = pd.DataFrame(
        {
            "input_tokens_target": [2048, 1024, 1024],
            "session_id": ["long", "zero-row", "valid-row"],
            "error": [None, None, ""],
            "token_calc_method": ["api", "api", "api"],
            "prefill_tokens": [2048, 1024, 1024],
            "prefill_speed": [0.0, 0.0, 500.0],
            "ttft": [2.0, 0.0, 1.25],
            "tpot": [0.020, 0.0, 0.012],
            "tpot_p95": [0.022, 0.0, 0.014],
            "tpot_p99": [0.024, 0.0, 0.018],
        }
    )

    summary = build_prefill_summary(df)

    assert summary.attrs["requests_per_level"] == 2
    assert summary["input_tokens_target"].tolist() == [1024, 2048]
    first = summary.iloc[0]
    assert first["Success_Rate"] == pytest.approx(100.0)
    assert first["Best_TTFT"] == 1.25
    assert first["Max_Prefill_Speed"] == 500.0
    assert first["TTFT_at_Best_Speed"] == 1.25
    assert first["x_label"] == "1.0k"
    assert pd.isna(summary.iloc[1]["Max_Prefill_Speed"])


def test_build_long_context_summary_includes_phase_throughputs_and_success_rate():
    df = pd.DataFrame(
        {
            "context_length_target": [4096, 4096, 8192],
            "session_id": ["zero-row", "valid-row", "failed"],
            "error": [None, "", "timeout"],
            "token_calc_method": ["api", "api", "api"],
            "prefill_tokens": [4096, 4096, 8192],
            "decode_tokens": [128, 128, 256],
            "ttft": [0.0, 1.25, 2.5],
            "prefill_speed": [0.0, 700.0, 800.0],
            "tps": [0.0, 80.0, 70.0],
            "tpot": [0.0, 0.012, 0.015],
            "tpot_p95": [0.0, 0.014, 0.017],
            "tpot_p99": [0.0, 0.018, 0.020],
            "system_input_throughput": [0.0, 300.0, 310.0],
            "system_output_throughput": [0.0, 200.0, 210.0],
            "system_throughput": [0.0, 500.0, 520.0],
        }
    )

    summary = build_long_context_summary(df)

    assert summary.attrs["requests_per_level"] == 2
    first = summary[summary["context_length_target"] == 4096].iloc[0]
    assert first["Success_Rate"] == pytest.approx(100.0)
    assert first["Best_TTFT"] == 1.25
    assert first["Max_Prefill_Speed"] == 700.0
    assert first["Max_TPS"] == 80.0
    assert first["Max_System_Input_Throughput"] == 300.0
    # Output throughput is the exact reciprocal of TPOT_Mean for long-context
    # (single-stream decode). TPOT_Mean = 0.012 s = 12.0 ms -> 1000/12.0 t/s.
    assert first["Max_System_Output_Throughput"] == pytest.approx(1000.0 / 12.0)
    assert first["Max_System_Throughput"] == 500.0
    assert first["x_label"] == "4.0k"


def test_build_long_context_summary_derives_output_throughput_from_tpot_mean():
    df = pd.DataFrame(
        {
            "context_length_target": [4096, 4096],
            "session_id": ["old-stale-row", "correct-row"],
            "error": ["", ""],
            "token_calc_method": ["api", "api"],
            "prefill_tokens": [4096, 4096],
            "decode_tokens": [512, 512],
            "ttft": [1.0, 1.1],
            "prefill_speed": [700.0, 690.0],
            "tps": [16.7, 15.8],
            "tpot": [0.0597, 0.0633],
            "tpot_p95": [0.060, 0.064],
            "tpot_p99": [0.061, 0.065],
            # Historical loaded results may still contain the pre-skip output
            # throughput, while tps/tpot already use the skip-first-token window.
            "system_output_throughput": [12.8, 12.4],
            "system_input_throughput": [208.0, 205.0],
            "system_throughput": [220.8, 217.4],
        }
    )

    summary = build_long_context_summary(df)

    first = summary[summary["context_length_target"] == 4096].iloc[0]
    assert first["Max_TPS"] == pytest.approx(16.7)
    # Output throughput is derived from TPOT_Mean (reciprocal), not from tps,
    # so it stays paired with the TPOT chart regardless of stale loaded tps.
    # mean(tpot) = (0.0597 + 0.0633)/2 = 0.0615 s -> 1000/61.5 t/s.
    assert first["TPOT_Mean"] == pytest.approx(61.5)
    assert first["Max_System_Output_Throughput"] == pytest.approx(1000.0 / 61.5)
