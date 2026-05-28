from ui.reporting.columns import COLUMN_RENAME_MAP, COLUMN_TOOLTIPS


def test_reporting_columns_include_success_rate_and_core_tooltips():
    assert COLUMN_RENAME_MAP["Success_Rate"] == "Success_Rate (%)"
    assert "Max_System_Output_Throughput" in COLUMN_TOOLTIPS
    assert "Success_Rate" in COLUMN_TOOLTIPS


def test_tpot_tail_latency_columns_are_labeled_as_stream_chunk_metrics():
    assert COLUMN_RENAME_MAP["TPOT_P95"] == "TPOT_Chunk_P95 (ms)"
    assert COLUMN_RENAME_MAP["TPOT_P99"] == "TPOT_Chunk_P99 (ms)"

    tooltip = COLUMN_TOOLTIPS["TPOT_P95"]
    assert "stream chunk" in tooltip.lower()
    assert "not guaranteed" in tooltip.lower()
