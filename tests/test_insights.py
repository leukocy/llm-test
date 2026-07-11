import pandas as pd

from ui.insights import _get_col, get_performance_grade
from ui.reporting.columns import COLUMN_RENAME_MAP
from ui.status import InsightSeverity, PerformanceInsight


def test_insight_column_lookup_handles_renamed_tpot_chunk_columns():
    tpot_p99_label = COLUMN_RENAME_MAP["TPOT_P99"]
    df = pd.DataFrame({tpot_p99_label: [120.0]})

    assert _get_col(df, "TPOT_P99") == tpot_p99_label


def test_performance_grade_uses_structured_severity_without_glyphs():
    insights = [
        PerformanceInsight(InsightSeverity.POSITIVE, "Throughput", "Strong"),
        PerformanceInsight(InsightSeverity.POSITIVE, "Latency", "Stable"),
        PerformanceInsight(InsightSeverity.NEUTRAL, "Capacity", "Measured"),
    ]

    grade, _, description = get_performance_grade(insights)

    assert grade == "A"
    assert description == "Excellent performance"


def test_critical_structured_insight_takes_priority():
    insights = [
        PerformanceInsight(InsightSeverity.POSITIVE, "Throughput", "Strong"),
        PerformanceInsight(InsightSeverity.CRITICAL, "Data", "Missing"),
    ]

    grade, _, description = get_performance_grade(insights)

    assert grade == "C"
    assert description == "Critical issue detected"
