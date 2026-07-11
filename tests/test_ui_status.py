"""Tests for structured, presentation-neutral insight status."""

import pytest


@pytest.mark.parametrize("severity", ["positive", "neutral", "warning", "critical"])
def test_performance_insight_preserves_markdown_and_severity(severity):
    from ui.status import InsightSeverity, PerformanceInsight

    item = PerformanceInsight(InsightSeverity(severity), "Capacity", "Near the limit")

    assert isinstance(item, str)
    assert str(item) == "**Capacity**: Near the limit"
    assert item.severity is InsightSeverity(severity)
    assert item.title == "Capacity"
    assert item.detail == "Near the limit"


def test_performance_insight_rejects_empty_title():
    from ui.status import InsightSeverity, PerformanceInsight

    with pytest.raises(ValueError, match="title"):
        PerformanceInsight(InsightSeverity.NEUTRAL, "", "No title")
