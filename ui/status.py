"""Structured status primitives shared by UI and export layers."""

from __future__ import annotations

from enum import Enum


class InsightSeverity(str, Enum):
    """Severity used to grade and render performance insights."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    WARNING = "warning"
    CRITICAL = "critical"


class PerformanceInsight(str):
    """Markdown-compatible insight that carries presentation-neutral severity."""

    severity: InsightSeverity
    title: str
    detail: str

    def __new__(
        cls,
        severity: InsightSeverity,
        title: str,
        detail: str,
    ) -> PerformanceInsight:
        clean_title = str(title or "").strip()
        clean_detail = str(detail or "").strip()
        if not clean_title:
            raise ValueError("Insight title cannot be empty")

        value = f"**{clean_title}**: {clean_detail}"
        instance = super().__new__(cls, value)
        instance.severity = InsightSeverity(severity)
        instance.title = clean_title
        instance.detail = clean_detail
        return instance
