"""Shared descriptive statistics; callers decide whether zero is a valid observation."""

from __future__ import annotations

import math
import statistics


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe_values(values: list[float]) -> dict:
    scale = max(values) if values else 0
    return {
        "count": len(values),
        "mean": scale * statistics.fmean(value / scale for value in values)
        if scale
        else (0.0 if values else None),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
