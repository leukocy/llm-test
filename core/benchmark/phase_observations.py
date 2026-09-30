"""Explicit client streaming clocks for descriptive phase-window estimates."""

from __future__ import annotations

import math
from typing import Any

PHASE_CONTRACT = "client-phase-v1"
PHASE_CLOCK = "client_monotonic"
PHASE_FIELDS = (
    "version",
    "clock",
    "start_seconds",
    "first_token_seconds",
    "end_seconds",
    "latency_offset_seconds",
)


def phase_observation(value: Any) -> dict[str, Any] | None:
    """Keep only validated numeric evidence; absence is never replaced by TTFT."""
    if not isinstance(value, dict):
        return None
    if value.get("version") != PHASE_CONTRACT or value.get("clock") != PHASE_CLOCK:
        return None
    keys = ("start_seconds", "end_seconds", "latency_offset_seconds")
    if any(
        isinstance(value.get(key), bool)
        or not isinstance(value.get(key), (int, float))
        or not math.isfinite(value[key])
        or value[key] < 0
        for key in keys
    ):
        return None
    if value["end_seconds"] < value["start_seconds"]:
        return None
    first = value.get("first_token_seconds")
    if first is not None and (
        isinstance(first, bool)
        or not isinstance(first, (int, float))
        or not math.isfinite(first)
        or not value["start_seconds"] <= first <= value["end_seconds"]
    ):
        return None
    return {key: value.get(key) for key in PHASE_FIELDS}


def provider_phase_observation(result: dict, latency_offset: float) -> dict[str, Any] | None:
    return phase_observation(
        {
            "version": PHASE_CONTRACT,
            "clock": result.get("timing_clock"),
            "start_seconds": result.get("start_time"),
            "first_token_seconds": result.get("first_token_time"),
            "end_seconds": result.get("end_time"),
            "latency_offset_seconds": latency_offset,
        }
    )
