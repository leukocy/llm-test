"""Explicit wall-window provenance for system throughput and API cache observations."""

from __future__ import annotations

import math
import uuid
from typing import Any

BATCH_CONTRACT = "batch-wall-v1"


def api_cached_tokens(usage: Any) -> int | None:
    """Zero is an observation only when the API explicitly returns a cache field."""
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    candidates = [details.get("cached_tokens")] if isinstance(details, dict) else []
    candidates.extend(
        usage.get(key)
        for key in [
            "cache_hit_tokens",
            "prompt_cache_hit_tokens",
            "disk_cache_hit_tokens",
            "cache_read_input_tokens",
        ]
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def tag_batch_window(
    results: list[Any],
    *,
    elapsed_seconds: float,
    expected_requests: int,
) -> None:
    """Attach one observation to each row; reports validate membership and deduplicate it.

    The monotonic window spans scheduling through drainage, including failures and client
    overhead. It excludes prompt preparation, warmup and pauses outside this batch. No
    calibration offset or clock-domain fallback is applied.
    """
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        return
    rows = [row for row in results if isinstance(row, dict)]
    observation = {
        "version": BATCH_CONTRACT,
        "id": uuid.uuid4().hex,
        "elapsed_seconds": elapsed_seconds,
        "expected_requests": expected_requests,
        "recorded_requests": len(rows),
    }
    for index, row in enumerate(rows):
        row["batch_id"] = observation["id"]
        row["request_index"] = index
        extra = row.get("extra_metrics")
        if not isinstance(extra, dict):
            extra = {}
            row["extra_metrics"] = extra
        extra["system_measurement"] = dict(observation)
