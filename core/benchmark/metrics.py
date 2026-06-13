"""Pure metric helpers for benchmark request timing and empty result rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RequestMetrics:
    """Timing metrics for one completed request."""

    ttft: float
    tps: float
    tpot: float
    tpot_p95: float
    tpot_p99: float
    generation_time: float

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        """Return the legacy tuple shape used by BenchmarkRunner."""
        return (
            self.ttft,
            self.tps,
            self.tpot,
            self.tpot_p95,
            self.tpot_p99,
            self.generation_time,
        )


def empty_metrics() -> dict[str, object]:
    """Return the legacy empty metrics payload used for failed/cancelled requests."""
    return {
        "ttft": 0,
        "tps": 0,
        "tpot": 0,
        "tpot_p95": 0,
        "tpot_p99": 0,
        "prefill_tokens": 0,
        "decode_tokens": 0,
        "decode_time": 0,
        "total_time": 0,
        "cache_hit_tokens": 0,
        "token_calc_method": "Error",
        "error": None,
    }


def calculate_request_metrics(
    start_time: float,
    first_token_time: float | None,
    end_time: float,
    completion_tokens: int,
    *,
    latency_offset: float = 0,
    token_timestamps: Iterable[float] | None = None,
    skip_first_token: bool = False,
) -> RequestMetrics:
    """Calculate TTFT, TPS, TPOT, and stream chunk latency percentiles.

    When *skip_first_token* is True **and** at least 2 token timestamps are
    available, the first streamed chunk is treated as part of the prefill phase.
    TPS / TPOT then use the **second** chunk timestamp as the start of decode, so
    the prefill-decode gap does not distort generation-speed metrics.  TTFT is
    **never** affected.
    """
    ttft = 0
    tps = 0
    tpot = 0
    tpot_p95 = 0
    tpot_p99 = 0
    generation_time = 0

    if first_token_time:
        ttft_raw = first_token_time - start_time
        ttft = max(0.000001, ttft_raw - latency_offset)

        timestamps = list(token_timestamps or [])
        use_skip = skip_first_token and len(timestamps) >= 2

        if use_skip:
            # Use second streamed chunk timestamp as generation start
            generation_time = end_time - timestamps[1]
            effective_tokens = completion_tokens - 1
            if effective_tokens > 0 and generation_time > 0:
                tps = effective_tokens / generation_time
            if effective_tokens > 1 and generation_time > 0:
                tpot = generation_time / (effective_tokens - 1)

            # Stream chunk latencies: skip the first interval (prefill→decode)
            if len(timestamps) > 2:
                latencies = []
                for i in range(2, len(timestamps)):
                    diff = timestamps[i] - timestamps[i - 1]
                    if diff >= 0:
                        latencies.append(diff)
                if latencies:
                    tpot_p95 = float(np.percentile(latencies, 95))
                    tpot_p99 = float(np.percentile(latencies, 99))
        else:
            generation_time = end_time - first_token_time
            tps = completion_tokens / generation_time if generation_time > 0 else 0

            if completion_tokens > 1 and generation_time > 0:
                tpot = generation_time / (completion_tokens - 1)

            if len(timestamps) > 1:
                latencies = []
                for i in range(1, len(timestamps)):
                    diff = timestamps[i] - timestamps[i - 1]
                    if diff >= 0:
                        latencies.append(diff)
                if latencies:
                    tpot_p95 = float(np.percentile(latencies, 95))
                    tpot_p99 = float(np.percentile(latencies, 99))

    return RequestMetrics(ttft, tps, tpot, tpot_p95, tpot_p99, generation_time)
