"""Pure metric helpers for benchmark request timing and empty result rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

METRIC_CONTRACT_VERSION = "decode-interval-v2"


def count_decode_intervals(
    completion_tokens: int,
    token_timestamps: Iterable[float] | None = None,
    *,
    skip_first_token: bool = False,
) -> int:
    """Return the output-token intervals represented by the decode window."""
    if completion_tokens <= 1:
        return 0
    timestamps = list(token_timestamps or [])
    if skip_first_token and completion_tokens >= 3 and len(timestamps) == completion_tokens:
        return completion_tokens - 2
    return completion_tokens - 1


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
        "metric_contract_version": METRIC_CONTRACT_VERSION,
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

    Decode throughput and TPOT share the same token intervals: N-1 intervals
    after the first output token. When *skip_first_token* is requested, the
    second timestamp can anchor the interval only if the stream supplies one
    timestamp for each counted output token. Otherwise chunk boundaries cannot
    identify a single token, so the normal first-token window is used.
    Percentiles describe streamed chunk gaps, not individual token gaps.
    """
    ttft = 0.0
    tps = 0.0
    tpot = 0.0
    tpot_p95 = 0.0
    tpot_p99 = 0.0
    generation_time = 0.0

    if first_token_time is not None:
        ttft_raw = first_token_time - start_time
        ttft = max(0.000001, ttft_raw - latency_offset)

        timestamps = list(token_timestamps or [])
        use_skip = (
            skip_first_token and completion_tokens >= 3 and len(timestamps) == completion_tokens
        )
        interval_tokens = count_decode_intervals(
            completion_tokens, timestamps, skip_first_token=skip_first_token
        )

        if use_skip:
            # Use second streamed output timestamp as generation start.
            generation_time = max(0.0, end_time - timestamps[1])
            if interval_tokens > 0 and generation_time > 0:
                tps = interval_tokens / generation_time
                tpot = generation_time / interval_tokens

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
            generation_time = max(0.0, end_time - first_token_time)
            if interval_tokens > 0 and generation_time > 0:
                tps = interval_tokens / generation_time
                tpot = generation_time / interval_tokens

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
