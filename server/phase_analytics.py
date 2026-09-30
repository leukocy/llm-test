"""Client phase windows sampled once per complete batch, separate from wall throughput."""

from __future__ import annotations

import json
import math
from typing import Any

from core.benchmark.phase_observations import PHASE_CONTRACT, phase_observation

PHASE_METRICS = {
    "phase_input": ("客户端输入阶段估计吞吐（含缓存）", "Client input phase", "token/s"),
    "phase_input_uncached_api": (
        "客户端未缓存输入阶段估计吞吐（API）",
        "Client uncached input (API)",
        "token/s",
    ),
    "phase_input_uncached_inferred": (
        "客户端未缓存输入阶段估计吞吐（含 TTFT 推断）",
        "Client uncached input (inferred)",
        "token/s",
    ),
    "phase_output": ("客户端输出阶段估计吞吐", "Client output phase", "token/s"),
    "phase_total": ("客户端请求窗口总吞吐", "Client request window", "token/s"),
    "phase_qpm": ("客户端请求窗口成功处理速率", "Client request QPM", "req/min"),
}


def phase_diagnostics() -> dict[str, Any]:
    return {
        "contract": PHASE_CONTRACT,
        "valid_batches": 0,
        "missing_clock_batches": 0,
        "invalid_clock_batches": 0,
        "no_success_batches": 0,
        "missing_first_token_batches": 0,
        "nonpositive_windows": {"input": 0, "output": 0, "total": 0},
    }


def collect_phase_estimates(successes, wall_seconds, values, diagnostics, token_count):
    if not successes:
        diagnostics["no_success_batches"] += 1
        return
    raw = [json.loads(row.get("extra_metrics") or "{}").get("request_phase") for row in successes]
    if any(item is None for item in raw):
        diagnostics["missing_clock_batches"] += 1
        return
    phases = []
    for item in raw:
        observation = phase_observation(item)
        if observation is None:
            diagnostics["invalid_clock_batches"] += 1
            return
        phases.append(observation)
    offsets = {item["latency_offset_seconds"] for item in phases}
    start = min(item["start_seconds"] for item in phases)
    end = max(item["end_seconds"] for item in phases)
    if len(offsets) != 1 or end - start > wall_seconds + 1e-6:
        diagnostics["invalid_clock_batches"] += 1
        return
    offset = offsets.pop()
    diagnostics["valid_batches"] += 1
    windows = {"input": None, "output": None, "total": end - start - offset}
    firsts = [item["first_token_seconds"] for item in phases]
    if all(value is not None for value in firsts):
        windows.update(input=max(firsts) - start - offset, output=end - min(firsts))
    else:
        diagnostics["missing_first_token_batches"] += 1
    for key, seconds in windows.items():
        if seconds is not None and seconds <= 0:
            diagnostics["nonpositive_windows"][key] += 1

    def total(field):
        counts = [token_count(row, field) for row in successes]
        return sum(counts) if all(count is not None for count in counts) else None

    def append(metric, numerator, window):
        seconds = windows[window]
        if numerator is None or seconds is None or seconds <= 0:
            return
        rate = numerator / seconds
        if math.isfinite(rate):
            values[metric].append(rate)

    inputs, outputs = total("prefill_tokens"), total("decode_tokens")
    append("phase_input", inputs, "input")
    append("phase_output", outputs, "output")
    append(
        "phase_total",
        inputs + outputs if inputs is not None and outputs is not None else None,
        "total",
    )
    append("phase_qpm", len(successes) * 60, "total")
    uncached = []
    sources = set()
    for row in successes:
        source = row.get("cache_hit_source")
        if source not in {"API", "TTFT_inferred"}:
            break
        denominator = (
            row.get("api_prefill") if source == "API" else row.get("effective_prefill_tokens")
        )
        hit = row.get("cache_hit_tokens")
        if (
            any(type(value) is not int or value < 0 for value in (denominator, hit))
            or hit > denominator
        ):
            break
        uncached.append(denominator - hit)
        sources.add(source)
    if len(uncached) == len(successes):
        metric = (
            "phase_input_uncached_api" if sources == {"API"} else "phase_input_uncached_inferred"
        )
        append(metric, sum(uncached), "input")
