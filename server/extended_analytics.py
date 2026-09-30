"""Batch throughput and cache strata with separate sampling units and provenance."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

from core.benchmark.batch_observations import BATCH_CONTRACT
from server.observations import describe_values
from server.phase_analytics import PHASE_METRICS, collect_phase_estimates, phase_diagnostics

SYSTEM_METRICS = {
    "system_input_wall": ("系统输入吞吐（含缓存）", "Input wall throughput", "token/s"),
    "system_output_wall": ("系统输出吞吐", "Output wall throughput", "token/s"),
    "system_total_wall": ("系统总吞吐", "Total wall throughput", "token/s"),
    "system_qpm": ("成功请求处理速率", "QPM", "req/min"),
}
CACHE_METRICS = {
    "cache_tokens_api": ("API 缓存命中 token", "API cached tokens", "token"),
    "cache_rate_api": ("API 缓存命中比例", "API cache ratio", "%"),
    "cache_tokens_inferred": ("TTFT 推断缓存 token", "Inferred cached tokens", "token"),
    "cache_rate_inferred": ("TTFT 推断缓存比例", "Inferred cache ratio", "%"),
    "ttft_zero_cache_api": ("API 明确零命中 TTFT", "API zero-cache TTFT", "s"),
    "ttft_cache_api": ("API 有命中 TTFT", "API cache-hit TTFT", "s"),
}


def _number(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and (value > 0 if positive else value >= 0) else None


def _tokens(row: dict, field: str) -> float | None:
    value = _number(row.get(field))
    # Failed rows contain placeholder zero; only successful rows reach this helper.
    if (
        value is None
        or value != int(value)
        or not row.get("token_source")
        or str(row["token_source"]).lower() in {"unknown", "未知"}
    ):
        return None
    if not value and row.get("token_source") not in {"API", "api_usage"}:
        return None
    return value


def extended_observations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        key: [] for key in SYSTEM_METRICS | PHASE_METRICS | CACHE_METRICS
    }
    phase = phase_diagnostics()
    batches: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    untagged = 0
    cache_unknown = cache_invalid = 0
    cache_sources: dict[str, int] = defaultdict(int)
    weighted_cache = {
        source: {"hits": 0, "tokens": 0, "count": 0} for source in ("API", "TTFT_inferred")
    }
    for row in rows:
        extra = json.loads(row.get("extra_metrics") or "{}")
        obs = extra.get("system_measurement") if isinstance(extra, dict) else None
        if (
            isinstance(obs, dict)
            and isinstance(obs.get("id"), str)
            and re.fullmatch(r"[0-9a-f]{32}", obs["id"])
        ):
            batches[obs["id"]].append((row, obs))
        else:
            untagged += 1
        if row.get("error"):
            continue
        source = row.get("cache_hit_source")
        if source not in {"API", "TTFT_inferred"}:
            cache_unknown += 1
            continue
        hit = _number(row.get("cache_hit_tokens"))
        if hit is None or hit != int(hit):
            cache_invalid += 1
            continue
        # API ratios require an API prompt-token denominator, never a local estimate.
        denominator = _number(
            row.get("api_prefill") if source == "API" else row.get("effective_prefill_tokens"),
        )
        if denominator is not None and (hit > denominator or denominator != int(denominator)):
            cache_invalid += 1
            continue
        cache_sources[source] += 1
        suffix = "api" if source == "API" else "inferred"
        values[f"cache_tokens_{suffix}"].append(hit)
        if denominator is not None and denominator > 0:
            values[f"cache_rate_{suffix}"].append(hit / denominator * 100)
            weighted_cache[source]["hits"] += int(hit)
            weighted_cache[source]["tokens"] += int(denominator)
            weighted_cache[source]["count"] += 1
        ttft = _number(row.get("ttft"), positive=True)
        if source == "API" and ttft is not None:
            values["ttft_zero_cache_api" if hit == 0 else "ttft_cache_api"].append(ttft)
    valid_batches = invalid_batches = 0
    for items in batches.values():
        obs = items[0][1]
        elapsed = _number(obs.get("elapsed_seconds"), positive=True)
        expected = obs.get("expected_requests")
        if (
            obs.get("version") != BATCH_CONTRACT
            or elapsed is None
            or isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected <= 0
            or len(items) != expected
            or obs.get("recorded_requests") != expected
            or any(item[1] != obs for item in items)
            or {_batch_index(row, obs) for row, _ in items} != set(range(expected))
            or any(row.get("batch_id") != obs["id"] for row, _ in items)
            or len(
                {
                    (
                        row.get("concurrency_level"),
                        row.get("input_tokens_target"),
                        row.get("context_length_target"),
                    )
                    for row, _ in items
                }
            )
            != 1
        ):
            invalid_batches += 1
            continue
        valid_batches += 1
        successes = [row for row, _ in items if not row.get("error")]
        collect_phase_estimates(successes, elapsed, values, phase, _tokens)
        qpm = len(successes) / elapsed * 60
        if math.isfinite(qpm):
            values["system_qpm"].append(qpm)
        totals: dict[str, float | None] = {}
        for field in ["prefill_tokens", "decode_tokens"]:
            counts = [_tokens(row, field) for row in successes]
            total = (
                sum(value for value in counts if value is not None)
                if all(value is not None for value in counts)
                else None
            )
            totals[field] = total
        for key, total in [
            ("system_input_wall", totals["prefill_tokens"]),
            ("system_output_wall", totals["decode_tokens"]),
            (
                "system_total_wall",
                sum(v for v in totals.values() if v is not None)
                if all(v is not None for v in totals.values())
                else None,
            ),
        ]:
            rate = total / elapsed if total is not None else None
            if rate is not None and math.isfinite(rate):
                values[key].append(rate)
    return {
        "metrics": {key: describe_values(samples) for key, samples in values.items()},
        "system": {
            "contract": BATCH_CONTRACT,
            "valid_batches": valid_batches,
            "invalid_batches": invalid_batches,
            "untagged_requests": untagged,
        },
        "cache": {
            "weighted": {
                source: {
                    **totals,
                    "rate": totals["hits"] / totals["tokens"] * 100 if totals["tokens"] else None,
                }
                for source, totals in weighted_cache.items()
            },
            "sources": dict(cache_sources),
            "unknown_successes": cache_unknown,
            "invalid_observations": cache_invalid,
        },
        "phase": phase,
    }


def _batch_index(row: dict, observation: dict) -> int | None:
    if observation.get("index_scope") != "stability_window":
        index = row.get("request_index")
        return index if type(index) is int else None
    timing = json.loads(row.get("extra_metrics") or "{}").get("timing_observation")
    if (
        not isinstance(timing, dict)
        or timing.get("version") != "stability-clock-v1"
        or timing.get("id") != observation["id"]
        or timing.get("window_state") != "completed"
        or timing.get("expected_requests") != observation["expected_requests"]
        or timing.get("window_seconds") != observation["elapsed_seconds"]
        or type(timing.get("index")) is not int
    ):
        return None
    index = timing["index"]
    return index if isinstance(index, int) else None
