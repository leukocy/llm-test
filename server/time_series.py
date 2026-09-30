"""Bounded completion-time slices from explicit, relative monotonic observations."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from html import escape
from typing import Any

from server.observations import describe_values

TIMING_FIELDS = (
    "version",
    "clock",
    "anchor",
    "id",
    "index",
    "start_seconds",
    "end_seconds",
    "window_seconds",
    "planned_seconds",
    "expected_requests",
)
REQUEST_METRICS = ("ttft", "tpot", "tps", "prefill_speed", "total_time")
TIMELINE_NOTES = [
    "横轴从调度开始计时，使用同一单调时钟的客户端请求包装器完成时刻；不是数据库写入时间，也不等同于服务端处理时间。",
    "按完成时间归入左闭右开时间窗，最终边界归入最后一窗；窗口含请求排空，可能超过计划发起时长。",
    "指标仅统计成功且有限、大于零的观测；失败仍计入完成请求数和失败数。空窗与未采集指标保留空值，不连线。",
    "自动划分最多 60 个时间窗；每窗有效 n、完成数和失败数分别记录。分位数为描述性统计，小样本尾部不可视为稳定性或显著性证明。",
    "时间来源缺失或无效的请求不参与曲线；完整性未通过时，计数只代表已记录部分。未完成请求不归入完成时间窗。",
]


def timing_observation(raw: Any) -> dict[str, Any] | None:
    """Whitelist timing provenance without exposing prompts or arbitrary extra fields."""
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None
    value = payload.get("timing_observation") if isinstance(payload, dict) else None
    return {key: value.get(key) for key in TIMING_FIELDS} if isinstance(value, dict) else None


def _finite(value: Any) -> bool:
    try:
        return (
            isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        )
    except OverflowError:
        return False


def _valid(value: dict[str, Any]) -> bool:
    return bool(
        value["version"] == "stability-clock-v1"
        and value["clock"] == "monotonic"
        and value["anchor"] == "scheduler_start"
        and isinstance(value["id"], str)
        and re.fullmatch(r"[0-9a-f]{32}", value["id"])
        and all(
            _finite(value[k])
            for k in ("start_seconds", "end_seconds", "window_seconds", "planned_seconds")
        )
        and 0 <= value["start_seconds"] <= value["end_seconds"] <= value["window_seconds"] <= 86400
        and 0 < value["planned_seconds"] <= 3600
        and value["window_seconds"] > 0
        and type(value["expected_requests"]) is int
        and value["expected_requests"] > 0
        and type(value["index"]) is int
        and 0 <= value["index"] < value["expected_requests"]
    )


def stability_time_series(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = []
    missing = invalid = 0
    for row in rows:
        value = timing_observation(row.get("extra_metrics"))
        if value is None:
            missing += 1
        elif not _valid(value):
            invalid += 1
        else:
            valid.append((row, value))
    signatures = {
        (v["id"], v["window_seconds"], v["planned_seconds"], v["expected_requests"])
        for _, v in valid
    }
    indices = [v["index"] for _, v in valid]
    conflict = len(signatures) > 1 or len(set(indices)) != len(indices)
    if conflict:
        invalid += len(valid)
        valid = []
    result: dict[str, Any] = {
        "contract": "stability-clock-v1",
        "timed_requests": len(valid),
        "missing_requests": missing,
        "invalid_requests": invalid,
        "conflict": conflict,
        "complete": False,
        "window_seconds": None,
        "planned_seconds": None,
        "bin_seconds": None,
        "bins": [],
        "notes": list(TIMELINE_NOTES),
    }
    if not valid:
        result["notes"].append("没有可核验的单调时钟时间记录，无法绘制稳定性时间序列。")
        return result
    _, window, planned, expected = next(iter(signatures))
    width = max(1, math.ceil(window / 60))
    count = max(1, math.ceil(window / width))
    by_bin: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row, value in valid:
        index = min(count - 1, math.floor(value["end_seconds"] / width))
        by_bin[index].append(row)
    bins = []
    for index in range(count):
        selected = by_bin[index]
        successful = [row for row in selected if not row.get("error")]
        metrics = {
            metric: describe_values(
                [
                    float(row[metric])
                    for row in successful
                    if _finite(row.get(metric)) and row[metric] > 0
                ]
            )
            for metric in REQUEST_METRICS
        }
        bins.append(
            {
                "start_seconds": index * width,
                "end_seconds": min(window, (index + 1) * width),
                "requests": len(selected),
                "successes": len(successful),
                "failures": len(selected) - len(successful),
                "metrics": metrics,
            }
        )
    result.update(
        complete=len(valid) == len(rows) == expected,
        window_seconds=window,
        planned_seconds=planned,
        bin_seconds=width,
        bins=bins,
    )
    return result


def timeline_svg(timeline: dict[str, Any], metric: str, label: str, unit: str) -> str:
    """Offline p50 curve; empty time slices split the path instead of inventing values."""
    bins = timeline["bins"]
    values = [b["metrics"][metric]["median"] for b in bins]
    observed = [v for v in values if v is not None]
    if not observed:
        return ""
    high = max(observed) or 1
    span = timeline["window_seconds"]
    segments: list[str] = []
    points: list[str] = []
    for b, value in zip(bins, values, strict=True):
        if value is None:
            if points:
                segments.append(" ".join(points))
                points = []
            continue
        x = 80 + 800 * ((b["start_seconds"] + b["end_seconds"]) / 2) / span
        y = 285 - 220 * (value / high)
        points.append(f"{x:.3f},{y:.3f}")
        segments.append(
            f"<circle cx='{x:.3f}' cy='{y:.3f}' r='4'><title>"
            f"{b['start_seconds']:.3f}–{b['end_seconds']:.3f} s · p50={value:.4g} · "
            f"n={b['metrics'][metric]['count']} · 完成={b['requests']} · 失败={b['failures']}"
            "</title></circle>"
        )
    if points:
        segments.append(" ".join(points))
    shapes = "".join(
        s if s.startswith("<") else f"<polyline points='{s}' fill='none'/>" for s in segments
    )
    title = escape(f"{label} p50 ({unit}) · 完成时间窗")
    return (
        f"<figure><svg viewBox='0 0 960 360' role='img' aria-label='{title}'>"
        f"<text x='80' y='30'>{title}</text>"
        "<path d='M80 60 V285 H880' fill='none' stroke='#8795a8'/>"
        f"<text x='15' y='65'>{high:.4g}</text><text x='45' y='290'>0</text>"
        f"<text x='80' y='315'>0</text><text x='840' y='315'>{span:.3f}</text>"
        "<text x='320' y='345'>距调度开始的时间（秒）· 时间窗中点</text>"
        f"<g stroke='#2463a6' fill='#2463a6' stroke-width='2'>{shapes}</g></svg>"
        f"<figcaption>每窗 {timeline['bin_seconds']} 秒；有效计时 {timeline['timed_requests']}，"
        f"缺失 {timeline['missing_requests']}，无效 {timeline['invalid_requests']}；缺测不连线。</figcaption></figure>"
    )
