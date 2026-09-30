"""Bounded completion-time slices from explicit, relative monotonic observations."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from html import escape
from typing import Any

from core.benchmark.batch_observations import BATCH_CONTRACT
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
    "window_state",
    "paused_seconds",
    "scheduling_seconds",
    "attempt",
    "admission_budget_seconds",
)
REQUEST_METRICS = ("ttft", "tpot", "tps", "prefill_speed", "total_time")
TIMELINE_NOTES = [
    "横轴从调度开始计时，使用同一单调时钟的客户端请求包装器完成时刻；不是数据库写入时间，也不等同于服务端处理时间。",
    "按完成时间归入左闭右开时间窗，最终边界归入最后一窗；窗口含请求排空，可能超过计划发起时长。",
    "指标仅统计成功且有限、大于零的观测；失败仍计入完成请求数和失败数。空窗与未采集指标保留空值，不连线。",
    "自动划分最多 60 个时间窗；每窗有效 n、完成数和失败数分别记录。分位数为描述性统计，小样本尾部不可视为稳定性或显著性证明。",
    "时间来源缺失或无效的请求不参与曲线；完整性未通过时，计数只代表已记录部分。未完成请求不归入完成时间窗。",
    "连续测量的实时窗口与请求在同一事务保存；尚在发起或排空时不标为完整。暂停等待不消耗计划发起时长，但保留在时间轴和最终系统墙钟速率的分母中。",
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


def _valid_window(window: Any) -> bool:
    return bool(
        isinstance(window, dict)
        and window.get("version") == "stability-clock-v1"
        and window.get("clock") == "monotonic"
        and window.get("anchor") == "scheduler_start"
        and isinstance(window.get("id"), str)
        and re.fullmatch(r"[0-9a-f]{32}", window["id"])
        and isinstance(window.get("state"), str)
        and window["state"] in {"running", "paused", "completed", "interrupted"}
        and all(
            _finite(window.get(k))
            for k in ["window_seconds", "planned_seconds", "paused_seconds", "scheduling_seconds"]
        )
        and 0 < window["window_seconds"] <= 86400
        and 0 < window["planned_seconds"] <= 3600
        and 0 <= window["paused_seconds"] <= window["window_seconds"]
        and 0 <= window["scheduling_seconds"] <= window["window_seconds"]
        and type(window.get("expected_requests")) is int
        and type(window.get("recorded_requests")) is int
        and 0 <= window["recorded_requests"] <= window["expected_requests"]
        and ("attempt" not in window or (type(window["attempt"]) is int and window["attempt"] > 0))
        and (
            "admission_budget_seconds" not in window
            or (
                _finite(window["admission_budget_seconds"])
                and 0 <= window["admission_budget_seconds"] <= window["planned_seconds"]
            )
        )
    )


def resolve_stability_row(row: dict[str, Any], window: Any) -> dict[str, Any]:
    """Resolve live rows using only the window from the same database read transaction."""
    if isinstance(window, list):
        value = timing_observation(row.get("extra_metrics"))
        matching = [w for w in window if _valid_window(w) and value and w["id"] == value["id"]]
        return resolve_stability_row(row, matching[0]) if len(matching) == 1 else row
    if not _valid_window(window):
        return row
    value = timing_observation(row.get("extra_metrics"))
    if value is None or any(value[k] != window[k] for k in ["id", "version", "clock", "anchor"]):
        return row
    for key in ["window_seconds", "planned_seconds", "expected_requests"]:
        if value[key] is not None and value[key] != window[key]:
            return row
        value[key] = window[key]
    value.update(
        window_state=window["state"],
        paused_seconds=window["paused_seconds"],
        scheduling_seconds=window["scheduling_seconds"],
    )
    for key in ("attempt", "admission_budget_seconds"):
        if key in window:
            value[key] = window[key]
    extra = json.loads(row.get("extra_metrics") or "{}")
    extra["timing_observation"] = value
    if (
        window["state"] == "completed"
        and _valid(value)
        and window.get("admission_budget_seconds", window["planned_seconds"]) > 0
    ):
        extra["system_measurement"] = {
            "version": BATCH_CONTRACT,
            "index_scope": "stability_window",
            "id": window["id"],
            "elapsed_seconds": window["window_seconds"],
            "expected_requests": window["expected_requests"],
            "recorded_requests": window["recorded_requests"],
        }
    return {**row, "extra_metrics": json.dumps(extra, ensure_ascii=False)}


def stability_time_series(rows: list[dict[str, Any]], window: Any = None) -> dict[str, Any]:
    if isinstance(window, list):
        return segmented_time_series(rows, window)
    rows = [resolve_stability_row(row, window) for row in rows]
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
    conflict = (
        len(signatures) > 1
        or len(set(indices)) != len(indices)
        or (window is not None and not _valid_window(window))
        or (_valid_window(window) and window["recorded_requests"] != len(rows))
    )
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
        "live": _valid_window(window) and window["state"] in {"running", "paused"},
        "window_state": window["state"] if _valid_window(window) else None,
        "attempt": window.get("attempt") if _valid_window(window) else None,
        "admission_budget_seconds": window.get("admission_budget_seconds")
        if _valid_window(window)
        else None,
    }
    if not valid:
        result["notes"].append("没有可核验的单调时钟时间记录，无法绘制稳定性时间序列。")
        return result
    if result.get("admission_budget_seconds") == 0:
        result["notes"].append("本窗口仅补做中断前未提交的请求，不产生连续负载系统速率。")
    _, span, planned, expected = next(iter(signatures))
    width = max(1, math.ceil(span / 60))
    count = max(1, math.ceil(span / width))
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
                "end_seconds": min(span, (index + 1) * width),
                "requests": len(selected),
                "successes": len(successful),
                "failures": len(selected) - len(successful),
                "metrics": metrics,
            }
        )
    result.update(
        complete=len(valid) == len(rows) == expected
        and (not _valid_window(window) or window["state"] == "completed"),
        window_seconds=span,
        planned_seconds=planned,
        bin_seconds=width,
        bins=bins,
    )
    return result


def segmented_time_series(rows: list[dict[str, Any]], windows: list[dict]) -> dict[str, Any]:
    """Keep each clock origin separate; no concatenated or cross-interruption curve."""
    ids = [window.get("id") if isinstance(window, dict) else None for window in windows]
    valid_windows = all(_valid_window(window) for window in windows) and len(ids) == len(set(ids))
    segments = []
    orphaned = []
    members: dict[str, list] = defaultdict(list)
    for row in rows:
        timing = timing_observation(row.get("extra_metrics"))
        if not timing or timing["id"] not in ids:
            orphaned.append(row)
        else:
            members[timing["id"]].append(row)
    if valid_windows:
        segments = [stability_time_series(members[window["id"]], window) for window in windows]
    latest = segments[-1] if segments else None
    return {
        "contract": "stability-clock-v1",
        "segments": segments,
        "cross_interruption": len(windows) > 1,
        "timed_requests": sum(s["timed_requests"] for s in segments),
        "missing_requests": sum(s["missing_requests"] for s in segments) + len(orphaned),
        "invalid_requests": sum(s["invalid_requests"] for s in segments)
        + (0 if valid_windows else len(rows)),
        "conflict": not valid_windows or bool(orphaned) or any(s["conflict"] for s in segments),
        "complete": bool(len(segments) == 1 and segments[0]["complete"] and not orphaned),
        "window_seconds": None,
        "planned_seconds": latest["planned_seconds"] if latest else None,
        "bin_seconds": None,
        "bins": [],
        "live": bool(latest and latest["live"]),
        "window_state": latest["window_state"] if latest else None,
        "notes": [
            *TIMELINE_NOTES,
            "每次执行尝试有独立时钟原点，窗口分别展示，不拼接、不跨中断连线；各窗口系统速率仅解释其原始观测。",
            "剩余发起时长以最后成功保存的调度时间计算；未保存的时间和未提交的请求可能重复执行，不能证明远端只执行一次。",
        ],
    }


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
