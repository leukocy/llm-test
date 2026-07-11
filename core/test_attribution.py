"""
状态 / 瓶颈 / 异常归因 —— 把 insights 与资源指标反推成结构化结论。

手册要求每条测试给出一句话归因：瓶颈在哪（带宽/算力/调度/通信/KV/数据质量），
状态如何（未测/异常/已通过/需复测/可对外）。本模块提供纯函数，供 UI 在测试后调用并写回 DB。

输入是 ui.insights.generate_performance_insights 的输出（纯文本 insight 列表 +
并行 severities 列表，severity ∈ positive/neutral/warning/critical）+
资源/成功率指标；不依赖 streamlit，可被 core 与 ui 共用。
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any


class TestStatusDetail(str, Enum):
    """测试状态明细（手册：未测/异常/已通过/需复测/可对外）。"""

    __test__ = False  # 防止 pytest 把本枚举误当作测试类收集

    UNTESTED = "untested"  # 未测
    ABNORMAL = "abnormal"  # 异常（关键指标缺失/失败率高）
    PASSED = "passed"  # 已通过
    NEEDS_RETEST = "needs_retest"  # 需复测（成功率边界/可疑）
    EXTERNAL_READY = "external_ready"  # 可对外（需 publish_gate 复核，见 core.publish_gate）


# insight 关键词 → 瓶颈标签（按优先级从前到后匹配）
_BOTTLENECK_RULES: list[tuple[list[str], str]] = [
    # 数据质量问题（最高优先级，结论不可信）
    (
        [
            "data anomaly",
            "throughput anomaly",
            "prefill speed anomaly",
            "ttft missing",
            "tps missing",
        ],
        "data_quality",
    ),
    # 算力 / prefill 瓶颈
    (
        [
            "long-text compute bottleneck",
            "low prefill speed",
            "high ttft",
            "compute bottleneck",
        ],
        "compute_prefill",
    ),
    # 并发饱和 / 调度
    (
        [
            "overload degradation",
            "low concurrency throughput efficiency",
            "queuing",
            "contention",
        ],
        "concurrency_saturation",
    ),
    # 延迟 / KV
    (["high latency", "tpot notable increase", "kv cache"], "latency"),
]


def derive_bottleneck(
    insights: list[str],
    severities: list[str] | None = None,
    bandwidth_utilization_pct: float | None = None,
) -> str | None:
    """从 insights + 带宽利用率反推主瓶颈标签。无明确结论返回 None。

    severities 是与 insights 平行的严重度列表（"positive"/"neutral"/
    "warning"/"critical"）。提供时用于判断正向洞察；否则回退到文本检测。
    """
    text = " ".join(insights or [])

    for keywords, label in _BOTTLENECK_RULES:
        if any(k in text.lower() for k in keywords):
            return label

    # 带宽利用率是强信号：decode 接近带宽上界 → memory_bandwidth 瓶颈
    if bandwidth_utilization_pct is not None and bandwidth_utilization_pct >= 80:
        return "memory_bandwidth"

    # 有正向洞察 → 无瓶颈（用 severities 优先，回退到 emoji 文本兼容旧数据）
    if severities is not None:
        if any(s == "positive" for s in severities):
            return None
    elif any("🚀" in i or "🏆" in i for i in (insights or [])):
        return None

    return None


def _has_critical(
    insights: list[str],
    severities: list[str] | None = None,
) -> bool:
    """是否存在关键问题（非“数据不足”类）。

    优先用 severities 判断；回退到 emoji 文本检测兼容旧数据。
    """
    if severities is not None:
        return any(
            sev == "critical" and "analysis skipped" not in txt.lower()
            for sev, txt in zip(severities, insights or [], strict=False)
        )
    return any("❌" in i and "analysis skipped" not in i.lower() for i in insights or [])


def derive_status_detail(
    success: bool,
    insights: list[str],
    severities: list[str] | None = None,
    success_rate: float | None = None,
) -> TestStatusDetail:
    """根据成功与否 / 关键问题 / 成功率判定状态。"""
    if not success:
        return TestStatusDetail.ABNORMAL

    if _has_critical(insights, severities):
        return TestStatusDetail.ABNORMAL

    if success_rate is not None:
        if success_rate < 0.5:
            return TestStatusDetail.ABNORMAL
        if success_rate < 0.95:
            return TestStatusDetail.NEEDS_RETEST

    return TestStatusDetail.PASSED


def derive_error_attribution(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """从失败请求的 error/error_type 聚合 run 级异常归因。"""
    if not results:
        return {"error_type": None, "error_detail": None, "count": 0}

    failed = [r for r in results if r and (r.get("error") or r.get("error_type"))]
    if not failed:
        return {"error_type": None, "error_detail": None, "count": 0}

    type_counts = Counter(
        (r.get("error_type") or _classify_error(r.get("error")) or "unknown") for r in failed
    )
    top_type, top_count = type_counts.most_common(1)[0]
    # 取该类型的一个代表 error 文本
    sample = next(
        (
            r.get("error")
            for r in failed
            if (r.get("error_type") or _classify_error(r.get("error")) or "unknown") == top_type
        ),
        None,
    )
    return {
        "error_type": top_type,
        "error_detail": (sample[:200] if sample else None),
        "count": len(failed),
        "top_type_count": top_count,
    }


_ERROR_KEYWORDS: list[tuple[list[str], str]] = [
    (["timeout", "timed out"], "timeout"),
    (["rate limit", "429", "too many requests"], "rate_limit"),
    (["unauthorized", "401", "authentication", "api key", "forbidden", "403"], "auth"),
    (["not found", "404", "model not found"], "not_found"),
    (
        ["overloaded", "503", "service unavailable", "502", "504", "connection"],
        "server",
    ),
    (["oom", "out of memory", "memory"], "oom"),
    (["context length", "too long", "maximum context"], "context_too_long"),
]


def _classify_error(msg: str | None) -> str | None:
    if not msg:
        return None
    low = str(msg).lower()
    for keywords, label in _ERROR_KEYWORDS:
        if any(k in low for k in keywords):
            return label
    return "unknown"
