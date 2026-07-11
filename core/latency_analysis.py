"""
客户端 vs 引擎侧延迟对照分析。

手册强调：不要只信客户端测的 TTFT。客户端 TTFT 是端到端（含网络 + 排队 + 引擎），
引擎侧 TTFT（vLLM/SGLang 的 /metrics 直方图）只是引擎内部处理时间。两者之差
= 网络/排队/调度开销，能定位“延迟在传输还是在引擎”。

数据来源：
- 客户端：results_df 的 ttft / tpot（中位数，抗尾部）。
- 引擎侧：engine_metrics 的 engine_means（末采样点累积直方图均值）。
"""

from __future__ import annotations

from typing import Any


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None and v > 0)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def compute_client_vs_engine_latency(
    results: list[dict[str, Any]] | object,
    engine_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """对照客户端中位数 TTFT/TPOT 与引擎侧整体 TTFT/TPOT。

    Args:
        results: 结果列表（每项含 ttft/tpot），或 pandas DataFrame（自动取列）。
        engine_metrics: EngineMetricsPoller 汇总（含 engine_means）。

    Returns:
        {client_ttft_s, engine_ttft_s, ttft_overhead_s, ttft_overhead_pct,
         client_tpot_ms, engine_tpot_ms, tpot_overhead_ms,
         verdict} —— 缺数据项为 None。
    """
    # 兼容 DataFrame
    ttft_vals: list[float] = []
    tpot_vals: list[float] = []
    if results is not None:
        if hasattr(results, "iterrows"):  # pandas DataFrame
            for _, row in results.iterrows():
                ttft_vals.append(row.get("ttft"))
                tpot_vals.append(row.get("tpot"))
        elif isinstance(results, list):
            for r in results:
                if not r:
                    continue
                ttft = r.get("ttft")
                tpot = r.get("tpot")
                if ttft is not None:
                    ttft_vals.append(ttft)
                if tpot is not None:
                    tpot_vals.append(tpot)

    client_ttft = _median(ttft_vals)
    # tpot 在结果里单位是毫秒（TPOT per output token, ms）
    client_tpot_ms = _median(tpot_vals)

    em = engine_metrics or {}
    engine_means = em.get("engine_means") or {}
    engine_ttft = engine_means.get("ttft_s")
    engine_tpot_ms = (
        (engine_means.get("tpot_s") or 0) * 1000.0
        if engine_means.get("tpot_s") is not None
        else None
    )

    def _overhead(client, engine):
        if client is None or engine is None:
            return None
        return client - engine

    ttft_overhead = _overhead(client_ttft, engine_ttft)
    tpot_overhead = _overhead(client_tpot_ms, engine_tpot_ms)

    ttft_overhead_pct = None
    if ttft_overhead is not None and engine_ttft and engine_ttft > 0:
        ttft_overhead_pct = round(ttft_overhead / engine_ttft * 100, 1)

    return {
        "client_ttft_s": round(client_ttft, 4) if client_ttft is not None else None,
        "engine_ttft_s": round(engine_ttft, 4) if engine_ttft is not None else None,
        "ttft_overhead_s": (
            round(ttft_overhead, 4) if ttft_overhead is not None else None
        ),
        "ttft_overhead_pct": ttft_overhead_pct,
        "client_tpot_ms": (
            round(client_tpot_ms, 3) if client_tpot_ms is not None else None
        ),
        "engine_tpot_ms": (
            round(engine_tpot_ms, 3) if engine_tpot_ms is not None else None
        ),
        "tpot_overhead_ms": (
            round(tpot_overhead, 3) if tpot_overhead is not None else None
        ),
        "verdict": _verdict(client_ttft, engine_ttft, ttft_overhead),
    }


def _verdict(client_ttft, engine_ttft, overhead) -> str:
    if client_ttft is None or engine_ttft is None or overhead is None:
        return "对照数据不全（缺客户端 TTFT 或引擎侧 /metrics TTFT），无法判定。"
    if engine_ttft <= 0:
        return "引擎侧 TTFT 异常为 0，请检查 /metrics 直方图是否上报。"
    ratio = overhead / engine_ttft
    if ratio <= 0.15:
        return f"客户端与引擎侧 TTFT 接近（开销仅 {ratio * 100:.0f}%），延迟主要在引擎内部。"
    if ratio <= 0.5:
        return (
            f"客户端比引擎侧高 {ratio * 100:.0f}%，存在一定网络/排队/调度开销，"
            f"可排查连接复用、负载均衡、请求排队。"
        )
    return (
        f"客户端比引擎侧高 {ratio * 100:.0f}%，开销显著，延迟主要在引擎之外"
        f"（网络/排队/客户端处理），而非模型推理本身。"
    )
