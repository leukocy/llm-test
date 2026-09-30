"""Shared scenario definitions and descriptive analysis over complete report slices."""

from __future__ import annotations

from html import escape
from typing import Any

METRICS = {
    "ttft": ("首字延迟", "TTFT", "s"),
    "tpot": ("每 token 延迟", "TPOT", "s"),
    "tps": ("逐请求生成速度", "TPS", "token/s"),
    "prefill_speed": ("逐请求输入处理速度", "Prefill speed", "token/s"),
    "total_time": ("请求总耗时", "Total time", "s"),
}
STATISTICS = {
    "median": "p50",
    "mean": "均值",
    "p95": "p95",
    "p99": "p99",
    "min": "观测最小值",
    "max": "观测最大值",
}
SCENARIOS = {
    "concurrency": ("并发扩展", "concurrency_level", "并发数"),
    "custom_text": ("自定义文本", "concurrency_level", "并发数"),
    "prefill": ("输入长度扫描", "input_tokens_target", "目标输入长度 (token)"),
    "long_context": ("长上下文", "context_length_target", "目标上下文长度 (token)"),
    "segmented_prefill": ("缓存分段", "context_length_target", "目标分段长度 (token)"),
    "throughput_matrix": ("并发 × 上下文矩阵", "context_length_target", "目标上下文长度 (token)"),
    "stability": ("稳定性分组概览", "concurrency_level", "并发数"),
}


def scenario_analysis(summary: dict[str, Any]) -> dict[str, Any]:
    """Describe observed extrema, never infer capacity, cache hits or significance."""
    kind = summary["run"]["test_type"]
    title, _, axis = SCENARIOS.get(kind, ("分组观测", "concurrency_level", summary["group_axis"]))
    notes = [
        "图形使用完整正式观测的分组统计，不使用详情页最近 50 条预览；预热不计入。",
        "p50 描述典型请求，p95 / p99 描述尾部；分位数为线性插值，不是置信区间。",
        "TPS 与输入处理速度为逐请求指标；不能乘以并发数作为系统吞吐或 QPM。",
        "均值与观测极值可用于核对初版图形；极值受样本数影响，不代表稳定性能上限。",
        "空值表示未采集或无有效样本，失败仍计入成功率分母；曲线不跨空值连线。",
    ]
    if summary.get("origin", {}).get("kind") == "saved_csv":
        notes[0] = "图形使用完整 CSV 的分组统计；原运行的预热、计划与执行条件未经核验。"
        notes[4] = "空值表示无有效样本；CSV 状态未知的行不计入成功率或性能统计。曲线不跨空值连线。"
    if kind == "throughput_matrix":
        notes.append("矩阵按并发分别绘制，热力图以并发 × 目标上下文定位；缺测组合保留为空。")
    elif kind == "segmented_prefill":
        notes.append(
            "分段 TTFT 变化仅为观测相关性；本图未按缓存命中分层，不能称为未缓存 TTFT 或缓存收益。"
        )
    elif kind == "stability":
        notes.append("此图按并发聚合，不是时间序列；不能据此判断随时间漂移或连续运行稳定性。")
    elif kind in {"prefill", "long_context"}:
        notes.append("横轴是计划目标长度，实际输入 token 和偏离程度须结合分组统计核对。")
    if not summary["integrity"]["verified"]:
        notes.append("运行未通过完整性核验，分析仅供诊断；缺失组合与中途停止可能影响比较。")
    observations = []
    for metric, (label, _, unit) in METRICS.items():
        usable = [g for g in summary["groups"] if g["metrics"][metric]["median"] is not None]
        if not usable:
            continue
        low = min(usable, key=lambda g: g["metrics"][metric]["median"])
        high = max(usable, key=lambda g: g["metrics"][metric]["median"])
        observations.append(
            {
                "metric": metric,
                "text": f"{label} p50 观测范围：{low['metrics'][metric]['median']:.4g}–"
                f"{high['metrics'][metric]['median']:.4g} {unit}；"
                f"最小值所在组 {low['label']}（n={low['metrics'][metric]['count']}），"
                f"最大值所在组 {high['label']}（n={high['metrics'][metric]['count']}）。"
                "并列时仅列首组，不是显著性或优劣排名。",
            }
        )
    return {"title": title, "axis": axis, "notes": notes, "observations": observations}


def profile_series(summary: dict[str, Any]) -> tuple[str, bool, list[dict[str, Any]]]:
    """Keep numeric dimensions separate from presentation labels, including matrix gaps."""
    kind = summary["run"]["test_type"]
    _, field, axis = SCENARIOS.get(kind, ("", "concurrency_level", summary["group_axis"]))
    groups = summary["groups"]
    required = [field] + (["concurrency_level"] if kind == "throughput_matrix" else [])
    numeric = all(
        all(isinstance(g.get("dimensions", {}).get(key), int) for key in required) for g in groups
    )
    if not numeric:
        return (
            summary["group_axis"],
            False,
            [{"name": "全部条件", "x": [g["label"] for g in groups], "groups": groups}],
        )
    planned = (summary.get("measurement_protocol") or {}).get("cells", [])
    plan_field = "concurrency" if field == "concurrency_level" else "input_tokens_target"
    planned_xs = {cell[plan_field] for cell in planned if isinstance(cell.get(plan_field), int)}
    xs = sorted({g["dimensions"][field] for g in groups} | planned_xs)
    levels = (
        sorted(
            {g["dimensions"]["concurrency_level"] for g in groups}
            | {cell["concurrency"] for cell in planned if isinstance(cell.get("concurrency"), int)}
        )
        if kind == "throughput_matrix"
        else [None]
    )
    series = []
    for level in levels:
        cells = {
            g["dimensions"][field]: g
            for g in groups
            if level is None or g["dimensions"]["concurrency_level"] == level
        }
        series.append(
            {
                "name": f"并发 {level}" if level is not None else "全部条件",
                "x": xs,
                "groups": [cells.get(x) for x in xs],
            }
        )
    return axis, True, series


def profile_svg(summary: dict[str, Any], metric: str) -> str:
    """Offline printable p50 profiles; SVG contains escaped text and no executable content."""
    axis, numeric, series = profile_series(summary)
    values = [
        g["metrics"][metric]["median"]
        for s in series
        for g in s["groups"]
        if g and g["metrics"][metric]["median"] is not None
    ]
    if not values:
        return ""
    label, abbreviation, unit = METRICS[metric]
    xs = series[0]["x"]
    low_x, high_x = (min(xs), max(xs)) if numeric else (0, max(0, len(xs) - 1))
    high_y = max(values)
    colors = ["#2463a6", "#188b79", "#b86a16", "#8754b3", "#ad3b62"]
    height = 430 + ((len(series) - 1) // 5) * 16
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img"',
        f' aria-label="{escape(label)} p50"><rect width="1000" height="{height}" fill="white"/>',
        f'<text x="80" y="30" fill="#17243b" font-size="20">{escape(label)} p50 ({unit})</text>',
        '<path d="M80 55V330H950" fill="none" stroke="#9aacbf"/>',
    ]
    for tick in range(5):
        value = high_y * (tick / 4)
        y = 330 - tick * 65
        elements.append(
            f'<path d="M80 {y}H950" stroke="#e6edf5"/><text x="70" y="{y + 4}" text-anchor="end" font-size="12">{value:.3g}</text>'
        )
    for index, x in enumerate(xs):
        position = 80 + 870 * ((x if numeric else index) - low_x) / (high_x - low_x or 1)
        if index in {0, len(xs) - 1} or len(xs) <= 8:
            text = f"{x:,}" if numeric else str(x)[:25]
            elements.append(
                f'<text x="{position:.2f}" y="350" text-anchor="middle" font-size="12">{escape(text)}</text>'
            )
    for index, s in enumerate(series):
        previous = None
        color = colors[index % len(colors)]
        for position, g in zip(s["x"], s["groups"], strict=True):
            value = g["metrics"][metric]["median"] if g else None
            if value is None:
                previous = None
                continue
            x = 80 + 870 * ((position if numeric else xs.index(position)) - low_x) / (
                high_x - low_x or 1
            )
            y = 330 - 260 * (value / high_y)
            if previous:
                elements.append(
                    f'<path d="M{previous[0]:.2f} {previous[1]:.2f}L{x:.2f} {y:.2f}" stroke="{color}" fill="none"/>'
                )
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>{escape(g["label"])}: {value:.4g} {unit}, n={g["metrics"][metric]["count"]}, 失败={g["failures"]}</title></circle>'
            )
            previous = (x, y)
        elements.append(
            f'<text x="{80 + (index % 5) * 170}" y="{397 + (index // 5) * 16}" fill="{color}" font-size="12">{escape(s["name"])}</text>'
        )
    elements.append(
        f'<text x="515" y="375" text-anchor="middle" font-size="13">{escape(axis)}</text></svg>'
    )
    return f"<figure><figcaption>{abbreviation} p50 · 正式成功请求；完整性与来源见报告</figcaption>{''.join(elements)}</figure>"
