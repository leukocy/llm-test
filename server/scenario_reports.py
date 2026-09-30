"""Shared scenario definitions and descriptive analysis over complete report slices."""

from __future__ import annotations

from html import escape
from typing import Any

from server.extended_analytics import CACHE_METRICS, SYSTEM_METRICS
from server.phase_analytics import PHASE_METRICS
from server.scenario_summaries import summary_cards

METRICS = {
    "ttft": ("首字延迟", "TTFT", "s"),
    "tpot": ("每 token 延迟", "TPOT", "s"),
    "tps": ("逐请求生成速度", "TPS", "token/s"),
    "prefill_speed": ("逐请求输入处理速度", "Prefill speed", "token/s"),
    "total_time": ("请求总耗时", "Total time", "s"),
    **SYSTEM_METRICS,
    **PHASE_METRICS,
    **CACHE_METRICS,
}


def sampling_unit(metric: str) -> str:
    return "测量批次" if metric in SYSTEM_METRICS or metric in PHASE_METRICS else "请求"


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
        "p50 描述典型样本（请求或测量批次），p95 / p99 描述尾部；分位数为线性插值，不是置信区间。",
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
            "总体分段 TTFT 变化仅为观测相关性；总体 TTFT 未按缓存命中分层，不能称为未缓存 TTFT 或缓存收益。可单独选择 API 缓存分层指标。"
        )
    elif kind == "stability":
        notes.append("此图按并发聚合，不是时间序列；不能据此判断随时间漂移或连续运行稳定性。")
    elif kind in {"prefill", "long_context"}:
        notes.append("横轴是计划目标长度，实际输入 token 和偏离程度须结合分组统计核对。")
    if not summary["integrity"]["verified"]:
        notes.append("运行未通过完整性核验，分析仅供诊断；缺失组合与中途停止可能影响比较。")
    observations = []
    for metric, (label, _, unit) in METRICS.items():
        usable = [
            g for g in summary["groups"] if g["metrics"].get(metric, {}).get("median") is not None
        ]
        if not usable:
            continue
        low = min(usable, key=lambda g: g["metrics"][metric]["median"])
        high = max(usable, key=lambda g: g["metrics"][metric]["median"])
        observations.append(
            {
                "metric": metric,
                "text": f"{label} p50 观测范围：{low['metrics'][metric]['median']:.4g}–"
                f"{high['metrics'][metric]['median']:.4g} {unit}；"
                f"最小值所在组 {low['label']}（n={low['metrics'][metric]['count']} {sampling_unit(metric)}），"
                f"最大值所在组 {high['label']}（n={high['metrics'][metric]['count']} {sampling_unit(metric)}）。"
                "并列时仅列首组，不是显著性或优劣排名。",
            }
        )
    notes.extend(
        [
            "系统输入/输出/总吞吐与 QPM 使用 batch-wall-v1：成功 token / 完整批次墙钟窗口；n 为批次数，每批一次，不按请求重复加权。QPM = 成功请求数 / 窗口秒数 × 60。",
            "批次窗口包含调度、失败等待和客户端开销，不扣延迟校准；不含批次外的提示词准备、预热或暂停。输入吞吐包含缓存 token；这是墙钟吞吐，不是引擎阶段吞吐或峰值容量。",
            "缺少批次来源、成员不齐或成员元数据冲突时不推算系统吞吐。历史重复的 system_* 字段不能代替该口径。",
            "client-phase-v1 按完整批次复算客户端阶段估计：输入窗口为最晚首 Token − 最早发送 − 延迟偏移；输出窗口为最晚成功结束 − 最早首 Token；总窗口为最晚成功结束 − 最早发送 − 延迟偏移。QPM = 成功请求数 / 总窗口 × 60。",
            "阶段估计使用成功请求的客户端单调时钟，每批一次。包含网络和服务商排队，不是引擎内部 Prefill/Decode 计时；成功请求窗口不包含失败等待。偏移扣除后窗口非正时不钳位、不产生速率。",
            "阶段输出分子为完整输出 token，含首 Token；逐请求 TPS 的跳过首 Token 选项不改变该分子。输入含缓存与未缓存分开；未缓存分子必须有全部成功请求的明确缓存来源和对应 token 分母，API 与含 TTFT 推断的估计分别统计。",
            "阶段时钟缺失、首 Token 缺失或批次成员不完整时，相关指标为空；不从旧 system_* 字段或校准后的 TTFT 反推原时钟。恢复后的独立窗口分别采样，不合并跨中断阶段时间。",
            "API 与 TTFT 推断的缓存值分开；API 明确返回零才计零命中，缺字段为未知。API 比例只使用 API prompt token 分母，未采集分母不补零。",
            "缓存分层 TTFT 是 API 明确零命中 / 有命中的观测对照，不证明缓存因果收益。系统速率和明确缓存零值是有效观测。",
        ]
    )
    extended = summary.get("extended_observations")
    if extended:
        system = extended["system"]
        cache = extended["cache"]
        phase = extended.get("phase")
        notes.extend(
            [
                f"本报告完整测量批次 {system['valid_batches']}；排除不完整/冲突批次 {system['invalid_batches']}；缺少批次来源的请求 {system['untagged_requests']}。",
                f"成功请求中，API 缓存记录 {cache['sources'].get('API', 0)}；TTFT 推断记录 {cache['sources'].get('TTFT_inferred', 0)}；缓存来源未知 {cache['unknown_successes']}；无效缓存观测 {cache['invalid_observations']}。",
            ]
        )
        if phase:
            notes.append(
                f"阶段时钟有效批次 {phase['valid_batches']}；缺少时钟 {phase['missing_clock_batches']}；无效时钟 {phase['invalid_clock_batches']}；首 Token 记录不齐 {phase['missing_first_token_batches']}；无成功请求 {phase['no_success_batches']}。"
            )
    return {
        "title": title,
        "axis": axis,
        "notes": notes,
        "observations": observations,
        "cards": summary_cards(summary),
    }


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
        if g and g["metrics"].get(metric, {}).get("median") is not None
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
            value = g["metrics"].get(metric, {}).get("median") if g else None
            if value is None:
                previous = None
                continue
            x = 80 + 870 * ((position if numeric else xs.index(position)) - low_x) / (
                high_x - low_x or 1
            )
            y = 330 - 260 * (value / (high_y or 1))
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
    return f"<figure><figcaption>{abbreviation} p50 · n 的单位：{sampling_unit(metric)}；完整性与来源见报告</figcaption>{''.join(elements)}</figure>"
