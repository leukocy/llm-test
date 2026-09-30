"""Scenario cards computed once for live, historical and offline reports."""

from __future__ import annotations

import math
import statistics
from typing import Any


def token_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    totals = {}
    for field in ("prefill_tokens", "decode_tokens"):
        values = [
            value
            for row in successful
            if isinstance(value := row.get(field), (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            and value == int(value)
        ]
        totals[field] = {
            "total": sum(int(value) for value in values) if values else None,
            "count": len(values),
            "successes": len(successful),
        }
    return totals


def summary_cards(summary: dict[str, Any]) -> list[dict[str, str]]:
    # Import at call time: scenario_reports owns the common axis and sampling contract.
    from server.scenario_reports import METRICS, STATISTICS, profile_series, sampling_unit

    cards: list[dict[str, str]] = []

    def show(value: float | None) -> str:
        return f"{value:.4g}" if value is not None and math.isfinite(value) else "未采集"

    def add(key: str, label: str, value: str, note: str) -> None:
        cards.append({"key": key, "label": label, "value": value, "note": note})

    overall = summary["overall"]
    add(
        "requests",
        "正式请求数",
        str(overall["requests"]),
        f"成功 {overall['successes']}；失败 {overall['failures']}。",
    )
    rate = overall.get("success_rate")
    add(
        "success_rate",
        "请求成功率",
        show(rate * 100 if rate is not None else None) + " %",
        f"成功 / 已知状态正式请求；未知状态 {overall.get('unknown_outcomes', 0)} 不计入分母。95% Wilson 区间另见总体统计。",
    )
    for field, label in (
        ("prefill_tokens", "成功输入 token 总量"),
        ("decode_tokens", "成功输出 token 总量"),
    ):
        total = overall.get("token_totals", {}).get(field, {})
        add(
            field,
            label,
            show(total.get("total")),
            f"有效 token 记录 {total.get('count', 0)} / {overall['successes']} 成功请求；缺失不补零，覆盖不全时为部分总量。",
        )

    def value(group: dict[str, Any] | None, metric: str, stat: str = "mean") -> float | None:
        raw = (group or {}).get("metrics", {}).get(metric, {}).get(stat)
        return (
            float(raw)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw)
            else None
        )

    def metric_card(
        key: str, label: str, group: dict[str, Any] | None, metric: str, stat: str = "mean"
    ) -> None:
        _, _, unit = METRICS[metric]
        m = (group or {}).get("metrics", {}).get(metric, {})
        condition = (group or {}).get("label", "计划条件未完成或无记录")
        add(
            key,
            label,
            show(value(group, metric, stat)) + f" {unit}",
            f"{condition}；{STATISTICS[stat]}，n={m.get('count', 0)} {sampling_unit(metric)}。",
        )

    def extreme(metric: str, highest: bool, stat: str = "mean", label: str | None = None) -> None:
        usable = [group for group in summary["groups"] if value(group, metric, stat) is not None]
        group = (
            (max if highest else min)(usable, key=lambda g: value(g, metric, stat) or 0.0)
            if usable
            else None
        )
        metric_card(
            f"{metric}_{stat}_{'high' if highest else 'low'}",
            label or f"{'最高' if highest else '最低'}条件均值 · {METRICS[metric][0]}",
            group,
            metric,
            stat,
        )
        cards[-1]["note"] += " 并列仅列首组；观测极值不是容量或显著性结论。"

    kind = summary["run"]["test_type"]
    axis, numeric, series = profile_series(summary)
    ordered = series[0]["groups"] if numeric and len(series) == 1 else []
    first, last = (ordered[0], ordered[-1]) if ordered else (None, None)

    def endpoints(metric: str, stat: str = "mean") -> None:
        label = METRICS[metric][0]
        for key, title, group in (("first", "最小计划条件", first), ("last", "最大计划条件", last)):
            metric_card(f"{metric}_{key}", f"{title} · {label}", group, metric, stat)

    def ratio(metric: str, label: str, percent: bool = False) -> None:
        a, b = value(first, metric), value(last, metric)
        result = b / a if a is not None and a > 0 and b is not None else None
        if result is not None and not math.isfinite(result):
            result = None
        add(
            metric + "_ratio",
            label,
            show(result * 100 if percent and result is not None else result)
            + (" %" if percent else " ×"),
            f"最大 / 最小计划条件的请求均值；{axis}。端点缺失或基准非正时不推算；不是因果效应。",
        )

    def adjacent(metric: str, decline: bool, threshold: float, absolute: bool = False) -> None:
        candidates = []
        change: float | None
        for before, after in zip(ordered, ordered[1:], strict=False):
            a, b = value(before, metric), value(after, metric)
            if a is not None and a > 0 and b is not None:
                change = (b / a - 1) * 100
                if math.isfinite(change):
                    candidates.append((change, before, after, b - a))
        chosen = (
            (min if decline else max)(candidates, key=lambda item: item[3] if absolute else item[0])
            if candidates
            else None
        )
        change, before, after, difference = chosen if chosen else (None, None, None, None)
        exceeded = change is not None and (-change if decline else change) > threshold
        add(
            metric + "_adjacent",
            f"最大相邻{'下降' if decline else '增长'} · {METRICS[metric][0]}",
            show(change) + " %",
            f"{(before or {}).get('label', '无有效相邻对')} → {(after or {}).get('label', '无有效相邻对')}；请求均值变化，正值增长、负值下降。{'按绝对增量选择，增量 ' + show(difference) + ' ' + METRICS[metric][2] + '。' if absolute else '按相对变化选择。'}初版描述阈值 {threshold:g}%：{'超过' if exceeded else '未超过或无法判断'}；不跨缺测条件，不称为数学拐点。",
        )

    if kind in {"concurrency", "custom_text"}:
        endpoints("ttft")
        endpoints("tps")
        ratio("ttft", "端点 TTFT 倍率")
        extreme("ttft", False)
        extreme("tps", True)
        extreme("phase_output", True, "max", "最高客户端阶段输出吞吐")
        adjacent("ttft", False, 10, absolute=True)
        gains = []
        for before, after in zip(ordered, ordered[1:], strict=False):
            a, b = value(before, "phase_output", "max"), value(after, "phase_output", "max")
            if a is not None and a > 0 and b is not None and math.isfinite(b / a):
                if b / a - 1 < 0.05:
                    gains.append((after["label"], (b / a - 1) * 100))
        add(
            "throughput_gain",
            "相邻吞吐增益低于 5%",
            gains[0][0] if len(ordered) >= 3 and gains else "未发现或条件不足",
            f"{'增益 ' + show(gains[0][1]) + '%' if len(ordered) >= 3 and gains else '至少需要三个计划条件与有效相邻批次观测'}；按各条件阶段输出吞吐观测最大值比较，下降也满足规则。仅为初版描述阈值，不证明饱和容量。",
        )
    elif kind in {"prefill", "long_context"}:
        endpoints("ttft")
        endpoints("prefill_speed")
        ratio("ttft", "端点 TTFT 倍率")
        extreme("prefill_speed", True)
        if kind == "prefill":
            ratio("prefill_speed", "输入处理速度保留比例", True)
            extreme("ttft", False)
            adjacent("prefill_speed", True, 10)
            xs = series[0]["x"] if ordered else []
            a, b = value(first, "ttft"), value(last, "ttft")
            length_ratio = xs[-1] / xs[0] if xs and xs[0] > 0 else None
            ttft_ratio = b / a if a is not None and a > 0 and b is not None else None
            pattern = "无法判断"
            if (
                len(ordered) >= 3
                and length_ratio
                and ttft_ratio is not None
                and math.isfinite(ttft_ratio)
            ):
                pattern = (
                    "超线性规则"
                    if ttft_ratio > length_ratio * 1.5
                    else "近线性规则"
                    if ttft_ratio > length_ratio * 0.8
                    else "次线性规则"
                )
            add(
                "ttft_pattern",
                "TTFT 长度增长对照",
                pattern,
                f"长度倍率 {show(length_ratio)}；TTFT 均值倍率 {show(ttft_ratio)}。初版规则：>长度倍率×1.5 / >×0.8 / 其余；至少三条件，端点完整。描述性阈值，不是回归拟合或算法复杂度。",
            )
        else:
            ratio("tps", "逐请求 TPS 保留比例", True)
            extreme("tps", True)
            adjacent("ttft", False, 20)
            values = [value(group, "tps") for group in ordered]
            usable = [v for v in values if v is not None]
            scale = max(usable) if usable else 0
            normalized = [v / scale for v in usable] if scale else []
            cv = (
                statistics.pstdev(normalized) / statistics.fmean(normalized) * 100
                if len(usable) >= 2 and len(usable) == len(values) and normalized
                else None
            )
            add(
                "tps_cv",
                "不同长度条件的 TPS 变异系数",
                show(cv) + " %",
                f"各条件请求均值等权，总体标准差 / 均值；有效条件 {len(usable)} / {len(values)}，缺测不跳过。初版阈值 >10%：{'超过' if cv is not None and cv > 10 else '未超过或无法判断'}。不是时间稳定性或显著性检验。",
            )
    elif kind == "segmented_prefill":
        endpoints("ttft_zero_cache_api")
        endpoints("prefill_speed", "max")
        extreme("ttft_zero_cache_api", False, label="最低 API 零缓存条件 TTFT 均值")
        extreme("prefill_speed", True, "max", "最高观测输入处理速度")

        def weighted(group: dict[str, Any] | None, source: str = "API") -> float | None:
            raw = (
                (group or {})
                .get("extended_observations", {})
                .get("cache", {})
                .get("weighted", {})
                .get(source, {})
                .get("rate")
            )
            return float(raw) if isinstance(raw, (int, float)) and math.isfinite(raw) else None

        for source in ("API", "TTFT_inferred"):
            totals = (
                summary.get("extended_observations", {})
                .get("cache", {})
                .get("weighted", {})
                .get(source, {})
            )
            add(
                "cache_total_" + source,
                source + " 缓存 token 加权命中比例",
                show(totals.get("rate")) + " %",
                f"合计命中 token / 合计对应输入 token；有效配对 {totals.get('count', 0)} / {overall['successes']} 成功请求；未知不补零。API 与 TTFT 推断分别统计。",
            )
        usable_cache = [g for g in ordered if weighted(g) is not None]
        peak = max(usable_cache, key=lambda g: weighted(g) or 0.0) if usable_cache else None
        add(
            "cache_peak",
            "最高条件 API token 加权命中比例",
            show(weighted(peak)) + " %",
            f"{(peak or {}).get('label', '无有效条件')}；并列只列首组，非请求比例的简单均值。",
        )
        a, b = weighted(first), weighted(last)
        add(
            "cache_change",
            "API 缓存比例端点变化",
            show(b - a if a is not None and b is not None else None) + " pp",
            "token 加权比例差；API 来源。目标长度顺序，不代表执行时间顺序；不是缓存因果收益。",
        )
        rates = [weighted(g) for g in ordered]
        trend = "无法判断"
        if len(rates) >= 3 and all(r is not None for r in rates):
            complete_rates = [r for r in rates if r is not None]
            deltas = [
                after - before
                for before, after in zip(complete_rates, complete_rates[1:], strict=False)
            ]
            trend = (
                "波动规则"
                if any(d < -1 for d in deltas)
                else "增长规则"
                if complete_rates[-1] - complete_rates[0] > 5
                else "稳定规则"
            )
        add(
            "cache_trend",
            "API 加权缓存比例趋势",
            trend,
            "至少三个完整条件；相邻下降超过 1 pp 为波动，否则端点增加超过 5 pp 为增长，其余为稳定。仅描述规则，不代表缓存收益或时间趋势。",
        )
    elif kind == "throughput_matrix":
        dims = [g.get("dimensions", {}) for g in summary["groups"]]
        add(
            "matrix_shape",
            "观测矩阵尺寸",
            f"{len({d.get('concurrency_level') for d in dims})} × {len({d.get('context_length_target') for d in dims})}",
            f"已记录条件 {len(dims)}；轴尺寸乘积不代表每个组合均完成，请核对热力图缺测格。",
        )
        extreme("phase_output", True, "max", "最高客户端阶段输出吞吐")
        extreme("phase_output", False, "max", "最低条件峰值阶段输出吞吐")
        extreme("ttft", False, "min", "最低观测请求 TTFT")
        extreme("ttft", True, "max", "最高观测请求 TTFT")
    else:
        for metric, stat in (
            ("ttft", "mean"),
            ("tps", "mean"),
            ("ttft", "p95"),
            ("prefill_speed", "mean"),
        ):
            metric_card(
                f"overall_{metric}_{stat}",
                f"整体 {METRICS[metric][0]} · {stat}",
                {**overall, "label": "全部正式成功请求"},
                metric,
                stat,
            )
    return cards
