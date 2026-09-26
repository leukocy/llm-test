"""数据仓库图表构建器（plotly）。

趋势 / 对比 / 分布三类图表的数据准备都是纯函数（不依赖 Streamlit），
便于单元测试；图形构建只做 plotly 装配。
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# 趋势/对比可用的投影数值指标（project_run 键 → 展示名）
TREND_METRICS: dict[str, str] = {
    "decode_tps": "decode TPS",
    "prefill_tps": "prefill TPS",
    "ttft_s": "TTFT (s)",
    "p50_latency_s": "p50 (s)",
    "p95_latency_s": "p95 (s)",
    "p99_latency_s": "p99 (s)",
    "effective_bandwidth_gbps": "等效带宽 (GB/s)",
    "bandwidth_utilization_pct": "带宽利用率 (%)",
    "gpu_vram_peak_gb": "显存峰值 (GB)",
    "system_memory_peak_gb": "内存峰值 (GB)",
    "gpu_util_pct": "GPU 利用率 (%)",
    "cpu_util_pct": "CPU 利用率 (%)",
}

# 趋势分组维度（project_run 键 → 展示名）
TREND_DIMS: dict[str, str] = {
    "model_name": "模型",
    "machine_id": "硬件",
    "engine": "引擎",
    "parallel_strategy": "并行",
    "quantization": "量化",
    "tester": "测试员",
}

# 对比指标分组（分组柱状图一节一组）
COMPARE_METRIC_GROUPS: dict[str, list[str]] = {
    "性能": [
        "decode_tps",
        "prefill_tps",
        "ttft_s",
        "p95_latency_s",
        "effective_bandwidth_gbps",
        "bandwidth_utilization_pct",
    ],
    "资源峰值": ["gpu_vram_peak_gb", "system_memory_peak_gb", "gpu_util_pct", "cpu_util_pct"],
}

_LAYOUT_DEFAULTS = {
    "height": 420,
    "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
    "hovermode": "x unified",
}


# ---------------------------------------------------------------------------
# 趋势图（指标 × 日期，按维度分组）
# ---------------------------------------------------------------------------


def build_trend_data(
    rows: list[dict[str, Any]], metric: str, group_dim: str
) -> dict[str, list[tuple[str, float, str]]]:
    """投影行 → {分组值: [(date, value, test_id), ...]}（按日期升序）。

    缺指标值或日期的行跳过；分组字段为空归入「未标注」。
    """
    groups: dict[str, list[tuple[str, float, str]]] = {}
    for row in rows:
        val = row.get(metric)
        date = row.get("date")
        if val in (None, "") or not date:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        key = str(row.get(group_dim) or "未标注")
        groups.setdefault(key, []).append((str(date), num, str(row.get("test_id") or "")))
    for key in groups:
        groups[key].sort(key=lambda t: t[0])
    return groups


def build_trend_figure(rows: list[dict[str, Any]], metric: str, group_dim: str) -> go.Figure:
    """指标 × 日期分组趋势图（折线 + 标记点，hover 带 test_id）。"""
    data = build_trend_data(rows, metric, group_dim)
    fig = go.Figure()
    for key, points in sorted(data.items()):
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="lines+markers",
                name=key,
                text=[p[2] for p in points],
                hovertemplate="%{x}<br>%{y:.2f}<br>%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"{TREND_METRICS.get(metric, metric)} × 日期（按{TREND_DIMS.get(group_dim, group_dim)}分组）",
        xaxis_title="日期",
        yaxis_title=TREND_METRICS.get(metric, metric),
        legend_title=TREND_DIMS.get(group_dim, group_dim),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 运行对比（指标 × 运行）
# ---------------------------------------------------------------------------


def run_label(row: dict[str, Any], max_len: int = 42) -> str:
    """投影行 → 对比用短标签（日期 · 模型 · 硬件 · test_id 前 8 位）。"""
    tid = str(row.get("test_id") or "")[:8]
    parts = [row.get("date") or "—", row.get("model_name") or "—", row.get("machine_id") or "—"]
    label = " · ".join(str(p) for p in parts) + (f" · {tid}" if tid else "")
    return label[:max_len]


def build_compare_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """投影行 → {指标展示名: {运行标签: 值}}（行=指标，列=运行）。"""
    labels = [run_label(r) for r in rows]
    table: dict[str, dict[str, Any]] = {}
    for group_metrics in COMPARE_METRIC_GROUPS.values():
        for metric in group_metrics:
            name = TREND_METRICS.get(metric, metric)
            table[name] = {label: row.get(metric) for label, row in zip(labels, rows, strict=False)}
    return table


def build_compare_bar_figure(rows: list[dict[str, Any]], group_name: str) -> go.Figure | None:
    """某指标分组的对比柱状图（x=指标，每运行一根柱）。"""
    metrics = COMPARE_METRIC_GROUPS.get(group_name) or []
    labels = [run_label(r) for r in rows]
    fig = go.Figure()
    has_any = False
    for label, row in zip(labels, rows, strict=False):
        values = []
        for metric in metrics:
            val = row.get(metric)
            try:
                values.append(float(val) if val not in (None, "") else None)
            except (TypeError, ValueError):
                values.append(None)
        if any(v is not None for v in values):
            has_any = True
        fig.add_trace(
            go.Bar(
                x=[TREND_METRICS.get(m, m) for m in metrics],
                y=values,
                name=label,
                hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
            )
        )
    if not has_any:
        return None
    fig.update_layout(
        title=f"{group_name}对比",
        barmode="group",
        legend_title="运行",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 分布图（单次运行的请求级指标分布）
# ---------------------------------------------------------------------------


def collect_distribution_values(results: list[Any], field: str) -> dict[str, list[float]]:
    """TestResult 列表 → {并发组: [值...]}（跳过空值与错误请求）。

    field: ttft / tps / tpot / prefill_speed 等 TestResult 数值属性。
    """
    groups: dict[str, list[float]] = {}
    for res in results:
        if getattr(res, "error", None):
            continue
        val = getattr(res, field, None)
        if val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        conc = getattr(res, "concurrency_level", None)
        key = f"并发 {conc}" if conc is not None else "全部"
        groups.setdefault(key, []).append(num)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def build_histogram_figure(values: list[float], title: str, x_title: str) -> go.Figure | None:
    """单指标直方图。"""
    if not values:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=min(40, max(10, len(values) // 5)),
            hovertemplate=f"{x_title}: %{{x}}<br>次数: %{{y}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="请求数",
        showlegend=False,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def build_box_figure(groups: dict[str, list[float]], title: str, y_title: str) -> go.Figure | None:
    """按并发分组的箱线图。"""
    if not groups:
        return None
    fig = go.Figure()
    for key, values in groups.items():
        if values:
            fig.add_trace(go.Box(y=values, name=key, boxmean=True))
    fig.update_layout(title=title, yaxis_title=y_title, showlegend=False, **_LAYOUT_DEFAULTS)
    return fig


# ---------------------------------------------------------------------------
# 引擎指标时间线（engine_metrics timeline）
# ---------------------------------------------------------------------------


def build_engine_timeline_figure(timeline: list[dict[str, Any]]) -> go.Figure | None:
    """引擎 /metrics 轮询时间线：KV 占用率 + 运行/等待请求数（双轴）。

    timeline 元素形如 EngineMetricsPoller._downsample 的输出：
    {t, gpu_cache_usage_perc, num_requests_running, num_requests_waiting, ...}。
    """
    if not timeline:
        return None
    xs = [point.get("t") for point in timeline]
    kv = [point.get("gpu_cache_usage_perc") for point in timeline]
    running = [point.get("num_requests_running") for point in timeline]
    waiting = [point.get("num_requests_waiting") for point in timeline]
    if not any(v is not None for v in kv + running + waiting):
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[(v * 100 if v is not None else None) for v in kv],
            name="KV 占用率 (%)",
            mode="lines",
            line={"color": "#e74c3c"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=running,
            name="运行中请求",
            mode="lines",
            line={"color": "#2ecc71"},
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=waiting,
            name="等待中请求",
            mode="lines",
            line={"color": "#f39c12", "dash": "dash"},
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="引擎运行时时间线",
        xaxis_title="时间 (s)",
        yaxis={"title": "KV 占用率 (%)", "range": [0, 100]},
        yaxis2={"title": "请求数", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": 1.12},
        **_LAYOUT_DEFAULTS,
    )
    return fig
