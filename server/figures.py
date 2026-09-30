"""Plotly figure JSON for full performance reports and shared warehouse charts."""

from __future__ import annotations

from html import escape
from typing import Any

from core.warehouse.charts import (
    COMPARE_METRIC_GROUPS,
    build_box_figure,
    build_compare_bar_figure,
    build_engine_timeline_figure,
    build_histogram_figure,
    build_trend_figure,
    collect_distribution_values,
)
from server.extended_analytics import SYSTEM_METRICS
from server.reports import REPORT_ENVIRONMENT_SCOPES
from server.scenario_reports import (
    METRICS,
    STATISTICS,
    profile_series,
    sampling_unit,
    scenario_analysis,
)
from server.specs import REPORT_ENVIRONMENT_FIELDS
from server.time_series import REQUEST_METRICS


def performance_report_figure(
    job: dict[str, Any],
    summary: dict[str, Any],
    metric: str = "ttft",
    *,
    view: str = "comparison",
    statistic: str = "median",
    attempt: int | None = None,
) -> dict:
    """Export the exact report statistics, retaining missing values and provenance."""
    groups = summary["groups"]
    if (
        metric not in METRICS
        or view not in {"comparison", "profile", "heatmap", "timeline"}
        or statistic not in STATISTICS
    ):
        raise ValueError("不支持的指标、图形或统计量")
    title, abbreviation, unit = METRICS[metric]
    if view != "timeline" and not any(
        group["metrics"].get(metric, {}).get("count") for group in groups
    ):
        raise ValueError(f"没有有效的 {abbreviation} 样本，无法导出性能图")
    labels = [escape(group["label"]) for group in groups]
    counts = [group["metrics"].get(metric, {}).get("count", 0) for group in groups]
    overall = summary["overall"]
    integrity = summary["integrity"]
    protocol = summary.get("measurement_protocol") or {}
    control = summary.get("execution_control") or {}
    status = "完整性通过" if integrity["verified"] else "仅供诊断 · 完整性未通过"
    footer = [
        f"{status} · 状态 {escape(job['status'])} · 指标契约 {escape(summary['metric_contract_version'])}",
        f"正式请求 {overall['requests']} · 失败 {overall['failures']} · "
        f"预热记录 {protocol.get('warmup_recorded', 0)}（不计入统计） · "
        f"暂停 {control.get('pause_count', 0)} 次 · 批次并行上限 {control.get('max_parallel', 1)}",
        f"柱顶 n 为有效 {abbreviation} 样本数（{sampling_unit(metric)}）；分位数采用线性插值。小样本尾部分位数分辨率有限，不代表显著性结论。",
        f"Token 来源：{escape(', '.join(summary['provenance']['token_sources']) or '未记录')} · "
        f"算法：{escape(', '.join(summary['provenance']['token_methods']) or '未记录')}",
    ]
    environment = summary.get("report_environment")
    if environment:
        scope = REPORT_ENVIRONMENT_SCOPES.get(environment["scope"], "未明确对象")
        footer.append(f"环境对象：{scope} · 用户填写，未经自动核验；自动硬件快照来自执行端。")
        values = [
            f"{label}: {escape(environment['fields'][key][:60])}"
            + ("…" if len(environment["fields"][key]) > 60 else "")
            for key, label in REPORT_ENVIRONMENT_FIELDS.items()
            if environment["fields"].get(key)
        ]
        footer.extend(" · ".join(values[index : index + 2]) for index in range(0, len(values), 2))
    footer.append(f"作业 {escape(job['job_id'])} · 完整条件和环境信息见 HTML / JSON 报告")
    origin = summary.get("origin") or {}
    if metric in SYSTEM_METRICS:
        system = summary.get("extended_observations", {}).get("system", {})
        footer.append(
            f"batch-wall-v1 · 完整批次 {system.get('valid_batches', 0)} · 排除不完整/冲突批次 {system.get('invalid_batches', 0)} · 未标记请求 {system.get('untagged_requests', 0)}；窗口包含失败等待与客户端开销。输入包含缓存 token。"
        )
    elif metric.startswith("cache_") or metric.startswith("ttft_"):
        footer.append(
            "API 与 TTFT 推断分开；缺失缓存字段不当作零命中。API 比例分母使用 API 输入 token；不是缓存因果收益。"
        )
    if origin.get("kind") == "saved_csv":
        footer[1] = (
            f"CSV 请求 {overall['requests']} · 失败 {overall['failures']} · 未知状态 {overall.get('unknown_outcomes', 0)}；原运行的预热、暂停和计划请求数未经核验。"
        )
        footer.append(
            f"历史 CSV：{escape(origin['filename'][:120])} · 已知状态样本；原执行条件未经核验。"
        )
        footer.append(f"源 CSV SHA-256：{origin['csv_sha256']}")
    result = {
        "figure": {
            "data": [
                {
                    "type": "bar",
                    "name": name,
                    "x": labels,
                    "y": [group["metrics"].get(metric, {}).get(stat) for group in groups],
                    "marker": {"color": color},
                    "text": [f"n={count}" if count else "" for count in counts]
                    if stat == "p95"
                    else [],
                    "textposition": "outside",
                    "cliponaxis": False,
                    "hovertemplate": "%{x}<br>%{y:.3f} " + unit + "<extra>" + name + "</extra>",
                }
                for name, stat, color in [
                    (f"{abbreviation} p50", "median", "#2463a6"),
                    (f"{abbreviation} p95", "p95", "#188b79"),
                ]
            ],
            "layout": {
                "title": {
                    "text": f"{escape(job['model_id'])} · {escape(job['test_type'])}<br>"
                    f"<sup>{title} · 样本单位：{sampling_unit(metric)}</sup>",
                    "x": 0.06,
                    "xanchor": "left",
                },
                "width": min(2400, max(1500, len(groups) * 80)),
                "height": 1050,
                "font": {
                    "family": "Noto Sans SC, Arial, sans-serif",
                    "size": 15,
                    "color": "#17243b",
                },
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "barmode": "group",
                "margin": {"l": 110, "r": 50, "t": 130, "b": 300},
                "xaxis": {
                    "title": {"text": escape(summary["group_axis"])},
                    "type": "category",
                    "automargin": True,
                },
                "yaxis": {
                    "title": {"text": f"{abbreviation} ({unit})"},
                    "rangemode": "tozero",
                    "gridcolor": "#e3eaf2",
                },
                "legend": {"orientation": "h", "x": 1, "xanchor": "right", "y": 1.12},
                "annotations": [
                    {
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0,
                        "y": -0.22,
                        "xanchor": "left",
                        "yanchor": "top",
                        "align": "left",
                        "showarrow": False,
                        "text": "<br>".join(footer),
                        "font": {"size": 13, "color": "#5c6e83"},
                    }
                ],
            },
        }
    }
    if view == "timeline":
        _timeline_view(result["figure"], summary, metric, statistic, attempt=attempt)
    elif view != "comparison":
        _scenario_view(result["figure"], summary, metric, view, statistic)
    result["analysis"] = scenario_analysis(summary)
    return result


def _timeline_view(
    figure: dict, summary: dict, metric: str, statistic: str, *, attempt: int | None = None
) -> None:
    timeline = summary.get("time_series")
    if timeline and "segments" in timeline:
        segments = timeline["segments"]
        selected = [s for s in segments if s["attempt"] == attempt] if attempt else segments[-1:]
        if not selected:
            raise ValueError("所选执行尝试没有可核验的计时窗口")
        timeline = selected[0]
    elif attempt is not None:
        raise ValueError("此测量没有按执行尝试保存的计时窗口")
    if summary["run"]["test_type"] != "stability" or metric not in REQUEST_METRICS:
        raise ValueError("时间序列仅支持稳定性测试的五种逐请求指标")
    if not timeline or not timeline["bins"]:
        raise ValueError("缺少可核验的单调时钟记录，无法绘制时间序列")
    bins = timeline["bins"]
    _, label, unit = METRICS[metric]
    x = [(b["start_seconds"] + b["end_seconds"]) / 2 for b in bins]
    custom = [
        [
            b["start_seconds"],
            b["end_seconds"],
            b["metrics"][metric]["count"],
            b["requests"],
            b["failures"],
        ]
        for b in bins
    ]
    figure["data"] = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": f"{label} {STATISTICS[statistic]}",
            "x": x,
            "y": [b["metrics"][metric][statistic] for b in bins],
            "connectgaps": False,
            "customdata": custom,
            "line": {"color": "#2463a6"},
            "hovertemplate": "%{customdata[0]:.3f}–%{customdata[1]:.3f} s<br>%{y:.4g} "
            + unit
            + "<br>有效 n=%{customdata[2]} · 完成=%{customdata[3]} · 失败=%{customdata[4]}<extra></extra>",
        },
        {
            "type": "bar",
            "name": "完成请求",
            "x": x,
            "y": [b["requests"] for b in bins],
            "yaxis": "y2",
            "marker": {"color": "#97b8dc"},
            "customdata": custom,
            "hovertemplate": "%{customdata[0]:.3f}–%{customdata[1]:.3f} s<br>完成=%{y}<extra></extra>",
        },
        {
            "type": "bar",
            "name": "失败请求",
            "x": x,
            "y": [b["failures"] for b in bins],
            "yaxis": "y2",
            "marker": {"color": "#cf675c"},
            "customdata": custom,
            "hovertemplate": "%{customdata[0]:.3f}–%{customdata[1]:.3f} s<br>失败=%{y}<extra></extra>",
        },
    ]
    layout = figure["layout"]
    layout["height"] = 1250
    layout["margin"]["b"] = 450
    layout["annotations"][0]["y"] = -0.10
    layout["title"]["text"] += (
        f"<br><sup>稳定性完成时间序列 · {STATISTICS[statistic]} · stability-clock-v1</sup>"
    )
    if timeline.get("attempt"):
        layout["title"]["text"] += f" · 执行尝试 #{timeline['attempt']}（独立时钟窗口）"
    layout["xaxis"] = {
        "title": {"text": "距调度开始的时间（秒）· 时间窗中点"},
        "type": "linear",
        "range": [0, timeline["window_seconds"]],
    }
    layout["yaxis"]["domain"] = [0.38, 1]
    layout["yaxis2"] = {
        "title": {"text": "完成 / 失败请求数"},
        "domain": [0, 0.22],
        "anchor": "x",
        "rangemode": "tozero",
        "tickformat": ",d",
        "nticks": 5,
    }
    if max((b["requests"] for b in bins), default=0) <= 4:
        layout["yaxis2"]["dtick"] = 1
    admission_budget = timeline.get("admission_budget_seconds")
    if admission_budget is None:
        admission_budget = timeline["planned_seconds"]
    layout["annotations"][0]["text"] = "<br>".join(
        [
            layout["annotations"][0]["text"].replace("柱顶 n 为", "悬停 n 为"),
            f"计时记录完整：{'是' if timeline['complete'] else '否'} · 有效计时 {timeline['timed_requests']} · 缺失 {timeline['missing_requests']} · 无效 {timeline['invalid_requests']}",
            f"窗口状态：{timeline.get('window_state') or '历史记录'}；运行中的时间窗仅供诊断，尚未最终确认。",
            f"本次发起预算 {admission_budget:.3f} 秒 · 调度至排空 {timeline['window_seconds']:.3f} 秒 · 每窗 {timeline['bin_seconds']} 秒",
            *timeline["notes"],
            f"作业 {escape(summary['run']['test_id'])} · 完整来源见 HTML / JSON 报告",
        ]
    )


def _scenario_view(figure: dict, summary: dict, metric: str, view: str, statistic: str) -> None:
    axis, numeric, series = profile_series(summary)
    _, abbreviation, unit = METRICS[metric]
    stat = STATISTICS[statistic]
    layout = figure["layout"]
    layout["title"]["text"] = (
        layout["title"]["text"].split("<br>")[0] + f"<br>{abbreviation} {stat} ({unit})"
    )
    layout["annotations"][0]["text"] = layout["annotations"][0]["text"].replace(
        "柱顶 n 为", "悬停 n 为"
    )
    layout["xaxis"] = {
        "title": {"text": axis},
        "type": "linear" if numeric else "category",
        "automargin": True,
    }
    layout["height"] = 950
    layout["width"] = 1500
    if not numeric:
        layout["annotations"][0]["text"] += "<br>结构化条件未记录，横轴使用分类标签。"
    if view == "heatmap":
        if summary["run"]["test_type"] != "throughput_matrix" or not numeric:
            raise ValueError("热力图需要并发 × 上下文矩阵的完整结构化条件")
        layout["xaxis"]["type"] = "category"
        layout["yaxis"] = {"title": {"text": "并发数"}, "type": "category", "automargin": True}
        layout["annotations"][0]["text"] += (
            "<br>色标为 " + abbreviation + " " + stat + "；空白为无有效样本或缺测组合。"
        )
        figure["data"] = [
            {
                "type": "heatmap",
                "x": [str(x) for x in series[0]["x"]],
                "y": [s["name"].removeprefix("并发 ") for s in series],
                "z": [
                    [
                        g["metrics"].get(metric, {}).get(statistic) if g else None
                        for g in s["groups"]
                    ]
                    for s in series
                ],
                "customdata": [[_hover_data(g, metric) for g in s["groups"]] for s in series],
                "colorscale": "Cividis",
                "hoverongaps": False,
                "colorbar": {"title": {"text": f"{abbreviation} ({unit})"}},
                "hovertemplate": "上下文 %{x} token · 并发 %{y}<br>%{z:.4g} "
                + unit
                + "<br>n=%{customdata[0]} · 请求=%{customdata[1]} · 失败=%{customdata[2]}<extra></extra>",
            }
        ]
        return
    figure["data"] = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": escape(s["name"]),
            "x": s["x"] if numeric else [escape(str(x)) for x in s["x"]],
            "y": [g["metrics"].get(metric, {}).get(statistic) if g else None for g in s["groups"]],
            "connectgaps": False,
            "customdata": [_hover_data(g, metric) for g in s["groups"]],
            "hovertemplate": "%{x}<br>%{y:.4g} "
            + unit
            + "<br>n=%{customdata[0]} · 请求=%{customdata[1]} · 失败=%{customdata[2]}<extra>%{fullData.name}</extra>",
        }
        for s in series
    ]


def _hover_data(group: dict | None, metric: str) -> list[int | None]:
    return (
        [group["metrics"].get(metric, {}).get("count", 0), group["requests"], group["failures"]]
        if group
        else [None, None, None]
    )


def _project_rows(db_manager, selection, limit: int = 2000) -> list[dict[str, Any]]:
    from core.warehouse import WarehouseFilter, project_run, query_runs

    flt = WarehouseFilter(
        machine_id=selection.machine_id,
        model_id=selection.model_id,
        test_type=selection.test_type,
        status_detail=selection.status,
        external_level=selection.external_level,
        search=selection.search,
        limit=limit,
    )
    runs = query_runs(db_manager, flt)
    return [project_run(r) for r in runs]


def trend_figure(
    db_manager, selection, metric: str, group_dim: str, publishable_only: bool
) -> dict:
    """趋势图：指标 × 日期 × 分组维度。"""
    from core.warehouse.charts import TREND_DIMS, TREND_METRICS

    if metric not in TREND_METRICS or group_dim not in TREND_DIMS:
        raise ValueError(f"未知指标或维度: {metric} / {group_dim}")
    rows = _project_rows(db_manager, selection)
    if publishable_only:
        rows = [r for r in rows if (r.get("external_level") or "internal") == "publishable"]
    fig = build_trend_figure(rows, metric, group_dim)
    return {"figure": fig.to_plotly_json(), "count": len(rows)}


def compare_figures(db_manager, test_ids: list[str]) -> dict:
    """运行对比：表 + 分组柱状图（按 COMPARE_METRIC_GROUPS 一节一组）。"""
    rows = []
    missing = []
    for test_id in test_ids:
        row = db_manager.db.fetch_one("SELECT * FROM test_runs WHERE test_id = ?", (test_id,))
        if row is None:
            missing.append(test_id)
            continue
        from core.models import TestRun
        from core.warehouse import project_run

        rows.append(project_run(TestRun.from_row(row)))

    table: dict[str, dict[str, Any]] = {}
    figures: dict[str, Any] = {}
    if rows:
        from core.warehouse.charts import build_compare_table

        table = build_compare_table(rows)
        for group_name in COMPARE_METRIC_GROUPS:
            fig = build_compare_bar_figure(rows, group_name)
            if fig:
                figures[group_name] = fig.to_plotly_json()
    return {"table": table, "figures": figures, "missing_test_ids": missing}


def run_detail_figures(db_manager, test_id: str) -> dict:
    """单运行：分布图（TTFT/TPS/TPOT 直方图+按并发箱线） + 引擎时间线。"""
    row = db_manager.db.fetch_one("SELECT id FROM test_runs WHERE test_id = ?", (test_id,))
    if row is None:
        return {"found": False}
    run_id = row["id"]

    results = db_manager.results.find_by_run_id(run_id, limit=5000)
    distributions: dict[str, Any] = {}
    for field, title, x_title in [
        ("ttft", "TTFT 分布", "TTFT (s)"),
        ("tps", "TPS 分布", "TPS (tok/s)"),
        ("tpot", "TPOT 分布", "TPOT (s)"),
    ]:
        groups = collect_distribution_values(results, field)
        all_values = [v for vals in groups.values() for v in vals]
        hist = build_histogram_figure(all_values, title, x_title)
        box = build_box_figure(groups, f"{title}（按并发）", x_title)
        distributions[field] = {
            "histogram": hist.to_plotly_json() if hist else None,
            "box": box.to_plotly_json() if box else None,
        }

    engine: dict[str, Any] = {"figure": None, "summary": {}}
    run_row = db_manager.db.fetch_one("SELECT * FROM test_runs WHERE id = ?", (run_id,))
    if run_row:
        import json

        raw = run_row.get("engine_metrics_json") or ""
        summary = {}
        if raw:
            try:
                summary = json.loads(raw)
            except (ValueError, TypeError):
                summary = {}
        if summary.get("sample_count"):
            fig = build_engine_timeline_figure(summary.get("timeline") or [])
            engine = {
                "figure": fig.to_plotly_json() if fig else None,
                "summary": {
                    "engine_family": summary.get("engine_family"),
                    "sample_count": summary.get("sample_count"),
                    "preemption_total": summary.get("preemption_total"),
                    "cache_config": summary.get("cache_config") or {},
                },
            }

    return {
        "found": True,
        "run_id": run_id,
        "result_count": len(results),
        "distributions": distributions,
        "engine": engine,
    }
