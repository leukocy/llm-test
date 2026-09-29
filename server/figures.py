"""Plotly 图表 JSON 端点：复用 ui/warehouse_charts 的纯构建器。

后端出 fig.to_plotly_json()，前端 plotly.js 直接渲染——图表逻辑零重写。
数据路径与 Streamlit 仓库页同口径（query_runs + project_run）。
"""

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
from server.reports import REPORT_ENVIRONMENT_SCOPES
from server.specs import REPORT_ENVIRONMENT_FIELDS


def performance_report_figure(job: dict[str, Any], summary: dict[str, Any]) -> dict:
    """Export the exact report statistics, retaining missing values and provenance."""
    groups = summary["groups"]
    if not any(group["metrics"]["ttft"]["count"] for group in groups):
        raise ValueError("没有有效的 TTFT 样本，无法导出性能图")
    labels = [escape(group["label"]) for group in groups]
    counts = [group["metrics"]["ttft"]["count"] for group in groups]
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
        "柱顶 n 为有效 TTFT 样本数；分位数采用线性插值。小样本尾部分位数分辨率有限，不代表显著性结论。",
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
    return {
        "figure": {
            "data": [
                {
                    "type": "bar",
                    "name": name,
                    "x": labels,
                    "y": [group["metrics"]["ttft"][stat] for group in groups],
                    "marker": {"color": color},
                    "text": [f"n={count}" if count else "" for count in counts]
                    if stat == "p95"
                    else [],
                    "textposition": "outside",
                    "cliponaxis": False,
                    "hovertemplate": "%{x}<br>%{y:.3f} s<extra>" + name + "</extra>",
                }
                for name, stat, color in [
                    ("TTFT p50", "median", "#2463a6"),
                    ("TTFT p95", "p95", "#188b79"),
                ]
            ],
            "layout": {
                "title": {
                    "text": f"{escape(job['model_id'])} · {escape(job['test_type'])}<br>"
                    "<sup>首字延迟 · 成功且有限、正值的逐请求观测</sup>",
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
                    "title": {"text": "TTFT (s)"},
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
