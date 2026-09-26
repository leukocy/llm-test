"""Plotly 图表 JSON 端点：复用 ui/warehouse_charts 的纯构建器。

后端出 fig.to_plotly_json()，前端 plotly.js 直接渲染——图表逻辑零重写。
数据路径与 Streamlit 仓库页同口径（query_runs + project_run）。
"""

from __future__ import annotations

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
