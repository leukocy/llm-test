"""
数据仓库浏览页（Warehouse Browser）。

手册核心论点：「报告是切片，仓库是全集；能在 10 分钟内从仓库抽出客户可用材料，
才算体系成立。」本页把已采集的八维 TestRun 数据变成可筛选、可追溯、可对外口径的
全集视图：

1. 跨八维筛选（硬件/模型/引擎/可对外等级/状态/类型/测试员/搜索）
2. 运行历史表（仓库列：machine_id/engine/external_level/bottleneck/等效带宽/decode_tps）
3. 硬件 × 模型透视矩阵（手册"不同硬件下的表现"）
4. 硬件盘点（每台机器一行）
5. 三套字段模板导出（CSV / JSON / ZIP）——手册 #templates 的口径
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.database import db_manager
from core.warehouse import (
    TEMPLATE_TITLES,
    WarehouseFilter,
    build_capability_markdown,
    build_capability_sheet,
    build_cross_matrix,
    build_hardware_inventory_rows,
    build_hm_test_rows,
    build_scaling_efficiency,
    count_runs,
    distinct_run_field_values,
    distinct_values,
    export_all_templates_zip,
    export_template_csv,
    export_template_json,
    interpret_efficiency,
    project_run,
    query_runs,
)
from ui.warehouse_charts import (
    COMPARE_METRIC_GROUPS,
    TREND_DIMS,
    TREND_METRICS,
    build_box_figure,
    build_compare_bar_figure,
    build_compare_table,
    build_engine_timeline_figure,
    build_histogram_figure,
    build_trend_figure,
    collect_distribution_values,
)

# 历史表展示的仓库列（顺序即展示顺序）
_HISTORY_COLUMNS = [
    "date",
    "machine_id",
    "model_name",
    "engine",
    "parallel_strategy",
    "concurrency",
    "decode_tps",
    "ttft_s",
    "effective_bandwidth_gbps",
    "bandwidth_utilization_pct",
    "gpu_vram_peak_gb",
    "bottleneck",
    "status",
    "external_level",
    "tester",
]


def render_warehouse_browser() -> None:
    """渲染数据仓库浏览页主入口。"""
    st.title("数据仓库")
    st.caption(
        "报告是切片，仓库是全集 —— 这里筛选、透视、导出历史测试的全集行。"
        "（口径：test-standard/端侧AI硬件与模型.html #templates）"
    )

    db = db_manager
    flt = _build_filter(db)
    try:
        runs = query_runs(db, flt)
        total = count_runs(db, flt)
    except Exception as e:
        # A corrupted DB / legacy row shape must not white-screen the page
        st.error(f"数据仓库查询失败：{e}")
        return

    _render_kpis(runs, total)
    st.markdown("---")

    if not runs:
        st.info("当前筛选条件下无记录。松开筛选条件，或先跑一次测试再回来。")
        return

    (
        tab_history,
        tab_trend,
        tab_matrix,
        tab_scaling,
        tab_inventory,
        tab_cases,
        tab_capability,
        tab_export,
        tab_admin,
    ) = st.tabs(
        [
            "运行历史",
            "趋势对比",
            "透视矩阵",
            "扩展效率",
            "硬件盘点",
            "应用用例",
            "客户能力表",
            "模板导出",
            "数据管理",
        ]
    )

    with tab_history:
        _render_history(runs, total)
    with tab_trend:
        _render_trend_and_compare(runs)
    with tab_matrix:
        _render_cross_matrix(runs)
    with tab_scaling:
        _render_scaling_efficiency(runs)
    with tab_inventory:
        _render_inventory(runs)
    with tab_cases:
        from ui.application_case_form import render_application_case_manager

        render_application_case_manager()
    with tab_capability:
        _render_capability_sheet()
    with tab_export:
        _render_export(runs)
    with tab_admin:
        from ui.warehouse_admin import render_warehouse_admin

        render_warehouse_admin()


# ---------------------------------------------------------------------------
# 筛选栏
# ---------------------------------------------------------------------------


def _build_filter(db) -> WarehouseFilter:
    st.sidebar.markdown("---")
    st.sidebar.subheader("仓库筛选")

    def _opts(field: str) -> list:
        return ["全部"] + distinct_values(db, field)

    machine = st.sidebar.selectbox("硬件 machine_id", _opts("machine_id"), key="wh_machine")
    model = st.sidebar.selectbox("模型", _opts("model_name"), key="wh_model")
    engine = st.sidebar.selectbox("引擎", _opts("engine"), key="wh_engine")
    level = st.sidebar.selectbox(
        "可对外等级",
        ["全部", "internal", "review", "publishable"],
        key="wh_level",
    )
    status = st.sidebar.selectbox("状态", _opts("status"), key="wh_status")
    tester = st.sidebar.selectbox("测试员", _opts("tester"), key="wh_tester")
    test_type = st.sidebar.selectbox(
        "测试类型",
        ["全部"] + distinct_run_field_values(db, "test_type"),
        key="wh_test_type",
    )
    cfg_hash = st.sidebar.selectbox(
        "config_hash（同配置）",
        _opts("config_hash"),
        key="wh_cfg_hash",
        help="CASE 02：同配置才能承诺。按配置指纹过滤同模型/引擎/并行/量化的一组测试。",
    )
    cmp_group = st.sidebar.selectbox(
        "对比组",
        ["全部"] + distinct_run_field_values(db, "comparison_group"),
        key="wh_cmp_group",
    )
    date_range = st.sidebar.date_input(
        "日期范围",
        value=[],
        key="wh_date_range",
        help="按测试创建日期过滤（留空 = 不限；选一个日期 = 从该日起）",
    )
    search = st.sidebar.text_input("模糊搜索", placeholder="备注 / 模型 / 测试员…", key="wh_search")

    with st.sidebar.expander("更多选项"):
        include_superseded = st.checkbox(
            "显示被复测取代的旧版",
            value=False,
            key="wh_include_superseded",
            help="默认复测链只留最新一版；勾选后连被取代的旧记录一并显示。",
        )
        limit = st.selectbox(
            "候选上限",
            [200, 500, 1000, 2000, 5000],
            index=1,
            key="wh_limit",
            help="从最近 N 条记录中过滤。调大可看更老的数据，调小加载更快。",
        )

    def _pick(v):
        return None if v in (None, "全部", "") else v

    # date_input(value=[]) 返回 0~2 个 date 的序列：1 个 = 仅起始日，2 个 = 完整区间
    date_from = date_to = None
    if isinstance(date_range, (tuple, list)):
        if len(date_range) >= 1:
            date_from = datetime.combine(date_range[0], datetime.min.time())
        if len(date_range) == 2:
            date_to = datetime.combine(date_range[1], datetime.max.time())

    return WarehouseFilter(
        machine_id=_pick(machine),
        model_id=_pick(model),
        engine=_pick(engine),
        external_level=_pick(level),
        status_detail=_pick(status),
        test_type=_pick(test_type),
        tester=_pick(tester),
        comparison_group=_pick(cmp_group),
        config_hash=_pick(cfg_hash),
        date_from=date_from,
        date_to=date_to,
        search=_pick(search),
        include_superseded=include_superseded,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------


def _render_kpis(runs, total: int) -> None:
    machines = {project_run(r)["machine_id"] for r in runs if project_run(r)["machine_id"]}
    models = {r.model_id for r in runs if r.model_id}
    publishable = sum(1 for r in runs if (r.external_level or "internal") == "publishable")
    completed = sum(1 for r in runs if r.status == "completed")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "记录数",
        f"{len(runs)} / {total}",
        help="分子 = 当前候选上限内显示的记录；分母 = 筛选条件命中的全部记录。",
    )
    c2.metric("硬件数", len(machines))
    c3.metric("模型数", len(models))
    c4.metric("已完成", completed)
    c5.metric("可对外", publishable)


# ---------------------------------------------------------------------------
# 运行历史
# ---------------------------------------------------------------------------


def _render_history(runs, total: int) -> None:
    st.subheader("运行历史（仓库列）")
    _show_flash()

    # 列可见性：顺序以 _HISTORY_COLUMNS 为准，选择持久在 session_state
    picked = st.multiselect(
        "展示列",
        _HISTORY_COLUMNS,
        default=st.session_state.get("wh_hist_cols") or _HISTORY_COLUMNS,
        key="wh_hist_cols",
        format_func=lambda k: _COLUMN_LABELS.get(k, k),
    )
    visible = [c for c in _HISTORY_COLUMNS if c in picked] or _HISTORY_COLUMNS

    rows = [project_run(r) for r in runs]
    df = pd.DataFrame(rows)[visible].rename(columns=_COLUMN_LABELS)

    # 行多选（streamlit ≥1.35）：勾选后启用下方批量操作
    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        selection_mode="multi-row",
        on_select="rerun",
        key="wh_hist_table",
    )
    sel = []
    if event:
        selection = getattr(event, "selection", None)  # stubs 未收录该属性, 运行时为官方 API
        if selection:
            sel = sorted(selection.rows)
    st.caption(f"显示 {len(rows)} / 共 {total} 条；复测链默认只留最新一版。勾选用批量操作。")

    if sel:
        _render_bulk_actions([runs[i] for i in sel if i < len(runs)])

    # 明细抽屉：选一条看全集（概览 / 请求明细 / 分布图 / 引擎指标 / 复核编辑）
    st.markdown("#### 查看单条全集")
    options = [
        f"{i}: {project_run(r)['date']} · {r.model_id} · {project_run(r)['machine_id'] or '—'}"
        for i, r in enumerate(runs)
    ]
    if options:
        choice = st.selectbox("选择记录", options, key="wh_detail_pick")
        # The selectbox can hold a stale index after filters shrink the run list
        idx = min(int(choice.split(":", 1)[0]), len(runs) - 1)
        _render_detail_drawer(runs[idx])


# ---------------------------------------------------------------------------
# 批量操作
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 操作反馈（闪存模式：消息入 session_state，rerun 后在历史页顶部稳定展示；
# 直接 st.success 后接 st.rerun 会让反馈在渲染前被整页重跑抹掉）
# ---------------------------------------------------------------------------


def _flash(kind: str, text: str) -> None:
    st.session_state["wh_flash"] = (kind, text)


def _show_flash() -> None:
    msg = st.session_state.pop("wh_flash", None)
    if msg:
        kind, text = msg
        getattr(st, kind, st.info)(text)


def _render_bulk_actions(selected_runs) -> None:
    """历史页勾选后的批量操作条：设置等级 / 追加标签 / 删除。"""
    ids = [r.id for r in selected_runs if r.id is not None]
    if not ids:
        return
    st.markdown(f"#### 批量操作（已选 {len(ids)} 条）")
    ac1, ac2, ac3 = st.columns(3)

    with ac1:
        level = st.selectbox(
            "设置外发等级",
            ["", "internal", "review", "publishable"],
            key="wh_bulk_level",
        )
        if st.button("应用等级", key="wh_bulk_level_btn", disabled=not level):
            ok = sum(
                1
                for rid in ids
                if db_manager.update_publish_metadata(rid, {"external_level": level})
            )
            _flash("success", f"已把 {ok} 条设为 {level}")
            st.rerun()

    with ac2:
        tag = st.text_input("追加标签", key="wh_bulk_tag", placeholder="如 baseline")
        if st.button("追加标签", key="wh_bulk_tag_btn", disabled=not tag.strip()):
            ok = 0
            for r in selected_runs:
                if r.id is None:
                    continue
                merged = f"{r.tags},{tag.strip()}" if r.tags else tag.strip()
                ok += 1 if db_manager.update_publish_metadata(r.id, {"tags": merged}) else 0
            _flash("success", f"已给 {ok} 条追加标签 {tag.strip()!r}")
            st.rerun()

    with ac3:
        _render_bulk_delete(ids)


def _render_bulk_delete(ids: list[int]) -> None:
    """删除所选：勾选确认 + 大批量（>10）需输入 DELETE。"""
    n = len(ids)
    confirm = st.checkbox(f"确认删除 {n} 条（不可恢复）", key="wh_bulk_del_confirm")
    phrase_ok = True
    if n > 10:
        phrase = st.text_input("输入 DELETE 确认", key="wh_bulk_del_phrase")
        phrase_ok = phrase.strip() == "DELETE"
    if st.button(
        "删除所选",
        key="wh_bulk_del_btn",
        disabled=not (confirm and phrase_ok),
        type="primary",
    ):
        result = db_manager.delete_runs(ids)
        if result["failed"]:
            _flash("error", f"删除完成：成功 {len(result['deleted'])} 条，失败 {result['failed']}")
        else:
            _flash("success", f"已删除 {len(result['deleted'])} 条")
        st.session_state.pop("wh_hist_table", None)  # 清空选择态
        st.rerun()


# ---------------------------------------------------------------------------
# 单条全集（详情抽屉 + 子标签）
# ---------------------------------------------------------------------------


def _render_detail_drawer(run) -> None:
    with st.expander(f"运行全集 · {run.test_id}", expanded=True):
        tab_overview, tab_results, tab_dist, tab_engine, tab_review = st.tabs(
            ["概览", "请求明细", "分布图", "引擎指标", "复核与编辑"]
        )
        with tab_overview:
            _render_detail_overview(project_run(run))
        with tab_results:
            _render_run_results_table(run)
        with tab_dist:
            _render_run_distributions(run)
        with tab_engine:
            _render_run_engine_metrics(run)
        with tab_review:
            _render_metadata_editor(run)
            st.markdown("---")
            _render_gate_review(run)


def _render_detail_overview(row: dict) -> None:
    """八维全集分组展示（原 detail drawer 的扁平字段区）。"""
    groups = [
        ("标识", ["test_id", "date", "tester", "machine_id", "external_level"]),
        (
            "模型 / 服务",
            [
                "model_name",
                "model_version",
                "model_type",
                "total_params",
                "active_params",
                "quantization",
                "dtype",
                "max_context",
                "engine",
                "engine_version",
                "parallel_strategy",
                "engine_params",
            ],
        ),
        (
            "性能",
            [
                "concurrency",
                "decode_tps",
                "prefill_tps",
                "ttft_s",
                "p50_latency_s",
                "p95_latency_s",
                "p99_latency_s",
                "effective_bandwidth_gbps",
                "bandwidth_utilization_pct",
            ],
        ),
        (
            "资源峰值",
            [
                "gpu_vram_peak_gb",
                "system_memory_peak_gb",
                "gpu_util_pct",
                "cpu_util_pct",
                "power_w",
                "temp_c",
            ],
        ),
        (
            "硬件指纹",
            [
                "cpu_model",
                "cpu_sockets",
                "memory_type",
                "memory_capacity_gb",
                "gpu_model",
                "gpu_count",
                "gpu_vram_gb",
                "gpu_bandwidth_gbps",
                "cuda_or_rocm",
                "driver",
            ],
        ),
        (
            "归因 / 下一步",
            [
                "status",
                "bottleneck",
                "error_type",
                "error_detail",
                "next_action",
                "supersedes_test_id",
                "log_path",
            ],
        ),
    ]
    cols = st.columns(len(groups))
    for col, (title, keys) in zip(cols, groups, strict=False):
        with col:
            st.markdown(f"**{title}**")
            for k in keys:
                v = row.get(k)
                display = "—" if v in (None, "", []) else v
                st.caption(f"{_COLUMN_LABELS.get(k, k)}")
                st.write(display)


def _render_run_results_table(run) -> None:
    """请求明细：test_results 逐请求表。"""
    if run.id is None:
        st.info("该记录无数据库 id。")
        return
    results = db_manager.results.find_by_run_id(run.id, limit=5000)
    if not results:
        st.info("无请求级数据（该运行可能只存了汇总行，或早于请求级入库）。")
        return

    only_errors = st.checkbox("只看错误请求", key=f"wh_res_err_{run.id}")
    rows = [
        {
            "session": res.session_id,
            "并发": res.concurrency_level,
            "轮次": res.round,
            "TTFT(s)": res.ttft,
            "TPS": res.tps,
            "TPOT(s)": res.tpot,
            "prefill速度": res.prefill_speed,
            "输入tok": res.prefill_tokens,
            "输出tok": res.decode_tokens,
            "缓存命中": res.cache_hit_tokens,
            "总时长(s)": res.total_time,
            "错误": res.error,
        }
        for res in results
    ]
    df = pd.DataFrame(rows)
    if only_errors:
        df = df[df["错误"].notna() & (df["错误"] != "")]
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"共 {len(results)} 条请求记录")


def _render_run_distributions(run) -> None:
    """分布图：用请求级真实数据画 TTFT/TPS/TPOT 直方图 + 按并发分组箱线图。"""
    if run.id is None:
        st.info("该记录无数据库 id。")
        return
    results = [r for r in db_manager.results.find_by_run_id(run.id, limit=5000) if not r.error]
    if not results:
        st.info("无成功请求数据，无法绘制分布。")
        return

    specs = [
        ("ttft", "TTFT 分布", "TTFT (s)"),
        ("tps", "TPS 分布", "TPS (tok/s)"),
        ("tpot", "TPOT 分布", "TPOT (s)"),
    ]
    for field, hist_title, x_title in specs:
        groups = collect_distribution_values(results, field)
        all_values = [v for vals in groups.values() for v in vals]
        c1, c2 = st.columns(2)
        with c1:
            fig = build_histogram_figure(all_values, hist_title, x_title)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"wh_hist_{field}_{run.id}")
        with c2:
            fig = build_box_figure(groups, f"{hist_title}（按并发）", x_title)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"wh_box_{field}_{run.id}")


def _render_run_engine_metrics(run) -> None:
    """引擎指标：engine_metrics 汇总 + 时间线（KV 占用 / 运行 / 等待队列）。"""
    summary = run.engine_metrics or {}
    if not summary or not summary.get("sample_count"):
        st.info("无引擎指标（该运行未轮询 /metrics，或早于引擎采集能力）。")
        return

    # 汇总行：引擎族 / KV 容量 / 抢占 / 采样数
    cc = summary.get("cache_config") or {}
    cols = st.columns(4)
    cols[0].metric("引擎", summary.get("engine_family") or "—")
    kv_cap = cc.get("kv_capacity_tokens")
    cols[1].metric("KV 容量", f"{kv_cap:,}" if kv_cap else "—")
    preempt = summary.get("preemption_total")
    cols[2].metric("窗口抢占", preempt if preempt is not None else "—")
    cols[3].metric("采样点数", summary.get("sample_count", 0))

    fig = build_engine_timeline_figure(summary.get("timeline") or [])
    if fig:
        st.plotly_chart(fig, use_container_width=True, key=f"wh_engine_tl_{run.id}")
    else:
        st.info("引擎指标无可用时间线。")


# ---------------------------------------------------------------------------
# 复核与编辑（元数据写回 + 发布门禁复评）
# ---------------------------------------------------------------------------

_EXTERNAL_LEVELS = ["internal", "review", "publishable"]


def _render_metadata_editor(run) -> None:
    """编辑可对外元数据（写回 update_publish_metadata；置空文本即清空字段）。"""
    st.markdown("##### 编辑元数据")
    with st.form(key=f"wh_edit_form_{run.id}"):
        c1, c2, c3 = st.columns(3)
        level_idx = (
            _EXTERNAL_LEVELS.index(run.external_level)
            if run.external_level in _EXTERNAL_LEVELS
            else 0
        )
        level = c1.selectbox("可对外等级", _EXTERNAL_LEVELS, index=level_idx)
        tester = c2.text_input("测试员", value=run.tester or "")
        cmp_group = c3.text_input("对比组", value=run.comparison_group or "")
        bottleneck = c1.text_input("瓶颈归因", value=run.bottleneck or "")
        status_detail = c2.text_input("状态明细", value=run.status_detail or "")
        next_action = c3.text_input("下一步", value=run.next_action or "")
        tags = st.text_input("标签（逗号分隔）", value=run.tags or "")
        notes = st.text_area("备注", value=run.notes or "", height=80)
        submitted = st.form_submit_button("保存")

    if submitted:
        fields = {
            "external_level": level,
            "tester": tester,
            "comparison_group": cmp_group,
            "bottleneck": bottleneck,
            "status_detail": status_detail,
            "next_action": next_action,
            "tags": tags,
            "notes": notes,
        }
        if db_manager.update_publish_metadata(run.id, fields):
            _flash("success", "元数据已保存")
            st.rerun()
        else:
            st.warning("未保存（无有效字段或更新失败）")


def _render_gate_review(run) -> None:
    """对存量运行重跑发布门禁（纯评估不写库；便于复核后手动改等级）。"""
    st.markdown("##### 发布门禁复评")
    st.caption("按当前记录重新评估四项门禁，不写库；通过后可到上方表单调整可对外等级。")
    if not st.button("重新评估门禁", key=f"wh_gate_{run.id}"):
        return
    try:
        from core.publish_gate import gate_from_run

        result = gate_from_run(run.to_dict())
    except Exception as e:  # noqa: BLE001  老记录字段缺失不应炸掉页面
        st.error(f"门禁评估失败：{e}")
        return

    if result.passed:
        st.success(f"门禁通过（{result.level}）")
    else:
        st.warning(f"门禁未通过（当前评定：{result.level}）")
    _GATE_LABELS = {
        "config_complete": "闸 1 · 配置齐全",
        "reproducible": "闸 2 · 可复现",
        "metrics_trustworthy": "闸 3 · 指标可信",
        "external_reviewed": "闸 4 · 人工复核",
    }
    for name, ok in result.gates.items():
        verdict = "通过" if ok else "未过"
        st.write(f"[{verdict}] {_GATE_LABELS.get(name, name)}")
    for reason in result.reasons:
        st.caption(f"· {reason}")


# ---------------------------------------------------------------------------
# 趋势与对比
# ---------------------------------------------------------------------------


def _render_trend_and_compare(runs) -> None:
    """趋势图（指标×日期×维度） + 运行对比（指标×运行）。"""
    st.subheader("趋势分析")
    st.caption("按日期看指标变化 —— 回归与改善都写在曲线上。数据来自当前筛选结果。")

    rows = [project_run(r) for r in runs]
    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.selectbox(
            "指标",
            list(TREND_METRICS),
            format_func=lambda k: TREND_METRICS[k],
            key="wh_trend_metric",
        )
    with c2:
        group_dim = st.selectbox(
            "分组维度",
            list(TREND_DIMS),
            format_func=lambda k: TREND_DIMS[k],
            key="wh_trend_dim",
        )
    with c3:
        publishable_only = st.checkbox(
            "仅可发布",
            value=False,
            key="wh_trend_pub",
            help="只保留 external_level = publishable 的记录",
        )

    trend_rows = (
        [r for r in rows if (r.get("external_level") or "internal") == "publishable"]
        if publishable_only
        else rows
    )
    fig = build_trend_figure(trend_rows, metric, group_dim)
    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("当前筛选下该指标无可绘数据（缺日期或指标值）。")

    # ---- 运行对比 ----
    st.markdown("---")
    st.subheader("运行对比")
    options = [
        f"{i}: {r.get('date')} · {r.get('model_name')} · {r.get('machine_id') or '—'}"
        for i, r in enumerate(rows)
    ]
    picked = st.multiselect(
        "选择 2~8 条运行做对比",
        options,
        max_selections=8,
        key="wh_compare_pick",
        placeholder="从当前筛选结果中挑选…",
    )
    idxs = [int(p.split(":", 1)[0]) for p in picked]
    if len(idxs) < 2:
        st.info("至少选择 2 条运行。")
        return

    selected = [rows[i] for i in idxs]
    table = build_compare_table(selected)
    df = pd.DataFrame(table).T
    st.dataframe(df, use_container_width=True)

    for group_name in COMPARE_METRIC_GROUPS:
        fig = build_compare_bar_figure(selected, group_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"wh_cmp_bar_{group_name}")


# ---------------------------------------------------------------------------
# 多卡扩展效率
# ---------------------------------------------------------------------------


def _render_scaling_efficiency(runs) -> None:
    """同模型 tp1→tpN 的扩展效率（手册诊断树 B：能跑但多卡没线性变快）。"""
    st.subheader("多卡扩展效率")
    st.caption(
        "手册：「4 卡只比 1 卡快 2 倍？」以 tp1 为基线，算 speedup 与 efficiency"
        "（理想线性=1.0；<1 亚线性，疑似通信/调度瓶颈）。"
    )

    metric = st.selectbox("指标", ["decode_tps", "effective_bandwidth_gbps"], key="se_metric")
    rows = build_scaling_efficiency(runs, metric=metric)
    if not rows:
        st.info("无可分析的多卡数据（需同模型在 tp1/tp2/tp4... 下各跑过测试）。")
        return

    import pandas as pd

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 归因：取效率最低的非 tp1 行
    non_baseline = [r for r in rows if r["tp_size"] != 1 and r["efficiency"] is not None]
    if non_baseline:
        worst = min(non_baseline, key=lambda r: r["efficiency"])
        st.warning(
            f"最低效率：{worst['model_name']} @ tp{worst['tp_size']} = "
            f"{worst['efficiency']:.2f}（speedup {worst['speedup_vs_tp1']:.2f}x / 理想 "
            f"{worst['linear_ideal_speedup']}x）。{interpret_efficiency(worst['efficiency'])}"
        )

    csv_str = _sheet_to_csv(rows)
    st.download_button(
        "导出 CSV",
        data=csv_str.encode("utf-8"),
        file_name="scaling_efficiency.csv",
        mime="text/csv",
        key="se_dl_csv",
    )


# ---------------------------------------------------------------------------
# 透视矩阵
# ---------------------------------------------------------------------------


def _render_cross_matrix(runs) -> None:
    st.subheader("硬件 × 模型 透视矩阵")
    # 维度可选：默认 machine×model；可切 quantization/engine/parallel 做量化/引擎/扩展对照
    dims = ["machine_id", "model_name", "engine", "quantization", "parallel_strategy"]
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        row_key = st.selectbox("行维度", dims, index=0, key="wh_matrix_row")
    with d2:
        col_opts = [d for d in dims if d != row_key]
        col_key = st.selectbox("列维度", col_opts, index=0, key="wh_matrix_col")
    with d3:
        metric = st.selectbox(
            "透视指标",
            [
                "decode_tps",
                "ttft_s",
                "effective_bandwidth_gbps",
                "bandwidth_utilization_pct",
                "gpu_vram_peak_gb",
            ],
            key="wh_matrix_metric",
        )
    with d4:
        agg = st.radio("聚合", ["best", "latest"], horizontal=True, key="wh_matrix_agg")
    agg_code = "best" if agg == "best" else "latest"

    mx = build_cross_matrix(runs, row_key=row_key, col_key=col_key, metric=metric, agg=agg_code)
    if not mx.row_labels or not mx.col_labels:
        st.info("该指标在当前筛选下无可透视的格（指标全缺测或无 machine_id/模型）。")
        return

    # 构建 DataFrame：行=machine_id，列=model_name
    table = {
        row: [mx.cells.get(row, {}).get(col) for col in mx.col_labels] for row in mx.row_labels
    }
    df = pd.DataFrame(table, index=mx.row_labels, columns=mx.col_labels).T
    st.caption(f"行 = {mx.row_key}，列 = {mx.col_key}，值 = {metric}（{agg_code}）")
    # 背景渐变：decode_tps/带宽/利用率 越高越好；ttft/显存越低越好——这里统一按数值大小渐变
    try:
        styled = df.style.background_gradient(cmap="YlGn", axis=None).format("{:.2f}", na_rep="—")
        st.dataframe(styled, use_container_width=True)
    except Exception:  # noqa: BLE001
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# 硬件盘点
# ---------------------------------------------------------------------------


def _render_inventory(runs) -> None:
    st.subheader("硬件盘点（每台机器一行）")
    rows = build_hardware_inventory_rows(runs)
    if not rows:
        st.info("无可用硬件指纹（记录缺 machine_id）。")
        return
    cols = [
        "machine_id",
        "cpu_model",
        "cpu_sockets",
        "cpu_cores",
        "memory_type",
        "memory_capacity_gb",
        "memory_channels_populated",
        "gpu_model",
        "gpu_count",
        "gpu_vram_gb",
        "gpu_bandwidth_gbps",
        "pcie_gen",
        "pcie_width",
        "cuda_or_rocm",
        "driver",
        "os",
        "owner",
        "location",
    ]
    df = pd.DataFrame(rows)[cols].rename(columns=_COLUMN_LABELS)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 客户能力表
# ---------------------------------------------------------------------------


def _render_capability_sheet() -> None:
    """按 customer_type × scenario 聚合应用用例，生成对外客户能力表。"""
    st.subheader("客户能力表")
    st.caption(
        "手册核心产出：「销售可以自动生成客户能力表」。按客户类型 × 场景聚合应用用例，"
        "10 分钟内从仓库抽出对外材料。"
    )

    c1, c2 = st.columns(2)
    with c1:
        min_level = st.selectbox(
            "对外口径下限",
            ["internal", "review", "publishable"],
            index=1,
            key="cap_min_level",
            help="只保留该等级及以上（默认 review 起，即可对外讨论）",
        )
    with c2:
        group_dim = st.multiselect(
            "聚合维度",
            ["customer_type", "scenario", "model_name"],
            default=["customer_type", "scenario", "model_name"],
            key="cap_group",
        )

    cases = db_manager.list_application_cases(limit=2000)
    if not cases:
        st.info("暂无应用用例。跑 Model Quality Test（自动采集）或录入应用用例后会生成。")
        return

    group_by = tuple(group_dim) if group_dim else ("customer_type", "scenario", "model_name")
    sheet = build_capability_sheet(cases, group_by=group_by, min_external_level=min_level)

    if not sheet:
        st.info(f"当前筛选下无达 {min_level} 口径的能力切片。降低口径下限或录入更多用例。")
        return

    import pandas as pd

    df = pd.DataFrame(sheet)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"共 {len(sheet)} 个能力切片（来自 {len(cases)} 条应用用例）")

    # 导出：CSV + 对外 Markdown
    e1, e2 = st.columns(2)
    csv_str = _sheet_to_csv(sheet)
    md_str = build_capability_markdown(sheet)
    e1.download_button(
        "导出 CSV",
        data=csv_str.encode("utf-8"),
        file_name="capability_sheet.csv",
        mime="text/csv",
        key="cap_dl_csv",
    )
    e2.download_button(
        "导出对外 Markdown",
        data=md_str.encode("utf-8"),
        file_name="capability_sheet.md",
        mime="text/markdown",
        key="cap_dl_md",
    )

    with st.expander("预览对外 Markdown"):
        st.markdown(md_str)


def _sheet_to_csv(sheet: list[dict]) -> str:
    """客户能力表 → CSV 字符串（UTF-8 BOM）。"""
    if not sheet:
        return ""
    # 取所有键的并集，但优先 CAPABILITY_COLUMNS 顺序
    from core.warehouse import CAPABILITY_COLUMNS

    fields = list(CAPABILITY_COLUMNS)
    extra = sorted({k for row in sheet for k in row} - set(fields))
    fields += extra
    from utils.spreadsheet import safe_csv_text

    return "\ufeff" + safe_csv_text(pd.DataFrame(sheet, columns=fields))


# ---------------------------------------------------------------------------
# 模板导出
# ---------------------------------------------------------------------------


def _render_export(runs) -> None:
    st.subheader("按手册三套字段模板导出")
    st.caption("导出的是仓库全集行（可筛选、可追溯、可对外口径），不是单次报告的图。")

    hw_rows = build_hardware_inventory_rows(runs)
    hm_rows = build_hm_test_rows(runs)
    # maTest 数据真源是 application_cases 表（自动采集 + 手动录入），不是 test_runs
    from core.warehouse import build_ma_test_rows_from_cases

    ma_rows = build_ma_test_rows_from_cases(db_manager)
    bundles = {"hwInventory": hw_rows, "hmTest": hm_rows, "maTest": ma_rows}

    # 每套模板一行：说明 + 行数 + CSV/JSON 下载
    for name, title in TEMPLATE_TITLES.items():
        rows_n = len(bundles[name])
        with st.container():
            c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
            c1.markdown(f"**{title}**　`{name}`　—　{rows_n} 行")
            csv_bytes = export_template_csv(name, bundles[name]).encode("utf-8")
            json_bytes = export_template_json(name, bundles[name]).encode("utf-8")
            c2.download_button(
                "CSV",
                data=csv_bytes,
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"wh_dl_csv_{name}",
            )
            c3.download_button(
                "JSON",
                data=json_bytes,
                file_name=f"{name}.json",
                mime="application/json",
                key=f"wh_dl_json_{name}",
            )
            c4.caption("—" if rows_n == 0 else f"{rows_n} 行")

    st.markdown("---")
    st.markdown("**一键打包全部模板（ZIP）**")
    zc, zj = st.columns(2)
    zip_csv = export_all_templates_zip(bundles, fmt="csv")
    zip_json = export_all_templates_zip(bundles, fmt="json")
    zc.download_button(
        "全部 CSV (ZIP)",
        data=zip_csv,
        file_name="warehouse_templates_csv.zip",
        mime="application/zip",
        key="wh_dl_zip_csv",
    )
    zj.download_button(
        "全部 JSON (ZIP)",
        data=zip_json,
        file_name="warehouse_templates_json.zip",
        mime="application/zip",
        key="wh_dl_zip_json",
    )


# ---------------------------------------------------------------------------
# 列名中英对照（表格表头用）
# ---------------------------------------------------------------------------


_COLUMN_LABELS = {
    "date": "日期",
    "machine_id": "machine_id",
    "model_name": "模型",
    "engine": "引擎",
    "parallel_strategy": "并行",
    "concurrency": "并发",
    "decode_tps": "decode TPS",
    "ttft_s": "TTFT(s)",
    "effective_bandwidth_gbps": "等效带宽(GB/s)",
    "bandwidth_utilization_pct": "带宽利用率(%)",
    "gpu_vram_peak_gb": "显存峰值(GB)",
    "bottleneck": "瓶颈",
    "status": "状态",
    "external_level": "可对外",
    "tester": "测试员",
    "test_id": "test_id",
    "model_version": "版本",
    "model_type": "类型",
    "total_params": "总参数(B)",
    "active_params": "激活参数(B)",
    "quantization": "量化",
    "dtype": "精度",
    "max_context": "上下文",
    "engine_version": "引擎版本",
    "engine_params": "引擎参数",
    "prefill_tps": "prefill TPS",
    "p50_latency_s": "p50(s)",
    "p95_latency_s": "p95(s)",
    "p99_latency_s": "p99(s)",
    "system_memory_peak_gb": "内存峰值(GB)",
    "gpu_util_pct": "GPU利用率(%)",
    "cpu_util_pct": "CPU利用率(%)",
    "power_w": "功耗(W)",
    "temp_c": "温度(℃)",
    "cpu_model": "CPU",
    "cpu_sockets": "路数",
    "cpu_cores": "核/路",
    "memory_type": "内存类型",
    "memory_capacity_gb": "内存(GB)",
    "memory_channels_populated": "通道",
    "gpu_model": "GPU",
    "gpu_count": "卡数",
    "gpu_vram_gb": "显存(GB)",
    "gpu_bandwidth_gbps": "显存带宽(GB/s)",
    "pcie_gen": "PCIe Gen",
    "pcie_width": "PCIe 宽",
    "cuda_or_rocm": "CUDA/ROCm",
    "driver": "驱动",
    "os": "OS",
    "owner": "负责人",
    "location": "位置",
    "error_type": "异常类型",
    "error_detail": "异常明细",
    "next_action": "下一步",
    "supersedes_test_id": "复测指向",
    "log_path": "日志路径",
}


# 模块自测入口：streamlit run ui/warehouse_browser.py 可独立预览
if __name__ == "__main__":
    render_warehouse_browser()
