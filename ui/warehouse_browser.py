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
    distinct_values,
    export_all_templates_zip,
    export_template_csv,
    export_template_json,
    interpret_efficiency,
    project_run,
    query_runs,
)

# 历史表展示的仓库列（顺序即展示顺序）
_HISTORY_COLUMNS = [
    "date", "machine_id", "model_name", "engine", "parallel_strategy",
    "concurrency", "decode_tps", "ttft_s", "effective_bandwidth_gbps",
    "bandwidth_utilization_pct", "gpu_vram_peak_gb", "bottleneck",
    "status", "external_level", "tester",
]


def render_warehouse_browser() -> None:
    """渲染数据仓库浏览页主入口。"""
    st.title("🗄️ 数据仓库")
    st.caption(
        "报告是切片，仓库是全集 —— 这里筛选、透视、导出历史测试的全集行。"
        "（口径：test-standard/端侧AI硬件与模型.html #templates）"
    )

    db = db_manager
    runs = query_runs(db, _build_filter(db))

    _render_kpis(runs)
    st.markdown("---")

    if not runs:
        st.info("当前筛选条件下无记录。松开筛选条件，或先跑一次测试再回来。")
        return

    tab_history, tab_matrix, tab_scaling, tab_inventory, tab_cases, tab_capability, tab_export = st.tabs(
        ["📋 运行历史", "🔲 透视矩阵", "📈 扩展效率", "🖥️ 硬件盘点", "🧪 应用用例",
         "📊 客户能力表", "📤 模板导出"]
    )

    with tab_history:
        _render_history(runs)
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


# ---------------------------------------------------------------------------
# 筛选栏
# ---------------------------------------------------------------------------


def _build_filter(db) -> WarehouseFilter:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗄️ 仓库筛选")

    def _opts(field: str, label: str) -> list:
        vals = ["全部"] + distinct_values(db, field)
        return vals

    machine = st.sidebar.selectbox("硬件 machine_id", _opts("machine_id", "硬件"), key="wh_machine")
    model = st.sidebar.selectbox("模型", _opts("model_name", "模型"), key="wh_model")
    engine = st.sidebar.selectbox("引擎", _opts("engine", "引擎"), key="wh_engine")
    level = st.sidebar.selectbox(
        "可对外等级",
        ["全部", "internal", "review", "publishable"],
        key="wh_level",
    )
    status = st.sidebar.selectbox("状态", _opts("status", "状态"), key="wh_status")
    tester = st.sidebar.selectbox("测试员", _opts("tester", "测试员"), key="wh_tester")
    cfg_hash = st.sidebar.selectbox(
        "config_hash（同配置）", _opts("config_hash", "配置"), key="wh_cfg_hash",
        help="CASE 02：同配置才能承诺。按配置指纹过滤同模型/引擎/并行/量化的一组测试。",
    )
    search = st.sidebar.text_input("模糊搜索", placeholder="备注 / 模型 / 测试员…", key="wh_search")

    def _pick(v):
        return None if v in (None, "全部", "") else v

    return WarehouseFilter(
        machine_id=_pick(machine),
        model_id=_pick(model),
        engine=_pick(engine),
        external_level=_pick(level),
        status_detail=_pick(status),
        tester=_pick(tester),
        config_hash=_pick(cfg_hash),
        search=_pick(search),
    )


# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------


def _render_kpis(runs) -> None:
    machines = {project_run(r)["machine_id"] for r in runs if project_run(r)["machine_id"]}
    models = {r.model_id for r in runs if r.model_id}
    publishable = sum(1 for r in runs if (r.external_level or "internal") == "publishable")
    completed = sum(1 for r in runs if r.status == "completed")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("记录数", len(runs))
    c2.metric("硬件数", len(machines))
    c3.metric("模型数", len(models))
    c4.metric("已完成", completed)
    c5.metric("可对外", publishable)


# ---------------------------------------------------------------------------
# 运行历史
# ---------------------------------------------------------------------------


def _render_history(runs) -> None:
    st.subheader("运行历史（仓库列）")
    rows = [project_run(r) for r in runs]
    df = pd.DataFrame(rows)[_HISTORY_COLUMNS].rename(columns=_COLUMN_LABELS)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"共 {len(rows)} 条；复测链默认只留最新一版。")

    # 明细抽屉：选一条看八维全集
    st.markdown("#### 查看单条全集")
    options = [
        f"{i}: {project_run(r)['date']} · {r.model_id} · {project_run(r)['machine_id'] or '—'}"
        for i, r in enumerate(runs)
    ]
    if options:
        choice = st.selectbox("选择记录", options, key="wh_detail_pick")
        idx = int(choice.split(":", 1)[0])
        _render_detail_drawer(runs[idx])


def _render_detail_drawer(run) -> None:
    row = project_run(run)
    with st.expander("八维全集（machine_id/指纹 → 资源 → 模型规格 → 服务配置 → 性能 → 引擎运行时 → 归因 → 可对外）", expanded=True):
        # 分组展示，避免一长串
        groups = [
            ("标识", ["test_id", "date", "tester", "machine_id", "external_level"]),
            ("模型 / 服务", ["model_name", "model_version", "model_type", "total_params",
                          "active_params", "quantization", "dtype", "max_context",
                          "engine", "engine_version", "parallel_strategy", "engine_params"]),
            ("性能", ["concurrency", "decode_tps", "prefill_tps", "ttft_s",
                    "p50_latency_s", "p95_latency_s", "p99_latency_s",
                    "effective_bandwidth_gbps", "bandwidth_utilization_pct"]),
            ("资源峰值", ["gpu_vram_peak_gb", "system_memory_peak_gb",
                       "gpu_util_pct", "cpu_util_pct", "power_w", "temp_c"]),
            ("硬件指纹", ["cpu_model", "cpu_sockets", "memory_type", "memory_capacity_gb",
                       "gpu_model", "gpu_count", "gpu_vram_gb", "gpu_bandwidth_gbps",
                       "cuda_or_rocm", "driver"]),
            ("归因 / 下一步", ["status", "bottleneck", "error_type", "error_detail",
                            "next_action", "supersedes_test_id", "log_path"]),
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


# ---------------------------------------------------------------------------
# 多卡扩展效率
# ---------------------------------------------------------------------------


def _render_scaling_efficiency(runs) -> None:
    """同模型 tp1→tpN 的扩展效率（手册诊断树 B：能跑但多卡没线性变快）。"""
    st.subheader("📈 多卡扩展效率")
    st.caption(
        "手册：「4 卡只比 1 卡快 2 倍？」以 tp1 为基线，算 speedup 与 efficiency"
        "（理想线性=1.0；<1 亚线性，疑似通信/调度瓶颈）。"
    )

    metric = st.selectbox("指标", ["decode_tps", "effective_bandwidth_gbps"],
                          key="se_metric")
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
    st.download_button("📥 导出 CSV", data=csv_str.encode("utf-8"),
                       file_name="scaling_efficiency.csv", mime="text/csv", key="se_dl_csv")


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
            ["decode_tps", "ttft_s", "effective_bandwidth_gbps", "bandwidth_utilization_pct",
             "gpu_vram_peak_gb"],
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
        row: [mx.cells.get(row, {}).get(col) for col in mx.col_labels]
        for row in mx.row_labels
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
        "machine_id", "cpu_model", "cpu_sockets", "cpu_cores", "memory_type",
        "memory_capacity_gb", "memory_channels_populated", "gpu_model", "gpu_count",
        "gpu_vram_gb", "gpu_bandwidth_gbps", "pcie_gen", "pcie_width",
        "cuda_or_rocm", "driver", "os", "owner", "location",
    ]
    df = pd.DataFrame(rows)[cols].rename(columns=_COLUMN_LABELS)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 客户能力表
# ---------------------------------------------------------------------------


def _render_capability_sheet() -> None:
    """按 customer_type × scenario 聚合应用用例，生成对外客户能力表。"""
    st.subheader("📊 客户能力表")
    st.caption(
        "手册核心产出：「销售可以自动生成客户能力表」。按客户类型 × 场景聚合应用用例，"
        "10 分钟内从仓库抽出对外材料。"
    )

    c1, c2 = st.columns(2)
    with c1:
        min_level = st.selectbox("对外口径下限", ["internal", "review", "publishable"],
                                 index=1, key="cap_min_level",
                                 help="只保留该等级及以上（默认 review 起，即可对外讨论）")
    with c2:
        group_dim = st.multiselect(
            "聚合维度", ["customer_type", "scenario", "model_name"],
            default=["customer_type", "scenario", "model_name"], key="cap_group",
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
    e1.download_button("📥 导出 CSV", data=csv_str.encode("utf-8"),
                       file_name="capability_sheet.csv", mime="text/csv", key="cap_dl_csv")
    e2.download_button("📥 导出对外 Markdown", data=md_str.encode("utf-8"),
                       file_name="capability_sheet.md", mime="text/markdown", key="cap_dl_md")

    with st.expander("预览对外 Markdown"):
        st.markdown(md_str)


def _sheet_to_csv(sheet: list[dict]) -> str:
    """客户能力表 → CSV 字符串（UTF-8 BOM）。"""
    import csv
    import io
    if not sheet:
        return ""
    # 取所有键的并集，但优先 CAPABILITY_COLUMNS 顺序
    from core.warehouse import CAPABILITY_COLUMNS
    fields = list(CAPABILITY_COLUMNS)
    extra = sorted({k for row in sheet for k in row} - set(fields))
    fields += extra
    buf = io.StringIO()
    buf.write("﻿")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    for row in sheet:
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return buf.getvalue()


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
                "CSV", data=csv_bytes, file_name=f"{name}.csv", mime="text/csv",
                key=f"wh_dl_csv_{name}",
            )
            c3.download_button(
                "JSON", data=json_bytes, file_name=f"{name}.json", mime="application/json",
                key=f"wh_dl_json_{name}",
            )
            c4.caption("—" if rows_n == 0 else f"{rows_n} 行")

    st.markdown("---")
    st.markdown("**一键打包全部模板（ZIP）**")
    zc, zj = st.columns(2)
    zip_csv = export_all_templates_zip(bundles, fmt="csv")
    zip_json = export_all_templates_zip(bundles, fmt="json")
    zc.download_button(
        "📥 全部 CSV (ZIP)", data=zip_csv, file_name="warehouse_templates_csv.zip",
        mime="application/zip", key="wh_dl_zip_csv",
    )
    zj.download_button(
        "📥 全部 JSON (ZIP)", data=zip_json, file_name="warehouse_templates_json.zip",
        mime="application/zip", key="wh_dl_zip_json",
    )


# ---------------------------------------------------------------------------
# 列名中英对照（表格表头用）
# ---------------------------------------------------------------------------


_COLUMN_LABELS = {
    "date": "日期", "machine_id": "machine_id", "model_name": "模型", "engine": "引擎",
    "parallel_strategy": "并行", "concurrency": "并发", "decode_tps": "decode TPS",
    "ttft_s": "TTFT(s)", "effective_bandwidth_gbps": "等效带宽(GB/s)",
    "bandwidth_utilization_pct": "带宽利用率(%)", "gpu_vram_peak_gb": "显存峰值(GB)",
    "bottleneck": "瓶颈", "status": "状态", "external_level": "可对外", "tester": "测试员",
    "test_id": "test_id", "model_version": "版本", "model_type": "类型",
    "total_params": "总参数(B)", "active_params": "激活参数(B)", "quantization": "量化",
    "dtype": "精度", "max_context": "上下文", "engine_version": "引擎版本",
    "engine_params": "引擎参数", "prefill_tps": "prefill TPS", "p50_latency_s": "p50(s)",
    "p95_latency_s": "p95(s)", "p99_latency_s": "p99(s)", "system_memory_peak_gb": "内存峰值(GB)",
    "gpu_util_pct": "GPU利用率(%)", "cpu_util_pct": "CPU利用率(%)", "power_w": "功耗(W)",
    "temp_c": "温度(℃)", "cpu_model": "CPU", "cpu_sockets": "路数", "cpu_cores": "核/路",
    "memory_type": "内存类型", "memory_capacity_gb": "内存(GB)",
    "memory_channels_populated": "通道", "gpu_model": "GPU", "gpu_count": "卡数",
    "gpu_vram_gb": "显存(GB)", "gpu_bandwidth_gbps": "显存带宽(GB/s)", "pcie_gen": "PCIe Gen",
    "pcie_width": "PCIe 宽", "cuda_or_rocm": "CUDA/ROCm", "driver": "驱动", "os": "OS",
    "owner": "负责人", "location": "位置", "error_type": "异常类型", "error_detail": "异常明细",
    "next_action": "下一步", "supersedes_test_id": "复测指向", "log_path": "日志路径",
}


# 模块自测入口：streamlit run ui/warehouse_browser.py 可独立预览
if __name__ == "__main__":
    render_warehouse_browser()
