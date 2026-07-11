"""
数据仓库富信息渲染（报告区）—— 把“八维记录”可视化呈现，告别干瘪图片。

渲染：
- 硬件指纹卡（CPU×socket/NUMA、内存、每卡 VRAM/带宽/PCIe、CUDA/驱动、machine_id）
- 资源监控时序图（GPU/CPU 利用率、显存/内存、功耗/温度 随测试时长）
- 等效带宽偏差分析（标称 vs 实测 vs 利用率 + 归因）
- 可对外闸门徽标 + 四道闸 checklist
- 单次测试 markdown 报告（9 节）导出

所有渲染只读 session_state（由 ui.test_runner._capture_post_run_artifacts 填充）。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.publish_gate import GATE_LABELS, LEVEL_BADGE, evaluate_publish_gate

# ---------------------------------------------------------------------------
# 硬件指纹卡
# ---------------------------------------------------------------------------

def render_hardware_fingerprint_card() -> None:
    sys_info = st.session_state.get("system_info") or {}
    fp = sys_info.get("hardware_fingerprint") or {}
    if not fp:
        return
    with st.expander("Hardware Fingerprint: 硬件指纹（配置冻结）", expanded=False):
        cpu = fp.get("cpu") or {}
        mem = fp.get("memory") or {}
        gpus = fp.get("gpus") or []
        cuda = fp.get("cuda") or {}

        c1, c2, c3 = st.columns(3)
        c1.metric("CPU", cpu.get("model_name") or "—",
                  f"{cpu.get('sockets', 1)}×{cpu.get('cores_per_socket') or '?'}核 / "
                  f"{cpu.get('threads_per_core') or '?'}线程 / NUMA {cpu.get('numa_nodes') or '?'}")
        c2.metric("内存", f"{mem.get('total_gb') or '?'} GB",
                  f"{mem.get('type') or '?'} / {mem.get('channels') or '?'}ch / "
                  f"{mem.get('speed_mt_s') or '?'}MT/s / ECC {mem.get('ecc') or '?'}")
        c3.metric("GPU 数量", len(gpus))

        if gpus:
            rows = []
            for g in gpus:
                rows.append({
                    "GPU": g.get("name") or "—",
                    "显存(GB)": g.get("vram_gb"),
                    "类型": g.get("memory_type") or "—",
                    "标称带宽(GB/s)": g.get("nominal_bandwidth_gbps"),
                    "PCIe": f"Gen{g.get('pcie_gen') or '?'} ×{g.get('pcie_width') or '?'}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        meta = st.columns(3)
        meta[0].metric("machine_id", fp.get("machine_id") or "—")
        meta[1].metric("CUDA", cuda.get("cuda_version") or "—")
        meta[2].metric("驱动", cuda.get("driver") or "—")


# ---------------------------------------------------------------------------
# 资源监控时序图
# ---------------------------------------------------------------------------

def render_resource_timeline() -> None:
    mon = st.session_state.get("resource_monitor")
    if not mon or not mon.get("timeline"):
        return
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    timeline = mon["timeline"]
    df = pd.DataFrame(timeline)
    if "t" not in df.columns:
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # 利用率（左轴）
    if "cpu_percent" in df:
        fig.add_trace(go.Scatter(x=df["t"], y=df["cpu_percent"], name="CPU%", line={"width": 1}), secondary_y=False)
    if "gpu_util_percent" in df:
        fig.add_trace(go.Scatter(x=df["t"], y=df["gpu_util_percent"], name="GPU%", line={"width": 1}), secondary_y=False)
    # 容量（右轴）
    if "gpu_vram_gb" in df:
        fig.add_trace(go.Scatter(x=df["t"], y=df["gpu_vram_gb"], name="显存(GB)", line={"dash": "dot"}), secondary_y=True)
    if "system_memory_gb" in df:
        fig.add_trace(go.Scatter(x=df["t"], y=df["system_memory_gb"], name="内存(GB)", line={"dash": "dot"}), secondary_y=True)

    fig.update_layout(
        title="资源监控时序（测试期间）", height=320,
        legend={"orientation": "h", "y": -0.2}, margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    fig.update_xaxes(title_text="时间 (s)")
    fig.update_yaxes(title_text="利用率 (%)", secondary_y=False, range=[0, 105])
    fig.update_yaxes(title_text="容量 (GB)", secondary_y=True)

    peaks = mon.get("peaks") or {}
    with st.expander("Scaling: 资源监控（利用率 / 显存 / 内存峰值）", expanded=False):
        st.plotly_chart(fig, use_container_width=True)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("GPU 利用率峰值", f"{peaks.get('gpu_util_percent') or '—'}%")
        p2.metric("显存峰值", f"{peaks.get('gpu_vram_gb') or '—'} GB")
        p3.metric("内存峰值", f"{peaks.get('system_memory_gb') or '—'} GB")
        p4.metric("功耗/温度", f"{peaks.get('gpu_power_w') or '—'}W / {peaks.get('gpu_temp_c') or '—'}℃")


# ---------------------------------------------------------------------------
# 等效带宽偏差分析
# ---------------------------------------------------------------------------

def render_deviation_analysis() -> None:
    bw = st.session_state.get("effective_bandwidth") or {}
    if not bw or bw.get("effective_bandwidth_gbps") is None:
        return
    from core.effective_bandwidth import summarize_gap
    with st.expander("Bandwidth: 等效带宽偏差分析", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("标称显存带宽", f"{bw.get('nominal_bandwidth_gbps') or '—'} GB/s")
        c2.metric("实测等效带宽", f"{bw.get('effective_bandwidth_gbps')} GB/s")
        c3.metric("带宽利用率", f"{bw.get('bandwidth_utilization_pct') or '—'}%")
        st.caption(summarize_gap(bw))


# ---------------------------------------------------------------------------
# 推理引擎运行时（KV 占用 / 调度队列 / 抢救）
# ---------------------------------------------------------------------------

def render_engine_runtime() -> None:
    """渲染引擎自身的运行视图：KV cache 占用、运行/等待队列、抢救数、KV 容量。"""
    eng = st.session_state.get("engine_metrics")
    if not eng or not eng.get("timeline"):
        return
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    df = pd.DataFrame(eng["timeline"])
    if "t" not in df.columns:
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "gpu_cache_usage_perc" in df:
        fig.add_trace(go.Scatter(x=df["t"], y=df["gpu_cache_usage_perc"], name="KV cache 占用", line={"width": 2}), secondary_y=False)
    if "num_requests_running" in df:
        fig.add_trace(go.Scatter(x=df["t"], y=df["num_requests_running"], name="运行请求", line={"dash": "dot"}), secondary_y=True)
    if "num_requests_waiting" in df:
        fig.add_trace(go.Scatter(x=df["t"], y=df["num_requests_waiting"], name="等待请求", line={"dash": "dot"}), secondary_y=True)
    fig.update_layout(
        title=f"推理引擎运行时（{eng.get('engine_family', '?')}）", height=320,
        legend={"orientation": "h", "y": -0.2}, margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    fig.update_xaxes(title_text="时间 (s)")
    fig.update_yaxes(title_text="KV cache 占用 (0~1)", secondary_y=False, range=[0, 1.05])
    fig.update_yaxes(title_text="请求数", secondary_y=True)

    peaks = eng.get("peaks") or {}
    cc = eng.get("cache_config") or {}
    with st.expander("Inference Engine: 推理引擎运行时（KV / 队列 / 抢救）", expanded=False):
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KV 占用峰值", f"{(peaks.get('gpu_cache_usage_perc') or 0)*100:.0f}%")
        c2.metric("抢救数(窗口)", eng.get("preemption_total") or 0)
        c3.metric("运行队列峰值", peaks.get("num_requests_running") or "—")
        c4.metric("KV 容量(tokens)", cc.get("kv_capacity_tokens") or st.session_state.get("kv_cache_capacity_tokens") or "—")
        if cc.get("num_gpu_blocks"):
            st.caption(f"block_size={cc.get('block_size')} · num_gpu_blocks={cc.get('num_gpu_blocks')} · num_cpu_blocks={cc.get('num_cpu_blocks')}")


# ---------------------------------------------------------------------------
# 客户端 vs 引擎侧 TTFT/TPOT 对照
# ---------------------------------------------------------------------------

def render_client_vs_engine_analysis() -> None:
    """客户端（端到端）vs 引擎侧（/metrics 直方图）TTFT/TPOT 对照，定位延迟在传输还是引擎。"""
    from core.latency_analysis import compute_client_vs_engine_latency

    df = st.session_state.get("results_df")
    eng = st.session_state.get("engine_metrics")
    if (df is None or getattr(df, "empty", True)) and not (eng or {}).get("engine_means"):
        return

    r = compute_client_vs_engine_latency(df, eng)
    if r["client_ttft_s"] is None and r["engine_ttft_s"] is None:
        return

    with st.expander("Latency: 客户端 vs 引擎侧延迟对照", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("客户端 TTFT(中位)", f"{r['client_ttft_s']*1000:.0f} ms" if r["client_ttft_s"] else "—")
        c2.metric("引擎侧 TTFT(整体)", f"{r['engine_ttft_s']*1000:.0f} ms" if r["engine_ttft_s"] else "—")
        oh = r["ttft_overhead_pct"]
        c3.metric("TTFT 开销占比", f"{oh}%" if oh is not None else "—",
                  help="(客户端-引擎)/引擎 —— 高则延迟在网络/排队，低则在引擎内部")
        c4, c5, c6 = st.columns(3)
        c4.metric("客户端 TPOT(中位)", f"{r['client_tpot_ms']:.1f} ms" if r["client_tpot_ms"] else "—")
        c5.metric("引擎侧 TPOT(整体)", f"{r['engine_tpot_ms']:.1f} ms" if r["engine_tpot_ms"] else "—")
        c6.metric("TPOT 开销", f"{r['tpot_overhead_ms']:.1f} ms" if r["tpot_overhead_ms"] is not None else "—")
        st.caption(r["verdict"])


# ---------------------------------------------------------------------------
# 可对外闸门徽标
# ---------------------------------------------------------------------------

def render_publish_gate_badge() -> None:
    sys_info = st.session_state.get("system_info") or {}
    fp = sys_info.get("hardware_fingerprint") or {}
    mon = st.session_state.get("resource_monitor")
    tm = st.session_state.get("test_metadata") or {}
    config = st.session_state.get("test_config") or {}

    result = evaluate_publish_gate(
        tester=tm.get("tester"),
        machine_id=fp.get("machine_id") or sys_info.get("machine_id"),
        has_hardware_fingerprint=bool(fp),
        seed_recorded=config.get("random_seed") is not None,
        insights=st.session_state.get("insights"),
        success_rate=_success_rate(),
        has_monitor=bool(mon and mon.get("timeline")),
        requested_external_level=tm.get("external_level", "internal"),
    )
    label, color = LEVEL_BADGE[result.level]
    st.markdown(
        f'<div style="padding:8px 12px;border-radius:6px;background:{color};color:#fff;'
        f'font-weight:700;display:inline-block;margin:4px 0;">{label}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for i, (key, label_cn) in enumerate(GATE_LABELS.items()):
        passed = result.gates.get(key, False)
        cols[i].metric(label_cn, "OK:" if passed else "Error:")
    if result.reasons:
        st.caption("未通过：" + "；".join(result.reasons))


def _success_rate() -> float | None:
    df = st.session_state.get("results_df")
    try:
        if df is None or getattr(df, "empty", True) or "error" not in df.columns:
            return None
        non_empty = df["error"].apply(lambda x: bool(x) and str(x).strip() not in ("", "nan", "None"))
        failed = int(non_empty.sum())
        return max(0.0, 1.0 - failed / len(df)) if len(df) else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 单次测试 markdown 报告（纯函数，可测可复用）
# ---------------------------------------------------------------------------

def build_single_test_report(ctx: dict[str, Any]) -> str:
    """从上下文字典生成单次测试 markdown 报告（9 节）。

    ctx 键：test_type, model_id, tester, machine_id, status_detail, bottleneck,
    hardware_fingerprint, test_config, summary_table, resource_monitor,
    effective_bandwidth, gate(result dict), insights, notes, next_action
    """
    lines: list[str] = []
    lines.append("# 单次测试报告\n")
    lines.append(f"- 测试类型: {ctx.get('test_type', '—')}")
    lines.append(f"- 模型: {ctx.get('model_id', '—')}")
    lines.append(f"- 测试人: {ctx.get('tester') or '—'}")
    lines.append(f"- machine_id: {ctx.get('machine_id') or '—'}")
    lines.append(f"- 状态: {ctx.get('status_detail') or '—'}")
    lines.append(f"- 瓶颈: {ctx.get('bottleneck') or '—（无明确瓶颈）'}")
    lines.append("")

    lines.append("## 硬件指纹\n")
    fp = ctx.get("hardware_fingerprint") or {}
    cpu = fp.get("cpu") or {}
    mem = fp.get("memory") or {}
    gpus = fp.get("gpus") or []
    lines.append(f"- CPU: {cpu.get('model_name') or '—'} ({cpu.get('sockets', 1)}×{cpu.get('cores_per_socket') or '?'}核)")
    lines.append(f"- 内存: {mem.get('total_gb') or '?'} GB / {mem.get('type') or '?'} / {mem.get('channels') or '?'}通道")
    for g in gpus:
        lines.append(f"- GPU: {g.get('name')} {g.get('vram_gb')}GB / 带宽 {g.get('nominal_bandwidth_gbps')}GB/s / PCIe Gen{g.get('pcie_gen')}×{g.get('pcie_width')}")
    cuda = fp.get("cuda") or {}
    lines.append(f"- CUDA/驱动: {cuda.get('cuda_version') or '—'} / {cuda.get('driver') or '—'}")
    lines.append("")

    lines.append("## 配置快照\n")
    for k, v in (ctx.get("test_config") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## 性能摘要\n")
    lines.append(ctx.get("summary_table") or "（见结果表）")
    lines.append("")

    lines.append("## 资源峰值\n")
    mon = ctx.get("resource_monitor") or {}
    peaks = mon.get("peaks") or {}
    lines.append(f"- GPU 利用率峰值: {peaks.get('gpu_util_percent') or '—'}%")
    lines.append(f"- 显存峰值: {peaks.get('gpu_vram_gb') or '—'} GB")
    lines.append(f"- 内存峰值: {peaks.get('system_memory_gb') or '—'} GB")
    lines.append(f"- 功耗/温度: {peaks.get('gpu_power_w') or '—'}W / {peaks.get('gpu_temp_c') or '—'}℃")
    lines.append("")

    lines.append("## 推理引擎运行时\n")
    eng = ctx.get("engine_metrics") or {}
    if eng and eng.get("sample_count", 0) > 0:
        ep = eng.get("peaks") or {}
        cc = eng.get("cache_config") or {}
        lines.append(f"- 引擎: {eng.get('engine_family', '?')} @ {eng.get('metrics_url', '?')}")
        lines.append(f"- KV cache 占用峰值: {(ep.get('gpu_cache_usage_perc') or 0)*100:.0f}%")
        lines.append(f"- 运行队列峰值: {ep.get('num_requests_running') or '—'} / 等待峰值: {ep.get('num_requests_waiting') or '—'}")
        lines.append(f"- 抢救数(窗口): {eng.get('preemption_total') or 0}")
        lines.append(f"- KV 容量: {cc.get('kv_capacity_tokens') or '—'} tokens (block_size={cc.get('block_size') or '?'})")
    else:
        lines.append("- 未采集到引擎运行时（未配置 /metrics 端点或端点不可达）。")
    lines.append("")

    lines.append("## 客户端 vs 引擎侧延迟对照\n")
    ce = ctx.get("client_vs_engine") or {}
    if ce.get("client_ttft_s") is not None or ce.get("engine_ttft_s") is not None:
        lines.append(f"- 客户端 TTFT(中位): {ce.get('client_ttft_s') or '—'} s | 引擎侧 TTFT(整体): {ce.get('engine_ttft_s') or '—'} s | 开销占比: {ce.get('ttft_overhead_pct') or '—'}%")
        lines.append(f"- 客户端 TPOT(中位): {ce.get('client_tpot_ms') or '—'} ms | 引擎侧 TPOT(整体): {ce.get('engine_tpot_ms') or '—'} ms")
        lines.append(f"- 判定: {ce.get('verdict') or '—'}")
    else:
        lines.append("- 对照数据不全（缺客户端 TTFT 或引擎侧 /metrics）。")
    lines.append("")

    lines.append("## 等效带宽偏差分析\n")
    from core.effective_bandwidth import summarize_gap
    lines.append(summarize_gap(ctx.get("effective_bandwidth") or {}))
    lines.append("")

    lines.append("## 可对外闸门\n")
    gate = ctx.get("gate") or {}
    lines.append(f"- 等级: **{gate.get('level', 'internal')}**")
    from core.publish_gate import GATE_LABELS as _GL
    for key, label_cn in _GL.items():
        passed = (gate.get("gates") or {}).get(key)
        lines.append(f"- {label_cn}: {'OK:' if passed else 'Error:'}")
    if gate.get("reasons"):
        lines.append(f"- 未通过：{'；'.join(gate['reasons'])}")
    lines.append("")

    lines.append("## 洞察 (insights)\n")
    for i in (ctx.get("insights") or []):
        lines.append(f"- {i}")
    lines.append("")

    lines.append("## 备注 / 下一步\n")
    lines.append(f"- 下一步动作: {ctx.get('next_action') or '—'}")
    lines.append(f"- 备注: {ctx.get('notes') or '—'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------

def render_warehouse_panel(test_type: str, model_id: str) -> None:
    """在结果区后渲染全部富信息 + 导出按钮。任一段失败不阻塞其它。"""
    try:
        st.markdown("---")
        st.header("Data Warehouse: 数据仓库记录（八维）")

        # 闸门徽标
        render_publish_gate_badge()
        # 指纹 / 时序 / 偏差 / 引擎运行时
        render_hardware_fingerprint_card()
        render_resource_timeline()
        render_deviation_analysis()
        render_engine_runtime()
        render_client_vs_engine_analysis()

        # 单次测试报告导出
        if st.button("Export: 导出单次测试报告 (Markdown)", key="export_single_report"):
            md = build_single_test_report(_collect_report_context(test_type, model_id))
            st.download_button(
                label="Download: 下载 .md",
                data=md.encode("utf-8"),
                file_name=f"test_report_{model_id}_{test_type}.md",
                mime="text/markdown",
            )
    except Exception as e:  # noqa: BLE001
        st.caption(f"数据仓库面板渲染跳过: {e}")


def _client_vs_engine_for_report(ss) -> dict[str, Any]:
    from core.latency_analysis import compute_client_vs_engine_latency

    return compute_client_vs_engine_latency(
        ss.get("results_df"), ss.get("engine_metrics")
    )


def _collect_report_context(test_type: str, model_id: str) -> dict[str, Any]:
    sys_info = st.session_state.get("system_info") or {}
    fp = sys_info.get("hardware_fingerprint") or {}
    mon = st.session_state.get("resource_monitor") or {}
    tm = st.session_state.get("test_metadata") or {}
    bw = st.session_state.get("effective_bandwidth") or {}

    result = evaluate_publish_gate(
        tester=tm.get("tester"),
        machine_id=fp.get("machine_id") or sys_info.get("machine_id"),
        has_hardware_fingerprint=bool(fp),
        seed_recorded=(st.session_state.get("test_config") or {}).get("random_seed") is not None,
        insights=st.session_state.get("insights"),
        success_rate=_success_rate(),
        has_monitor=bool(mon and mon.get("timeline")),
        requested_external_level=tm.get("external_level", "internal"),
    )
    return {
        "test_type": test_type,
        "model_id": model_id,
        "tester": tm.get("tester"),
        "machine_id": fp.get("machine_id"),
        "status_detail": st.session_state.get("status_detail") or tm.get("status_detail"),
        "bottleneck": st.session_state.get("bottleneck"),
        "hardware_fingerprint": fp,
        "test_config": st.session_state.get("test_config") or {},
        "resource_monitor": mon,
        "engine_metrics": st.session_state.get("engine_metrics") or {},
        "client_vs_engine": _client_vs_engine_for_report(st.session_state),
        "effective_bandwidth": bw,
        "gate": {"level": result.level, "gates": result.gates, "reasons": result.reasons},
        "insights": st.session_state.get("insights") or [],
        "notes": tm.get("notes"),
        "next_action": tm.get("next_action"),
    }
