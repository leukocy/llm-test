"""
数据仓库输入面板（sidebar）—— 让用户填写“事无巨细”记录所需的模型规格 / 服务配置 / 测试元数据。

这些面板写入 session_state，由 core.benchmark_runner._build_warehouse_extra_fields 读取并
随测试写入 test_runs 的 model_spec_json / serving_config_json / 元数据列。
"""

from __future__ import annotations

import streamlit as st

from core.model_spec import resolve_spec
from core.serving_config import ATTENTION_BACKENDS, ENGINES, MOE_BACKENDS
from core.test_attribution import TestStatusDetail


def render_engine_runtime_panel(api_base_url: str) -> None:
    """Inference Engine: 推理引擎接入面板——记录引擎自身的运行（/metrics 轮询 + 启动日志 KV 容量）。"""
    with st.sidebar.expander("Inference Engine: 推理引擎接入（记录引擎运行）", expanded=False):
        st.caption(
            "测试期间轮询 vLLM/SGLang 的 Prometheus `/metrics`，记录引擎 KV cache 占用、"
            "调度队列、抢救数；并解析启动日志拿 KV 容量。留空则按 api_base 自动推导端点。"
        )
        from core.engine_metrics import default_metrics_url

        er = st.session_state.get("engine_runtime") or {}
        if not isinstance(er, dict):
            er = {}

        default_url = default_metrics_url(api_base_url)
        metrics_url = st.text_input(
            "引擎 metrics 端点", value=er.get("metrics_url", "") or default_url,
            help="如 http://gpu-host:8000/metrics", key="er_metrics_url")
        log_path = st.text_input(
            "引擎启动日志路径（可选）", value=er.get("log_path", ""),
            help="vLLM/SGLang 启动 stdout 日志，用于解析 KV 容量/max_num_seqs", key="er_log_path")
        enabled = st.checkbox("启用引擎运行时采集", value=er.get("enabled", True), key="er_enabled")

        st.session_state.engine_runtime = {
            "metrics_url": metrics_url.strip() or None,
            "log_path": log_path.strip() or None,
            "enabled": enabled,
        }


def render_model_spec_panel(model_id: str) -> None:
    """Model Architecture: 模型架构规格面板（自动从注册表预填，可覆盖；用于等效带宽与模型身份证）。"""
    with st.sidebar.expander("Model Architecture: 模型架构规格（等效带宽）", expanded=False):
        st.caption("自动从注册表预填；覆盖后用于等效带宽计算与模型身份证。留空则用注册表默认。")

        spec = resolve_spec(model_id)
        if spec:
            st.caption(f"命中注册表：**{spec.name}** ({spec.architecture})")
        else:
            st.caption("未命中注册表，请手动填写关键字段（尤其激活参数 / 精度）。")

        c1, c2 = st.columns(2)
        with c1:
            total = st.number_input(
                "总参数 (B)", min_value=0.0, value=float(spec.total_params_b or 0) if spec else 0.0,
                step=1.0, key="ms_total_params", help="total_params_b")
            experts = st.number_input(
                "专家数", min_value=0, value=int(spec.num_experts or 0) if spec else 0, step=1,
                key="ms_num_experts")
            ctx = st.number_input(
                "上下文上限", min_value=0, value=int(spec.max_position_embeddings or 0) if spec else 0,
                step=1024, key="ms_max_context", help="max_position_embeddings")
        with c2:
            active = st.number_input(
                "激活参数 (B)", min_value=0.0, value=float(spec.active_params_b or 0) if spec else 0.0,
                step=1.0, key="ms_active_params", help="active_params_b (MoE 每次激活)")
            topk = st.number_input(
                "top-k 专家", min_value=0, value=int(spec.num_experts_per_tok or 0) if spec else 0, step=1,
                key="ms_top_k")
            dtype = st.selectbox(
                "权重精度", ["", "bf16", "fp16", "fp8", "int4", "fp4", "int8"],
                index=(["", "bf16", "fp16", "fp8", "int4", "fp4", "int8"].index(spec.weight_dtype)
                       if spec and spec.weight_dtype in ["bf16", "fp16", "fp8", "int4", "fp4", "int8"] else 0),
                key="ms_weight_dtype")

        cc1, cc2 = st.columns(2)
        with cc1:
            kv_dtype = st.selectbox(
                "KV 精度", ["", "fp16", "fp8", "int4"],
                index=(["", "fp16", "fp8", "int4"].index(spec.kv_dtype)
                       if spec and spec.kv_dtype in ["fp16", "fp8", "int4"] else 0),
                key="ms_kv_dtype", help="KV cache 精度（可与权重精度不同）")
        with cc2:
            multimodal = st.checkbox(
                "多模态", value=bool(spec.is_multimodal) if spec else False, key="ms_multimodal")

        mtp = st.checkbox(
            "支持 MTP / 推测解码", value=bool(spec.supports_mtp) if spec else False,
            key="ms_supports_mtp", help="Multi-Token Prediction / 投机解码")

        # 只收集非零/非空字段作为 override（None 字段交由 resolve_spec 用注册表默认）
        override = {}
        if total > 0:
            override["total_params_b"] = float(total)
        if active > 0:
            override["active_params_b"] = float(active)
        if experts > 0:
            override["num_experts"] = int(experts)
        if topk > 0:
            override["num_experts_per_tok"] = int(topk)
        if ctx > 0:
            override["max_position_embeddings"] = int(ctx)
        if dtype:
            override["weight_dtype"] = dtype
        if kv_dtype:
            override["kv_dtype"] = kv_dtype
        override["is_multimodal"] = bool(multimodal)
        override["supports_mtp"] = bool(mtp)
        st.session_state.model_spec_override = override


def render_serving_config_panel() -> None:
    """Service Config: 服务/启动配置面板（引擎/并行/后端/调度/MTP）。"""
    with st.sidebar.expander("Service Config: 服务/启动配置（引擎/并行/后端）", expanded=False):
        st.caption("记录模型是怎么被服务起来的：引擎版本、TP/DP/EP/PP、注意力/MoE 后端等。")

        sc = st.session_state.get("serving_config", {})
        if not isinstance(sc, dict):
            sc = {}

        engine = st.selectbox(
            "引擎", [""] + ENGINES,
            index=([""] + ENGINES).index(sc.get("engine", "")) if sc.get("engine", "") in ENGINES else 0,
            key="sc_engine")
        engine_version = st.text_input("引擎确切版本", value=sc.get("engine_version", ""), key="sc_engine_version")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tp = st.number_input("TP", min_value=0, value=int(sc.get("tp_size") or 0), step=1, key="sc_tp")
        with c2:
            dp = st.number_input("DP", min_value=0, value=int(sc.get("dp_size") or 0), step=1, key="sc_dp")
        with c3:
            ep = st.number_input("EP", min_value=0, value=int(sc.get("ep_size") or 0), step=1, key="sc_ep")
        with c4:
            pp = st.number_input("PP", min_value=0, value=int(sc.get("pp_size") or 0), step=1, key="sc_pp")

        attn = st.selectbox(
            "注意力后端", [""] + ATTENTION_BACKENDS,
            index=([""] + ATTENTION_BACKENDS).index(sc.get("attention_backend", ""))
            if sc.get("attention_backend", "") in ATTENTION_BACKENDS else 0,
            key="sc_attn", help="如 vLLM VLLM_ATTENTION_BACKEND")
        moe = st.selectbox(
            "MoE 后端", [""] + MOE_BACKENDS,
            index=([""] + MOE_BACKENDS).index(sc.get("moe_backend", ""))
            if sc.get("moe_backend", "") in MOE_BACKENDS else 0,
            key="sc_moe")
        kv_dtype = st.selectbox(
            "KV cache dtype", ["", "auto", "fp16", "fp8"],
            index=(["", "auto", "fp16", "fp8"].index(sc.get("kv_cache_dtype", ""))
                   if sc.get("kv_cache_dtype", "") in ["auto", "fp16", "fp8"] else 0),
            key="sc_kv_dtype")

        cc1, cc2 = st.columns(2)
        with cc1:
            max_len = st.number_input(
                "max_model_len", min_value=0, value=int(sc.get("max_model_len") or 0),
                step=4096, key="sc_max_len")
        with cc2:
            gpu_util = st.number_input(
                "gpu_memory_utilization", min_value=0.0, max_value=1.0, step=0.05,
                value=float(sc.get("gpu_memory_utilization") or 0.0), key="sc_gpu_util")

        chunked = st.checkbox("chunked prefill", value=bool(sc.get("chunked_prefill")), key="sc_chunked")
        prefix = st.checkbox("prefix caching", value=bool(sc.get("prefix_caching")), key="sc_prefix")
        mtp_on = st.checkbox("开启 MTP / 推测解码", value=bool(sc.get("mtp_enabled")), key="sc_mtp_on")
        if mtp_on:
            spec_method = st.text_input("推测方法", value=sc.get("speculative_method", ""),
                                        placeholder="EAGLE3 / ngram / MTP", key="sc_spec_method")
            num_spec = st.number_input("num_speculative_tokens", min_value=0,
                                       value=int(sc.get("num_speculative_tokens") or 0), step=1, key="sc_num_spec")
        else:
            spec_method = ""
            num_spec = 0

        st.session_state.serving_config = {
            "engine": engine, "engine_version": engine_version,
            "tp_size": tp or None, "dp_size": dp or None, "ep_size": ep or None, "pp_size": pp or None,
            "attention_backend": attn, "moe_backend": moe, "kv_cache_dtype": kv_dtype,
            "max_model_len": max_len or None, "gpu_memory_utilization": gpu_util or None,
            "chunked_prefill": chunked, "prefix_caching": prefix,
            "mtp_enabled": mtp_on, "speculative_method": spec_method,
            "num_speculative_tokens": num_spec or None,
        }


def render_test_metadata_panel() -> None:
    """Test Metadata: 测试元数据 & 可对外等级面板（tester/notes/状态/可对外）。"""
    with st.sidebar.expander("Test Metadata: 测试元数据 & 可对外等级", expanded=False):
        st.caption("测试人、备注、状态、可对外等级与下一步动作（数据仓库可筛选/可追溯字段）。")

        tm = st.session_state.get("test_metadata", {})
        if not isinstance(tm, dict):
            tm = {}

        tester = st.text_input("测试人 (tester)", value=tm.get("tester", ""), key="tm_tester")
        status = st.selectbox(
            "状态明细", [""] + [s.value for s in TestStatusDetail],
            index=([""] + [s.value for s in TestStatusDetail]).index(tm.get("status_detail", ""))
            if tm.get("status_detail", "") in [s.value for s in TestStatusDetail] else 0,
            key="tm_status")
        external = st.selectbox(
            "可对外等级 (external_level)", ["internal", "review", "publishable"],
            index=(["internal", "review", "publishable"].index(tm.get("external_level", "internal"))
                   if tm.get("external_level", "internal") in ["internal", "review", "publishable"] else 0),
            key="tm_external",
            help="publishable 需通过四道闸（配置完整/可复现/指标可信/人工复核）")
        next_action = st.text_input("下一步动作 (next_action)", value=tm.get("next_action", ""), key="tm_next")
        comparison_group = st.text_input(
            "对照组 (comparison_group)", value=tm.get("comparison_group", ""),
            help="如 mtp_on / mtp_off 用于对比", key="tm_group")
        supersedes = st.text_input(
            "复测指向 (supersedes_test_id)", value=tm.get("supersedes_test_id", ""), key="tm_supersedes")
        notes = st.text_area("备注 (notes)", value=tm.get("notes", ""), height=60, key="tm_notes")

        st.session_state.test_metadata = {
            "tester": tester, "status_detail": status, "external_level": external,
            "next_action": next_action, "comparison_group": comparison_group,
            "supersedes_test_id": supersedes, "notes": notes,
        }
