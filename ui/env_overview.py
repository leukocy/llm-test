"""环境信息总览页面 — 测试前查看当前静态环境采集结果。

从 sidebar 的 api_base / model_id 出发,自动采集并展示:
  硬件指纹(CPU/内存/GPU/拓扑/磁盘/CUDA)
  系统信息(Python/OS/git/库版本)
  引擎配置(Docker/裸进程/API 三级降级 → TP/quant/backends/MTP/版本)
  服务配置(归一化 ServingConfig)
  模型架构(注册表 + config.json 合并)

带"刷新"按钮重新采集;带"复制 JSON"按钮导出全量。
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


def _safe_capture(fn, *args, **kwargs) -> tuple[Any, str | None]:
    """安全采集:返回 (result, error_msg)。失败不崩。"""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _capture_all(api_base: str, model_id: str | None) -> dict[str, Any]:
    """采集全部静态环境信息。"""
    env: dict[str, Any] = {}

    # 1. 硬件指纹
    from core.hardware_fingerprint import capture_hardware_fingerprint

    fp, err = _safe_capture(capture_hardware_fingerprint)
    env["hardware"] = fp or {}
    if err:
        env["_hardware_error"] = err

    # 2. 系统信息
    from core.system_info import capture_system_info, get_library_versions

    si, err = _safe_capture(capture_system_info)
    env["system"] = si or {}
    if err:
        env["_system_error"] = err

    # 3. 引擎配置
    from core.engine_capture import capture_engine_config, find_vllm_container

    container = find_vllm_container(api_base)
    ec, err = _safe_capture(capture_engine_config, api_base, container_name=container)
    env["engine_config"] = ec or {}
    env["container"] = container
    if err:
        env["_engine_error"] = err

    # 4. 服务配置(归一化)
    from core.serving_config import from_engine_capture

    sc, err = _safe_capture(from_engine_capture, env["engine_config"])
    env["serving_config"] = sc.to_dict() if sc else {}
    if err:
        env["_serving_error"] = err

    # 5. 模型架构(注册表优先 + config.json)
    from core.model_spec import from_local_config, resolve_spec

    model_root = (
        (env["engine_config"].get("model") or {}).get("model_root")
        if isinstance(env["engine_config"].get("model"), dict)
        else None
    )
    # /v1/models 的 root 字段
    if not model_root and env["engine_config"].get("model"):
        model_root = (
            env["engine_config"]["model"].get("model_root")
            if isinstance(env["engine_config"]["model"], dict)
            else None
        )

    base_spec = None
    if model_id:
        base_spec, _ = _safe_capture(resolve_spec, model_id)
    local_spec = None
    if model_root:
        import os

        cfg_path = os.path.join(str(model_root), "config.json")
        if os.path.exists(cfg_path):
            local_spec, _ = _safe_capture(from_local_config, cfg_path)

    if base_spec and local_spec:
        merged = base_spec.to_dict()
        for k, v in local_spec.to_dict().items():
            if merged.get(k) in (None, "", [], {}) and v not in (None, "", [], {}):
                merged[k] = v
        env["model_spec"] = merged
        env["model_spec_source"] = "registry + local_config"
    elif base_spec:
        env["model_spec"] = base_spec.to_dict()
        env["model_spec_source"] = "registry"
    elif local_spec:
        env["model_spec"] = local_spec.to_dict()
        env["model_spec_source"] = "local_config"
    else:
        env["model_spec"] = {}
        env["model_spec_source"] = "none"

    return env


def _render_hardware(hw: dict):
    """渲染硬件指纹。"""
    st.subheader("硬件")
    cpu = hw.get("cpu") or {}
    mem = hw.get("memory") or {}
    gpus = hw.get("gpus") or []
    cuda = hw.get("cuda") or {}
    topo = hw.get("gpu_topology") or {}
    disks = hw.get("disks") or []

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**CPU**")
        st.text(f"{cpu.get('model_name', '?')}")
        st.text(
            f"{cpu.get('sockets', '?')}s × {cpu.get('cores_per_socket', '?')}c × {cpu.get('threads_per_core', '?')}t"
        )
        st.text(f"NUMA: {cpu.get('numa_nodes', '?')} | 逻辑核: {cpu.get('logical_cores', '?')}")
    with col2:
        st.markdown("**内存**")
        st.text(f"{mem.get('total_gb', '?')} GB {mem.get('type', '')}")
        st.text(f"{mem.get('channels', '?')} 通道 {mem.get('speed_mt_s', '?')} MT/s")
        st.text(f"ECC: {mem.get('ecc', '?')}")
    with col3:
        st.markdown("**CUDA / 驱动**")
        st.text(f"CUDA: {cuda.get('cuda_version', '?')}")
        st.text(f"驱动: {cuda.get('driver', '?')}")
        st.text(f"machine_id: {hw.get('machine_id', '?')}")

    if gpus:
        st.markdown("**GPU**")
        g0 = gpus[0]
        st.text(f"{len(gpus)}× {g0.get('name', '?')}")
        st.text(
            f"  {g0.get('vram_gb', '?')} GB | {g0.get('nominal_bandwidth_gbps', '?')} GB/s | PCIe Gen{g0.get('pcie_gen', '?')}x{g0.get('pcie_width', '?')}"
        )
        if topo:
            st.text(
                f"  NVLink: {'Yes' if topo.get('has_nvlink') else 'No'} | 拓扑矩阵: {len(topo.get('matrix', []))} 行"
            )

    if disks:
        st.markdown("**磁盘**")
        for d in disks[:4]:
            st.text(
                f"  {d.get('name', '?')} {d.get('model', '?')} {d.get('size_tb', '?')}TB {'SSD' if d.get('is_ssd') else 'HDD'}"
            )


def _render_system(si: dict):
    """渲染系统信息。"""
    st.subheader("系统 / 运行时")
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"Python: {si.get('python_version', '?')}")
        st.text(f"OS: {si.get('os_name', '?')} {si.get('os_version', '')}")
        st.text(f"hostname: {si.get('hostname', '?')}")
        st.text(f"git: {si.get('git_hash', '?')} | 版本: {si.get('project_version', '?')}")
    with col2:
        lv = si.get("library_versions") or {}
        if lv:
            libs = ", ".join(f"{k}={v}" for k, v in sorted(lv.items()))
            st.text(f"库版本({len(lv)}):")
            st.text(f"  {libs[:120]}")
        fp = si.get("hardware_fingerprint") or {}
        if fp:
            st.text(
                f"hardware_fingerprint: {'已采集' if fp.get('machine_id') else '缺 machine_id'}"
            )


def _render_engine(ec: dict, container: str | None):
    """渲染引擎配置。"""
    st.subheader("引擎")
    sources = ec.get("capture_source") or []
    st.caption(f"采集来源: {', '.join(sources) or '无'} | 容器: {container or '未找到'}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**引擎**")
        st.text(f"名称: {ec.get('engine', '?')}")
        st.text(f"镜像: {(ec.get('image', '') or '?')[-50:]}")
        rt = ec.get("container_runtime") or {}
        st.text(f"vllm: {rt.get('vllm', '?')}")
        st.text(f"torch: {rt.get('torch', '?')}")
        st.text(f"cuda: {rt.get('cuda', '?')}")
    with col2:
        st.markdown("**并行 / 调度**")
        par = ec.get("parallel_strategy") or {}
        st.text(f"TP: {par.get('tp', '?')} | DCP: {par.get('dcp', '?')} | EP: {par.get('ep', '?')}")
        sched = ec.get("schedule") or {}
        st.text(f"max_seqs: {sched.get('max_num_seqs', '?')}")
        st.text(f"gpu_mem: {sched.get('gpu_memory_utilization', '?')}")
        st.text(f"prefix_cache: {sched.get('enable_prefix_caching', '?')}")
    with col3:
        st.markdown("**后端 / 运行时**")
        backends = ec.get("backends") or {}
        st.text(f"quant: {backends.get('quantization', '?')}")
        st.text(f"kv_dtype: {backends.get('kv_cache_dtype', '?')}")
        st.text(f"attn: {backends.get('attention_backend', '?')}")
        st.text(f"moe: {backends.get('moe_backend', '?')}")
        runtime = ec.get("runtime") or {}
        st.text(f"KV tokens: {runtime.get('kv_cache_tokens', '?')}")
        st.text(f"冷启动: {runtime.get('cold_start_s_est', '?')}s")

    mtp = ec.get("mtp") or {}
    if mtp:
        st.markdown(
            f"**MTP**: enabled={mtp.get('mtp_enabled', '?')} tokens={mtp.get('num_speculative_tokens', '?')} method={mtp.get('speculative_config', {}).get('method', '?')}"
        )

    env_vars = ec.get("env") or {}
    if env_vars:
        with st.expander(f"环境变量({len(env_vars)} 个)"):
            for k in sorted(env_vars):
                st.text(f"  {k}={env_vars[k]}")


def _render_serving(sc: dict):
    """渲染服务配置(归一化)。"""
    st.subheader("服务配置 (ServingConfig)")
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"engine: {sc.get('engine', '?')}")
        st.text(
            f"tp: {sc.get('tp_size', '?')} | dcp: {sc.get('dp_size', '?')} | pp: {sc.get('pp_size', '?')}"
        )
        st.text(f"quant: {sc.get('serving_quant', '?')}")
        st.text(f"kv_dtype: {sc.get('kv_cache_dtype', '?')}")
        st.text(f"attn: {sc.get('attention_backend', '?')}")
        st.text(f"moe: {sc.get('moe_backend', '?')}")
    with col2:
        st.text(
            f"mtp: {sc.get('mtp_enabled', '?')} tokens: {sc.get('num_speculative_tokens', '?')}"
        )
        st.text(f"method: {sc.get('speculative_method', '?')}")
        st.text(f"torch: {sc.get('torch_version', '?')}")
        st.text(f"cuda: {sc.get('cuda_version', '?')}")
        st.text(f"block_size: {sc.get('block_size', '?')}")
        st.text(f"env_flags: {len(sc.get('env_flags', {}))} 个")


def _render_model(ms: dict, source: str):
    """渲染模型架构。"""
    st.subheader(f"模型架构(来源: {source})")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text(f"名称: {ms.get('name', '?')}")
        st.text(f"架构: {ms.get('architecture', '?')}")
        st.text(f"注意力: {ms.get('attention_type', '?')}")
        st.text(f"层数: {ms.get('num_layers', '?')}")
        st.text(f"hidden: {ms.get('hidden_size', '?')}")
    with col2:
        st.text(f"总参: {ms.get('total_params_b', '?')}B")
        st.text(f"激活: {ms.get('active_params_b', '?')}B")
        st.text(f"专家: {ms.get('num_experts', '?')} top{ms.get('num_experts_per_tok', '?')}")
        st.text(f"shared: {ms.get('num_shared_experts', '?')}")
        st.text(f"vocab: {ms.get('vocab_size', '?')}")
    with col3:
        st.text(f"精度: {ms.get('weight_dtype', '?')}")
        st.text(f"量化: {ms.get('quant_method', '?')}")
        st.text(f"KV: {ms.get('kv_dtype', '?')}")
        st.text(f"上下文: {ms.get('max_position_embeddings', '?')}")
        st.text(f"MTP: {ms.get('supports_mtp', '?')} depth={ms.get('mtp_depth', '?')}")

    bw = ms.get("bytes_per_param")
    if bw and ms.get("active_params_b"):
        read_gb = ms["active_params_b"] * 1e9 * bw / 1e9
        st.caption(f"每 token 权重读取量(roofline): ~{read_gb:.0f} GB")


def render_env_overview(api_base: str, model_id: str | None):
    """渲染环境信息总览页面。"""
    st.markdown("## 环境信息总览")
    st.caption("测试前查看当前静态环境采集结果。点击「刷新」重新采集。")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("刷新采集", type="primary", key="env_refresh"):
            st.session_state.pop("_env_snapshot", None)
            st.rerun()

    # 采集(缓存到 session_state,避免每次 rerun 都重新采集)
    if "_env_snapshot" not in st.session_state:
        with st.spinner("正在采集环境信息..."):
            st.session_state["_env_snapshot"] = _capture_all(api_base, model_id)

    env = st.session_state["_env_snapshot"]

    # 错误提示
    for k, v in env.items():
        if k.startswith("_") and k.endswith("_error"):
            st.warning(f"{k[1:-6]} 采集失败: {v}")

    # 各维度渲染
    if env.get("hardware"):
        _render_hardware(env["hardware"])
        st.divider()
    if env.get("system"):
        _render_system(env["system"])
        st.divider()
    if env.get("engine_config"):
        _render_engine(env["engine_config"], env.get("container"))
        st.divider()
    if env.get("serving_config"):
        _render_serving(env["serving_config"])
        st.divider()
    if env.get("model_spec"):
        _render_model(env["model_spec"], env.get("model_spec_source", "?"))
        st.divider()

    # 全量 JSON 导出
    with st.expander("全量 JSON (复制用)"):
        st.json(env)
        st.download_button(
            "下载 JSON",
            data=json.dumps(env, ensure_ascii=False, indent=2, default=str),
            file_name="env_snapshot.json",
            mime="application/json",
        )
