"""
服务 / 启动配置（ServingConfig）—— “事无巨细”记录“模型是怎么被服务起来的”。

手册强调：同一硬件 + 同一模型，换引擎 / 换并行 / 换后端，性能可能差几倍。
所以每条测试都要冻结：引擎名 + 确切版本、TP/DP/EP/PP、注意力后端、MoE 后端、
serving 量化与 KV dtype、调度参数、MTP/推测解码开关、runtime（CUDA/torch）。

数据来源：sidebar 的“服务/启动配置”面板（用户填写）+ 尽力探测本地引擎版本。
持久化：Phase 4 经 extra_fields 写入 test_runs.serving_config_json。
"""

from __future__ import annotations

import importlib.metadata as md
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

# 已知的推理引擎（与手册 #engine 对齐）
ENGINES = [
    "vllm", "sglang", "tensorrt-llm", "trt-llm", "lmdeploy",
    "llama.cpp", "tgi", "heyi", "openai-compatible", "unknown",
]

# 常见注意力后端（vLLM: VLLM_ATTENTION_BACKEND；SGLang 类似）
ATTENTION_BACKENDS = [
    "FLASH_ATTN", "FLASH_ATTN_MLA", "FLASHINFER", "TRITON_ATTN",
    "XFORMERS", "TORCH_SDPA", "CUTLASS", "_IVYFD", "FLASHINFER_MLA",
]

# MoE 后端
MOE_BACKENDS = ["fused_moe", "triton", "deekseekm1", "flashinfer"]


@dataclass
class ServingConfig:
    """单次测试的服务/启动配置快照。"""

    # 引擎
    engine: str = ""  # vllm / sglang / ...
    engine_version: str = ""
    engine_build: str = ""

    # 并行
    tp_size: int | None = None
    dp_size: int | None = None
    ep_size: int | None = None
    pp_size: int | None = None
    world_size: int | None = None  # 总卡数

    # 后端
    attention_backend: str = ""  # FLASH_ATTN / FLASHINFER / TRITON / ...
    moe_backend: str = ""  # fused_moe / triton / ...

    # serving 量化
    serving_quant: str = ""  # fp8 / awq / gptq / marlin / ""
    kv_cache_dtype: str = ""  # auto / fp8 / fp16 / ""

    # 调度
    max_model_len: int | None = None  # 实际生效的最大上下文
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    enable_chunked_prefill: bool | None = None
    enable_prefix_caching: bool | None = None
    block_size: int | None = None
    enforce_eager: bool | None = None

    # MTP / 推测解码
    mtp_enabled: bool = False
    num_speculative_tokens: int | None = None
    speculative_method: str = ""  # EAGLE / EAGLE3 / ngram / MTP / ""
    draft_model: str = ""
    guided_decoding_backend: str = ""

    # runtime
    cuda_version: str = ""
    torch_version: str = ""
    env_flags: dict[str, Any] = field(default_factory=dict)  # VLLM_ATTENTION_BACKEND=... 等

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ServingConfig:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def from_sidebar(state: dict[str, Any]) -> ServingConfig:
    """从 session_state（sidebar 收集）构造 ServingConfig。

    state 键名约定（sidebar 面板写入）：
    engine, engine_version, tp_size, dp_size, ep_size, pp_size,
    attention_backend, moe_backend, kv_cache_dtype, max_model_len,
    gpu_memory_utilization, max_num_seqs, chunked_prefill, prefix_caching,
    mtp_enabled, num_speculative_tokens, speculative_method, draft_model
    """
    def g(k, default=None):
        v = state.get(k, default)
        return v if v not in ("", None) else default

    sc = ServingConfig(
        engine=g("engine", ""),
        engine_version=g("engine_version", ""),
        tp_size=_to_int(g("tp_size")),
        dp_size=_to_int(g("dp_size")),
        ep_size=g("ep_size"),  # ep 可能需要单独面板
        pp_size=_to_int(g("pp_size")),
        attention_backend=g("attention_backend", ""),
        moe_backend=g("moe_backend", ""),
        kv_cache_dtype=g("kv_cache_dtype", ""),
        max_model_len=_to_int(g("max_model_len")),
        gpu_memory_utilization=_to_float(g("gpu_memory_utilization")),
        max_num_seqs=_to_int(g("max_num_seqs")),
        enable_chunked_prefill=g("chunked_prefill"),
        enable_prefix_caching=g("prefix_caching"),
        mtp_enabled=bool(g("mtp_enabled", False)),
        num_speculative_tokens=_to_int(g("num_speculative_tokens")),
        speculative_method=g("speculative_method", ""),
        draft_model=g("draft_model", ""),
    )
    sc.world_size = _compute_world_size(sc)
    sc.engine_version = sc.engine_version or detect_engine_version(sc.engine)
    return sc


def _compute_world_size(sc: ServingConfig) -> int | None:
    parts = [x for x in (sc.tp_size, sc.dp_size, sc.ep_size, sc.pp_size) if x]
    if not parts:
        return None
    # world_size 一般 = tp*dp*pp（ep 与 tp 重叠时取较大）—— 这里给一个保守的估算
    ws = 1
    for p in (sc.tp_size, sc.dp_size, sc.pp_size):
        if p:
            ws *= p
    return ws


def detect_engine_version(engine: str) -> str:
    """尽力探测本地安装的引擎版本（探测失败返回空串，不抛异常）。"""
    engine = (engine or "").lower()
    if not engine:
        return ""
    # 1) importlib.metadata（pip 安装的包）
    pkg_candidates = {
        "vllm": ["vllm"],
        "sglang": ["sglang", "sgl-kernel"],
        "lmdeploy": ["lmdeploy"],
        "tgi": ["text-generation-inference", "huggingface_tgi"],
    }
    for key, pkgs in pkg_candidates.items():
        if key in engine:
            for pkg in pkgs:
                try:
                    return md.version(pkg)
                except md.PackageNotFoundError:
                    continue
    # 2) CLI --version（llama.cpp 等）
    bin_candidates = {
        "llama.cpp": ["llama-server", "llama-cli", "main"],
    }
    for key, bins in bin_candidates.items():
        if key in engine:
            for b in bins:
                exe = shutil.which(b)
                if exe:
                    try:
                        out = subprocess.run(
                            [exe, "--version"], capture_output=True, text=True, timeout=5, check=False
                        )
                        line = (out.stdout or out.stderr or "").strip().splitlines()
                        if line:
                            return line[0]
                    except (OSError, subprocess.SubprocessError):
                        pass
    return ""


def _to_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
