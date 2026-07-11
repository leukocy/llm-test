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
    "vllm",
    "sglang",
    "tensorrt-llm",
    "trt-llm",
    "lmdeploy",
    "llama.cpp",
    "tgi",
    "heyi",
    "openai-compatible",
    "unknown",
]

# 常见注意力后端（vLLM: VLLM_ATTENTION_BACKEND；SGLang 类似）
ATTENTION_BACKENDS = [
    "FLASH_ATTN",
    "FLASH_ATTN_MLA",
    "FLASHINFER",
    "TRITON_ATTN",
    "XFORMERS",
    "TORCH_SDPA",
    "CUTLASS",
    "_IVYFD",
    "FLASHINFER_MLA",
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
        known = set(cls.__dataclass_fields__)
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


def from_engine_capture(engine_config: dict[str, Any]) -> ServingConfig:
    """从 engine_capture.capture_engine_config() 的输出构造/补充 ServingConfig。

    填充 sidebar 无法覆盖的字段:torch_version/cuda_version(从容器 env)、
    serving_quant/block_size/enforce_eager(从 vLLM 日志/launch_cmd)、env_flags。
    """
    sc = ServingConfig()
    env = engine_config.get("env") or {}
    # torch/cuda 版本:优先从 container_runtime(docker exec 探测),其次从容器 env
    rt = engine_config.get("container_runtime") or {}
    if rt.get("torch"):
        sc.torch_version = str(rt["torch"])
    if rt.get("cuda"):
        sc.cuda_version = str(rt["cuda"])
    if rt.get("vllm"):
        sc.engine_version = str(rt["vllm"])
    elif rt.get("sglang"):
        sc.engine_version = str(rt["sglang"])
    # 容器 env 兜底
    if not sc.cuda_version:
        for k, v in env.items():
            if k.upper() in ("CUDA_VERSION", "CUDA_HOME_VERSION"):
                sc.cuda_version = str(v)
                break
    sc.env_flags = dict(env)

    # 从 schedule/parallel_strategy 补充
    sched = engine_config.get("schedule") or {}
    if sched.get("max_num_seqs"):
        sc.max_num_seqs = sched["max_num_seqs"]
    if sched.get("gpu_memory_utilization"):
        sc.gpu_memory_utilization = sched["gpu_memory_utilization"]
    if sched.get("max_model_len") and sched["max_model_len"] != -1:
        sc.max_model_len = sched["max_model_len"]
    if "enable_prefix_caching" in sched:
        sc.enable_prefix_caching = bool(sched["enable_prefix_caching"])

    par = engine_config.get("parallel_strategy") or {}
    if par.get("tp"):
        sc.tp_size = par["tp"]
    if par.get("dcp"):
        sc.dp_size = par["dcp"]
    if par.get("ep") is not None:
        sc.ep_size = par["ep"]

    # 从 launch_cmd 解析 vLLM 特有参数
    cmd = engine_config.get("launch_cmd") or ""
    sc.serving_quant = _flag_value(cmd, "quantization") or sc.serving_quant
    sc.kv_cache_dtype = _flag_value(cmd, "kv-cache-dtype") or sc.kv_cache_dtype
    sc.attention_backend = (
        _flag_value(cmd, "attention-backend") or ""
    ).upper() or sc.attention_backend
    sc.moe_backend = _flag_value(cmd, "moe-backend") or sc.moe_backend
    bs = _flag_value(cmd, "block-size")
    if bs:
        sc.block_size = _to_int(bs)
    if "--enforce-eager" in cmd:
        sc.enforce_eager = True

    # 从适配器解析的 backends 字典覆盖(日志解析比正则更可靠)
    backends = engine_config.get("backends") or {}
    if backends.get("quantization"):
        sc.serving_quant = str(backends["quantization"])
    if backends.get("kv_cache_dtype"):
        sc.kv_cache_dtype = str(backends["kv_cache_dtype"])
    if backends.get("attention_backend"):
        sc.attention_backend = str(backends["attention_backend"]).upper()
    if backends.get("moe_backend"):
        sc.moe_backend = str(backends["moe_backend"])
    # block_size / enforce_eager 从 schedule
    if sched.get("block_size"):
        sc.block_size = _to_int(sched["block_size"])
    if sched.get("enforce_eager") is not None:
        sc.enforce_eager = bool(sched["enforce_eager"])

    # MTP/推测解码从 mtp 字典
    mtp = engine_config.get("mtp") or {}
    if mtp.get("mtp_enabled"):
        sc.mtp_enabled = True
    if mtp.get("num_speculative_tokens"):
        sc.num_speculative_tokens = _to_int(mtp["num_speculative_tokens"])
    if mtp.get("speculative_model"):
        sc.draft_model = str(mtp["speculative_model"])
    spec_cfg = mtp.get("speculative_config") or {}
    if spec_cfg.get("method"):
        sc.speculative_method = str(spec_cfg["method"]).upper()
    if spec_cfg.get("num_speculative_tokens") and not sc.num_speculative_tokens:
        sc.num_speculative_tokens = _to_int(spec_cfg["num_speculative_tokens"])

    # 引擎名/版本/镜像
    sc.engine = engine_config.get("engine") or sc.engine
    sc.engine_build = engine_config.get("image") or sc.engine_build
    sc.world_size = _compute_world_size(sc)
    return sc


def _flag_value(cmd: str, flag: str) -> str | None:
    """从命令字符串解析 --flag value 或 --flag=value。"""
    import re

    m = re.search(rf"--{flag}[ =](\S+)", cmd)
    return m.group(1).strip("'\"") if m else None


def merge_serving_configs(base: ServingConfig, override: ServingConfig) -> ServingConfig:
    """用 override 的非空字段覆盖 base 的对应字段(override 优先)。

    布尔字段:override 显式设了 True 或 False 都覆盖 base(只有 None 才跳过)。
    """
    d = base.to_dict()
    # 布尔字段集合(显式 False 也是有效覆盖值)。
    # mtp_enabled 排除:其默认 False 有歧义(未设置 vs 明确关闭),由 from_engine_capture
    # 通过 mtp 字典显式设 True,不走 merge 的布尔覆盖。
    bool_fields = {"enforce_eager", "enable_chunked_prefill", "enable_prefix_caching"}
    for k, v in override.to_dict().items():
        if k in bool_fields:
            if v is not None:
                d[k] = v
        elif k == "mtp_enabled":
            # mtp_enabled 默认 False 有歧义:只在 override 显式 True 时覆盖
            if v is True:
                d[k] = v
        elif v not in (None, "", [], {}):
            d[k] = v
    return ServingConfig.from_dict(d)


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
                            [exe, "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            check=False,
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
