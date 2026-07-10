"""
模型架构规格（ModelSpec）—— “事无巨细”记录模型本体的结构化身份证。

覆盖手册要求的全量模型字段：架构(dense/MoE)、总参数/激活参数、MoE(experts/top_k)、
transformer(layers/hidden/heads/kv_heads/head_dim)、上下文上限、是否多模态、
权重精度(BF16/FP8/INT4)、KV精度(fp8_kv)、MTP(是否支持/深度/方法)、tokenizer。

用途：
1. 等效带宽计算（core.effective_bandwidth）—— 需要激活参数 × 每参字节数。
2. 报告 / 数据仓库里“模型这一侧”的全量记录。
3. 跨硬件 × 模型矩阵的分组维度。

数据来源优先级：sidebar 用户覆盖 > 本地模型 config.json 解析 > 预填注册表 > None(跳过)。
注册表数值为公开规格的常用值，可被 from_local_config / 用户覆盖替换。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# 每参数字节数（按权重精度）
BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "fp4": 0.5,
    "nf4": 0.5,
    "awq": 0.5,  # 默认 int4
    "gptq": 0.5,
    "bnb": 0.5,
}


@dataclass
class ModelSpec:
    """模型架构规格（结构化身份证）。"""

    # identity
    name: str = ""
    version: str = ""
    family: str = ""
    source: str = ""
    license: str = ""

    # architecture
    architecture: str = ""  # "dense" | "moe"

    # params
    total_params_b: float | None = None  # 总参数(十亿)
    active_params_b: float | None = None  # 激活参数(十亿)；MoE 每次只激活 top-k 专家

    # MoE
    num_experts: int | None = None
    num_experts_per_tok: int | None = None  # top_k
    num_shared_experts: int | None = None
    routed_scaling: float | None = None

    # transformer
    num_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    attention_type: str = ""  # "mha" | "gqa" | "mla" | ""

    # context
    max_position_embeddings: int | None = None  # 官方上下文上限

    # modality
    is_multimodal: bool = False
    modalities: list[str] = field(default_factory=list)  # ["text","vision","audio","video"]
    vision_encoder: str = ""

    # precision
    weight_dtype: str = ""  # bf16 / fp16 / fp8 / int4 / fp4 ...
    quant_method: str = ""  # gptq / awq / fp8 / bnb / ""
    calibration: str = ""
    group_size: int | None = None

    # KV cache precision（手册强调：KV 精度与权重精度可不同）
    kv_dtype: str = ""  # fp16 / fp8 ...
    kv_cache_dtype: str = ""  # 同 kv_dtype，便于兼容命名
    kv_quant_method: str = ""

    # MTP / 推测解码
    supports_mtp: bool = False
    mtp_depth: int | None = None  # num_predict / num_speculative_tokens
    mtp_method: str = ""  # "DeepSeek-MTP" / "EAGLE" / "EAGLE3" / "ngram" ...

    # tokenizer
    tokenizer_type: str = ""
    tokenizer_vocab: int | None = None

    # 每参数字节数（由 weight_dtype 推导，可被 quant 覆盖）
    bytes_per_param: float | None = None
    # 每 token KV 读取字节的估算系数（可选，用于更精确的 roofline）
    kv_bytes_per_token_factor: float | None = None

    def __post_init__(self) -> None:
        if self.bytes_per_param is None:
            self.bytes_per_param = self._infer_bytes_per_param()

    def _infer_bytes_per_param(self) -> float | None:
        if self.weight_dtype:
            key = self.weight_dtype.lower()
            if key in BYTES_PER_PARAM:
                return BYTES_PER_PARAM[key]
        if self.quant_method:
            key = self.quant_method.lower()
            if key in BYTES_PER_PARAM:
                return BYTES_PER_PARAM[key]
        return None

    def bytes_per_token_read(self) -> float | None:
        """decode 阶段每生成一个 token 需读取的权重字节量(roofline 近似)。

        ≈ active_params_b * 1e9 * bytes_per_param。
        MoE 只读激活专家权重；dense 读全部权重。
        """
        active = self.active_params_b if self.active_params_b is not None else self.total_params_b
        bpp = self.bytes_per_param
        if active is None or bpp is None:
            return None
        return active * 1e9 * bpp

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        known = set(ModelSpec.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# 预填注册表（公开规格常用值；可被 from_local_config / 用户覆盖替换）
# 键名与 config/settings.py HF_MODEL_MAPPING 同口径（小写模糊匹配）。
# ---------------------------------------------------------------------------
MODEL_SPEC_REGISTRY: dict[str, ModelSpec] = {
    "deepseek-v3": ModelSpec(
        name="DeepSeek-V3",
        family="DeepSeek",
        architecture="moe",
        total_params_b=671,
        active_params_b=37,
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        routed_scaling=2.5,
        num_layers=61,
        hidden_size=7168,
        num_attention_heads=128,
        num_kv_heads=128,
        attention_type="mla",
        vocab_size=129280,
        max_position_embeddings=128000,
        weight_dtype="fp8",
        quant_method="fp8",
        kv_dtype="fp8",
        supports_mtp=True,
        mtp_depth=1,
        mtp_method="DeepSeek-MTP",
    ),
    "deepseek-v3.1": ModelSpec(
        name="DeepSeek-V3.1",
        family="DeepSeek",
        architecture="moe",
        total_params_b=671,
        active_params_b=37,
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        num_layers=61,
        hidden_size=7168,
        num_attention_heads=128,
        attention_type="mla",
        vocab_size=129280,
        max_position_embeddings=128000,
        weight_dtype="fp8",
        quant_method="fp8",
        kv_dtype="fp8",
        supports_mtp=True,
        mtp_depth=1,
        mtp_method="DeepSeek-MTP",
    ),
    "deepseek-v3.2": ModelSpec(
        name="DeepSeek-V3.2-Exp",
        family="DeepSeek",
        architecture="moe",
        total_params_b=685,
        active_params_b=37,
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        num_layers=61,
        hidden_size=7168,
        num_attention_heads=128,
        attention_type="mla",
        max_position_embeddings=128000,
        weight_dtype="fp8",
        quant_method="fp8",
        kv_dtype="fp8",
        supports_mtp=True,
        mtp_depth=1,
        mtp_method="DeepSeek-MTP",
    ),
    "qwen3-235b": ModelSpec(
        name="Qwen3-235B-A22B",
        family="Qwen3",
        architecture="moe",
        total_params_b=235,
        active_params_b=22,
        num_experts=128,
        num_experts_per_tok=8,
        num_layers=94,
        hidden_size=5120,
        num_attention_heads=64,
        num_kv_heads=4,
        attention_type="gqa",
        max_position_embeddings=262144,
        weight_dtype="bf16",
        kv_dtype="fp16",
    ),
    "qwen3-30b": ModelSpec(
        name="Qwen3-30B-A3B",
        family="Qwen3",
        architecture="moe",
        total_params_b=30,
        active_params_b=3,
        num_experts=128,
        num_experts_per_tok=8,
        num_layers=48,
        hidden_size=2048,
        num_attention_heads=32,
        num_kv_heads=4,
        attention_type="gqa",
        max_position_embeddings=262144,
        weight_dtype="bf16",
        kv_dtype="fp16",
    ),
    "qwen2.5-72b": ModelSpec(
        name="Qwen2.5-72B",
        family="Qwen2.5",
        architecture="dense",
        total_params_b=72,
        active_params_b=72,
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        attention_type="gqa",
        max_position_embeddings=131072,
        weight_dtype="bf16",
        kv_dtype="fp16",
    ),
    "kimi-k2": ModelSpec(
        # 架构字段逐字取自 Kimi-K2.7-Code/config.json 的 text_config（权威源）；
        # total/active 取 Moonshot 公开规格（1T 总参 / 32B 激活，与 384 专家 top-8 自洽）。
        # 注：K2.5/K2.6/K2.7 是混合精度量化(compressed-tensors)：routed experts=int4,
        # 但 attention/shared_experts/dense/mlp_gate/lm_head=bf16(在 quantization ignore 列表)。
        # 故 bytes_per_param ≠ 纯 int4 的 0.5 —— 有效约 0.91(73% int4@0.5 + 27% bf16@2.0)。
        name="Kimi-K2/K2.5/K2.7",
        family="Moonshot",
        architecture="moe",
        total_params_b=1000,
        active_params_b=32,
        num_experts=384,
        num_experts_per_tok=8,
        num_shared_experts=1,
        routed_scaling=2.827,
        num_layers=61,
        hidden_size=7168,
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=128,
        intermediate_size=18432,
        attention_type="mla",
        vocab_size=163840,
        max_position_embeddings=262144,
        weight_dtype="int4+bf16(mixed)",
        quant_method="compressed-tensors",
        group_size=32,
        bytes_per_param=0.91,  # 混合精度有效值(非纯int4 0.5)
        kv_dtype="bf16",
        is_multimodal=True,
        modalities=["text", "vision"],
        # num_nextn_predict_layers=0 → 该 checkpoint 未启用 MTP
        supports_mtp=False,
    ),
    "glm-4.5": ModelSpec(
        name="GLM-4.5",
        family="GLM",
        architecture="moe",
        total_params_b=355,
        active_params_b=32,
        num_experts=256,
        num_experts_per_tok=32,  # 含 shared
        num_layers=61,
        hidden_size=7168,
        num_attention_heads=64,
        num_kv_heads=8,
        attention_type="gqa",
        max_position_embeddings=131072,
        weight_dtype="bf16",
        kv_dtype="fp16",
    ),
    "glm-z1": ModelSpec(
        name="GLM-Z1",
        family="GLM",
        architecture="moe",
        total_params_b=355,
        active_params_b=32,
        num_layers=61,
        hidden_size=7168,
        attention_type="gqa",
        max_position_embeddings=131072,
        weight_dtype="bf16",
        kv_dtype="fp16",
    ),
    "llama-3.1-405b": ModelSpec(
        name="Llama-3.1-405B",
        family="Llama",
        architecture="dense",
        total_params_b=405,
        active_params_b=405,
        num_layers=126,
        hidden_size=16384,
        num_attention_heads=128,
        num_kv_heads=8,
        attention_type="gqa",
        max_position_embeddings=131072,
        weight_dtype="bf16",
        kv_dtype="fp16",
    ),
    "minimax-m3": ModelSpec(
        name="MiniMax-M3",
        family="MiniMax",
        architecture="moe",
        total_params_b=245,
        active_params_b=10,
        num_layers=62,
        hidden_size=5120,
        attention_type="lightning",
        max_position_embeddings=1000000,
        weight_dtype="bf16",
        kv_dtype="fp16",
    ),
}


def _normalize_key(model_id: str) -> str:
    return (model_id or "").lower().strip()


def resolve_spec(model_id: str, override: dict[str, Any] | None = None) -> ModelSpec | None:
    """按 model_id 模糊匹配注册表；override 中的字段覆盖命中规格。未命中返回 None。"""
    key = _normalize_key(model_id)
    if not key:
        return None

    base: ModelSpec | None = None
    # 精确子串匹配（键名通常是 'deepseek-v3.1' 这类）
    for reg_key, spec in MODEL_SPEC_REGISTRY.items():
        if reg_key in key or key in reg_key:
            base = spec
            break

    if base is None:
        # 没有命中注册表：若 override 给了关键架构字段，仍构造一个（供用户自填）
        if override and any(
            override.get(k) for k in ("active_params_b", "total_params_b", "weight_dtype")
        ):
            base = ModelSpec(name=model_id)
        else:
            return None

    merged = base.to_dict()
    merged["name"] = base.name or model_id
    if override:
        for k, v in override.items():
            if v not in (None, "", []):
                merged[k] = v
    return ModelSpec.from_dict(merged)


def from_local_config(config_path: str | Path) -> ModelSpec | None:
    """从本地 HF/Ollama 模型的 config.json 推导 ModelSpec。

    读取 transformer 结构字段与官方上下文上限；MoE / 精度字段尽力而为（config.json 不一定全有）。
    """
    path = Path(config_path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    arch_list = cfg.get("architectures") or []
    is_moe = bool(
        cfg.get("num_experts")
        or cfg.get("moe_intermediate_size")
        or any("moe" in str(a).lower() for a in arch_list)
    )
    spec = ModelSpec(
        name=cfg.get("model_type", path.parent.name),
        architecture="moe" if is_moe else "dense",
        num_experts=cfg.get("num_experts") or cfg.get("n_routed_experts"),
        num_experts_per_tok=cfg.get("num_experts_per_tok") or cfg.get("num_selected_experts"),
        num_shared_experts=cfg.get("num_shared_experts") or cfg.get("n_shared_experts"),
        num_layers=cfg.get("num_hidden_layers") or cfg.get("n_layers"),
        hidden_size=cfg.get("hidden_size") or cfg.get("n_embd"),
        num_attention_heads=cfg.get("num_attention_heads") or cfg.get("n_head"),
        num_kv_heads=cfg.get("num_key_value_heads"),
        head_dim=cfg.get("head_dim"),
        intermediate_size=cfg.get("intermediate_size"),
        vocab_size=cfg.get("vocab_size"),
        max_position_embeddings=cfg.get("max_position_embeddings")
        or cfg.get("max_sequence_length")
        or cfg.get("model_max_length"),
        weight_dtype=(cfg.get("torch_dtype") or "").replace("torch.", ""),
    )
    if is_moe and spec.num_experts and spec.num_experts_per_tok:
        # 估算激活参数比例（粗略，仅供 roofline，用户应覆盖）
        pass
    return spec
