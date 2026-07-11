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
    moe_intermediate_size: int | None = None  # MoE 专家的 intermediate size(可与 dense 层不同)
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
    # decode 每 token 权重读取字节数的直接覆盖（字节，非 GB）。
    # 当 active_params_b 不含 attention(DeepSeek/GLM MoE 口径)时,active*bpp 会漏算
    # attention 读取;此时直接写精确值更准。优先于 active*bpp 近似。
    bytes_per_token_read_override: float | None = None

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

        优先用 bytes_per_token_read_override(精确值,含 attention 等 active 未覆盖的部分);
        否则回退到 active_params_b * 1e9 * bytes_per_param 近似。
        MoE 只读激活专家权重；dense 读全部权重。
        """
        if self.bytes_per_token_read_override is not None:
            return self.bytes_per_token_read_override
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
    # GLM-5 / 5.2:MoE + DeepSeek Sparse Attention(DSA)+ MLA,78 层(77 MoE + 1 MTP)。
    # 744B 总参 / 40B 激活(256 专家 top-8 + 1 shared);DSA 即 NSA 的 GLM 实现。
    # 架构字段逐字取自 lukealonso/GLM-5.2-NVFP4/config.json;参数量取自 rtx6kpro wiki。
    # 量化:modelopt_fp4 不是一刀切——routed experts 的 Linear 量化(fp4@0.5),但
    # shared_experts / MLA attention / lm_head / layer 0-2 保留 bf16。
    # active=40B 不含 attention(DeepSeek MoE 口径:仅 routed+shared),但 decode 每 token
    # 实际还读 attention(15B bf16)+ lm_head,故 active*bpp 会漏算。这里直接写精确的
    # bytes_per_token_read_override。
    #   主模型每前向:expert 11.8(fp4) + shared 35.3(bf16) + attn 30.0(bf16) + lm_head 1.9 ≈ 79 GB
    # MTP(DeepSeek-MTP, num_spec=5):draft 1 层,每轮前向 5 次,主模型 1 次验证;
    #   接受率 ~56% → 每轮产出 ~3.8 verified token → 每 verified token 摊薄为 79/3.8 ≈ 20.8 GB
    #   (draft 开销 ~1.3 GB,占比小,含入约 22 GB)
    "glm-5": ModelSpec(
        name="GLM-5",
        family="GLM",
        architecture="moe",
        total_params_b=744,
        active_params_b=40,
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        routed_scaling=2.5,
        num_layers=78,
        hidden_size=6144,
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=192,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        attention_type="mla",
        vocab_size=154880,
        max_position_embeddings=1048576,
        weight_dtype="fp4+bf16(mixed)",
        quant_method="modelopt_fp4",
        group_size=16,
        kv_dtype="fp8",
        bytes_per_token_read_override=22.0e9,  # 混合精度+MTP摊薄,每 verified token ~22 GB
        supports_mtp=True,
        mtp_depth=1,
        mtp_method="DeepSeek-MTP",
    ),
    "glm-5.2": ModelSpec(
        name="GLM-5.2",
        family="GLM",
        architecture="moe",
        total_params_b=744,
        active_params_b=40,
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        routed_scaling=2.5,
        num_layers=78,
        hidden_size=6144,
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=192,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        attention_type="mla",
        vocab_size=154880,
        max_position_embeddings=1048576,
        weight_dtype="fp4+bf16(mixed)",
        quant_method="modelopt_fp4",
        group_size=16,
        kv_dtype="fp8",
        bytes_per_token_read_override=22.0e9,  # 同 glm-5
        supports_mtp=True,
        mtp_depth=1,
        mtp_method="DeepSeek-MTP",
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
    # Tencent Hunyuan Hy3:295B MoE(21B 激活 + 3.8B MTP),首代重构 Hy 架构。
    # 架构字段逐字取自 tencent/Hy3/config.json(Architecture=HYV3ForCausalLM)。
    # 80 层(1 dense + 79 MoE),192 专家 top-8 + 1 shared,routed_scaling=2.826;
    # GQA(64 q-head / 8 kv-head / head_dim 128),ctx 256K,vocab 120832。
    # 官方权重 bf16(无 quantization_config);num_nextn_predict_layers=1 → MTP 单层。
    "hy3": ModelSpec(
        name="Hy3",
        family="Hunyuan",
        architecture="moe",
        total_params_b=295,
        active_params_b=21,
        num_experts=192,
        num_experts_per_tok=8,
        num_shared_experts=1,
        routed_scaling=2.826,
        num_layers=80,
        hidden_size=4096,
        num_attention_heads=64,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=13312,
        moe_intermediate_size=1536,
        attention_type="gqa",
        vocab_size=120832,
        max_position_embeddings=262144,
        weight_dtype="bf16",
        kv_dtype="fp16",
        supports_mtp=True,
        mtp_depth=1,
        mtp_method="Hunyuan-MTP",
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


def _mla_head_dim(cfg: dict) -> int | None:
    """MLA 模型无顶层 head_dim,合成 qk_nope_head_dim + qk_rope_head_dim。"""
    nope = cfg.get("qk_nope_head_dim")
    rope = cfg.get("qk_rope_head_dim")
    if nope is not None and rope is not None:
        return nope + rope
    return None


def _detect_multimodal(cfg: dict) -> tuple[bool, list[str], str]:
    """从 config.json 推断多模态:返回 (is_multimodal, modalities, vision_encoder)。"""
    modalities: list[str] = []
    vision_encoder = ""
    # 显式字段
    if cfg.get("multi_modal_projector") or cfg.get("vision_config") or cfg.get("image_token_id"):
        modalities = ["text", "vision"]
        vc = cfg.get("vision_config") or {}
        vision_encoder = (
            vc.get("model_type") or vc.get("architectures", [""])[0] if isinstance(vc, dict) else ""
        )
    # 架构名暗示
    arch_str = " ".join(str(a).lower() for a in (cfg.get("architectures") or []))
    if "vl" in arch_str or "vision" in arch_str or "visual" in arch_str or "multimodal" in arch_str:
        if "vision" not in modalities:
            modalities = ["text", "vision"]
    is_mm = bool(modalities)
    if not is_mm:
        modalities = ["text"]
    return is_mm, modalities, vision_encoder


def _detect_mtp(cfg: dict) -> tuple[bool, int | None, str]:
    """从 config.json 推断 MTP/推测解码:返回 (supports_mtp, mtp_depth, mtp_method)。

    DeepSeek/GLM MTP:num_nextn_predict_layers > 0。
    EAGLE:architecture 含 'eagle' 或 speculator_config。
    """
    depth = cfg.get("num_nextn_predict_layers")
    if depth and depth > 0:
        return True, depth, "DeepSeek-MTP"
    # EAGLE / speculator
    if cfg.get("speculator_config") or cfg.get("eagle_config"):
        sc = cfg.get("speculator_config") or cfg.get("eagle_config") or {}
        d = sc.get("num_predict_tokens") or sc.get("num_speculative_tokens")
        return True, d, "EAGLE"
    arch_str = " ".join(str(a).lower() for a in (cfg.get("architectures") or []))
    if "eagle" in arch_str:
        return True, None, "EAGLE"
    return False, None, ""


def _detect_attention_type(cfg: dict, arch_list: list) -> str:
    """从 config.json 字段推断 attention 类型:mha/gqa/mla/lightning/dsa。

    优先级:DSA > MLA 结构信号 > Lightning > GQA/MHA 结构信号。
    Lightning 检查在 GQA/MHA 之前,避免被 kv_heads≠attn_heads 误判为 GQA。
    """
    arch_str = " ".join(str(a).lower() for a in arch_list)
    model_type = (cfg.get("model_type") or "").lower()
    # GLM DSA(DeepSeek Sparse Attention)基于 MLA
    if "dsa" in arch_str or "dsa" in model_type:
        return "mla"  # DSA 是 MLA + 稀疏索引,底层仍是 MLA
    # MLA:DeepSeek/GLM 的 qk_nope_head_dim + qk_rope_head_dim 分裂
    if cfg.get("qk_nope_head_dim") is not None or cfg.get("q_lora_rank") is not None:
        return "mla"
    # Lightning(MiniMax):检查 model_type + architectures(在 GQA/MHA 之前,避免误判)
    if "lightning" in model_type or "lightning" in arch_str or "minimax" in model_type:
        return "lightning"
    kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_kv_heads")
    attn_heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    if kv_heads and attn_heads:
        if kv_heads == attn_heads:
            return "mha"
        return "gqa"
    return ""


def _read_quant_config(model_dir: Path, embedded: dict | None = None) -> dict[str, Any]:
    """读取量化配置。优先级:独立 hf_quant_config.json > config.json 内嵌 quantization_config。
    返回 {weight_dtype, quant_method, kv_dtype, group_size} 尽力而为。"""
    out: dict[str, Any] = {}
    qcfg: dict | None = None
    # 1. 独立的 hf_quant_config.json(NVIDIA ModelOpt 风格)
    qpath = model_dir / "hf_quant_config.json"
    if qpath.exists():
        try:
            with open(qpath, encoding="utf-8") as f:
                qcfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            qcfg = None
    # 2. config.json 内嵌的 quantization_config(compressed-tensors / ModelOpt / GPTQ / AWQ)
    if qcfg is None:
        qcfg = embedded if isinstance(embedded, dict) else None
    if qcfg is None:
        return out
    qm = qcfg.get("quant_method") or qcfg.get("quant_algo")
    if qm:
        qm_lower = str(qm).lower()
        # modelopt NVFP4 → weight_dtype=fp4, quant_method=modelopt_fp4
        if "nvfp4" in qm_lower or "fp4" in qm_lower:
            out["weight_dtype"] = "fp4"
            out["quant_method"] = "modelopt_fp4" if "modelopt" in qm_lower else qm
        elif "modelopt" in qm_lower:
            g = qcfg.get("config_groups", {}).get("group_0", {})
            w = g.get("weights", {})
            bits = w.get("num_bits")
            if bits == 4 and w.get("type") == "float":
                out["weight_dtype"] = "fp4"
                out["quant_method"] = "modelopt_fp4"
            elif bits == 8:
                out["weight_dtype"] = "fp8"
                out["quant_method"] = "modelopt_fp8"
            if w.get("group_size"):
                out["group_size"] = w["group_size"]
        elif "compressed" in qm_lower:
            out["quant_method"] = "compressed-tensors"
            g = qcfg.get("config_groups", {}).get("group_0", {})
            w = g.get("weights", {})
            bits = w.get("num_bits")
            wtype = (w.get("type") or "").lower()
            if bits == 4:
                out["weight_dtype"] = "int4" if "int" in wtype else "fp4"
            elif bits == 8:
                out["weight_dtype"] = "int8" if "int" in wtype else "fp8"
            if w.get("group_size"):
                out["group_size"] = w["group_size"]
        else:
            out["quant_method"] = str(qm)
    # KV cache scheme
    kv = qcfg.get("kv_cache_scheme") or {}
    if isinstance(kv, dict):
        kv_bits = kv.get("num_bits")
        kv_type = (kv.get("type") or "").lower()
        if kv_bits == 8 and "float" in kv_type:
            out["kv_dtype"] = "fp8"
        elif kv_bits == 8:
            out["kv_dtype"] = "int8"
    return out


def from_local_config(config_path: str | Path) -> ModelSpec | None:
    """从本地 HF/Ollama 模型的 config.json 推导 ModelSpec。

    读取 transformer 结构字段与官方上下文上限；MoE / 精度字段尽力而为。
    额外读取相邻的 hf_quant_config.json 补充 weight_dtype / quant_method / kv_dtype。
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
        or cfg.get("n_routed_experts")
        or cfg.get("moe_intermediate_size")
        or any("moe" in str(a).lower() for a in arch_list)
    )
    model_dir = path.parent
    quant = _read_quant_config(model_dir, embedded=cfg.get("quantization_config"))
    spec = ModelSpec(
        name=cfg.get("model_type", path.parent.name),
        architecture="moe" if is_moe else "dense",
        num_experts=cfg.get("num_experts") or cfg.get("n_routed_experts"),
        num_experts_per_tok=cfg.get("num_experts_per_tok") or cfg.get("num_selected_experts"),
        num_shared_experts=cfg.get("num_shared_experts") or cfg.get("n_shared_experts"),
        routed_scaling=cfg.get("routed_scaling_factor"),
        num_layers=cfg.get("num_hidden_layers") or cfg.get("n_layers"),
        hidden_size=cfg.get("hidden_size") or cfg.get("n_embd"),
        num_attention_heads=cfg.get("num_attention_heads") or cfg.get("n_head"),
        num_kv_heads=cfg.get("num_key_value_heads"),
        head_dim=cfg.get("head_dim")
        or _mla_head_dim(cfg),  # MLA 无顶层 head_dim,用 qk_nope + qk_rope 合成
        intermediate_size=cfg.get("intermediate_size"),
        moe_intermediate_size=cfg.get("moe_intermediate_size"),
        vocab_size=cfg.get("vocab_size"),
        max_position_embeddings=cfg.get("max_position_embeddings")
        or cfg.get("max_sequence_length")
        or cfg.get("model_max_length"),
        attention_type=_detect_attention_type(cfg, arch_list),
        weight_dtype=quant.get("weight_dtype")
        or (cfg.get("torch_dtype") or "").replace("torch.", ""),
        quant_method=quant.get("quant_method", ""),
        group_size=quant.get("group_size"),
        kv_dtype=quant.get("kv_dtype", ""),
        is_multimodal=_detect_multimodal(cfg)[0],
        modalities=_detect_multimodal(cfg)[1],
        vision_encoder=_detect_multimodal(cfg)[2],
        supports_mtp=_detect_mtp(cfg)[0],
        mtp_depth=_detect_mtp(cfg)[1],
        mtp_method=_detect_mtp(cfg)[2],
    )
    if is_moe and spec.num_experts and spec.num_experts_per_tok:
        # 估算激活参数量(粗略,仅供 roofline,用户/注册表应覆盖):
        # active ≈ dense 部分 + top_k 专家 + shared 专家
        # 每层 MoE 参数 ≈ num_experts_per_tok × moe_intermediate_size × hidden_size × 3(up/down/gate)
        # shared 专家 ≈ 1 × moe_intermediate_size × hidden_size × 3
        # dense 层(attention + 非 MoE MLP)≈ hidden² × 约 12(粗估 q/k/v/o + 2×MLP if dense)
        h = spec.hidden_size or 0
        mi = spec.moe_intermediate_size or spec.intermediate_size or 0
        n_layers = spec.num_layers or 0
        topk = spec.num_experts_per_tok
        n_shared = spec.num_shared_experts or 0
        if h and mi and n_layers:
            moe_per_layer = (topk + n_shared) * mi * h * 3
            # attention 粗估:q/k/v/o 各 h²,加上 MLA 压缩维度修正后约 2×h²
            attn_per_layer = 2 * h * h
            dense_per_layer = attn_per_layer  # 无 dense MLP 时(MoE 层)
            estimated_active = (moe_per_layer + dense_per_layer) * n_layers
            # lm_head + embed
            estimated_active += (spec.vocab_size or 0) * h
            spec.active_params_b = round(estimated_active / 1e9, 1)
    return spec
