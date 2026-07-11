"""core.model_spec / core.serving_config / core.effective_bandwidth 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.effective_bandwidth import compute_effective_bandwidth, summarize_gap
from core.model_spec import MODEL_SPEC_REGISTRY, ModelSpec, from_local_config, resolve_spec
from core.serving_config import ServingConfig, from_sidebar

# ---------- ModelSpec ----------


def test_bytes_per_token_read_dense():
    spec = ModelSpec(
        name="x",
        architecture="dense",
        total_params_b=72,
        active_params_b=72,
        weight_dtype="bf16",
    )
    # 72B * 2 bytes = 144 GB per token
    assert spec.bytes_per_token_read() == pytest.approx(144e9, rel=1e-6)


def test_bytes_per_token_read_moe_uses_active():
    spec = ModelSpec(
        name="x",
        architecture="moe",
        total_params_b=671,
        active_params_b=37,
        weight_dtype="fp8",
    )
    # 37B * 1 byte = 37 GB per token（MoE 只读激活专家）
    assert spec.bytes_per_token_read() == pytest.approx(37e9, rel=1e-6)


def test_bytes_per_token_read_int4():
    spec = ModelSpec(name="x", total_params_b=30, active_params_b=30, weight_dtype="int4")
    assert spec.bytes_per_token_read() == pytest.approx(15e9, rel=1e-6)


def test_resolve_spec_fuzzy_match_and_override():
    spec = resolve_spec("DeepSeek-V3.1", override={"active_params_b": 40})
    assert spec is not None
    assert spec.family == "DeepSeek"
    assert spec.num_experts == 256
    assert spec.supports_mtp is True
    assert spec.active_params_b == 40  # override 覆盖


def test_resolve_spec_unknown_returns_none_without_override():
    assert resolve_spec("totally-unknown-model-xyz") is None


def test_resolve_spec_unknown_with_override_builds_spec():
    spec = resolve_spec("my-custom-model", override={"active_params_b": 10, "weight_dtype": "bf16"})
    assert spec is not None
    assert spec.active_params_b == 10
    assert spec.bytes_per_token_read() == pytest.approx(20e9, rel=1e-6)


def test_registry_has_known_models():
    keys = list(MODEL_SPEC_REGISTRY.keys())
    assert any("deepseek" in k for k in keys)
    assert any("qwen3" in k for k in keys)


def test_model_spec_roundtrip():
    spec = MODEL_SPEC_REGISTRY["qwen3-235b"]
    spec2 = ModelSpec.from_dict(spec.to_dict())
    assert spec2.total_params_b == 235
    assert spec2.active_params_b == 22


def test_from_local_config_parses_hf_config(tmp_path: Path):
    cfg = {
        "model_type": "qwen3",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 94,
        "hidden_size": 5120,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "max_position_embeddings": 262144,
        "torch_dtype": "bfloat16",
        "vocab_size": 151936,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    spec = from_local_config(p)
    assert spec is not None
    assert spec.architecture == "moe"
    assert spec.num_layers == 94
    assert spec.max_position_embeddings == 262144
    assert spec.weight_dtype == "bfloat16"


def test_from_local_config_missing_file_returns_none(tmp_path: Path):
    assert from_local_config(tmp_path / "nope.json") is None


# ---------- ServingConfig ----------


def test_serving_config_from_sidebar_and_world_size():
    state = {
        "engine": "vLLM",
        "tp_size": 4,
        "dp_size": 2,
        "pp_size": 1,
        "attention_backend": "FLASHINFER",
        "moe_backend": "fused_moe",
        "kv_cache_dtype": "fp8",
        "max_model_len": 131072,
        "gpu_memory_utilization": 0.9,
        "chunked_prefill": True,
        "mtp_enabled": True,
        "num_speculative_tokens": 3,
        "speculative_method": "EAGLE3",
    }
    sc = from_sidebar(state)
    assert sc.engine == "vLLM"
    assert sc.tp_size == 4 and sc.dp_size == 2
    assert sc.world_size == 8  # tp*dp*pp
    assert sc.attention_backend == "FLASHINFER"
    assert sc.mtp_enabled is True
    assert sc.num_speculative_tokens == 3


def test_serving_config_roundtrip():
    sc = ServingConfig(engine="sglang", tp_size=8, attention_backend="FLASH_ATTN")
    sc2 = ServingConfig.from_dict(sc.to_dict())
    assert sc2.engine == "sglang"
    assert sc2.tp_size == 8


def test_serving_config_unknown_keys_ignored():
    sc = ServingConfig.from_dict({"engine": "vllm", "bogus_field": 123})
    assert sc.engine == "vllm"


# ---------- effective bandwidth ----------


def test_effective_bandwidth_deepseek():
    spec = resolve_spec("DeepSeek-V3.1")  # 37B active, fp8 → 37 GB/token
    bw = compute_effective_bandwidth(decode_tps=50.0, spec=spec, nominal_gpu_bandwidth_gbps=3350.0)
    # 50 * 37e9 / 1e9 = 1850 GB/s
    assert bw["effective_bandwidth_gbps"] == pytest.approx(1850.0, rel=1e-3)
    assert bw["bandwidth_utilization_pct"] == pytest.approx(1850 / 3350 * 100, rel=1e-3)
    assert bw["bytes_per_token_read"] == pytest.approx(37e9, rel=1e-6)


def test_effective_bandwidth_skips_when_missing():
    assert compute_effective_bandwidth(None, None, None)["effective_bandwidth_gbps"] is None
    spec = resolve_spec("DeepSeek-V3.1")
    # 无 nominal → 有 effective，无 utilization
    bw = compute_effective_bandwidth(50.0, spec, None)
    assert bw["effective_bandwidth_gbps"] is not None
    assert bw["bandwidth_utilization_pct"] is None
    # 无 decode_tps → 全 None
    bw2 = compute_effective_bandwidth(0.0, spec, 3350.0)
    assert bw2["effective_bandwidth_gbps"] is None


def test_summarize_gap_branches():
    spec = resolve_spec("DeepSeek-V3.1")
    hi = compute_effective_bandwidth(150.0, spec, 3350.0)  # 高利用
    assert "上界" in summarize_gap(hi) or "带宽" in summarize_gap(hi)
    lo = compute_effective_bandwidth(5.0, spec, 3350.0)  # 低利用
    assert "偏低" in summarize_gap(lo)
    none_ = compute_effective_bandwidth(None, None, None)
    assert "未知" in summarize_gap(none_)
