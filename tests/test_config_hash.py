"""core.config_hash 单元测试。"""

from __future__ import annotations

from core.config_hash import CONFIG_HASH_FIELDS, compute_config_hash


def test_same_fields_same_hash():
    a = compute_config_hash(model_name="M1", engine="vllm", parallel_strategy="tp8",
                            quantization="fp8", dtype="bf16", max_context=128000)
    b = compute_config_hash(model_name="M1", engine="vllm", parallel_strategy="tp8",
                            quantization="fp8", dtype="bf16", max_context=128000)
    assert a == b


def test_different_model_different_hash():
    a = compute_config_hash(model_name="M1", engine="vllm")
    b = compute_config_hash(model_name="M2", engine="vllm")
    assert a != b


def test_different_engine_different_hash():
    a = compute_config_hash(model_name="M1", engine="vllm")
    b = compute_config_hash(model_name="M1", engine="sglang")
    assert a != b


def test_different_parallel_different_hash():
    a = compute_config_hash(model_name="M1", parallel_strategy="tp8")
    b = compute_config_hash(model_name="M1", parallel_strategy="tp4")
    assert a != b


def test_different_quantization_different_hash():
    a = compute_config_hash(model_name="M1", quantization="fp8")
    b = compute_config_hash(model_name="M1", quantization="bf16")
    assert a != b


def test_none_and_empty_string_same_hash():
    # None 与 "" 规范化一致（保证缺测不误判为不同配置）
    a = compute_config_hash(model_name="M1", quantization=None)
    b = compute_config_hash(model_name="M1", quantization="")
    assert a == b


def test_hash_length_and_hex():
    h = compute_config_hash(model_name="M1")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_concurrency_not_in_hash():
    # concurrency 不算“同配置”字段（同配置可在不同并发测）
    fields = set(CONFIG_HASH_FIELDS)
    assert "concurrency" not in fields
    assert "model_name" in fields and "parallel_strategy" in fields


def test_no_args_still_stable():
    a = compute_config_hash()
    b = compute_config_hash()
    assert a == b  # 全空配置也有稳定 hash
