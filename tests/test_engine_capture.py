"""core.engine_capture 单元测试(多引擎适配器架构)。"""
from __future__ import annotations

from unittest.mock import patch

import core.engine_capture as ec
from core.engine_capture import (
    VLLMAdapter, SGLangAdapter, LlamaCppAdapter, KTransformersAdapter,
    detect_adapter, capture_engine_config, get_adapters,
)

SAMPLE_VLLM_LOG = """\
[APIServer] INFO non-default args: {'model_tag': '/m', 'tensor_parallel_size': 8, 'decode_context_parallel_size': 8, 'enable_expert_parallel': True, 'gpu_memory_utilization': 0.94, 'max_num_seqs': 64, 'enable_prefix_caching': True, 'max_model_len': 262144}
[Worker] Loading weights took 148.52 seconds
[Worker] Model loading took 71.98 GiB memory and 156.39 seconds
[EngineCore] init engine (profile, create kv cache, warmup model) took 75.49 s (compilation: 16.98 s)
[Worker] Graph capturing finished in 32 secs
[EngineCore] GPU KV cache size: 1,529,216 tokens
"""


# ---------- 适配器探测 ----------
def test_detect_adapters():
    assert detect_adapter("vllm/vllm-openai:v0.23", "vllm serve /m", None).name == "vLLM"
    assert detect_adapter("lmsysorg/sglang:latest", "python -m sglang", None).name == "SGLang"
    assert detect_adapter("ghcr.io/ggerganov/llama.cpp:server", "llama-server -m x.gguf", None).name == "llama.cpp"
    assert detect_adapter("ktransformers/ktransformers", "ktransformers", None).name == "ktransformers"
    assert detect_adapter("custom/image", "/custom", None) is None


def test_get_adapters_lists_all():
    names = get_adapters()
    assert {"vLLM", "SGLang", "llama.cpp", "ktransformers", "fastllm"} <= set(names)


# ---------- vLLM 适配器 ----------
def test_vllm_parse_logs():
    p = VLLMAdapter.parse_logs(SAMPLE_VLLM_LOG)
    assert p["args"]["max_num_seqs"] == 64
    assert p["args"]["tensor_parallel_size"] == 8
    assert p["args"]["gpu_memory_utilization"] == 0.94
    assert p["runtime"]["weight_load_s"] == 148.52
    assert p["runtime"]["kv_cache_tokens"] == 1529216
    assert p["runtime"]["cold_start_s_est"] == 256.0


def test_vllm_normalize_params():
    p = VLLMAdapter.parse_logs(SAMPLE_VLLM_LOG)
    n = VLLMAdapter.normalize_params("vllm serve /m", p)
    assert n["schedule"]["max_num_seqs"] == 64
    assert n["parallel"]["tp"] == 8
    assert n["parallel"]["dcp"] == 8


def test_vllm_parse_empty():
    assert VLLMAdapter.parse_logs("no args") == {}


# ---------- SGLang 适配器(从 launch_cmd 解析)----------
def test_sglang_normalize_from_cmd():
    cmd = "python -m sglang.launch_server --model-path /m --tp 8 --max-running-requests 64 --mem-fraction-static 0.9"
    n = SGLangAdapter.normalize_params(cmd, {})
    assert n["schedule"]["max_running_requests"] == 64
    assert n["schedule"]["gpu_memory_utilization"] == 0.9
    assert n["parallel"]["tp"] == 8


# ---------- llama.cpp 适配器(-c/-ngl 解析)----------
def test_llamacpp_normalize():
    cmd = "llama-server -m model.gguf -c 8192 -ngl 99 -t 8"
    n = LlamaCppAdapter.normalize_params(cmd, {})
    assert n["schedule"]["context_length"] == 8192
    assert n["schedule"]["gpu_layers"] == 99
    assert n["runtime"]["kv_cache_tokens_est"] == 8192


# ---------- 容器查找 + 优雅降级 ----------
def test_find_container_by_port():
    class R:
        returncode = 0
        stdout = "kimi-k27\t0.0.0.0:10814->10814/tcp\nother\t0.0.0.0:80->80/tcp\n"
    with patch.object(ec.subprocess, "run", return_value=R()):
        assert ec.find_vllm_container("http://localhost:10814/v1") == "kimi-k27"


def test_capture_graceful_no_docker():
    with patch.object(ec, "_run", return_value=None):
        result = capture_engine_config("http://127.0.0.1:1/v1")
    assert "captured_at" in result  # 不抛异常


def test_capture_garbage_logs():
    with patch.object(ec, "_run", return_value="garbage"):
        with patch.object(ec, "find_vllm_container", return_value="x"):
            result = capture_engine_config("http://127.0.0.1:1/v1", container_name="x")
    assert "captured_at" in result
