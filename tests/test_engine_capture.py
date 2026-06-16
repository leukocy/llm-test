"""core.engine_capture 单元测试。

覆盖:
- _parse_nondefault_args / _parse_startup_breakdown:vLLM 日志解析(纯函数,canned 日志)。
- capture_engine_config:docker/网络不可用时优雅降级,永不抛异常。
- find_vllm_container:按端口匹配(mock subprocess)。
"""
from __future__ import annotations

from unittest.mock import patch

import core.engine_capture as ec


SAMPLE_LOG = """\
[APIServer pid=1] INFO non-default args: {'model_tag': '/DATA/Model/K', 'tensor_parallel_size': 8, 'decode_context_parallel_size': 8, 'enable_expert_parallel': True, 'gpu_memory_utilization': 0.94, 'max_num_seqs': 64, 'enable_prefix_caching': True, 'max_model_len': 262144}
[Worker pid=629] Loading weights took 148.52 seconds
[Worker pid=629] Model loading took 71.98 GiB memory and 156.39 seconds
[EngineCore pid=424] init engine (profile, create kv cache, warmup model) took 75.49 s (compilation: 16.98 s)
[Worker pid=629] Graph capturing finished in 32 secs, took 1.72 GiB
[EngineCore pid=424] GPU KV cache size: 1,529,216 tokens
[APIServer pid=1] Application startup complete.
"""


def test_parse_nondefault_args():
    args = ec._parse_nondefault_args(SAMPLE_LOG)
    assert args["tensor_parallel_size"] == 8
    assert args["decode_context_parallel_size"] == 8
    assert args["max_num_seqs"] == 64
    assert args["gpu_memory_utilization"] == 0.94
    assert args["max_model_len"] == 262144


def test_parse_nondefault_args_empty():
    assert ec._parse_nondefault_args("no args here") == {}


def test_parse_startup_breakdown():
    b = ec._parse_startup_breakdown(SAMPLE_LOG)
    assert b["weight_load_s"] == 148.52
    assert b["model_load_s"] == 156.39
    assert b["init_engine_s"] == 75.49
    assert b["graph_capture_s"] == 32
    assert b["kv_cache_tokens"] == 1529216  # 去逗号


def test_parse_startup_takes_last_occurrence():
    # 多次启动:取最后一条
    log = "Loading weights took 100.0 seconds\nLoading weights took 148.52 seconds\n"
    assert ec._parse_startup_breakdown(log)["weight_load_s"] == 148.52


def test_find_vllm_container_by_port():
    def fake_run(args, **kw):
        class R:
            returncode = 0
            stdout = "kimi-k27\t0.0.0.0:10814->10814/tcp\nother\t0.0.0.0:80->80/tcp\n"
        return R()
    with patch.object(ec.subprocess, "run", side_effect=fake_run):
        assert ec.find_vllm_container("http://localhost:10814/v1") == "kimi-k27"


def test_capture_engine_config_graceful_no_docker():
    # docker 不可用 + httpx 不可达:不抛异常,返回含 captured_at 的 dict
    with patch.object(ec, "_run", return_value=None), \
         patch.dict("sys.modules", {"httpx": None}):
        result = ec.capture_engine_config("http://127.0.0.1:1/v1")
    assert "captured_at" in result
    assert isinstance(result, dict)


def test_capture_engine_config_no_exception_on_garbage_logs():
    # 日志是垃圾:不抛异常,返回能拿到的部分
    with patch.object(ec, "_run", return_value="garbage no args"):
        with patch.object(ec, "find_vllm_container", return_value="x"):
            result = ec.capture_engine_config("http://127.0.0.1:1/v1", container_name="x")
    assert "captured_at" in result
