"""core.engine_log_parser 单元测试（用真实的 vLLM 启动日志片段）。"""

from __future__ import annotations

import pytest

from core.engine_log_parser import parse_engine_log, parse_engine_log_file


VLLM_LOG = """INFO 06-13 12:00:01 model_runner.py:123] Loading model took 15.23 seconds
INFO 06-13 12:00:02 weight_loader.py:45] Loading weights took 32.45 GiB and 12.34 seconds
INFO 06-13 12:00:03 llm_engine.py:200] # GPU blocks: 91,750
INFO 06-13 12:00:03 llm_engine.py:200] # CPU blocks: 0
INFO 06-13 12:00:03 llm_engine.py:201] Maximum concurrency for 131072 tokens per request: 2.54x | CPU-KV cache size: 0 | GPU-KV cache size: 89,600.00 MiB
INFO 06-13 12:00:03 llm_engine.py:202] KV cache size: 1468000 tokens
INFO 06-13 12:00:04 llm_engine.py:300] Maximum number of sequences (max_num_seqs = 256)
INFO 06-13 12:00:04 config.py:50] max_model_len = 131072, block_size = 16
"""

SGLANG_LOG = """[sglang] max_total_tokens=1048576, context_len=131072
[sglang] mem pool size: 16384
[sglang] model loading took 8.5 seconds
"""


def test_parse_vllm_log_all_fields():
    r = parse_engine_log(VLLM_LOG)
    assert r["engine"] == "vllm"
    assert r["num_gpu_blocks"] == 91750
    assert r["num_cpu_blocks"] == 0
    assert r["block_size"] == 16
    assert r["max_num_seqs"] == 256
    assert r["max_model_len"] == 131072
    assert r["kv_cache_size_tokens"] == 1468000
    # 89,600 MiB → 87.5 GiB
    assert r["kv_cache_size_gib"] == pytest.approx(89600 / 1024)
    assert r["weight_load_seconds"] == pytest.approx(12.34)
    assert r["model_load_seconds"] == pytest.approx(15.23)


def test_parse_derives_kv_tokens_when_only_blocks():
    log = """INFO # GPU blocks: 1000
INFO block_size = 16
"""
    r = parse_engine_log(log)
    assert r["kv_cache_size_tokens"] == 16000  # 1000 * 16
    assert r.get("kv_cache_tokens_derived") is True


def test_parse_sglang_engine_detected():
    r = parse_engine_log(SGLANG_LOG)
    assert r["engine"] == "sglang"
    assert r["model_load_seconds"] == pytest.approx(8.5)


def test_parse_empty_log():
    r = parse_engine_log("")
    assert "engine" not in r  # 无引擎标识
    assert "num_gpu_blocks" not in r


def test_parse_file_missing(tmp_path):
    assert parse_engine_log_file(tmp_path / "nope.log") == {}


def test_parse_file_reads(tmp_path):
    p = tmp_path / "engine.log"
    p.write_text(VLLM_LOG, encoding="utf-8")
    r = parse_engine_log_file(p)
    assert r["num_gpu_blocks"] == 91750
    assert r["engine"] == "vllm"


def test_last_match_wins_on_repeated_field():
    log = "INFO max_num_seqs = 128\nINFO max_num_seqs = 256\n"
    r = parse_engine_log(log)
    assert r["max_num_seqs"] == 256
