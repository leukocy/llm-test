"""core.prometheus_parser 单元测试（用真实的 vLLM /metrics 文本片段）。"""

from __future__ import annotations

import pytest

from core.prometheus_parser import (
    detect_engine_family,
    extract_sglang_runtime,
    extract_vllm_runtime,
    parse_prometheus,
)

VLLM_SAMPLE = """# HELP vllm:gpu_cache_usage_perc Ratio of GPU KV cache used.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.4231
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 8
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 3
# TYPE vllm:num_preemption counter
vllm:num_preemption 12
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",gpu_memory_utilization="0.9",num_gpu_blocks="91750",num_cpu_blocks="0",kv_cache_size_tokens="6150106",kv_cache_max_concurrency="5.865198"} 1.0
# TYPE vllm:gpu_prefix_cache_hits_total counter
vllm:gpu_prefix_cache_hits_total 120
# TYPE vllm:gpu_prefix_cache_queries_total counter
vllm:gpu_prefix_cache_queries_total 200
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} 50
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 100
vllm:time_to_first_token_seconds_count 100
vllm:time_to_first_token_seconds_sum 5.0
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_count 1000
vllm:time_per_output_token_seconds_sum 20.0
"""

SGLANG_SAMPLE = """# TYPE sglang:token_usage gauge
sglang:token_usage 0.55
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs 4
# TYPE sglang:num_queue_req gauge
sglang:num_queue_req 1
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput 2500.5
"""


def test_parse_simple_values_and_labeled():
    parsed = parse_prometheus(VLLM_SAMPLE)
    assert parsed["values"]["vllm:gpu_cache_usage_perc"] == pytest.approx(0.4231)
    assert parsed["values"]["vllm:num_requests_running"] == 8
    assert parsed["values"]["vllm:num_preemption"] == 12
    # labeled cache_config
    cfg = parsed["labeled"]["vllm:cache_config_info"]
    assert cfg[0]["labels"]["num_gpu_blocks"] == "91750"


def test_parse_histogram_aggregation():
    parsed = parse_prometheus(VLLM_SAMPLE)
    h = parsed["histograms"]["vllm:time_to_first_token_seconds"]
    assert h["count"] == 100
    assert h["sum"] == 5.0
    assert h["mean"] == pytest.approx(0.05)
    tpot = parsed["histograms"]["vllm:time_per_output_token_seconds"]
    assert tpot["mean"] == pytest.approx(0.02)


def test_parse_ignores_comments_and_blank():
    parsed = parse_prometheus("# comment\n\n  \n")
    assert parsed["values"] == {}


def test_extract_vllm_runtime_fields():
    parsed = parse_prometheus(VLLM_SAMPLE)
    rt = extract_vllm_runtime(parsed)
    assert rt["gpu_cache_usage_perc"] == pytest.approx(0.4231)
    assert rt["num_requests_running"] == 8
    assert rt["num_requests_waiting"] == 3
    assert rt["num_preemption"] == 12
    assert rt["gpu_prefix_cache_hit_rate"] == pytest.approx(0.6)
    assert rt["ttft_mean_s"] == pytest.approx(0.05)
    assert rt["tpot_mean_s"] == pytest.approx(0.02)
    assert rt["cache_config"]["block_size"] == 16
    assert rt["cache_config"]["num_gpu_blocks"] == 91750
    assert rt["cache_config"]["kv_cache_size_tokens"] == 6150106
    assert rt["cache_config"]["kv_cache_max_concurrency"] == pytest.approx(5.865198)


def test_extract_sglang_runtime_fields():
    parsed = parse_prometheus(SGLANG_SAMPLE)
    rt = extract_sglang_runtime(parsed)
    assert rt["gpu_cache_usage_perc"] == pytest.approx(0.55)
    assert rt["num_requests_running"] == 4
    assert rt["gen_throughput"] == pytest.approx(2500.5)


def test_detect_engine_family():
    assert detect_engine_family(parse_prometheus(VLLM_SAMPLE)) == "vllm"
    assert detect_engine_family(parse_prometheus(SGLANG_SAMPLE)) == "sglang"
    assert detect_engine_family(parse_prometheus("")) == "unknown"


def test_missing_fields_are_none():
    rt = extract_vllm_runtime(parse_prometheus(""))
    assert rt["gpu_cache_usage_perc"] is None
    assert rt["cache_config"]["num_gpu_blocks"] is None
    assert rt["gpu_prefix_cache_hit_rate"] is None
