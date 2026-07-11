"""core.engine_metrics 单元测试（mock httpx 响应）。"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from core.engine_metrics import EngineMetricsPoller, default_metrics_url

VLLM_METRICS = """# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.40
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 8
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 2
# TYPE vllm:num_preemption counter
vllm:num_preemption 5
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",gpu_memory_utilization="0.9",num_gpu_blocks="1000",num_cpu_blocks="0"} 1.0
"""


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


def _run_poller(url, samples_text, duration=0.25, interval=0.05):
    """构造 poller，mock httpx.Client.get 返回给定 metrics 文本，跑一小段后 stop。"""
    call = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def get(self, url):
            call["n"] += 1
            # 让 num_preemption 随采样递增，验证窗口增量
            text = samples_text.replace(
                "vllm:num_preemption 5", f"vllm:num_preemption {5 + call['n']}"
            )
            return FakeResponse(text)

        def close(self):
            pass

    with patch("httpx.Client", FakeClient):
        p = EngineMetricsPoller(url, interval=interval)
        p.start()
        time.sleep(duration)
        return p.stop()


# ---------- default_metrics_url ----------


def test_default_metrics_url_strips_path():
    assert default_metrics_url("http://localhost:8000/v1") == "http://localhost:8000/metrics"
    assert default_metrics_url("https://gpu-host:443/openai/v1") == "https://gpu-host:443/metrics"
    assert default_metrics_url("localhost:8000") == "http://localhost:8000/metrics"
    assert default_metrics_url(None) is None
    assert default_metrics_url("") is None


# ---------- 轮询与汇总 ----------


def test_poller_captures_peaks_and_timeline():
    summary = _run_poller("http://h/metrics", VLLM_METRICS)
    assert summary["sample_count"] >= 2
    assert summary["engine_family"] == "vllm"
    assert summary["peaks"]["gpu_cache_usage_perc"] == pytest.approx(0.40)
    assert summary["peaks"]["num_requests_running"] == 8
    assert summary["peaks"]["num_requests_waiting"] == 2
    assert summary["cache_config"]["block_size"] == 16
    assert summary["cache_config"]["num_gpu_blocks"] == 1000
    assert summary["cache_config"]["kv_capacity_tokens"] == 16000  # 1000 * 16
    assert len(summary["timeline"]) >= 2


def test_poller_preemption_is_window_delta():
    # interval 钳到最小 0.2s，故 duration 需足够长以拿到 ≥2 个递增样本
    summary = _run_poller("http://h/metrics", VLLM_METRICS, duration=0.6)
    # num_preemption 每次采样 +1，窗口内增量 >= 1
    assert summary["preemption_total"] is not None
    assert summary["preemption_total"] >= 1


def test_poller_no_url_is_noop():
    p = EngineMetricsPoller(None)
    p.start()  # 不启动线程
    summary = p.stop()
    assert summary["sample_count"] == 0
    assert summary["timeline"] == []


def test_poller_unreachable_endpoint_returns_empty():
    class ErrClient:
        def __init__(self, *a, **kw):
            pass

        def get(self, url):
            raise OSError("connection refused")

        def close(self):
            pass

    with patch("httpx.Client", ErrClient):
        p = EngineMetricsPoller("http://dead/metrics", interval=0.05)
        p.start()
        time.sleep(0.2)
        summary = p.stop()
    # 启动了线程但采不到样
    assert summary["sample_count"] == 0
    assert summary["timeline"] == []
