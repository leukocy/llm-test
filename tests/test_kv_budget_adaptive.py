"""自适应 KV 预算测试（core.engine_metrics.probe_kv_capacity / estimate_kv_need）。

覆盖：
- estimate_kv_need 口径与 live_bench.build_phases 一致：conc*(ctx+max_tokens)。
- probe_kv_capacity 从 /metrics cache_config 解析 KV 容量（vLLM/SGLang）。
- probe_kv_capacity /v1/models max_model_len 兜底。
- probe_kv_capacity 不可达端点降级为 None（不抛异常）。
- 优先级：手动 > /metrics > /v1/models。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from core.engine_metrics import estimate_kv_need, probe_kv_capacity


def _install_fake_httpx(cm_factory):
    """probe_kv_capacity 内部 `import httpx` 走 sys.modules，故注入 fake 模块。

    cm_factory(metrics_resp, models_resp) → 一个 context manager，其 __enter__ 返回
    一个 .get(url) 按 URL 返回 /metrics 或 /models 响应的 client。
    """
    fake = MagicMock()
    fake.Client.side_effect = cm_factory
    sys.modules["httpx"] = fake
    return fake


def _restore_httpx():
    sys.modules.pop("httpx", None)


# ---------- estimate_kv_need ----------


def test_estimate_kv_need_formula():
    assert estimate_kv_need(1, 4096, 512) == 4608
    assert estimate_kv_need(8, 65536, 512) == 528384
    assert estimate_kv_need(16, 131072, 1024) == 2113536


def test_estimate_kv_need_zero_concurrency():
    assert estimate_kv_need(0, 4096, 512) == 0


# ---------- probe_kv_capacity: /metrics 路径（vLLM）----------

_VLLM_METRICS = """
# HELP vllm:cache_config_info Cache config information
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",num_gpu_blocks="32096",num_cpu_blocks="0",gpu_memory_utilization="0.9"} 1.0
# HELP vllm:num_requests_running Gauge
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 0
"""

_VLLM_GROUP_AWARE_METRICS = """
# HELP vllm:cache_config_info Cache config information
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="4",num_gpu_blocks="44206",num_cpu_blocks="0",kv_cache_size_tokens="6150106",kv_cache_max_concurrency="5.865198"} 1.0
"""


@pytest.fixture(autouse=True)
def _restore_httpx_after():
    yield
    _restore_httpx()


def _make_cm(metrics_resp: str | None, models_resp: dict | None = None):
    """造一个 httpx.Client() 返回的 context manager。"""
    client = MagicMock()
    resp_m = MagicMock()
    resp_m.status_code = 200 if metrics_resp is not None else 404
    resp_m.text = metrics_resp or ""
    resp_models = MagicMock()
    resp_models.status_code = 200 if models_resp is not None else 404
    resp_models.json.return_value = models_resp or {}

    def _get(url):
        if url.endswith("/metrics"):
            return resp_m
        if url.endswith("/models"):
            return resp_models
        r = MagicMock()
        r.status_code = 404
        return r

    client.get.side_effect = _get
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=client)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_probe_kv_capacity_from_metrics_vllm():
    """vLLM /metrics cache_config: block_size=16 × num_gpu_blocks=32096 = 513536 tokens。"""
    _install_fake_httpx(lambda *a, **kw: _make_cm(_VLLM_METRICS))
    with patch(
        "core.engine_metrics.default_metrics_url", return_value="http://x/metrics"
    ):
        r = probe_kv_capacity("http://x/v1")
    assert r["kv_capacity_tokens"] == 16 * 32096
    assert r["source"] == "metrics"
    assert r["metrics_ok"] is True


def test_probe_prefers_group_aware_capacity_from_new_vllm():
    """Hybrid KV cache must use the explicit group-aware capacity."""
    _install_fake_httpx(lambda *a, **kw: _make_cm(_VLLM_GROUP_AWARE_METRICS))
    with patch(
        "core.engine_metrics.default_metrics_url", return_value="http://x/metrics"
    ):
        r = probe_kv_capacity("http://x/v1")
    assert r["kv_capacity_tokens"] == 6150106
    assert r["kv_capacity_tokens"] != 4 * 44206
    assert r["source"] == "metrics"
    assert r["metrics_ok"] is True


def test_probe_kv_capacity_falls_back_to_models_max_model_len():
    """/metrics 不可达 → /v1/models 的 max_model_len 兜底。"""
    _install_fake_httpx(
        lambda *a, **kw: _make_cm(
            None, {"data": [{"id": "m", "max_model_len": 131072}]}
        )
    )
    with patch(
        "core.engine_metrics.default_metrics_url", return_value="http://x/metrics"
    ):
        r = probe_kv_capacity("http://x/v1")
    assert r["kv_capacity_tokens"] == 131072
    assert r["source"] == "models"
    assert r["max_model_len"] == 131072


def test_probe_kv_capacity_none_when_both_fail():
    """都不可达 → kv_capacity_tokens=None（调用方回退到全跑，不跳过）。"""
    _install_fake_httpx(lambda *a, **kw: _make_cm(None, None))
    with patch(
        "core.engine_metrics.default_metrics_url", return_value="http://x/metrics"
    ):
        r = probe_kv_capacity("http://x/v1")
    assert r["kv_capacity_tokens"] is None
    assert r["source"] is None


def test_probe_kv_capacity_no_exception_on_unreachable():
    """无 httpx / 网络异常时绝不抛（探测失败比中断测试更糟）。"""
    _restore_httpx()  # 确保无 fake httpx，触发 import 失败路径
    fake = MagicMock()
    fake.Client.side_effect = ImportError("no httpx")
    sys.modules["httpx"] = fake
    try:
        r = probe_kv_capacity("http://127.0.0.1:1/v1", timeout=0.5)
    finally:
        _restore_httpx()
    assert r["kv_capacity_tokens"] is None


def test_probe_kv_capacity_metrics_preferred_over_models():
    """/metrics 有值时不用 /v1/models 兜底（source=metrics）。"""
    _install_fake_httpx(
        lambda *a, **kw: _make_cm(
            _VLLM_METRICS, {"data": [{"id": "m", "max_model_len": 131072}]}
        )
    )
    with patch(
        "core.engine_metrics.default_metrics_url", return_value="http://x/metrics"
    ):
        r = probe_kv_capacity("http://x/v1")
    assert r["kv_capacity_tokens"] == 16 * 32096  # 用 /metrics，非 131072
    assert r["source"] == "metrics"
