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
    """vLLM /metrics 旧版无 kv_cache_size_tokens：num_gpu_blocks×block_size 是 per-rank 值，
    不写入 kv_capacity_tokens（避免 TP>1 误杀），单独放 per_rank_capacity，source 标
    metrics_per_rank。"""
    _install_fake_httpx(lambda *a, **kw: _make_cm(_VLLM_METRICS))
    with patch("core.engine_metrics.default_metrics_url", return_value="http://x/metrics"):
        r = probe_kv_capacity("http://x/v1")
    assert r["kv_capacity_tokens"] is None  # per-rank 值不直接当 budget
    assert r["per_rank_capacity"] == 16 * 32096
    assert r["source"] == "metrics_per_rank"
    assert r["metrics_ok"] is True


def test_probe_prefers_group_aware_capacity_from_new_vllm():
    """Hybrid KV cache must use the explicit group-aware capacity."""
    _install_fake_httpx(lambda *a, **kw: _make_cm(_VLLM_GROUP_AWARE_METRICS))
    with patch("core.engine_metrics.default_metrics_url", return_value="http://x/metrics"):
        r = probe_kv_capacity("http://x/v1")
    assert r["kv_capacity_tokens"] == 6150106
    assert r["kv_capacity_tokens"] != 4 * 44206
    assert r["source"] == "metrics"
    assert r["metrics_ok"] is True


def test_probe_kv_capacity_falls_back_to_models_max_model_len():
    """/metrics 不可达 → /v1/models 的 max_model_len 兜底。"""
    _install_fake_httpx(
        lambda *a, **kw: _make_cm(None, {"data": [{"id": "m", "max_model_len": 131072}]})
    )
    with patch("core.engine_metrics.default_metrics_url", return_value="http://x/metrics"):
        r = probe_kv_capacity("http://x/v1")
    assert r["kv_capacity_tokens"] == 131072
    assert r["source"] == "models"
    assert r["max_model_len"] == 131072


def test_probe_kv_capacity_none_when_both_fail():
    """都不可达 → kv_capacity_tokens=None（调用方回退到全跑，不跳过）。"""
    _install_fake_httpx(lambda *a, **kw: _make_cm(None, None))
    with patch("core.engine_metrics.default_metrics_url", return_value="http://x/metrics"):
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
    """per-rank /metrics 不可靠时：不把 max_model_len 当 budget，但记录供参考。

    source 保持 metrics_per_rank，kv_capacity_tokens 仍为 None（调用方据此不跳过），
    max_model_len 记 131072 供参考。
    """
    _install_fake_httpx(
        lambda *a, **kw: _make_cm(_VLLM_METRICS, {"data": [{"id": "m", "max_model_len": 131072}]})
    )
    with patch("core.engine_metrics.default_metrics_url", return_value="http://x/metrics"):
        r = probe_kv_capacity("http://x/v1")
    assert r["kv_capacity_tokens"] is None  # per-rank 不当 budget，也不回退 models
    assert r["per_rank_capacity"] == 16 * 32096
    assert r["source"] == "metrics_per_rank"
    assert r["max_model_len"] == 131072  # 参考值仍记录


# ---------- _probe_kv_budget 多源编排（BenchmarkRunner 层）----------


def _make_runner(warehouse_context=None):
    """构造一个轻量 BenchmarkRunner，仅用于 _probe_kv_budget / should_skip_cell 测试。"""
    from unittest.mock import MagicMock

    from core.benchmark_runner import BenchmarkRunner

    return BenchmarkRunner(
        placeholder=MagicMock(),
        progress_bar=MagicMock(),
        status_text=MagicMock(),
        api_base_url="http://test/v1",
        model_id="test-model",
        tokenizer_option="字符数 (Fallback)",
        csv_filename="test.csv",
        api_key="test-key",  # pragma: allowlist secret
        log_placeholder=MagicMock(),
        provider="TestProvider",
        warehouse_context=warehouse_context or {},
    )


def test_probe_kv_budget_per_rank_does_not_skip():
    """旧版 vLLM 只给 per-rank 容量时：_kv_budget=None，should_skip_cell 永不跳过。

    回归 bug：per-rank 值低估 TP 全局容量，曾导致高并发大上下文 cell 被误跳过。
    """
    runner = _make_runner(
        {"engine_runtime": {"disable_kv_skip": False}}
    )  # 无手动 kv_budget、无 log_path
    with patch(
        "core.engine_metrics.get_cached_kv_capacity",
        return_value={
            "kv_capacity_tokens": None,
            "source": "metrics_per_rank",
            "per_rank_capacity": 181368,
            "max_model_len": 1048576,
            "metrics_ok": True,
        },
    ):
        runner._probe_kv_budget()
    assert runner._kv_budget is None  # per-rank 不可靠 → 不跳过
    # 高并发大上下文 cell 也不应被跳过
    skip, reason = runner.should_skip_cell(8, 65536, 512)
    assert skip is False
    assert reason == ""


def test_probe_kv_budget_falls_back_to_engine_log():
    """/metrics 缺全局 kv_cache_size_tokens 时，用启动日志的全局 KV 容量作 budget。"""
    runner = _make_runner(
        {"engine_runtime": {"disable_kv_skip": False, "log_path": "/some/engine.log"}}
    )
    with (
        patch(
            "core.engine_metrics.get_cached_kv_capacity",
            return_value={
                "kv_capacity_tokens": None,
                "source": "metrics_per_rank",
                "per_rank_capacity": 181368,
                "max_model_len": 1048576,
                "metrics_ok": True,
            },
        ),
        patch.object(runner, "_parse_engine_log_kv", return_value=1450944),
    ):
        runner._probe_kv_budget()
    assert runner._kv_budget == 1450944
    assert runner._kv_budget_source == "engine_log"
    # budget 来自日志全局值，超预算 cell 仍正常跳过
    skip, reason = runner.should_skip_cell(32, 65536, 512)
    assert skip is True
    assert "engine_log" in reason


def test_probe_kv_budget_global_metrics_preferred_over_log():
    """新版 vLLM 给了全局 kv_cache_size_tokens 时，优先用它，不看日志。"""
    runner = _make_runner(
        {"engine_runtime": {"disable_kv_skip": False, "log_path": "/some/engine.log"}}
    )
    with (
        patch(
            "core.engine_metrics.get_cached_kv_capacity",
            return_value={
                "kv_capacity_tokens": 2017024,
                "source": "metrics",
                "per_rank_capacity": 504256,
                "max_model_len": 1048576,
                "metrics_ok": True,
            },
        ),
        patch.object(runner, "_parse_engine_log_kv", return_value=999) as mock_log,
    ):
        runner._probe_kv_budget()
    assert runner._kv_budget == 2017024
    assert runner._kv_budget_source == "metrics"
    mock_log.assert_not_called()  # 全局值已够，不查日志


def test_probe_kv_budget_disable_skip_runs_all():
    """勾选 disable_kv_skip → 无视预算，跑全部 cell，不探测不跳过（最高优先级）。"""
    runner = _make_runner({"engine_runtime": {"disable_kv_skip": True, "kv_budget": 100000}})
    with patch("core.engine_metrics.get_cached_kv_capacity") as mock_probe:
        runner._probe_kv_budget()
    assert runner._kv_budget is None
    assert runner._kv_budget_source is None
    mock_probe.assert_not_called()  # 禁用时根本不探测
    # 即使给了手动 kv_budget，禁用也优先 → 不跳过
    skip, reason = runner.should_skip_cell(128, 262144, 4096)
    assert skip is False
    assert reason == ""
