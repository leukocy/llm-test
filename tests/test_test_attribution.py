"""core.test_attribution 单元测试。"""

from __future__ import annotations

from core.test_attribution import (
    TestStatusDetail,
    derive_bottleneck,
    derive_error_attribution,
    derive_status_detail,
)

# ---------- derive_bottleneck ----------


def test_bottleneck_data_quality_priority():
    insights = ["❌ **Throughput Anomaly**: System throughput is 0."]
    assert derive_bottleneck(insights) == "data_quality"


def test_bottleneck_compute_prefill():
    insights = ["⚠️ **Long-Text Compute Bottleneck**: dropped 40% at longest input."]
    assert derive_bottleneck(insights) == "compute_prefill"


def test_bottleneck_concurrency_saturation():
    insights = ["⚠️ **Overload Degradation**: throughput dropped 30%."]
    assert derive_bottleneck(insights) == "concurrency_saturation"


def test_bottleneck_latency():
    insights = ["⚠️ **High Latency**: P99 TPOT exceeds 500ms."]
    assert derive_bottleneck(insights) == "latency"


def test_bottleneck_memory_bandwidth_from_util():
    insights = ["🚀 **Good**: steady."]
    assert derive_bottleneck(insights, bandwidth_utilization_pct=88) == "memory_bandwidth"


def test_bottleneck_none_when_positive():
    insights = ["🚀 **Blazing Fast Prefill**.", "🏆 **Excellent scaling**."]
    assert derive_bottleneck(insights) is None


def test_bottleneck_empty():
    assert derive_bottleneck([]) is None


# ---------- derive_status_detail ----------


def test_status_abnormal_on_failure():
    assert derive_status_detail(False, []) == TestStatusDetail.ABNORMAL


def test_status_abnormal_on_critical():
    assert derive_status_detail(True, ["❌ **High Latency**: bad"]) == TestStatusDetail.ABNORMAL


def test_status_needs_retest_on_borderline_success_rate():
    s = derive_status_detail(True, ["⚠️ minor"], success_rate=0.8)
    assert s == TestStatusDetail.NEEDS_RETEST


def test_status_abnormal_on_low_success_rate():
    assert derive_status_detail(True, [], success_rate=0.3) == TestStatusDetail.ABNORMAL


def test_status_passed_when_clean():
    assert derive_status_detail(True, [], success_rate=0.99) == TestStatusDetail.PASSED


# ---------- derive_error_attribution ----------


def test_error_attribution_counts_and_classifies():
    results = [
        {"error": "Request timed out after 60s"},
        {"error": "Connection timeout"},
        {"error": "rate limit exceeded (429)"},
        {"error": None},
        {"error": None},
        {"error": "Request timed out again"},
    ]
    attr = derive_error_attribution(results)
    assert attr["count"] == 4
    assert attr["error_type"] == "timeout"
    assert attr["top_type_count"] == 3
    assert attr["error_detail"]


def test_error_attribution_no_failures():
    assert derive_error_attribution([{"error": None}, {}])["count"] == 0


def test_error_attribution_empty():
    assert derive_error_attribution([])["count"] == 0
