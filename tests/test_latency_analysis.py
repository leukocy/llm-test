"""core.latency_analysis 单元测试。"""

from __future__ import annotations

import pytest

from core.latency_analysis import compute_client_vs_engine_latency


def test_overhead_computation():
    results = [
        {"ttft": 0.6, "tpot": 30.0},
        {"ttft": 0.8, "tpot": 35.0},
        {"ttft": 0.7, "tpot": 32.0},
    ]
    engine = {"engine_means": {"ttft_s": 0.3, "tpot_s": 0.025}}  # engine TPOT 25ms
    r = compute_client_vs_engine_latency(results, engine)
    # 客户端 TTFT 中位数 = 0.7，引擎 0.3 → 开销 0.4
    assert r["client_ttft_s"] == pytest.approx(0.7)
    assert r["engine_ttft_s"] == pytest.approx(0.3)
    assert r["ttft_overhead_s"] == pytest.approx(0.4)
    assert r["ttft_overhead_pct"] == pytest.approx(0.4 / 0.3 * 100, abs=0.1)
    # 引擎 TPOT 0.025s = 25ms
    assert r["engine_tpot_ms"] == pytest.approx(25.0)
    assert "开销" in r["verdict"] or "高" in r["verdict"]


def test_missing_engine_means():
    results = [{"ttft": 0.5, "tpot": 20.0}]
    r = compute_client_vs_engine_latency(results, None)
    assert r["client_ttft_s"] == pytest.approx(0.5)
    assert r["engine_ttft_s"] is None
    assert r["ttft_overhead_s"] is None
    assert "不全" in r["verdict"]


def test_low_overhead_verdict():
    results = [{"ttft": 0.31, "tpot": 25.0}]
    engine = {"engine_means": {"ttft_s": 0.30, "tpot_s": 0.025}}
    r = compute_client_vs_engine_latency(results, engine)
    assert "接近" in r["verdict"] or "引擎内部" in r["verdict"]


def test_empty_results():
    r = compute_client_vs_engine_latency([], {"engine_means": {"ttft_s": 0.3}})
    assert r["client_ttft_s"] is None


def test_accepts_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"ttft": [0.6, 0.8, 0.7], "tpot": [30.0, 35.0, 32.0]})
    engine = {"engine_means": {"ttft_s": 0.3, "tpot_s": 0.025}}
    r = compute_client_vs_engine_latency(df, engine)
    assert r["client_ttft_s"] == pytest.approx(0.7)
