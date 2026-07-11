"""core.warehouse.scaling_efficiency 单元测试。"""

from __future__ import annotations

from core.models import TestRun
from core.warehouse.scaling_efficiency import (
    build_scaling_efficiency,
    interpret_efficiency,
    parse_tp_size,
)


def _mk(model_id="M1", parallel="", avg_tps=40.0):
    # 直接构造 TestRun，serving_config 带 engine/tp 供 project_run 解析 parallel_strategy
    run = TestRun(test_type="concurrency", model_id=model_id, avg_tps=avg_tps)
    run.serving_config = {"engine": "vllm"} if parallel else {}
    # parallel_strategy 由 project_run 从 tp_size 等推导；这里直接塞 parallel_strategy 不可，
    # 故改用 serving_config.tp_size（_parallel_strategy 读 tp_size）
    if parallel:
        # parallel 形如 "tp2" / "tp4"
        n = int(parallel[2:])
        run.serving_config = {"engine": "vllm", "tp_size": n}
    return run


def test_parse_tp_size():
    assert parse_tp_size("tp8-dp1-ep1-pp1") == 8
    assert parse_tp_size("tp4") == 4
    assert parse_tp_size("") == 1
    assert parse_tp_size(None) == 1
    assert parse_tp_size("dp2-ep2") == 1  # 无 tp 视为单卡


def test_linear_scaling():
    # tp1=40, tp4=160 → speedup 4.0, efficiency 1.0（线性）
    runs = [_mk("M1", "tp1", 40.0), _mk("M1", "tp4", 160.0)]
    rows = build_scaling_efficiency(runs)
    by_tp = {r["tp_size"]: r for r in rows}
    assert by_tp[1]["speedup_vs_tp1"] == 1.0
    assert by_tp[4]["speedup_vs_tp1"] == 4.0
    assert by_tp[4]["efficiency"] == 1.0


def test_sublinear_scaling():
    # tp1=40, tp4=80 → speedup 2.0, efficiency 0.5（亚线性）
    runs = [_mk("M1", "tp1", 40.0), _mk("M1", "tp4", 80.0)]
    rows = build_scaling_efficiency(runs)
    by_tp = {r["tp_size"]: r for r in rows}
    assert by_tp[4]["speedup_vs_tp1"] == 2.0
    assert by_tp[4]["efficiency"] == 0.5


def test_best_metric_per_tp():
    # 同 tp 多次取 best
    runs = [_mk("M1", "tp1", 40.0), _mk("M1", "tp1", 45.0), _mk("M1", "tp2", 80.0)]
    rows = build_scaling_efficiency(runs)
    by_tp = {r["tp_size"]: r for r in rows}
    assert by_tp[1]["decode_tps"] == 45.0  # best
    # baseline=45, tp2=80 → speedup=80/45
    assert by_tp[2]["speedup_vs_tp1"] == round(80 / 45, 3)


def test_no_tp1_baseline_efficiency_none():
    # 只有 tp2/tp4，无 tp1 → speedup/efficiency None
    runs = [_mk("M1", "tp2", 70.0), _mk("M1", "tp4", 130.0)]
    rows = build_scaling_efficiency(runs)
    assert all(r["speedup_vs_tp1"] is None for r in rows)
    assert all(r["efficiency"] is None for r in rows)


def test_multiple_models_separate():
    runs = [
        _mk("M1", "tp1", 40.0),
        _mk("M1", "tp2", 70.0),
        _mk("M2", "tp1", 30.0),
        _mk("M2", "tp2", 50.0),
    ]
    rows = build_scaling_efficiency(runs)
    models = {r["model_name"] for r in rows}
    assert models == {"M1", "M2"}


def test_interpret_efficiency_thresholds():
    assert "线性" in interpret_efficiency(1.0)
    assert "亚线性" in interpret_efficiency(0.8)
    assert "通信" in interpret_efficiency(0.6)
    assert "很低" in interpret_efficiency(0.3)
    assert "基线" in interpret_efficiency(None)
