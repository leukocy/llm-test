"""core.publish_gate 单元测试。"""

from __future__ import annotations

from core.publish_gate import LEVEL_BADGE, evaluate_publish_gate, gate_from_run


def _ok(**overrides):
    base = {
        "tester": "alice",
        "machine_id": "host1",
        "has_hardware_fingerprint": True,
        "seed_recorded": True,
        "insights": ["🚀 good"],
        "success_rate": 0.99,
        "has_monitor": True,
        "requested_external_level": "publishable",
    }
    base.update(overrides)
    return evaluate_publish_gate(**base)


def test_all_pass_publishable():
    r = _ok()
    assert r.level == "publishable"
    assert r.passed is True
    assert all(r.gates.values())


def test_review_when_gates_1_3_pass_but_not_human_promoted():
    r = _ok(requested_external_level="internal")
    assert r.level == "review"
    assert r.passed is False


def test_internal_when_config_incomplete():
    r = _ok(tester="")
    assert r.level == "internal"
    assert r.gates["config_complete"] is False
    assert any("tester" in reason for reason in r.reasons)


def test_internal_when_missing_monitor():
    r = _ok(has_monitor=False)
    assert r.level == "internal"
    assert r.gates["metrics_trustworthy"] is False


def test_internal_when_critical_insight():
    r = _ok(insights=["❌ **High Latency**: bad"])
    assert r.level == "internal"


def test_internal_when_low_success_rate():
    r = _ok(success_rate=0.7)
    # 成功率 0.7 < 0.95 → metrics_trustworthy 不过 → internal
    assert r.level == "internal"
    assert r.gates["metrics_trustworthy"] is False


def test_internal_when_not_reproducible_no_seed():
    r = _ok(seed_recorded=False)
    assert r.level == "internal"
    assert r.gates["reproducible"] is False


# ---- CASE 03 红线：缺强制字段不可对外 ----


def _fp(**overrides):
    base = {
        "machine_id": "m1",
        "gpus": [{"name": "H100", "pcie_gen": 5, "pcie_width": 16}],
        "memory": {"type": "DDR5", "channels": 24, "speed_mt_s": 6400},
    }
    base.update(overrides)
    return base


def test_case03_complete_hw_fields_still_publishable():
    r = _ok(hardware_fingerprint=_fp())
    assert r.gates["config_complete"] is True
    assert r.level == "publishable"


def test_case03_missing_pcie_blocks_publishable():
    fp = _fp(gpus=[{"name": "H100"}])  # 缺 pcie_gen/pcie_width
    r = _ok(hardware_fingerprint=fp)
    assert r.gates["config_complete"] is False
    assert r.level == "internal"
    assert any("PCIe" in reason for reason in r.reasons)


def test_case03_missing_memory_channels_blocks_publishable():
    fp = _fp(memory={"type": "DDR5", "speed_mt_s": 6400})  # 缺 channels
    r = _ok(hardware_fingerprint=fp)
    assert r.gates["config_complete"] is False
    assert any("内存通道数" in reason for reason in r.reasons)


def test_case03_not_checked_when_no_fingerprint():
    # 远程 API / 旧调用：不传 hardware_fingerprint → 仅 tester+machine_id，向后兼容
    r = _ok()  # hardware_fingerprint 默认 None
    assert r.gates["config_complete"] is True


def test_never_auto_promote_to_publishable_without_human():
    # 即使前三闸全过，未人工置 publishable → 最多 review
    r = _ok(requested_external_level="review")
    assert r.level == "review"
    assert not r.passed


def test_gate_from_run_convenience():
    run = {
        "tester": "bob",
        "machine_id": "m1",
        "system_info": {"hardware_fingerprint": _fp()},
        "config": {"random_seed": 42},
        "resource_monitor_json": '{"peaks": {}}',
        "external_level": "publishable",
        "success_rate": 0.98,
    }
    r = gate_from_run(run, insights=["🚀 ok"])
    assert r.gates["config_complete"] is True
    assert r.gates["reproducible"] is True
    assert r.level == "publishable"


def test_gate_from_run_case03_blocks_when_required_fields_missing():
    run = {
        "tester": "bob",
        "machine_id": "m1",
        "system_info": {"hardware_fingerprint": {"machine_id": "m1"}},  # 缺 PCIe/通道/频率
        "config": {"random_seed": 42},
        "resource_monitor_json": '{"peaks": {}}',
        "external_level": "publishable",
        "success_rate": 0.98,
    }
    r = gate_from_run(run, insights=["🚀 ok"])
    assert r.gates["config_complete"] is False  # CASE 03 拦截
    assert r.level != "publishable"


# ---- 统计严谨性：小样本高成功率不可信 ----


def test_small_sample_blocks_publishable():
    # 5 样本 100% 成功率——样本量不足，不可信
    r = _ok(success_rate=1.0, sample_size=5)
    assert r.gates["metrics_trustworthy"] is False
    assert r.level == "internal"
    assert any("样本量" in reason for reason in r.reasons)


def test_sufficient_sample_passes():
    # 100 样本 99% 成功率——样本量足够
    r = _ok(success_rate=0.99, sample_size=100)
    assert r.gates["metrics_trustworthy"] is True
    assert r.level == "publishable"


def test_sample_size_none_no_check_back_compat():
    # 不传 sample_size → 不做样本量校验（向后兼容，现有 _ok 默认）
    r = _ok(success_rate=0.99)
    assert r.gates["metrics_trustworthy"] is True


def test_wilson_lower_bound_blocks_small_sample():
    # 5 样本全对（success_count=5, sample_size=5）→ Wilson 下界远低于 95%
    r = _ok(success_rate=1.0, sample_size=5, success_count=5, use_wilson=True)
    assert r.gates["metrics_trustworthy"] is False
    assert any("Wilson" in reason for reason in r.reasons)


def test_wilson_lower_bound_passes_large_sample():
    # 1000 样本 98% 对（success_count=980）→ Wilson 下界 ≈ 0.97，明确 ≥ 95%
    r = _ok(success_rate=0.98, sample_size=1000, success_count=980, use_wilson=True)
    assert r.gates["metrics_trustworthy"] is True
    assert r.level == "publishable"


def test_gate_from_run_passes_sample_size_from_run():
    run = {
        "tester": "bob",
        "machine_id": "m1",
        "system_info": {"hardware_fingerprint": _fp()},
        "config": {"random_seed": 42},
        "resource_monitor_json": '{"peaks": {}}',
        "external_level": "publishable",
        "success_rate": 0.98,
        "total_requests": 3,  # 小样本
    }
    r = gate_from_run(run, insights=["🚀 ok"])
    assert r.gates["metrics_trustworthy"] is False  # 样本量 3 不足
    assert any("样本量" in reason for reason in r.reasons)


def test_level_badge_lookup():
    assert LEVEL_BADGE["publishable"][1] == "green"
    assert LEVEL_BADGE["internal"][1] == "gray"
