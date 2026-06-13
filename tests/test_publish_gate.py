"""core.publish_gate 单元测试。"""

from __future__ import annotations

from core.publish_gate import (
    LEVEL_BADGE,
    evaluate_publish_gate,
    gate_from_run,
)


def _ok(**overrides):
    base = dict(
        tester="alice", machine_id="host1", has_hardware_fingerprint=True,
        seed_recorded=True, insights=["🚀 good"], success_rate=0.99,
        has_monitor=True, requested_external_level="publishable",
    )
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


def test_never_auto_promote_to_publishable_without_human():
    # 即使前三闸全过，未人工置 publishable → 最多 review
    r = _ok(requested_external_level="review")
    assert r.level == "review"
    assert not r.passed


def test_gate_from_run_convenience():
    run = {
        "tester": "bob", "machine_id": "m1",
        "system_info": {"hardware_fingerprint": {"machine_id": "m1"}},
        "config": {"random_seed": 42},
        "resource_monitor_json": '{"peaks": {}}',
        "external_level": "publishable",
        "success_rate": 0.98,
    }
    r = gate_from_run(run, insights=["🚀 ok"])
    assert r.gates["config_complete"] is True
    assert r.gates["reproducible"] is True
    assert r.level == "publishable"


def test_level_badge_lookup():
    assert LEVEL_BADGE["publishable"][1] == "green"
    assert LEVEL_BADGE["internal"][1] == "gray"
