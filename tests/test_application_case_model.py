"""core.models.ApplicationCase 单元测试（to_dict / from_row 往返）。"""

from __future__ import annotations

from core.models import ApplicationCase


def _sample_case(**overrides):
    base = {
        "case_id": "c1",
        "source": "manual",
        "scenario": "coding",
        "model_name": "M1",
        "machine_id": "m1",
        "tester": "alice",
        "success": True,
        "quality_score": 8.5,
        "citation_score": 0.9,
        "tool_success_rate": 0.8,
        "retrieval_latency_s": 0.3,
        "ttft_s": 0.4,
        "total_latency_s": 1.2,
        "decode_tps": 42.0,
        "external_level": "review",
        "failure_reason": "oom",
        "extra": {"reasoning_quality_overall": 7.0, "category": "intro"},
    }
    base.update(overrides)
    return ApplicationCase(**base)


def test_to_dict_serializes_extra_and_success():
    d = _sample_case().to_dict()
    assert d["extra_json"] is not None  # extra 非空 → JSON
    assert "extra" not in d  # 原始 extra 已 pop
    assert d["success"] == 1  # bool → int
    assert d["external_level"] == "review"


def test_to_dict_empty_extra_yields_none():
    c = ApplicationCase(case_id="c2", extra={})
    d = c.to_dict()
    assert d["extra_json"] is None
    assert "extra" not in d


def test_round_trip_preserves_fields():
    import json

    c = _sample_case()
    d = c.to_dict()
    # 模拟 DB 行（extra_json 是字符串）
    row = dict(d)
    row["extra_json"] = json.dumps(c.extra, ensure_ascii=False)
    restored = ApplicationCase.from_row(row)
    assert restored.case_id == "c1"
    assert restored.success is True
    assert restored.scenario == "coding"
    assert restored.quality_score == 8.5
    assert restored.citation_score == 0.9
    assert restored.external_level == "review"
    assert restored.extra.get("reasoning_quality_overall") == 7.0


def test_from_row_handles_none_success_and_missing():
    c = ApplicationCase.from_row(
        {"case_id": "c3", "success": None, "external_level": None}
    )
    assert c.success is None
    assert c.external_level == "internal"  # 默认
    assert c.extra == {}


def test_from_row_parses_int_success():
    c = ApplicationCase.from_row({"case_id": "c4", "success": 0})
    assert c.success is False
    c2 = ApplicationCase.from_row({"case_id": "c4b", "success": 1})
    assert c2.success is True


def test_default_case_id_is_uuid():
    c = ApplicationCase()
    assert c.case_id and len(c.case_id) >= 32  # uuid4 字符串
    assert c.external_level == "internal"
    assert c.source == "manual"
