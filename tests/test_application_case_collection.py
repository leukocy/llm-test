"""quality_evaluator 自动采集映射（纯函数 _build_case_from_sample）单元测试。"""

from __future__ import annotations

from core.quality_evaluator import _build_case_from_sample
from evaluators.base_evaluator import SampleResult


def _sample(**kw):
    base = {
        "sample_id": "q1",
        "question": "q",
        "correct_answer": "A",
        "model_response": "A",
        "predicted_answer": "A",
        "is_correct": True,
    }
    base.update(kw)
    return SampleResult(**base)


def test_mapping_success_and_performance():
    s = _sample(
        is_correct=True,
        ttft_ms=300.0,
        total_time_ms=1200.0,
        tps=42.0,
        input_tokens=10,
        output_tokens=20,
    )
    c = _build_case_from_sample(s, "humaneval", "M1", "2026-06-14")
    assert c.success is True
    assert c.ttft_s == 0.3
    assert c.prefill_latency_s == 0.3
    assert c.total_latency_s == 1.2
    assert c.decode_tps == 42.0
    assert c.input_tokens == 10 and c.output_tokens == 20


def test_application_quality_fields_left_none():
    """evaluator 不产出的字段必须留 None（手册诚实口径）。"""
    s = _sample()
    c = _build_case_from_sample(s, "humaneval", "M1", "2026-06-14")
    assert c.quality_score is None
    assert c.citation_score is None
    assert c.tool_success_rate is None
    assert c.retrieval_latency_s is None


def test_scenario_mapping_per_evaluator():
    assert (
        _build_case_from_sample(_sample(), "humaneval", "M", "d").scenario == "coding"
    )
    assert (
        _build_case_from_sample(_sample(), "longbench", "M", "d").scenario == "long_doc"
    )
    assert (
        _build_case_from_sample(_sample(), "needle_haystack", "M", "d").scenario
        == "retrieval"
    )
    assert (
        _build_case_from_sample(_sample(), "arena_hard", "M", "d").scenario
        == "dialogue"
    )
    assert (
        _build_case_from_sample(_sample(), "mmlu", "M", "d").scenario == "knowledge_qa"
    )


def test_case_id_format_and_metadata():
    s = _sample(sample_id="abc")
    c = _build_case_from_sample(
        s,
        "humaneval",
        "DeepSeek-V3",
        "2026-06-14",
        tester="alice",
        machine_id="m1",
        engine="vllm",
        engine_version="0.6",
    )
    assert c.case_id == "DeepSeek-V3:humaneval:abc"
    assert c.source == "auto"
    assert c.evaluator_name == "humaneval"
    assert c.sample_id == "abc"
    assert c.tester == "alice"
    assert c.machine_id == "m1"
    assert c.engine == "vllm"
    assert c.external_level == "internal"
    assert c.extra.get("engine_version") == "0.6"


def test_failure_reason_from_failure_category():
    s = _sample(is_correct=False, failure_category="concept_error")
    c = _build_case_from_sample(s, "mmlu", "M", "d")
    assert c.success is False
    assert c.failure_reason == "concept_error"


def test_failure_reason_falls_back_to_error():
    s = _sample(is_correct=False, error="timeout")
    c = _build_case_from_sample(s, "mmlu", "M", "d")
    assert c.failure_reason == "timeout"
