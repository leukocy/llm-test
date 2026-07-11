"""core.application_scenario 单元测试。"""

from __future__ import annotations

from core.application_scenario import (
    APPLICATION_SCENARIOS,
    is_application_scenario,
    scenario_from_dataset,
)


def test_coding_scenarios():
    for name in (
        "humaneval",
        "HumanEval",
        "human_eval",
        "mbpp",
        "swebench_lite",
        "sw_bench",
    ):
        assert scenario_from_dataset(name) == "coding", name


def test_long_doc_and_retrieval_and_dialogue():
    assert scenario_from_dataset("longbench") == "long_doc"
    assert scenario_from_dataset("long_bench") == "long_doc"
    assert scenario_from_dataset("needle_haystack") == "retrieval"
    assert scenario_from_dataset("custom_needle") == "retrieval"
    assert scenario_from_dataset("arena_hard") == "dialogue"


def test_knowledge_qa_is_not_application():
    for name in (
        "mmlu",
        "gsm8k",
        "math500",
        "aime2025",
        "arc",
        "hellaswag",
        "winogrande",
        "truthfulqa",
        "gpqa",
        "global_piqa",
    ):
        assert scenario_from_dataset(name) == "knowledge_qa", name
        assert not is_application_scenario(scenario_from_dataset(name))


def test_is_application_scenario():
    for s in ("coding", "long_doc", "retrieval", "dialogue", "agent"):
        assert is_application_scenario(s) is True
    assert is_application_scenario("knowledge_qa") is False
    assert is_application_scenario("other") is False
    assert is_application_scenario(None) is False


def test_unknown_defaults_to_other():
    assert scenario_from_dataset("totally_unknown_eval") == "other"
    assert scenario_from_dataset("") == "other"
    assert scenario_from_dataset(None) == "other"


def test_application_scenarios_frozenset():
    assert isinstance(APPLICATION_SCENARIOS, frozenset)
    assert "coding" in APPLICATION_SCENARIOS
