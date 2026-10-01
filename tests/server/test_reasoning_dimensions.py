"""Versioned reasoning diagnostics persist before checkpoints and retain real zero."""

import json

import pytest

from core.reasoning_assessment import DIMENSIONS, VERSION, assess_reasoning
from evaluators.base_evaluator import SampleResult
from evaluators.gsm8k_evaluator import GSM8KEvaluator
from server.reasoning_reports import reasoning_html, reasoning_markdown, reasoning_summary


def test_missing_reasoning_or_reference_does_not_invent_dimensions():
    empty = assess_reasoning("Q", "", "8", "8", True)
    assert empty["overall"] is None and all(v is None for v in empty["dimensions"].values())
    partial = assess_reasoning("Q", "First add 3 and 5. Therefore 3 + 5 = 8.", "8", "", None)
    assert partial["dimensions"]["correctness"] is None and partial["overall"] is None
    assert all(partial["dimensions"][d] is not None for d in DIMENSIONS if d != "correctness")
    complete = assess_reasoning("Q", "First add 3 and 5. Therefore 3 + 5 = 8.", "8", "8", True)
    assert complete["overall"] == sum(
        complete["dimensions"][d] * complete["weights"][d] for d in DIMENSIONS
    )
    assert complete["input_sha256"] != partial["input_sha256"]


def test_report_uses_common_population_and_never_rescores_legacy_data():
    record = {
        "version": VERSION,
        "source": "local_rule_heuristics",
        "dimensions": dict.fromkeys(DIMENSIONS, 0),
    }
    partial = {
        **record,
        "dimensions": {**record["dimensions"], "correctness": None, "efficiency": 10},
    }
    rows = [
        {"reasoning_assessment": record},
        {"reasoning_assessment": partial},
        {"reasoning_quality_overall": 9},
        {"reasoning_assessment": record, "error": "request failed"},
    ]
    summary = reasoning_summary(rows)
    assert summary["radar_n"] == 1 and summary["ignored"] == 2
    assert summary["dimensions"]["correctness"] == {"label": "正确性代理", "n": 1, "mean": 0}
    assert summary["dimensions"]["efficiency"]["mean"] == 5
    assert summary["figure"]["data"][0]["r"] == [0] * 6
    assert "n=1" in reasoning_markdown("lab", summary)
    assert "<svg" in reasoning_html("<script>", summary) and "<script>" not in reasoning_html(
        "<script>", summary
    )
    assert reasoning_summary([rows[2]])["figure"] is None


@pytest.mark.asyncio
async def test_real_evaluator_batch_persists_reasoning_before_commit_and_reuses_saved_diagnostic():
    evaluator = GSM8KEvaluator(num_shots=0)
    calls = 0

    async def response(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"content": "#### 8", "reasoning_content": "First add 3 and 5. Therefore 3 + 5 = 8."}

    saved = {}

    def commit(index, result):
        saved[index] = result.to_dict()
        assert saved[index]["reasoning_assessment"]["version"] == VERSION

    samples = [{"question": "What is 3 + 5?", "answer": "8", "correct_answer": "8"}]
    results = await evaluator.evaluate_batch(
        samples, response, concurrency=1, commit_callback=commit
    )
    assert calls == 1 and results[0].reasoning_assessment["overall"] is not None
    frozen = json.dumps(saved, sort_keys=True)
    restored = {index: SampleResult(**value) for index, value in saved.items()}
    again = await evaluator.evaluate_batch(samples, response, restored_results=restored)
    assert calls == 1 and json.dumps({0: again[0].to_dict()}, sort_keys=True) == frozen
