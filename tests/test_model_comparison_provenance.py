"""A/B rankings require complete results from the same official sample set."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.model_comparator import ModelComparator, ModelConfig
from evaluators.base_evaluator import DatasetUnavailableError, EvaluationResult, SampleResult


def _result(model_id: str, source: str = "configured_dataset", fingerprint: str = "abc"):
    details = [
        SampleResult(
            sample_id="0",
            question="Question 0",
            correct_answer="B",
            model_response="A",
            predicted_answer="A",
            is_correct=model_id == "a",
        ),
        SampleResult(
            sample_id="1",
            question="Question 1",
            correct_answer="B",
            model_response="B",
            predicted_answer="B",
            is_correct=True,
        ),
    ]
    return EvaluationResult(
        dataset_name="mmlu",
        model_id=model_id,
        accuracy=sum(detail.is_correct for detail in details) / len(details),
        total_samples=len(details),
        correct_samples=sum(detail.is_correct for detail in details),
        details=details,
        config={
            "dataset_provenance": {
                "source": source,
                "sample_count": len(details),
                "sample_sha256": fingerprint,
                "few_shot_count": 0,
                "few_shot_sha256": "empty-shot-fingerprint",
                "selection_seed": 42,
            }
        },
    )


def _comparator(tmp_path):
    comparator = ModelComparator(output_dir=str(tmp_path))
    comparator.add_model("a", ModelConfig(model_id="a", api_base_url="https://api.openai.com"))
    comparator.add_model("b", ModelConfig(model_id="b", api_base_url="https://api.openai.com"))
    comparator.results = {"a": {"mmlu": _result("a")}, "b": {"mmlu": _result("b")}}
    return comparator


def test_matching_official_samples_are_compared_by_sample_id(tmp_path):
    comparison = _comparator(tmp_path)._analyze_comparison(["mmlu"])

    assert comparison.datasets["mmlu"].total_samples == 2
    assert comparison.datasets["mmlu"].sample_comparisons[0].expected_answer == "B"
    assert comparison.datasets["mmlu"].sample_comparisons[0].predictions == {
        "a": "A",
        "b": "A",
    }
    assert comparison.summary["overall_winner"] == "a"


def test_demo_and_mismatched_sample_sets_cannot_be_ranked(tmp_path):
    comparator = _comparator(tmp_path)
    comparator.results["b"]["mmlu"] = _result("b", source="embedded_demo")
    with pytest.raises(ValueError, match="verified benchmark provenance"):
        comparator._analyze_comparison(["mmlu"])

    comparator.results["b"]["mmlu"] = _result("b", fingerprint="different")
    with pytest.raises(ValueError, match="different sample sets"):
        comparator._analyze_comparison(["mmlu"])


@pytest.mark.asyncio
async def test_failed_model_evaluation_does_not_publish_ranking(monkeypatch, tmp_path):
    engine = MagicMock()
    engine.run_evaluation = AsyncMock(side_effect=DatasetUnavailableError("official data missing"))
    monkeypatch.setattr("core.quality_evaluator.QualityEvaluator", lambda **kwargs: engine)
    comparator = _comparator(tmp_path)

    with pytest.raises(RuntimeError, match="official data missing"):
        await comparator.run_comparison(["mmlu"])

    assert comparator.comparison_result is None
    assert not list(tmp_path.glob("comparison_*.json"))
