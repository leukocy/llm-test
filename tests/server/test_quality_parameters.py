"""Quality parity options reach the engine and preserve scoring provenance."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from core.cancel_state import reset_all
from core.quality_evaluator import QualityEvaluator, QualityTestConfig
from evaluators.base_evaluator import BaseEvaluator, EvaluationResult
from server.quality_catalog import GROUPS, quality_catalog
from server.runner_adapter import execute_job
from server.settings import Endpoint, Settings
from server.specs import QualitySpec, expected_requests
from server.store import JobStore


def test_full_sampling_is_explicit_null_and_has_unknown_initial_request_count():
    spec = QualitySpec(
        datasets=["mmlu", "gsm8k"],
        max_samples=None,
        model_type="thinking",
        thinking_enabled=True,
        use_llm_judge=True,
        thinking_budget=4096,
        max_tokens=131072,
    )
    config = QualityTestConfig(**spec.model_dump())
    assert config.max_samples is None
    assert config.to_dict()["use_llm_judge"] is True
    assert config.thinking_enabled and config.model_type == "thinking"
    assert expected_requests("quality", spec.model_dump()) == 0


def test_medium_and_multiple_dataset_sampling_no_longer_has_1000_total_limit():
    names = [item["id"] for item in quality_catalog() if item["available"]][:8]
    spec = QualitySpec(datasets=names, max_samples=500)
    assert expected_requests("quality", spec.model_dump()) == 4000
    with pytest.raises(ValidationError):
        QualitySpec(datasets=names + names, max_samples=100)
    with pytest.raises(ValidationError):
        QualitySpec(datasets=names, max_samples=10001)


def test_initial_eighteen_dataset_controls_remain_distinct_from_registration():
    assert sum(map(len, GROUPS.values())) == 18
    catalog = quality_catalog()
    assert len({item["id"] for item in catalog}) == len(catalog)
    assert next(item for item in catalog if item["id"] == "ceval")["available"] is True


class SyntheticEvaluator(BaseEvaluator):
    def load_dataset(self, subset=None):
        return []

    def format_prompt(self, sample, include_answer=False):
        return sample["question"]

    def parse_response(self, response):
        return response

    def check_answer(self, predicted, correct):
        return predicted == correct


@pytest.mark.parametrize(
    "verdict,correct,method",
    [
        ("YES", True, "llm_judge_passed"),
        (" yes\n", True, "llm_judge_passed"),
        ("NO", False, "llm_judge_rejected"),
        ("NOT YES", False, "llm_judge_invalid"),
        ("YES but actually NO", False, "llm_judge_invalid"),
        ("", False, "llm_judge_invalid"),
    ],
)
def test_judge_requires_complete_verdict_and_preserves_standard_score(verdict, correct, method):
    evaluator = SyntheticEvaluator(dataset_name="synthetic", dataset_path="unused", num_shots=0)
    evaluator.use_llm_judge = True
    responses = iter(["wrong", verdict])

    async def response(*args, **kwargs):
        return {"content": next(responses)}

    sample = asyncio.run(
        evaluator.evaluate_single({"id": "one", "question": "1+1?", "answer": "2"}, response)
    )
    assert sample.is_correct is correct
    assert sample.evaluation_method == method
    assert sample.judge_verdict == verdict.strip().upper()
    assert sample.error is None
    report = EvaluationResult(
        dataset_name="synthetic",
        model_id="test",
        accuracy=float(correct),
        total_samples=1,
        correct_samples=int(correct),
        details=[sample],
    ).to_dict()
    assert report["standard_correct_samples"] == 0
    assert report["judge_corrected_samples"] == int(correct)


def test_missing_details_do_not_fabricate_standard_zero_score():
    report = EvaluationResult(
        dataset_name="legacy", model_id="test", accuracy=1, total_samples=2, correct_samples=2
    ).to_dict()
    assert report["standard_correct_samples"] is None


@pytest.mark.asyncio
async def test_full_adapter_evaluates_more_than_1000_real_frozen_samples(tmp_path, monkeypatch):
    reset_all()
    monkeypatch.setenv("UNUSED", "synthetic")
    samples = [{"id": str(i), "question": f"question-{i}", "answer": "42"} for i in range(1001)]
    evaluator = SyntheticEvaluator("synthetic", "outside-project", num_shots=0)
    evaluator.load_dataset = lambda subset=None: samples
    monkeypatch.setattr(QualityEvaluator, "get_evaluator", lambda *_: evaluator)
    calls = []

    async def response(self, prompt="", **kwargs):
        calls.append(kwargs)
        return {"content": "42"}

    monkeypatch.setattr(QualityEvaluator, "_get_response_with_metrics", response)
    endpoint = Endpoint(
        "lab", "Synthetic", "OpenAI", "http://127.0.0.1:9/v1", "synthetic", "UNUSED"
    )
    store = JobStore(tmp_path / "isolated.db")
    settings = Settings(
        api_token="test-" + "a" * 40,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={"lab": endpoint},
    )
    parameters = QualitySpec(
        datasets=["mmlu"],
        max_samples=None,
        model_type="thinking",
        thinking_enabled=True,
        thinking_budget=8192,
        reasoning_effort="high",
    ).model_dump()
    store.submit(
        test_type="quality", endpoint_id="lab", model_id="synthetic", parameters=parameters
    )
    job = store.claim("fixture")
    output = await execute_job(job, endpoint, settings, store, "fixture")
    report = json.loads((settings.artifact_root / output.result_artifact).read_text())
    assert output.total == output.completed == len(calls) == 1001
    assert report["checkpoint"]["planned_units"] == 1001
    assert report["datasets"]["mmlu"]["config"]["max_samples"] is None
    assert report["datasets"]["mmlu"]["standard_correct_samples"] == 1001
    assert calls[0]["thinking_budget"] == 8192 and calls[-1]["reasoning_effort"] == "high"
    reset_all()
