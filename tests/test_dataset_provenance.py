"""Production quality scores must come from identified benchmark samples."""

from unittest.mock import MagicMock

import pytest

from core.quality_evaluator import (
    ProviderRequestError,
    QualityEvaluator,
    QualityTestConfig,
    fingerprint_samples,
)
from evaluators.base_evaluator import DatasetUnavailableError, EvaluationResult, SampleResult
from evaluators.mmlu_evaluator import MMLUEvaluator
from ui.quality_reports import generate_quality_summary


def test_missing_official_dataset_fails_closed(monkeypatch):
    monkeypatch.delenv("LLM_TEST_ALLOW_EMBEDDED_SAMPLES", raising=False)
    monkeypatch.setattr("core.dataset_manager.get_dataset", lambda **kwargs: [])

    evaluator = MMLUEvaluator(num_shots=0)
    with pytest.raises(DatasetUnavailableError, match="Install the official dataset"):
        evaluator.load_dataset()


def test_demo_samples_require_opt_in_and_are_marked(monkeypatch):
    monkeypatch.setenv("LLM_TEST_ALLOW_EMBEDDED_SAMPLES", "1")
    monkeypatch.setattr("core.dataset_manager.get_dataset", lambda **kwargs: [])

    evaluator = MMLUEvaluator(num_shots=0)
    samples = evaluator.load_dataset()

    assert len(samples) == 2
    assert evaluator.dataset_source == "embedded_demo"


def test_sample_fingerprint_is_stable_and_order_sensitive():
    sample_a = {"question": "甲", "answer": 1}
    sample_a_reordered = {"answer": 1, "question": "甲"}
    sample_b = {"question": "乙", "answer": 2}

    assert fingerprint_samples([sample_a, sample_b]) == fingerprint_samples(
        [sample_a_reordered, sample_b]
    )
    assert fingerprint_samples([sample_a, sample_b]) != fingerprint_samples([sample_b, sample_a])


def test_empty_model_response_counts_as_incorrect_answer():
    evaluator = MMLUEvaluator(num_shots=0)
    results = [
        SampleResult("0", "Q0", "A", "A", "A", True, category="test"),
        SampleResult("1", "Q1", "B", "", "", False, category="test"),
    ]

    accuracy, by_category = evaluator.compute_metrics(results)

    assert accuracy == 0.5
    assert by_category["test"]["count"] == 2


def test_failed_code_sample_stays_in_accuracy_interval():
    details = [
        SampleResult("0", "Q0", "A", "A", "A", True),
        SampleResult("1", "Q1", "B", "bad code", "", False, error="AssertionError"),
    ]
    result = EvaluationResult("humaneval", "test-model", 0.5, 2, 1, details=details)

    result.compute_performance_stats()

    assert result.extended_metrics["stderr"] == pytest.approx(0.5)


@pytest.mark.parametrize("demo_mode", [False, True])
def test_ui_precheck_does_not_silently_skip_selected_dataset(monkeypatch, demo_mode):
    from ui import advanced_panels

    if demo_mode:
        monkeypatch.setenv("LLM_TEST_ALLOW_EMBEDDED_SAMPLES", "1")
    else:
        monkeypatch.delenv("LLM_TEST_ALLOW_EMBEDDED_SAMPLES", raising=False)
    manager = MagicMock()
    manager.configs = {"mmlu": object()}
    manager.is_available.return_value = False
    manager.download.return_value = False
    monkeypatch.setattr("core.dataset_manager.get_manager", lambda: manager)
    ui = MagicMock()
    monkeypatch.setattr(advanced_panels, "st", ui)
    monkeypatch.setattr(advanced_panels, "_get_quality_eval_module", lambda: None)

    result = advanced_panels._run_quality_test(
        config={},
        selected_datasets=["mmlu"],
        max_samples=1,
        num_shots=0,
        model_type="standard",
        temperature=0,
        max_tokens=16,
        concurrency=1,
    )

    assert result is False
    if demo_mode:
        manager.download.assert_not_called()
        assert "Demonstration mode" in ui.warning.call_args.args[0]
    else:
        manager.download.assert_called_once_with("mmlu")
        assert any(
            "selected datasets are unavailable" in call.args[0] for call in ui.error.call_args_list
        )


def _quality_engine(monkeypatch, tmp_path, evaluator):
    monkeypatch.setattr("core.quality_evaluator.get_provider", lambda *args: MagicMock())
    monkeypatch.setattr(QualityEvaluator, "_init_tokenizer", lambda self, model_id: None)
    engine = QualityEvaluator(
        api_base_url="https://api.openai.com/v1",
        model_id="test-model",
        output_dir=str(tmp_path),
        enable_cache=False,
    )
    monkeypatch.setattr(engine, "get_evaluator", lambda *args, **kwargs: evaluator)
    return engine


@pytest.mark.asyncio
async def test_missing_dataset_aborts_run_before_model_requests(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_TEST_ALLOW_EMBEDDED_SAMPLES", raising=False)
    monkeypatch.setattr("core.dataset_manager.get_dataset", lambda **kwargs: [])
    evaluator = MMLUEvaluator(num_shots=0)
    engine = _quality_engine(monkeypatch, tmp_path, evaluator)

    with pytest.raises(DatasetUnavailableError, match="Install the official dataset"):
        await engine.run_evaluation(QualityTestConfig(datasets=["mmlu"], num_shots=0))

    engine.provider.get_completion.assert_not_called()


@pytest.mark.asyncio
async def test_demo_result_carries_provenance_and_is_labeled_in_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_TEST_ALLOW_EMBEDDED_SAMPLES", "1")
    monkeypatch.setattr("core.dataset_manager.get_dataset", lambda **kwargs: [])
    evaluator = MMLUEvaluator(num_shots=0)
    engine = _quality_engine(monkeypatch, tmp_path, evaluator)

    async def evaluate_batch(samples, **kwargs):
        return [
            SampleResult(
                sample_id=str(index),
                question=sample["question"],
                correct_answer=str(sample["answer"]),
                model_response="A",
                predicted_answer="A",
                is_correct=True,
            )
            for index, sample in enumerate(samples)
        ]

    monkeypatch.setattr(evaluator, "evaluate_batch", evaluate_batch)
    result = await engine.evaluate_dataset("mmlu", QualityTestConfig(num_shots=0))

    assert result is not None
    provenance = result.config["dataset_provenance"]
    assert provenance["source"] == "embedded_demo"
    assert provenance["sample_count"] == result.total_samples == 2
    assert provenance["sample_sha256"] == fingerprint_samples(evaluator.samples)
    assert provenance["few_shot_count"] == 0
    assert provenance["few_shot_sha256"] == fingerprint_samples([])
    assert provenance["selection_seed"] == evaluator.seed
    summary = generate_quality_summary({"mmlu": result}).iloc[0]
    assert summary["Dataset Source"] == "embedded_demo"
    assert summary["Sample SHA-256"] == provenance["sample_sha256"]


@pytest.mark.asyncio
async def test_model_api_failure_does_not_become_wrong_answer_score(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_TEST_ALLOW_EMBEDDED_SAMPLES", "1")
    monkeypatch.setattr("core.dataset_manager.get_dataset", lambda **kwargs: [])
    evaluator = MMLUEvaluator(num_shots=0)
    engine = _quality_engine(monkeypatch, tmp_path, evaluator)

    async def failed_request(*args, **kwargs):
        return {"content": "", "error": "HTTP 503"}

    monkeypatch.setattr(engine, "_get_response_with_metrics", failed_request)

    with pytest.raises(ProviderRequestError, match="no accuracy score was issued"):
        await engine.evaluate_dataset("mmlu", QualityTestConfig(num_shots=0))


@pytest.mark.asyncio
async def test_judge_api_failure_does_not_publish_uncorrected_score(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_TEST_ALLOW_EMBEDDED_SAMPLES", "1")
    monkeypatch.setattr("core.dataset_manager.get_dataset", lambda **kwargs: [])
    evaluator = MMLUEvaluator(num_shots=0)
    evaluator.use_llm_judge = True
    engine = _quality_engine(monkeypatch, tmp_path, evaluator)
    calls = 0

    async def first_answer_then_failed_judge(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"content": "A", "error": None} if calls % 2 else {"content": "", "error": "503"}

    monkeypatch.setattr(engine, "_get_response_with_metrics", first_answer_then_failed_judge)

    with pytest.raises(ProviderRequestError, match="no accuracy score was issued"):
        await engine.evaluate_dataset(
            "mmlu", QualityTestConfig(num_shots=0, use_llm_judge=True, concurrency=1)
        )
