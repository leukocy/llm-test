"""MMLU split/subject isolation and persisted worker/report evidence."""

import json
import random

import pytest

from core.dataset_manager import DatasetManager
from core.quality_evaluator import QualityEvaluator
from evaluators.base_evaluator import DatasetUnavailableError
from evaluators.mmlu_evaluator import MMLUEvaluator
from server.runner_adapter import execute_job
from server.settings import Endpoint
from server.specs import QualitySpec
from server.worker import run_claimed_job
from tests.server.test_dataset_preparation import lab, submit  # noqa: F401


def rows(subject="anatomy", split="test", count=2):
    return [
        {
            "question": f"{split}-{subject}-{i}",
            "subject": subject,
            "choices": ["one", "two", "three", "four"],
            "answer": 1,
        }
        for i in range(count)
    ]


def test_dev_alias_subject_prompts_and_global_random_state(monkeypatch):
    data = {
        "test": rows() + rows("astronomy"),
        "dev": rows(split="dev", count=5) + rows("astronomy", "dev", 5),
    }
    calls = []

    def load(name, split, **kwargs):
        calls.append(split)
        return data[split]

    monkeypatch.setattr("core.dataset_manager.get_dataset", load)
    state = random.getstate()
    evaluator = MMLUEvaluator(num_shots=5)
    samples = evaluator.load_dataset()
    assert state == random.getstate() and calls == ["test", "dev"]
    assert evaluator.evaluation_split == "test" and evaluator.few_shot_split == "dev"
    for sample in samples:
        messages = evaluator.build_chat_messages(sample)
        assert len(messages) == 12
        assert all(f"dev-{sample['subject']}" in message["content"] for message in messages[1:-1:2])
        assert sample["question"] not in "\n".join(message["content"] for message in messages[:-1])
    assert DatasetManager(auto_download=False).configs["mmlu"].split_mapping["dev"] == "dev"


def test_missing_dev_never_borrows_test_or_silently_uses_zero_shot(monkeypatch):
    monkeypatch.setattr(
        "core.dataset_manager.get_dataset",
        lambda name, split, **kw: rows(count=20) if split == "test" else [],
    )
    with pytest.raises(DatasetUnavailableError, match="insufficient"):
        MMLUEvaluator(num_shots=5, max_samples=2).load_dataset()
    assert len(MMLUEvaluator(num_shots=0, max_samples=2).load_dataset()) == 2


def test_overlap_and_invalid_rows_fail_before_scoring():
    samples = MMLUEvaluator.normalize(rows(), "test")
    examples = MMLUEvaluator.normalize(rows(count=5), "dev")
    with pytest.raises(DatasetUnavailableError, match="overlaps"):
        MMLUEvaluator.validate_dev(samples, examples, 5)
    for field in ["subject", "question", "answer", "choices"]:
        bad = rows()
        bad[0].pop(field)
        with pytest.raises(DatasetUnavailableError):
            MMLUEvaluator.normalize(bad, "test")
    bad = rows()
    bad[0]["answer"] = True
    with pytest.raises(DatasetUnavailableError):
        MMLUEvaluator.normalize(bad, "test")


@pytest.mark.asyncio
async def test_preparation_quality_worker_freezes_source_policy_and_reports(lab, monkeypatch):  # noqa: F811
    store, settings, manager, client, auth = lab

    def download(self, name, force=False, progress_callback=None):
        path = self.get_local_path(name)
        path.mkdir(parents=True, exist_ok=True)
        for split, count in [("test", 2), ("dev", 5)]:
            (path / f"{split}.json").write_text(
                json.dumps(rows(split=split, count=count) + rows("astronomy", split, count))
            )
        progress_callback(1, "synthetic MMLU")
        return True

    monkeypatch.setattr(DatasetManager, "download", download)
    prepare = submit(store, "mmlu")
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    assert store.get(prepare["job_id"])["status"] == "completed"
    calls = []

    async def response(self, prompt="", **kwargs):
        calls.append(kwargs["messages"])
        return {"content": "B"}

    monkeypatch.setattr(QualityEvaluator, "_get_response_with_metrics", response)
    endpoint = Endpoint(
        "lab",
        "Synthetic",
        "OpenAI",
        "http://127.0.0.1:9/v1",
        "synthetic",
        "UNUSED",
        api_key_value="synthetic",  # pragma: allowlist secret
    )  # pragma: allowlist secret
    p = QualitySpec(datasets=["mmlu"], max_samples=None, num_shots=5).model_dump()
    job = store.submit(test_type="quality", endpoint_id="lab", model_id="synthetic", parameters=p)
    output = await execute_job(store.claim("fixture"), endpoint, settings, store, "fixture")
    payload = json.loads((settings.artifact_root / output.result_artifact).read_text())
    data = payload["datasets"]["mmlu"]
    source = data["config"]["dataset_provenance"]
    assert data["total_samples"] == data["correct_samples"] == 4
    assert (
        source["source"] == "mmlu_test"
        and source["evaluation_split"] == "test"
        and source["few_shot_split"] == "dev"
    )
    assert (
        source["few_shot_policy"] == "same_subject_dev"
        and source["few_shot_per_sample"] == 5
        and source["few_shot_count"] == 10
    )
    assert source["answer_protocol"] == "generated_text_choice" and all(
        len(messages) == 12 for messages in calls
    )
