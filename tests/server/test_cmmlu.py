"""CMMLU archive, split isolation and the persisted worker/report path."""

import csv
import io
import json
import random
import zipfile
from dataclasses import replace

import pytest

from core.cmmlu_data import prepare_archive
from core.dataset_manager import DatasetManager
from core.quality_evaluator import QualityEvaluator
from evaluators import get_evaluator
from evaluators.base_evaluator import DatasetUnavailableError
from evaluators.cmmlu_evaluator import CMMLUEvaluator
from server.runner_adapter import execute_job
from server.settings import Endpoint
from server.specs import QualitySpec
from server.worker import run_claimed_job
from tests.server.test_dataset_preparation import lab, submit  # noqa: F401


def rows(subject="anatomy", count=2, split="test"):
    return [
        {
            "id": str(i),
            "subject": subject,
            "Question": f"{split}-{subject}-{i}",
            "A": "一",
            "B": "二",
            "C": "三",
            "D": "四",
            "Answer": "B",
        }
        for i in range(count)
    ]


def make_archive(path, bad=None):
    with zipfile.ZipFile(path, "w") as archive:
        for subject in range(67):
            for split, count in (("test", 2), ("dev", 5)):
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["", "Question", "A", "B", "C", "D", "Answer"])
                for i in range(count):
                    writer.writerow([i, f"{split}-{subject}-{i}", "一", "二", "三", "四", "B"])
                archive.writestr(f"{split}/subject_{subject}.csv", output.getvalue())
        if bad:
            archive.writestr(bad, "bad")


def test_archive_and_pinned_download_never_execute_dataset_script(tmp_path, monkeypatch):
    archive = tmp_path / "official-format.zip"
    make_archive(archive)
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("test/.zip", "ignored upstream artifact")
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return str(archive)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: pytest.fail("remote script load"))
    manager = DatasetManager(cache_dir=str(tmp_path / "data"), auto_download=False)
    manager.configs["cmmlu"] = replace(manager.configs["cmmlu"], version="a" * 40)
    assert manager.download("cmmlu", force=True)
    assert calls[0]["revision"] == "a" * 40 and calls[0]["token"] is False
    assert len(manager.load("cmmlu", split="test")) == 134
    assert len(manager.load("cmmlu", split="dev")) == 335


@pytest.mark.parametrize("bad", ["../escape.csv", "dev/subject_0.csv", "script.py"])
def test_archive_rejects_unsupported_or_duplicate_entries(tmp_path, bad):
    archive = tmp_path / "bad.zip"
    make_archive(archive, bad)
    with pytest.raises(ValueError):
        prepare_archive(str(archive), tmp_path)
    assert not (tmp_path / "test.json").exists()


def test_registered_evaluator_subject_dev_and_no_global_random_side_effect(monkeypatch):
    data = {"test": rows() + rows("law"), "dev": rows(count=5, split="dev") + rows("law", 5, "dev")}
    monkeypatch.setattr("core.dataset_manager.get_dataset", lambda name, split: data[split])
    assert get_evaluator("cmmlu") is CMMLUEvaluator
    evaluator = CMMLUEvaluator(num_shots=5)
    state = random.getstate()
    samples = evaluator.load_dataset("anatomy")
    assert len(samples) == 2 and state == random.getstate()
    for sample in samples:
        messages = evaluator.build_chat_messages(sample)
        assert len(messages) == 12
        assert all("dev-anatomy" in msg["content"] for msg in messages[1:-1:2])
        assert sample["question"] not in "\n".join(msg["content"] for msg in messages[:-1])
        assert evaluator.get_sample_category(sample) == "anatomy"
        assert evaluator.check_answer(
            evaluator.parse_response("答案：B"), evaluator.get_correct_answer(sample)
        )
    data["dev"] = []
    with pytest.raises(DatasetUnavailableError, match="insufficient"):
        evaluator.load_dataset()
    assert len(CMMLUEvaluator(num_shots=0).load_dataset()) == 4


@pytest.mark.parametrize("field", ["id", "Question", "Answer", "A", "subject"])
def test_bad_rows_fail_before_model_calls(field):
    records = rows()
    records[0].pop(field)
    with pytest.raises(DatasetUnavailableError):
        CMMLUEvaluator.normalize(records)


@pytest.mark.asyncio
async def test_preparation_worker_quality_report_and_catalog(lab, monkeypatch):  # noqa: F811
    store, settings, manager, client, auth = lab

    def download(self, name, force=False, progress_callback=None):
        path = self.get_local_path(name)
        path.mkdir(parents=True, exist_ok=True)
        for split, count in (("test", 2), ("dev", 5)):
            (path / f"{split}.json").write_text(json.dumps(rows(count=count, split=split)))
        progress_callback(1, "synthetic CMMLU")
        return True

    monkeypatch.setattr(DatasetManager, "download", download)
    submit(store, "cmmlu")
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    item = next(
        item
        for item in client.get("/api/v1/quality/datasets", headers=auth).json()["items"]
        if item["id"] == "cmmlu"
    )
    assert item["available"] and item["local_available"] and item["downloadable"]
    assert item["preparation"]["status"] == "completed"
    calls = []

    async def response(self, prompt="", **kwargs):
        calls.append(kwargs["messages"])
        return {"content": "B"}

    monkeypatch.setattr(QualityEvaluator, "_get_response_with_metrics", response)
    monkeypatch.setenv("UNUSED", "synthetic")
    endpoint = Endpoint(
        "lab", "Synthetic", "OpenAI", "http://127.0.0.1:9/v1", "synthetic", "UNUSED"
    )
    parameters = QualitySpec(datasets=["cmmlu"], max_samples=None, num_shots=5).model_dump()
    store.submit(
        test_type="quality", endpoint_id="lab", model_id="synthetic", parameters=parameters
    )
    output = await execute_job(store.claim("fixture"), endpoint, settings, store, "fixture")
    report = json.loads((settings.artifact_root / output.result_artifact).read_text())
    result = report["datasets"]["cmmlu"]
    assert result["total_samples"] == result["correct_samples"] == 2
    provenance = result["config"]["dataset_provenance"]
    assert provenance["source"] == "cmmlu_test"
    assert provenance["evaluation_split"] == "test" and provenance["few_shot_split"] == "dev"
    assert all(len(messages) == 12 for messages in calls)
    assert len(provenance["sample_sha256"]) == 64
