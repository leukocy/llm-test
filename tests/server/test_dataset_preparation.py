"""Real queue, worker and filesystem boundaries with synthetic public data only."""

import asyncio
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.dataset_manager import DatasetManager
from core.quality_evaluator import QualityEvaluator
from evaluators.base_evaluator import DatasetUnavailableError
from evaluators.ceval_evaluator import CEvalEvaluator
from server.api import create_app
from server.dataset_preparation import execute_preparation
from server.settings import Settings
from server.specs import QualitySpec
from server.store import JobStore, LeaseLost
from server.worker import run_claimed_job


def ceval_rows(subject="physics", count=2, prefix="validation"):
    return [
        {
            "id": i,
            "question": f"{prefix}-{subject}-{i}",
            "subject": subject,
            "A": "一",
            "B": "二",
            "C": "三",
            "D": "四",
            "answer": "B",
        }
        for i in range(count)
    ]


@pytest.mark.parametrize("split", ["test", "val"])
def test_ceval_split_selection_and_same_subject_dev_without_scoring_leak(monkeypatch, split):
    data = {
        split: ceval_rows() + ceval_rows("law"),
        "dev": ceval_rows(count=5, prefix="dev") + ceval_rows("law", count=5, prefix="dev"),
    }
    calls = []

    def load(name, split):
        calls.append((name, split))
        return data[split]

    monkeypatch.setattr("core.dataset_manager.get_dataset", load)
    state = random.getstate()
    evaluator = CEvalEvaluator(num_shots=5, max_samples=None)
    evaluator.evaluation_split = split
    samples = evaluator.load_dataset()
    assert len(samples) == 4 and len({row["id"] for row in samples}) == 4
    assert random.getstate() == state
    assert calls == [("ceval", split), ("ceval", "dev")]
    for sample in samples:
        messages = evaluator.build_chat_messages(sample)
        assert len(messages) == 12
        assert all(
            sample["subject"] in message["content"]
            for message in messages[1:-1]
            if message["role"] == "user"
        )
        assert sample["question"] not in "\n".join(message["content"] for message in messages[:-1])
        assert "dev-" in evaluator.build_full_prompt(sample)
        assert evaluator.get_correct_answer(sample) == evaluator.parse_response("答案：B")
    assert {sample["id"] for sample in samples}.isdisjoint(
        {sample["id"] for sample in evaluator.few_shot_examples}
    )


@pytest.mark.parametrize("bad", ["answer", "subject", "A", "id"])
def test_ceval_rejects_unlabeled_or_invalid_records(bad):
    rows = ceval_rows()
    rows[0].pop(bad)
    with pytest.raises(DatasetUnavailableError):
        CEvalEvaluator.normalize(rows)


def test_ceval_refuses_missing_dev_and_does_not_borrow_validation(monkeypatch):
    monkeypatch.setattr(
        "core.dataset_manager.get_dataset",
        lambda name, split: ceval_rows() if split != "dev" else [],
    )
    with pytest.raises(DatasetUnavailableError, match="insufficient"):
        CEvalEvaluator(num_shots=5).load_dataset()
    assert len(CEvalEvaluator(num_shots=0).load_dataset()) == 2


@pytest.fixture
def lab(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    monkeypatch.setenv("LLM_TEST_DATASET_ROOT", str(root))
    manager = DatasetManager(auto_download=False)
    monkeypatch.setattr("core.dataset_manager._global_manager", manager)
    monkeypatch.setattr("server.dataset_preparation.pinned_revision", lambda repo: "a" * 40)
    store = JobStore(tmp_path / "isolated.db")
    settings = Settings(
        api_token="test-" + "a" * 40,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={},
    )

    def download(self, name, force=False, progress_callback=None):
        path = self.get_local_path(name)
        path.mkdir(parents=True, exist_ok=True)
        progress_callback(0.2, "synthetic download")
        if name == "ceval":
            (path / "test.json").write_text(json.dumps(ceval_rows(prefix="test")))
            (path / "val.json").write_text(json.dumps(ceval_rows()))
            (path / "dev.json").write_text(json.dumps(ceval_rows(count=5, prefix="dev")))
        else:
            (path / "test.json").write_text(json.dumps([{"question": "synthetic", "answer": "42"}]))
        progress_callback(1, "synthetic download complete")
        return True

    monkeypatch.setattr(DatasetManager, "download", download)
    client = TestClient(create_app(settings, store))
    return store, settings, manager, client, {"Authorization": "Bearer " + settings.api_token}


def submit(store, name="ceval"):
    return store.submit(
        test_type="dataset_prepare",
        endpoint_id="dataset-manager",
        model_id=name,
        parameters={"name": name},
        progress_total=1,
    )


@pytest.mark.asyncio
async def test_worker_prepares_without_model_endpoint_and_publishes_receipt(lab):
    store, settings, manager, client, auth = lab
    job = submit(store)
    claimed = store.claim("fixture")
    await run_claimed_job(claimed, settings, store, "fixture")
    final = store.get(job["job_id"])
    assert final["status"] == "completed"
    assert final["progress_completed"] == final["progress_total"] == 1
    assert manager.is_available("ceval")
    response = client.get(f"/api/v1/jobs/{job['job_id']}/report", headers=auth)
    receipt = response.json()["dataset_preparation"]
    assert receipt["revision"] == "a" * 40 and receipt["evaluation_split"] == "test"
    assert len(receipt["files"]) == 3 and all(
        len(file["sha256"]) == 64 for file in receipt["files"]
    )
    item = next(
        item
        for item in client.get("/api/v1/quality/datasets", headers=auth).json()["items"]
        if item["id"] == "ceval"
    )
    assert (
        item["available"]
        and item["local_available"]
        and item["preparation"]["status"] == "completed"
    )
    assert any(event["event"] == "DATASET_PREPARED" for event in store.events(job["job_id"]))


@pytest.mark.asyncio
async def test_prepared_ceval_runs_real_registered_evaluator_and_freezes_split_source(
    lab, monkeypatch
):
    from server.runner_adapter import execute_job
    from server.settings import Endpoint

    store, settings, manager, _, _ = lab
    submit(store)
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    calls = []

    async def response(self, prompt="", **kwargs):
        calls.append(kwargs.get("messages"))
        return {"content": "B"}

    monkeypatch.setattr(QualityEvaluator, "_get_response_with_metrics", response)
    monkeypatch.setenv("UNUSED", "synthetic")
    endpoint = Endpoint(
        "lab", "Synthetic", "OpenAI", "http://127.0.0.1:9/v1", "synthetic", "UNUSED"
    )
    parameters = QualitySpec(datasets=["ceval"], max_samples=None, num_shots=5).model_dump()
    store.submit(
        test_type="quality", endpoint_id="lab", model_id="synthetic", parameters=parameters
    )
    output = await execute_job(store.claim("fixture"), endpoint, settings, store, "fixture")
    report = json.loads((settings.artifact_root / output.result_artifact).read_text())
    result = report["datasets"]["ceval"]
    assert result["total_samples"] == result["correct_samples"] == 2
    provenance = result["config"]["dataset_provenance"]
    assert (
        provenance["source"] == "ceval_test"
        and provenance["evaluation_split"] == "test"
        and provenance["few_shot_split"] == "dev"
    )
    assert provenance["path"] == str(manager.get_local_path("ceval"))
    assert all(len(messages) == 12 for messages in calls)
    assert report["checkpoint"]["committed_units"] == 2


def test_api_auth_registered_only_deduplicates_and_all_missing(lab):
    store, _, _, client, auth = lab
    assert (
        client.post("/api/v1/quality/datasets/preparations", json={"names": ["ceval"]}).status_code
        == 401
    )
    for names in (["ceval", "ceval"], ["../outside"], ["custom_needle"]):
        assert (
            client.post(
                "/api/v1/quality/datasets/preparations", json={"names": names}, headers=auth
            ).status_code
            == 422
        )
    first = client.post(
        "/api/v1/quality/datasets/preparations", json={"names": ["ceval"]}, headers=auth
    ).json()
    second = client.post(
        "/api/v1/quality/datasets/preparations", json={"names": ["ceval"]}, headers=auth
    ).json()
    assert first["items"][0]["job_id"] == second["items"][0]["job_id"]
    all_tasks = client.post("/api/v1/quality/datasets/preparations", json={}, headers=auth).json()[
        "items"
    ]
    assert "ceval" in {task["model_id"] for task in all_tasks}
    assert len({task["model_id"] for task in all_tasks}) == len(all_tasks)


def test_prepare_and_measurement_claims_are_mutually_exclusive(lab):
    store, *_ = lab
    measurement = store.submit(test_type="quality", endpoint_id="lab", model_id="m", parameters={})
    running = store.claim("measure")
    assert running["job_id"] == measurement["job_id"]
    preparation = submit(store)
    assert store.claim("prepare") is None
    store.finish(measurement["job_id"], "measure", outcome="completed")
    assert store.claim("prepare")["job_id"] == preparation["job_id"]
    store.submit(test_type="quality", endpoint_id="lab", model_id="m", parameters={})
    assert store.claim("measure") is None


def test_cancel_at_download_boundary_cannot_publish(lab, monkeypatch):
    store, settings, manager, *_ = lab
    job = submit(store)
    claimed = store.claim("fixture")

    def cancelled(self, name, force=False, progress_callback=None):
        store.request_cancel(job["job_id"])
        progress_callback(0.3, "download boundary")

    monkeypatch.setattr(DatasetManager, "download", cancelled)
    with pytest.raises(asyncio.CancelledError):
        execute_preparation(claimed, settings, store, "fixture")
    assert not manager.is_available("ceval")


@pytest.mark.parametrize("reason", ["lease", "quota", "empty"])
def test_failed_or_expired_preparation_never_publishes(lab, monkeypatch, reason):
    store, settings, manager, *_ = lab
    submit(store)
    claimed = store.claim("fixture")
    if reason == "lease":
        with store._connection() as conn:
            conn.execute("UPDATE control_jobs SET lease_until=0")
        expected = LeaseLost
    elif reason == "quota":
        monkeypatch.setattr("server.dataset_preparation.MAX_BYTES", 1)
        expected = ValueError
    else:
        monkeypatch.setattr(DatasetManager, "download", lambda *args, **kwargs: True)
        expected = ValueError
    with pytest.raises(expected):
        execute_preparation(claimed, settings, store, "fixture")
    assert not manager.is_available("ceval")


def test_metadata_alone_is_not_available(lab):
    _, _, manager, *_ = lab
    path = manager.get_local_path("ceval")
    path.mkdir(parents=True)
    (path / "metadata.json").write_text("{}")
    (path / "preparation.json").write_text("{}")
    assert not manager.is_available("ceval")


def test_ceval_public_loader_pins_revision_and_all_published_splits(lab, monkeypatch):
    _, _, manager, *_ = lab
    calls = []

    def load(repo, subject, split, **options):
        calls.append((repo, subject, split, options))
        return ceval_rows(subject, count=5 if split == "dev" else 2)

    def configs(repo, **options):
        assert options == {"token": False, "revision": "a" * 40}
        return ["physics", "law"]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=load, get_dataset_config_names=configs),
    )
    config = replace(manager.configs["ceval"], version="a" * 40)
    assert manager._download_from_hf("ceval", config)
    assert {split for _, _, split, _ in calls} == {"test", "val", "dev"}
    assert all(
        options == {"token": False, "trust_remote_code": False, "revision": "a" * 40}
        for *_, options in calls
    )
    rows = manager.load("ceval", "val")
    assert len(CEvalEvaluator.normalize(rows)) == 4


def test_single_split_loader_preserves_dataset_rows_instead_of_treating_rows_as_splits(
    lab, monkeypatch
):
    _, _, manager, *_ = lab
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=lambda *args, **kwargs: [{"question": "single", "answer": "42"}]
        ),
    )
    assert manager._download_from_hf("aime2025", manager.configs["aime2025"])
    assert manager.load("aime2025", "test")[0]["answer"] == "42"


def test_failed_publication_restores_existing_partial_directory(lab, monkeypatch):
    store, settings, manager, *_ = lab
    target = manager.get_local_path("ceval")
    target.mkdir(parents=True)
    (target / "metadata.json").write_text("original")
    submit(store)
    job = store.claim("fixture")
    rename = Path.rename

    def fail_stage(path, destination):
        if ".staging" in str(path):
            raise OSError("synthetic rename failure")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_stage)
    with pytest.raises(OSError, match="synthetic"):
        execute_preparation(job, settings, store, "fixture")
    assert (target / "metadata.json").read_text() == "original"
    assert not (target / "val.json").exists()
