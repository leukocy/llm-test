"""Real SQLite, engine and API recovery boundaries; no external model requests."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from core.cancel_state import reset_all
from core.evaluation_control import run_samples
from core.quality_evaluator import QualityEvaluator, QualityTestConfig, fingerprint_samples
from core.robustness_tester import PerturbationType, RobustnessTester
from core.run_lifecycle import RunStatus
from evaluators.base_evaluator import BaseEvaluator, ProviderRequestError
from server.api import create_app
from server.checkpoints import CheckpointConflict, JobJournal, checkpoint_info, recover_job
from server.control import JobControl
from server.reports import render_quality_html, render_quality_markdown, render_robustness_markdown
from server.runner_adapter import execute_job
from server.settings import Endpoint, Settings
from server.store import JobStore, LeaseLost


@pytest.fixture
def lab(tmp_path):
    reset_all()
    endpoint = Endpoint(
        "lab",
        "Synthetic",
        "OpenAI",
        "http://127.0.0.1:9010/v1",
        "m",
        "TEST_KEY",
        api_key_value="synthetic",  # pragma: allowlist secret
    )
    store = JobStore(tmp_path / "jobs.db")
    settings = Settings(
        api_token="checkpoint-test-token-" + "a" * 32,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={"lab": endpoint},
    )
    yield store, endpoint, settings
    reset_all()


def claim(store, test_type="quality", parameters=None, worker="w"):
    store.submit(test_type=test_type, endpoint_id="lab", model_id="m", parameters=parameters or {})
    return store.claim(worker)


def interrupt(store, job, worker="w"):
    store.finish(job["job_id"], worker, outcome=RunStatus.FAILED, error_code="WORKER_LOST")
    return store.get(job["job_id"])


def plan():
    return {"few_shot": ["original"]}, [{"question": str(i)} for i in range(3)]


def test_recovery_reuses_inputs_and_results_and_fences_old_attempt(lab):
    store, endpoint, _ = lab
    job = claim(store)
    journal = JobJournal(store, job, "w", endpoint)
    frozen = journal.prepare_scope("scope", plan)
    journal.start("scope", 0)
    journal.commit("scope", 0, {"answer": "original"})
    journal.start("scope", 1)  # Unknown outcome at interruption.
    with pytest.raises(CheckpointConflict):
        journal.start("scope", 1)
    failed = interrupt(store, job)
    assert checkpoint_info(store, failed)["can_recover"]
    queued = recover_job(store, failed, endpoint)
    assert recover_job(store, queued, endpoint)["status"] == "queued"
    resumed = store.claim("w")  # Even reusing a worker ID must fence its old epoch.
    with pytest.raises(LeaseLost):
        journal.commit("scope", 1, {"answer": "stale"})
    new = JobJournal(JobStore(store.path), resumed, "w", replace(endpoint, api_key_value="rotated"))
    assert new.prepare_scope("scope", lambda: pytest.fail("must not reload dataset")) == frozen
    assert new.results("scope") == {0: {"answer": "original"}}
    new.start("scope", 1)
    new.commit("scope", 1, {"answer": "new"})
    with pytest.raises(CheckpointConflict):
        new.commit("scope", 1, {"answer": "duplicate"})
    info = new.describe()
    assert (info["committed_units"], info["recoveries"], info["repeated_unit_attempts"]) == (
        2,
        1,
        1,
    )
    assert [e["event"] for e in store.events(job["job_id"])].count("recover") == 1
    assert "synthetic" not in json.dumps(info)


@pytest.mark.parametrize(
    "change", ["input", "result", "metadata", "membership", "endpoint", "parameters", "code"]
)
def test_recovery_rejects_corrupt_or_changed_provenance(lab, change, monkeypatch):
    store, endpoint, _ = lab
    job = claim(store)
    journal = JobJournal(store, job, "w", endpoint)
    journal.prepare_scope("scope", plan)
    journal.start("scope", 0)
    journal.commit("scope", 0, {"answer": 1})
    failed = interrupt(store, job)
    with store._connection() as conn:
        if change in {"input", "result"}:
            conn.execute(f"UPDATE checkpoint_units SET {change}_json='{{}}' WHERE unit_index=0")
        elif change == "metadata":
            conn.execute("UPDATE checkpoint_scopes SET metadata_json='{}'")
        elif change == "membership":
            conn.execute("DELETE FROM checkpoint_units WHERE unit_index=1")
    if change == "endpoint":
        endpoint = replace(endpoint, model_id="different")
    if change == "parameters":
        failed = {**failed, "parameters": {"changed": True}}
    if change == "code":
        monkeypatch.setattr("server.checkpoints.execution_signature", lambda *_: "changed-code")
    with pytest.raises(CheckpointConflict):
        recover_job(store, failed, endpoint)
    assert store.get(job["job_id"])["status"] == "failed"


@pytest.mark.parametrize(
    "outcome,error",
    [
        (RunStatus.COMPLETED, None),
        (RunStatus.FAILED, "EXECUTION_FAILED"),
    ],
)
def test_recovery_does_not_retry_scores_or_api_failures(lab, outcome, error):
    store, endpoint, _ = lab
    job = claim(store)
    journal = JobJournal(store, job, "w", endpoint)
    journal.prepare_scope("scope", plan)
    store.finish(job["job_id"], "w", outcome=outcome, error_code=error)
    failed = store.get(job["job_id"])
    assert not checkpoint_info(store, failed)["can_recover"]
    with pytest.raises(CheckpointConflict):
        recover_job(store, failed, endpoint)


def test_expired_lease_cannot_commit_and_reaper_preserves_checkpoint(lab):
    store, endpoint, _ = lab
    job = claim(store)
    journal = JobJournal(store, job, "w", endpoint)
    journal.prepare_scope("scope", plan)
    journal.start("scope", 0)
    with store._connection() as conn:
        conn.execute("UPDATE control_jobs SET lease_until=0")
    with pytest.raises(LeaseLost):
        journal.commit("scope", 0, {"answer": "late"})
    assert store.reap_expired() == 1
    failed = store.get(job["job_id"])
    assert failed["error_code"] == "WORKER_LOST"
    assert recover_job(store, failed, endpoint)["status"] == "queued"


@pytest.mark.asyncio
async def test_sample_pause_drains_and_restores_without_readmitting_saved_units(lab):
    store, _, _ = lab
    job = claim(store)
    control = JobControl(store, job["job_id"], "w")
    gate, two_started = asyncio.Event(), asyncio.Event()
    calls, committed = [], []

    async def evaluate(index):
        calls.append(index)
        if len(calls) == 2:
            two_started.set()
        await gate.wait()
        return index * 10

    task = asyncio.create_task(
        run_samples(
            evaluate,
            total=5,
            concurrency=2,
            restored={0: 0},
            checkpoint=control.checkpoint,
            pause_requested=control.pause_requested,
            commit=lambda i, v: committed.append(i),
        )
    )
    await asyncio.wait_for(two_started.wait(), 2)
    store.request_pause(job["job_id"])
    gate.set()
    for _ in range(30):
        if store.get(job["job_id"])["status"] == "paused":
            break
        await asyncio.sleep(0.02)
    assert store.get(job["job_id"])["status"] == "paused"
    assert calls == [1, 2] and committed == [1, 2]
    store.resume(job["job_id"])
    assert await asyncio.wait_for(task, 2) == [0, 10, 20, 30, 40]
    assert calls == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_durable_commit_failure_stops_bounded_admission():
    called = []

    async def evaluate(index):
        called.append(index)
        return index

    def commit(*_):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        await run_samples(evaluate, total=100, concurrency=1, commit=commit)
    assert called == [0]


@pytest.mark.asyncio
async def test_cancelled_sample_aborts_before_admitting_more_work():
    calls = []

    async def evaluate(index):
        calls.append(index)
        raise asyncio.CancelledError("provider task cancelled")

    with pytest.raises(asyncio.CancelledError):
        await run_samples(evaluate, total=100, concurrency=1)
    assert calls == [0]


class SyntheticEvaluator(BaseEvaluator):
    def __init__(self, samples):
        super().__init__("synthetic", "outside-project", num_shots=1)
        self.source_samples = samples

    def load_dataset(self, subset=None):
        self.few_shot_examples = [{"question": "original-example", "answer": "42"}]
        return self.source_samples

    def format_prompt(self, sample, include_answer=False):
        return sample["question"] + (":" + sample["answer"] if include_answer else "")

    def parse_response(self, response):
        return response

    def check_answer(self, predicted, correct):
        return predicted == correct


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["interruption", "stop"])
async def test_quality_adapter_recovers_exact_samples_and_few_shot_without_repeat(
    lab, monkeypatch, termination
):
    store, endpoint, settings = lab
    config = QualityTestConfig(
        datasets=["synthetic"], max_samples=3, concurrency=1, use_cache=False
    )
    samples = [{"question": f"original-{i}", "answer": "42"} for i in range(3)]
    monkeypatch.setattr(QualityEvaluator, "_init_tokenizer", lambda *_: None)
    monkeypatch.setattr(QualityEvaluator, "_save_results", lambda *_: None)
    monkeypatch.setattr(QualityEvaluator, "get_evaluator", lambda *_: SyntheticEvaluator(samples))
    calls, started = [], asyncio.Event()

    async def response(self, prompt="", **kwargs):
        calls.append(prompt or json.dumps(kwargs["messages"]))
        if len(calls) == 2:
            started.set()
            await asyncio.Event().wait()
        return {"content": "42"}

    monkeypatch.setattr(QualityEvaluator, "_get_response_with_metrics", response)
    job = claim(store, parameters=config.to_dict())
    task = asyncio.create_task(execute_job(job, endpoint, settings, store, "w"))
    await asyncio.wait_for(started.wait(), 3)
    if termination == "stop":
        store.request_cancel(job["job_id"])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert checkpoint_info(store, store.get(job["job_id"]))["committed_units"] == 1
    if termination == "stop":
        store.finish(job["job_id"], "w", outcome=RunStatus.CANCELLED)
        stopped = store.get(job["job_id"])
    else:
        stopped = interrupt(store, job)
    recover_job(store, stopped, endpoint)
    monkeypatch.setattr(
        QualityEvaluator,
        "get_evaluator",
        lambda *_: SyntheticEvaluator([{"question": "changed", "answer": "wrong"}]),
    )
    resumed = store.claim("w2")
    output = await execute_job(resumed, endpoint, settings, store, "w2")
    payload = json.loads((settings.artifact_root / output.result_artifact).read_text())
    assert output.result_artifact.endswith("report-attempt-2.json")
    assert output.completed == output.total == 3
    data = payload["datasets"]["synthetic"]
    assert data["accuracy"] == 1 and len(data["details"]) == 3
    assert data["config"]["dataset_provenance"]["sample_sha256"] == fingerprint_samples(samples)
    assert "original-0" in calls[0] and all("original-0" not in call for call in calls[1:])
    assert calls[1] == calls[2] and len(calls) == 4
    assert all("original-example" in call for call in calls)
    assert (
        payload["checkpoint"]["recoveries"] == payload["checkpoint"]["repeated_unit_attempts"] == 1
    )
    assert "执行中断" in render_quality_html(resumed, payload)
    assert "不能作为连续吞吐" in render_quality_markdown(resumed, payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["interruption", "stop"])
async def test_robustness_freezes_perturbations_and_skips_committed_samples(
    lab, monkeypatch, termination
):
    store, endpoint, settings = lab
    params = {
        "samples": [
            {"question": f"Please answer {i} plus 2", "correct_answer": "42"} for i in range(3)
        ],
        "perturbation_types": ["typo", "whitespace"],
    }
    calls, blocked = [], asyncio.Event()

    async def response(**kwargs):
        calls.append(kwargs["prompt"])
        if len(calls) == 4:
            blocked.set()
            await asyncio.Event().wait()
        return {"full_response_content": "42"}

    provider = type("FakeProvider", (), {"get_completion": staticmethod(response)})()
    monkeypatch.setattr("core.providers.factory.get_provider", lambda *_: provider)
    job = claim(store, "robustness", params)
    task = asyncio.create_task(execute_job(job, endpoint, settings, store, "w"))
    await asyncio.wait_for(blocked.wait(), 3)
    if termination == "stop":
        store.request_cancel(job["job_id"])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    journal = JobJournal(store, job, "w", endpoint)
    _, frozen = journal.prepare_scope("robustness", lambda: pytest.fail("existing plan"))
    if termination == "stop":
        store.finish(job["job_id"], "w", outcome=RunStatus.CANCELLED)
        stopped = store.get(job["job_id"])
    else:
        stopped = interrupt(store, job)
    recover_job(store, stopped, endpoint)
    output = await execute_job(store.claim("w2"), endpoint, settings, store, "w2")
    payload = json.loads((settings.artifact_root / output.result_artifact).read_text())
    expected = [
        prompt
        for unit in frozen[1:]
        for prompt in [
            unit["sample"]["question"],
            *[p["perturbed_question"] for p in unit["perturbations"]],
        ]
    ]
    assert calls[4:] == expected and len(calls) == 10
    assert output.completed == output.total == 9
    assert payload["robustness"]["total_samples"] == 3
    assert "重复发起 1 次" in render_robustness_markdown(job, payload)


@pytest.mark.asyncio
async def test_robustness_accuracy_uses_perturbed_answers_and_transport_errors_abort():
    tester = RobustnessTester([PerturbationType.CASE_CHANGE])
    responses = iter(["wrong", "42", "42", "wrong"])

    async def response(_):
        return next(responses)

    report = await tester.test_batch([{"question": "HELLO", "correct_answer": "42"}] * 2, response)
    assert report.original_accuracy == report.perturbed_accuracy == 0.5
    assert report.accuracy_drop == 0
    failing = AsyncMock(side_effect=RuntimeError("API unavailable"))
    with pytest.raises(ProviderRequestError):
        await tester.test_batch([{"question": "HELLO", "correct_answer": "42"}], failing)
    assert failing.await_count == 1


def test_checkpoint_api_auth_metadata_and_recovery(lab):
    store, endpoint, settings = lab
    job = claim(store)
    journal = JobJournal(store, job, "w", endpoint)
    journal.prepare_scope("scope", plan)
    interrupt(store, job)
    client = TestClient(create_app(settings, store))
    path = f"/api/v1/jobs/{job['job_id']}"
    headers = {"Authorization": f"Bearer {settings.api_token}"}
    assert client.get(f"{path}/checkpoint").status_code == 401
    assert client.post(f"{path}/recover").status_code == 401
    info = client.get(f"{path}/checkpoint", headers=headers).json()
    assert info["can_recover"] and info["planned_units"] == 3
    assert "question" not in json.dumps(info)
    assert client.post(f"{path}/recover", headers=headers).json()["status"] == "queued"
    assert client.post(f"{path}/recover", headers=headers).status_code == 200
    assert client.get("/api/v1/jobs/missing/checkpoint", headers=headers).status_code == 404


def test_schema_upgrade_preserves_jobs_and_checkpoint_foreign_keys(lab):
    store, _, _ = lab
    job = claim(store)
    with store._connection() as conn:
        for table in ["checkpoint_units", "checkpoint_scopes", "job_checkpoints"]:
            conn.execute(f"DROP TABLE {table}")
        conn.execute("UPDATE db_meta SET value='1.11.0' WHERE key='schema_version'")
    upgraded = JobStore(store.path)
    assert upgraded.get(job["job_id"])["parameters"] == job["parameters"]
    with upgraded._connection() as conn:
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
        assert (
            conn.execute("SELECT value FROM db_meta WHERE key='schema_version'").fetchone()[0]
            == "1.14.0"
        )


@pytest.mark.asyncio
async def test_custom_needle_concurrent_samples_keep_their_own_scoring_context():
    from evaluators.custom_needle_evaluator import CustomNeedleEvaluator

    evaluator = CustomNeedleEvaluator()
    both_started = asyncio.Event()
    started = 0

    async def respond(prompt):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), 2)
        return "alpha" if "retrieve-alpha" in prompt else "beta"

    samples = [
        {"question": "retrieve-alpha", "keywords": ["alpha"]},
        {"question": "retrieve-beta", "keywords": ["beta"]},
    ]
    results = await evaluator.evaluate_batch(samples, respond, concurrency=2)
    assert [r.is_correct for r in results] == [True, True]
    assert not hasattr(evaluator, "_current_keywords")
