"""Shared A/B plans exercise real API, worker, checkpoints and evaluator without network."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from core.dataset_manager import DatasetManager
from core.quality_evaluator import QualityEvaluator
from core.standard_report import StandardReport
from evaluators.base_evaluator import EvaluationResult, SampleResult
from server.api import create_app
from server.checkpoints import CheckpointConflict, recover_job
from server.settings import Endpoint, Settings
from server.shared_quality import SharedQualityJournal
from server.store import JobStore
from server.worker import run_claimed_job


def records(count=6):
    return [
        {
            "id": i,
            "subject": "physics",
            "question": f"question-{i}",
            "A": "1",
            "B": "2",
            "C": "3",
            "D": "4",
            "answer": "B",
        }
        for i in range(count)
    ]


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_TEST_DATASET_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("UNUSED", "synthetic")
    manager = DatasetManager(auto_download=False)
    monkeypatch.setattr("core.dataset_manager._global_manager", manager)
    path = manager.get_local_path("ceval")
    path.mkdir(parents=True)
    (path / "test.json").write_text(json.dumps(records()))
    (path / "dev.json").write_text(json.dumps(records(5)))
    endpoints = {
        role: Endpoint(role, role, "OpenAI", "http://127.0.0.1:9/v1", f"synthetic-{role}", "UNUSED")
        for role in ("a", "b")
    }
    store = JobStore(tmp_path / "isolated.db")
    settings = Settings(
        api_token="test-" + "a" * 40,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints=endpoints,
    )
    calls = []

    async def respond(self, prompt="", **kwargs):
        calls.append((self.model_id, kwargs.get("messages"), prompt))
        return {"content": "B" if self.model_id == "synthetic-a" else "A"}

    monkeypatch.setattr(QualityEvaluator, "_get_response_with_metrics", respond)
    client = TestClient(create_app(settings, store))
    auth = {"Authorization": "Bearer " + settings.api_token}
    body = {
        "endpoint_id_a": "a",
        "endpoint_id_b": "b",
        "parameters": {
            "datasets": ["ceval"],
            "max_samples": None,
            "num_shots": 0,
            "use_cache": False,
            "concurrency": 1,
        },
    }
    return store, settings, manager, client, auth, body, calls


def submit(lab, **headers):
    _, _, _, client, auth, body, _ = lab
    response = client.post("/api/v1/comparisons", json=body, headers={**auth, **headers})
    assert response.status_code == 201, response.text
    return response.json()


def report(settings, job):
    return json.loads((settings.artifact_root / job["result_artifact"]).read_text())


@pytest.mark.asyncio
async def test_two_models_reuse_exact_frozen_plan_even_after_source_changes(lab):
    store, settings, manager, client, auth, _, calls = lab
    group = submit(lab)
    a, b = group["jobs"]
    first = store.claim("fixture")
    assert first["job_id"] == a["job_id"]
    assert store.claim("other") is None
    await run_claimed_job(first, settings, store, "fixture")
    (manager.get_local_path("ceval") / "test.json").write_text("[]")
    second = store.claim("fixture")
    assert second["job_id"] == b["job_id"]
    await run_claimed_job(second, settings, store, "fixture")
    final_a, final_b = store.get(a["job_id"]), store.get(b["job_id"])
    assert final_a["status"] == final_b["status"] == "completed"
    left, right = (
        report(settings, final_a)["datasets"]["ceval"],
        report(settings, final_b)["datasets"]["ceval"],
    )
    assert left["config"]["shared_plan"] == right["config"]["shared_plan"]
    assert (
        left["config"]["dataset_provenance"]["sample_sha256"]
        == right["config"]["dataset_provenance"]["sample_sha256"]
    )
    assert left["config"]["scoring_contract"] == right["config"]["scoring_contract"]
    assert [item["sample_id"] for item in left["details"]] == [
        item["sample_id"] for item in right["details"]
    ]
    assert len(calls) == 12 and [call[1] for call in calls[:6]] == [call[1] for call in calls[6:]]
    result = client.post(
        "/api/v1/compare", json={"job_id_a": a["job_id"], "job_id_b": b["job_id"]}, headers=auth
    )
    assert result.status_code == 200, result.text
    entry = result.json()["datasets"]["ceval"]
    assert (
        entry["verified"] and entry["samples"] == 6 and entry["p_value"] == pytest.approx(0.03125)
    )


def test_submission_auth_validation_and_atomic_idempotency(lab):
    store, _, _, client, auth, body, _ = lab
    assert client.post("/api/v1/comparisons", json=body).status_code == 401
    assert (
        client.post(
            "/api/v1/comparisons", json={**body, "endpoint_id_b": "missing"}, headers=auth
        ).status_code
        == 422
    )
    assert store.list()[1] == 0
    first = submit(lab, **{"Idempotency-Key": "same-comparison"})
    second = submit(lab, **{"Idempotency-Key": "same-comparison"})
    assert first == second and store.list()[1] == 2
    changed = {**body, "parameters": {**body["parameters"], "num_shots": 1}}
    assert (
        client.post(
            "/api/v1/comparisons",
            json=changed,
            headers={**auth, "Idempotency-Key": "same-comparison"},
        ).status_code
        == 409
    )
    assert "synthetic" not in json.dumps(first).replace("synthetic-a", "").replace(
        "synthetic-b", ""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["input", "membership", "metadata"])
async def test_corrupt_shared_plan_blocks_all_model_b_calls(lab, damage):
    store, settings, _, _, _, _, calls = lab
    group = submit(lab)
    a, b = group["jobs"]
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    with store._connection() as conn:
        if damage == "input":
            conn.execute(
                "UPDATE checkpoint_units SET input_json='{}' WHERE job_id=? AND unit_index=0",
                (a["job_id"],),
            )
        elif damage == "membership":
            conn.execute(
                "DELETE FROM checkpoint_units WHERE job_id=? AND unit_index=0", (a["job_id"],)
            )
        else:
            conn.execute(
                "UPDATE checkpoint_scopes SET metadata_json='{}' WHERE job_id=?", (a["job_id"],)
            )
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    assert store.get(b["job_id"])["status"] == "failed" and len(calls) == 6


@pytest.mark.asyncio
async def test_missing_complete_leader_plan_does_not_reload_or_download(lab):
    store, settings, manager, _, _, _, calls = lab
    group = submit(lab)
    (manager.get_local_path("ceval") / "test.json").unlink()
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    assert all(store.get(job["job_id"])["status"] == "failed" for job in group["jobs"])
    assert not calls and manager.auto_download is False


@pytest.mark.asyncio
async def test_scoring_source_change_blocks_b_before_any_request(lab, monkeypatch):
    store, settings, _, _, _, _, calls = lab
    group = submit(lab)
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    monkeypatch.setattr("core.quality_evaluator.scoring_fingerprint", lambda evaluator: "f" * 64)
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    assert store.get(group["jobs"][1]["job_id"])["status"] == "failed" and len(calls) == 6


@pytest.mark.asyncio
async def test_stopped_b_recovers_same_plan_without_repeating_committed_sample(lab, monkeypatch):
    store, settings, _, _, _, _, calls = lab
    group = submit(lab)
    b = group["jobs"][1]
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    original = QualityEvaluator._get_response_with_metrics

    async def stop_after_one(self, *args, **kwargs):
        response = await original(self, *args, **kwargs)
        if len(calls) == 7:
            store.request_cancel(b["job_id"])
        return response

    monkeypatch.setattr(QualityEvaluator, "_get_response_with_metrics", stop_after_one)
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    stopped = store.get(b["job_id"])
    assert stopped["status"] == "cancelled"
    monkeypatch.setattr(QualityEvaluator, "_get_response_with_metrics", original)
    recover_job(store, stopped, settings.endpoints["b"])
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    assert store.get(b["job_id"])["status"] == "completed" and len(calls) == 12


def test_group_cancel_cancels_both_queued_sides(lab):
    store, _, _, client, auth, _, _ = lab
    group = submit(lab)
    result = client.post(f"/api/v1/jobs/batch/{group['batch_id']}/cancel", headers=auth)
    assert result.status_code == 200
    assert all(store.get(job["job_id"])["status"] == "cancelled" for job in group["jobs"])


@pytest.mark.asyncio
async def test_every_selected_dataset_is_frozen_before_first_model_request(lab, monkeypatch):
    store, settings, manager, _, _, body, _ = lab
    path = manager.get_local_path("gsm8k")
    path.mkdir(parents=True)
    (path / "test.json").write_text(json.dumps([{"question": "1+1?", "answer": "2"}]))
    body["parameters"]["datasets"] = ["ceval", "gsm8k"]
    group = submit(lab)
    leader = group["jobs"][0]["job_id"]
    original = QualityEvaluator._get_response_with_metrics
    observed = []

    async def check_plan(self, *args, **kwargs):
        with store._connection() as conn:
            observed.append(
                conn.execute(
                    "SELECT COUNT(*) FROM checkpoint_scopes WHERE job_id=?", (leader,)
                ).fetchone()[0]
            )
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(QualityEvaluator, "_get_response_with_metrics", check_plan)
    await run_claimed_job(store.claim("fixture"), settings, store, "fixture")
    assert observed and all(count == 2 for count in observed)


def test_changed_endpoint_is_rejected_before_execution(lab):
    store, settings, _, _, _, _, _ = lab
    submit(lab)
    job = store.claim("fixture")
    endpoint = replace(settings.endpoints["a"], api_base_url="http://127.0.0.1:10/v1")
    with pytest.raises(CheckpointConflict, match="endpoint identity"):
        SharedQualityJournal(store, job, "fixture", endpoint)


def test_standard_quality_export_uses_current_sample_fields_for_failures():
    sample = SampleResult(
        sample_id="one",
        question="1+1?",
        correct_answer="2",
        model_response="3",
        predicted_answer="3",
        is_correct=False,
    )
    result = EvaluationResult(
        dataset_name="synthetic",
        model_id="test",
        accuracy=0,
        total_samples=1,
        correct_samples=0,
        details=[sample],
    )
    report = StandardReport.from_evaluation_result(result)
    failure = report.failure_analysis["synthetic"][0]
    assert (
        failure.expected_answer == "2"
        and failure.predicted_answer == "3"
        and failure.model_response == "3"
        and failure.question == "1+1?"
    )
