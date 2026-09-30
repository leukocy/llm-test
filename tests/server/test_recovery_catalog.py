"""Recovery candidates are filtered before counting/paging, in one read snapshot."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from core.run_lifecycle import RunStatus
from server.api import create_app
from server.checkpoints import (
    CheckpointConflict,
    JobJournal,
    checkpoint_info,
    delete_checkpoint,
    recover_job,
)
from server.settings import Endpoint, Settings
from server.store import JobStore


@pytest.fixture
def catalog(tmp_path):
    store = JobStore(tmp_path / "catalog.db")
    endpoint = Endpoint("lab", "Synthetic", "OpenAI", "http://127.0.0.1:9010/v1", "m", "TEST_KEY")
    settings = Settings(
        api_token="synthetic-catalog-" + "a" * 32,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={"lab": endpoint},
    )
    return (
        store,
        endpoint,
        TestClient(create_app(settings, store)),
        {"Authorization": f"Bearer {settings.api_token}"},
    )


def saved_job(store, endpoint, *, test_type="quality", error="INTERRUPTED", plan=True):
    store.submit(test_type=test_type, endpoint_id="lab", model_id="m", parameters={})
    job = store.claim("fixture")
    journal = JobJournal(store, job, "fixture", endpoint)
    if plan:
        journal.prepare_scope("scope", lambda: ({}, [{"question": "synthetic"}]))
    store.finish(job["job_id"], "fixture", outcome=RunStatus.FAILED, error_code=error)
    return store.get(job["job_id"])


def test_catalog_finds_older_candidates_and_paginates_before_recent_job_window(catalog):
    store, endpoint, client, headers = catalog
    expected = [saved_job(store, endpoint)["job_id"] for _ in range(55)]
    # Newer non-recoverable records must not crowd candidates out of the first page.
    for _ in range(70):
        store.submit(test_type="concurrency", endpoint_id="lab", model_id="m", parameters={})
    assert not set(expected) & {job["job_id"] for job in store.list()[0]}
    path = "/api/v1/jobs?recoverable=true&limit=50"
    assert client.get(path).status_code == 401
    first = client.get(path, headers=headers).json()
    second = client.get(path + "&offset=50", headers=headers).json()
    assert first["total"] == second["total"] == 55
    assert len(first["items"]) == 50 and len(second["items"]) == 5
    assert [job["job_id"] for job in first["items"] + second["items"]] == expected[::-1]
    assert client.get(path + "&status=paused", headers=headers).json()["total"] == 0


def test_catalog_and_detail_use_same_recovery_eligibility(catalog):
    store, endpoint, client, headers = catalog
    valid = saved_job(store, endpoint, test_type="robustness", error="WORKER_LOST")
    invalid = [
        saved_job(store, endpoint, error="EXECUTION_FAILED"),
        saved_job(store, endpoint, plan=False),
        saved_job(store, endpoint, test_type="stability"),
        saved_job(store, endpoint),
    ]
    with store._connection() as conn:
        conn.execute(
            "UPDATE job_checkpoints SET contract='unknown' WHERE job_id=?", (invalid[-1]["job_id"],)
        )
    result = client.get("/api/v1/jobs?recoverable=true", headers=headers).json()
    assert [item["job_id"] for item in result["items"]] == [valid["job_id"]]
    assert checkpoint_info(store, valid)["can_recover"]
    assert all(not checkpoint_info(store, job)["can_recover"] for job in invalid)


def test_catalog_preserves_batch_scope_and_normal_status_filters(catalog):
    store, endpoint, client, headers = catalog
    first = saved_job(store, endpoint)
    second = saved_job(store, endpoint)
    with store._connection() as conn:
        conn.execute(
            "UPDATE control_jobs SET parent_job_id='one' WHERE job_id=?", (first["job_id"],)
        )
        conn.execute(
            "UPDATE control_jobs SET parent_job_id='two' WHERE job_id=?", (second["job_id"],)
        )
    response = client.get("/api/v1/jobs?recoverable=true&parent_job_id=one", headers=headers).json()
    assert response["total"] == 1 and response["items"][0]["job_id"] == first["job_id"]
    assert store.list(status=RunStatus.FAILED)[1] == 2
    assert store.list(recoverable=True, offset=100)[0] == []


def test_count_and_page_share_read_snapshot_during_concurrent_submission(catalog, monkeypatch):
    store, _, _, _ = catalog
    original = store.submit(test_type="concurrency", endpoint_id="lab", model_id="m", parameters={})
    writer = JobStore(store.path)
    connection = store._connection

    class ConcurrentInsert:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, *args):
            cursor = self.conn.execute(sql, *args)
            if sql.startswith("SELECT COUNT(*) FROM control_jobs"):
                writer.submit(
                    test_type="concurrency", endpoint_id="lab", model_id="new", parameters={}
                )
            return cursor

    @contextmanager
    def concurrent_connection():
        with connection() as conn:
            yield ConcurrentInsert(conn)

    monkeypatch.setattr(store, "_connection", concurrent_connection)
    items, total = store.list()
    assert total == len(items) == 1 and items[0]["job_id"] == original["job_id"]
    assert writer.list()[1] == 2


def test_saved_history_includes_unrecoverable_progress_and_stable_page_order(catalog):
    store, endpoint, client, headers = catalog
    first = saved_job(store, endpoint)
    second = saved_job(store, endpoint, error="EXECUTION_FAILED")
    third = saved_job(store, endpoint, plan=False)
    store.submit(test_type="concurrency", endpoint_id="lab", model_id="m", parameters={})
    with store._connection() as conn:
        conn.execute("UPDATE control_jobs SET created_at=1")
    response = client.get("/api/v1/jobs?saved_progress=true&limit=2", headers=headers).json()
    rest = client.get("/api/v1/jobs?saved_progress=true&limit=2&offset=2", headers=headers).json()
    assert response["total"] == rest["total"] == 3
    assert [r["job_id"] for r in response["items"] + rest["items"]] == [
        third["job_id"],
        second["job_id"],
        first["job_id"],
    ]
    assert response["items"][0]["saved_progress_planned"] == 0
    assert response["items"][1]["saved_progress_planned"] == 1
    assert response["items"][1]["saved_progress_committed"] == 0
    assert response["items"][1]["saved_progress_at"] > 0
    assert (
        client.get("/api/v1/jobs?saved_progress=true&recoverable=true", headers=headers).json()[
            "total"
        ]
        == 1
    )


@pytest.mark.parametrize("test_type", ["quality", "robustness"])
def test_user_stopped_run_requires_explicit_recovery_and_retains_cancel_event(catalog, test_type):
    store, endpoint, client, headers = catalog
    store.submit(test_type=test_type, endpoint_id="lab", model_id="m", parameters={})
    job = store.claim("fixture")
    journal = JobJournal(store, job, "fixture", endpoint)
    journal.prepare_scope("scope", lambda: ({}, [{"question": "fixed"}]))
    journal.start("scope", 0)
    journal.commit("scope", 0, {"answer": "fixed"})
    store.request_cancel(job["job_id"])
    store.finish(job["job_id"], "fixture", outcome=RunStatus.CANCELLED)
    cancelled = store.get(job["job_id"])
    assert checkpoint_info(store, cancelled)["can_recover"]
    assert store.claim("other") is None  # No automatic requeue on cancellation.
    assert client.get("/api/v1/jobs?recoverable=true", headers=headers).json()["total"] == 1
    restored = client.post(f"/api/v1/jobs/{job['job_id']}/recover", headers=headers)
    assert restored.status_code == 200 and restored.json()["status"] == "queued"
    claimed = store.claim("new-worker")
    assert JobJournal(store, claimed, "new-worker", endpoint).results("scope") == {
        0: {"answer": "fixed"}
    }
    events = store.events(job["job_id"])
    assert [e["event"] for e in events].count("cancel") == 1
    assert [e["event"] for e in events].count("recover") == 1
    assert next(e for e in events if e["event"] == "recover")["from_status"] == "cancelled"


def test_batch_stop_cancellation_is_not_a_user_stopped_recovery(catalog):
    store, endpoint, _, _ = catalog
    job = saved_job(store, endpoint)
    with store._connection() as conn:
        conn.execute(
            "UPDATE control_jobs SET status='cancelled',error_code='BATCH_STOP_ON_ERROR' WHERE job_id=?",
            (job["job_id"],),
        )
    current = store.get(job["job_id"])
    assert not checkpoint_info(store, current)["can_recover"]
    assert store.list(recoverable=True)[1] == 0
    with pytest.raises(CheckpointConflict):
        recover_job(store, current, endpoint)


@pytest.mark.parametrize("reason", [None, "BATCH_STOP_ON_ERROR"])
def test_expired_cancelling_lease_retains_stop_intent_and_reason(catalog, reason):
    store, endpoint, _, _ = catalog
    store.submit(test_type="quality", endpoint_id="lab", model_id="m", parameters={})
    job = store.claim("fixture")
    journal = JobJournal(store, job, "fixture", endpoint)
    journal.prepare_scope("scope", lambda: ({}, [{"question": "fixed"}]))
    store.request_cancel(job["job_id"])
    with store._connection() as conn:
        conn.execute(
            "UPDATE control_jobs SET lease_until=0,error_code=? WHERE job_id=?",
            (reason, job["job_id"]),
        )
    assert store.reap_expired() == 1
    current = store.get(job["job_id"])
    assert current["status"] == "cancelled" and current["error_code"] == reason
    assert checkpoint_info(store, current)["can_recover"] == (reason is None)
    assert store.events(job["job_id"])[-1]["event"] == "cancel"


def test_checkpoint_delete_requires_version_and_preserves_results_report_and_audit(catalog):
    store, endpoint, client, headers = catalog
    job = saved_job(store, endpoint)
    artifact = store.path.parent / "artifacts" / "report.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        {"datasets": {"synthetic": {"accuracy": 1, "total_samples": 1, "correct_samples": 1}}}
    )
    artifact.write_text(raw)
    with store._connection() as conn:
        run_id = conn.execute(
            "INSERT INTO test_runs(test_id,test_type,model_id) VALUES (?,'quality','m')",
            (job["job_id"],),
        ).lastrowid
        conn.execute(
            "INSERT INTO test_results(run_id,output_text) VALUES (?,'preserve observation')",
            (run_id,),
        )
        conn.execute(
            "UPDATE control_jobs SET result_artifact='report.json' WHERE job_id=?",
            (job["job_id"],),
        )
    path = f"/api/v1/jobs/{job['job_id']}/checkpoint"
    info = client.get(path, headers=headers)
    assert info.json()["can_delete"] and info.headers["etag"] == f'"{info.json()["revision"]}"'
    assert client.delete(path).status_code == 401
    assert client.delete(path, headers=headers).status_code == 428
    assert client.delete(path, headers={**headers, "If-Match": '"wrong"'}).status_code == 409
    before_events = store.events(job["job_id"])
    deleted = client.delete(path, headers={**headers, "If-Match": info.headers["etag"]})
    assert deleted.status_code == 200 and not deleted.json()["available"]
    assert (
        client.delete(path, headers={**headers, "If-Match": info.headers["etag"]}).status_code
        == 200
    )
    assert (
        client.get(f"/api/v1/jobs/{job['job_id']}/report", headers=headers).json()["datasets"][
            "synthetic"
        ]["accuracy"]
        == 1
    )
    assert artifact.read_text() == raw
    assert store.get(job["job_id"])["status"] == "failed"
    assert len(store.events(job["job_id"])) == len(before_events) + 1
    with store._connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM test_results WHERE run_id=?", (run_id,)).fetchone()[
                0
            ]
            == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM checkpoint_units").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM checkpoint_scopes").fetchone()[0] == 0
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    assert store.list(saved_progress=True)[1] == store.list(recoverable=True)[1] == 0
    assert client.post(f"/api/v1/jobs/{job['job_id']}/recover", headers=headers).status_code == 409


@pytest.mark.parametrize("status", ["queued", "running", "pausing", "paused", "cancelling"])
def test_active_checkpoint_cannot_be_deleted(catalog, status):
    store, endpoint, client, headers = catalog
    job = saved_job(store, endpoint)
    version = checkpoint_info(store, job)["revision"]
    with store._connection() as conn:
        conn.execute("UPDATE control_jobs SET status=? WHERE job_id=?", (status, job["job_id"]))
    response = client.delete(
        f"/api/v1/jobs/{job['job_id']}/checkpoint", headers={**headers, "If-Match": version}
    )
    assert response.status_code == 409
    assert (
        checkpoint_info(store, job)["available"] and not checkpoint_info(store, job)["can_delete"]
    )


def test_delete_rejects_stale_checkpoint_version(catalog):
    store, endpoint, _, _ = catalog
    job = saved_job(store, endpoint)
    previous = checkpoint_info(store, job)["revision"]
    with store._connection() as conn:
        conn.execute(
            "UPDATE job_checkpoints SET updated_at=updated_at+1 WHERE job_id=?", (job["job_id"],)
        )
    with pytest.raises(CheckpointConflict, match="已变更"):
        delete_checkpoint(store, job, previous)
    current = checkpoint_info(store, job)
    assert current["revision"] != previous and current["available"]


def test_recovery_and_deletion_are_mutually_exclusive_transactions(catalog):
    store, endpoint, _, _ = catalog
    job = saved_job(store, endpoint)
    revision = checkpoint_info(store, job)["revision"]

    def recover_or_delete(action):
        try:
            if action == "recover":
                recover_job(JobStore(store.path), job, endpoint)
            else:
                delete_checkpoint(JobStore(store.path), job, revision)
            return action
        except CheckpointConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(recover_or_delete, ["recover", "delete"]))
    assert outcomes.count("conflict") == 1
    info = checkpoint_info(store, job)
    current = store.get(job["job_id"])
    assert (current["status"] == "queued" and info["available"]) or (
        current["status"] == "failed" and not info["available"]
    )
