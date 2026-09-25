"""Run lifecycle invariants at the domain and persistence boundaries."""

from unittest.mock import patch

import pytest

from core.database.connection import Database
from core.models.test_run import TestRun
from core.repositories.test_run import TestRunRepository as RunRepository
from core.run_lifecycle import (
    InvalidRunTransition,
    RunEvent,
    RunStatus,
    advance_run,
)


def test_pause_resume_and_terminal_states():
    assert advance_run(RunStatus.CREATED, RunEvent.ENQUEUE) == RunStatus.QUEUED
    assert advance_run(RunStatus.QUEUED, RunEvent.START) == RunStatus.RUNNING
    assert advance_run(RunStatus.RUNNING, RunEvent.REQUEST_PAUSE) == RunStatus.PAUSING
    assert advance_run(RunStatus.PAUSING, RunEvent.PAUSE) == RunStatus.PAUSED
    assert advance_run(RunStatus.PAUSED, RunEvent.RESUME) == RunStatus.RUNNING
    assert advance_run(RunStatus.RUNNING, RunEvent.COMPLETE) == RunStatus.COMPLETED
    with pytest.raises(InvalidRunTransition):
        advance_run(RunStatus.COMPLETED, RunEvent.START)
    with pytest.raises(InvalidRunTransition):
        advance_run(RunStatus.CANCELLED, RunEvent.COMPLETE)


def test_model_uses_validated_transitions():
    run = TestRun.create("concurrency", "model")
    run.pause()
    run.resume()
    run.cancel()
    assert run.status == RunStatus.CANCELLED.value
    with pytest.raises(InvalidRunTransition):
        run.complete()


@pytest.fixture
def run_repo(tmp_path):
    Database._instance = None
    db = Database(str(tmp_path / "lifecycle.db"))
    try:
        yield RunRepository(db), db
    finally:
        Database._instance = None


def test_repository_rejects_progress_and_completion_after_cancellation(run_repo):
    repo, db = run_repo
    run_id = repo.insert(TestRun.create("concurrency", "model"))
    assert run_id is not None
    assert repo.advance(run_id, RunEvent.REQUEST_CANCEL)
    assert repo.advance(run_id, RunEvent.CANCEL)
    assert repo.update_progress(run_id, completed=9, total=10) is False
    with pytest.raises(InvalidRunTransition):
        repo.update_status(run_id, RunStatus.CANCELLED.value, progress=90)
    with pytest.raises(InvalidRunTransition):
        repo.complete(run_id)
    row = db.fetch_one("SELECT status, completed_requests FROM test_runs WHERE id = ?", (run_id,))
    assert row is not None
    assert row["status"] == RunStatus.CANCELLED.value
    assert row["completed_requests"] == 0


def test_repository_compare_and_set_preserves_concurrent_cancellation(run_repo):
    repo, db = run_repo
    run_id = repo.insert(TestRun.create("concurrency", "model"))
    assert run_id is not None
    update_by = repo.update_by

    def cancel_before_complete(data, where, params):
        db.execute(
            "UPDATE test_runs SET status = ? WHERE id = ?",
            (RunStatus.CANCELLED.value, run_id),
        )
        return update_by(data, where, params)

    with patch.object(repo, "update_by", side_effect=cancel_before_complete):
        assert repo.complete(run_id) is False
    row = db.fetch_one("SELECT status, completed_at FROM test_runs WHERE id = ?", (run_id,))
    assert row is not None
    assert row["status"] == RunStatus.CANCELLED.value
    assert row["completed_at"] is None


def test_full_model_update_cannot_resurrect_terminal_run(run_repo):
    repo, db = run_repo
    stale_run = TestRun.create("concurrency", "model")
    stale_run.id = repo.insert(stale_run)
    assert stale_run.id is not None
    assert repo.advance(stale_run.id, RunEvent.CANCEL)

    with pytest.raises(InvalidRunTransition):
        repo.update(stale_run)

    row = db.fetch_one("SELECT status FROM test_runs WHERE id = ?", (stale_run.id,))
    assert row is not None
    assert row["status"] == RunStatus.CANCELLED.value
