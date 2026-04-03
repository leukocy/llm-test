"""Tests for database persistence (SQLAlchemy ORM)."""

import pytest
from engine.persistence.models import (
    Base, TestRunModel, TestResultModel, ExecLogModel,
    create_db_engine, init_database,
)
from engine.persistence.db_writer import DatabaseWriter


@pytest.fixture
def db_writer(tmp_path):
    """Create a DatabaseWriter with an in-memory SQLite database."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    return DatabaseWriter(db_url)


def test_start_run(db_writer):
    run = db_writer.start_run(
        test_type="concurrency",
        model_id="gpt-4",
        provider="openai",
    )
    assert run.id is not None
    assert run.test_type == "concurrency"
    assert run.model_id == "gpt-4"
    assert run.status == "running"


def test_save_result(db_writer):
    run = db_writer.start_run("concurrency", "gpt-4")
    
    db_writer.save_result(run.id, {
        "session_id": 1,
        "ttft": 0.15,
        "tps": 50.0,
        "prefill_tokens": 100,
        "decode_tokens": 50,
    })
    
    # Verify via a raw query
    from sqlalchemy.orm import Session
    with db_writer._session() as session:
        results = session.query(TestResultModel).filter_by(run_id=run.id).all()
        assert len(results) == 1
        assert results[0].ttft == 0.15
        assert results[0].tps == 50.0


def test_save_results_batch(db_writer):
    run = db_writer.start_run("prefill", "gpt-4")
    
    results = [
        {"session_id": i, "ttft": 0.1 * i, "tps": 50.0 - i}
        for i in range(1, 6)
    ]
    
    db_writer.save_results_batch(run.id, results)
    
    with db_writer._session() as session:
        count = session.query(TestResultModel).filter_by(run_id=run.id).count()
        assert count == 5


def test_complete_run(db_writer):
    run = db_writer.start_run("concurrency", "gpt-4")
    
    db_writer.complete_run(
        run.id,
        success=True,
        stats={"avg_ttft": 0.15, "avg_tps": 48.0, "total_requests": 10},
        csv_path="/results/test.csv",
        duration=12.5,
    )
    
    with db_writer._session() as session:
        updated = session.get(TestRunModel, run.id)
        assert updated.status == "completed"
        assert updated.avg_ttft == 0.15
        assert updated.avg_tps == 48.0
        assert updated.duration_seconds == 12.5
        assert updated.csv_path == "/results/test.csv"


def test_complete_run_failed(db_writer):
    run = db_writer.start_run("concurrency", "gpt-4")
    db_writer.complete_run(run.id, success=False)
    
    with db_writer._session() as session:
        updated = session.get(TestRunModel, run.id)
        assert updated.status == "failed"


def test_save_run_convenience(db_writer):
    results = [
        {"session_id": 1, "ttft": 0.1, "tps": 50.0},
        {"session_id": 2, "ttft": 0.2, "tps": 45.0},
        {"session_id": 3, "ttft": 0.3, "tps": 40.0},
    ]
    
    run_id = db_writer.save_run(
        test_type="concurrency",
        model_id="qwen-7b",
        results=results,
        duration=5.0,
        provider="local",
    )
    
    assert run_id is not None
    
    with db_writer._session() as session:
        run = session.get(TestRunModel, run_id)
        assert run.status == "completed"
        assert run.total_requests == 3
        assert run.duration_seconds == 5.0
        
        result_count = session.query(TestResultModel).filter_by(run_id=run_id).count()
        assert result_count == 3


def test_log_execution(db_writer):
    run = db_writer.start_run("concurrency", "gpt-4")
    
    db_writer.log_execution(run.id, "Test started", level="INFO")
    db_writer.log_execution(run.id, "Request failed", level="ERROR", metrics={"retry": 1})
    
    with db_writer._session() as session:
        logs = session.query(ExecLogModel).filter_by(run_id=run.id).all()
        assert len(logs) == 2
        assert logs[0].level == "INFO"
        assert logs[1].level == "ERROR"


def test_get_recent_runs(db_writer):
    # Create a few runs
    for i in range(3):
        db_writer.save_run(
            test_type="concurrency",
            model_id=f"model-{i}",
            results=[{"session_id": 1, "ttft": 0.1}],
            duration=1.0,
        )
    
    recent = db_writer.get_recent_runs(limit=2)
    assert len(recent) == 2
    assert recent[0]["model_id"] == "model-2"  # Most recent first


def test_init_database(tmp_path):
    db_url = f"sqlite:///{tmp_path}/init_test.db"
    engine = init_database(db_url)
    
    # Verify tables exist
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    assert "test_runs" in tables
    assert "test_results" in tables
    assert "api_logs" in tables
    assert "execution_logs" in tables
    assert "reports" in tables
    assert "db_meta" in tables
