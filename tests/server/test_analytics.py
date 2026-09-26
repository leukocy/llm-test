"""Statistical contract and escaped standalone report."""

import sqlite3
from pathlib import Path

from core.database.schema import create_tables
from server.analytics import run_summary
from server.reports import render_html, render_quality_html


def test_summary_excludes_failed_latency_and_keeps_failure_rate(tmp_path: Path):
    path = tmp_path / "results.db"
    with sqlite3.connect(path) as conn:
        create_tables(conn)
        conn.execute(
            "INSERT INTO test_runs(test_id,test_type,model_id) VALUES ('j','concurrency','m')"
        )
        for value, error in [(0.2, None), (0.4, None), (99.0, "timeout")]:
            conn.execute(
                """INSERT INTO test_results(run_id,concurrency_level,ttft,tps,error)
                   VALUES (1,4,?,10,?)""",
                (value, error),
            )
    data = run_summary(str(path), 1)
    assert data["overall"]["requests"] == 3
    assert data["overall"]["failures"] == 1
    assert data["overall"]["metrics"]["ttft"]["median"] == 0.30000000000000004
    assert data["overall"]["metrics"]["ttft"]["count"] == 2
    assert data["group_axis"] == "并发"
    assert data["groups"][0]["label"] == "4 并发"
    assert data["overall"]["success_rate_ci95"][0] < 2 / 3
    html = render_html(
        {
            "job_id": "j",
            "model_id": "<script>alert(1)</script>",
            "test_type": "concurrency",
            "status": "completed",
        },
        data,
    )
    assert "&lt;script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "66.7%" in html


def test_quality_report_escapes_dataset_and_displays_uncertainty():
    html = render_quality_html(
        {"job_id": "j", "model_id": "m", "status": "completed"},
        {
            "datasets": {
                "<script>alert(1)</script>": {
                    "accuracy": 0.7,
                    "correct_samples": 7,
                    "total_samples": 10,
                    "config": {
                        "dataset_provenance": {"source": "verified", "sample_sha256": "abc"}
                    },
                    "extended_metrics": {"wilson_ci_lower": 0.4, "wilson_ci_upper": 0.9},
                }
            }
        },
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "40.0%–90.0%" in html
