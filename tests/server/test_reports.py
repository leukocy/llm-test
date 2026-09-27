"""The report API reads persisted observations and redacts raw prompt content."""

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from core.benchmark.metrics import METRIC_CONTRACT_VERSION
from core.run_lifecycle import RunStatus
from server.api import create_app
from server.settings import Endpoint, Settings
from server.store import JobStore


def test_report_survives_job_completion_and_excludes_prompt(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    store = JobStore(db_path)
    job = store.submit(
        test_type="prefill",
        endpoint_id="lab",
        model_id="test-model",
        parameters={"token_levels": [512], "requests_per_level": 1, "max_tokens": 10},
        progress_total=1,
    )
    job_id = job["job_id"]
    store.claim("test-worker")
    version = json.dumps({"metric_contract_version": METRIC_CONTRACT_VERSION})
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO test_runs(test_id,test_type,status,model_id,config_json)
               VALUES (?, 'prefill', 'completed', 'test-model', ?)""",
            (job_id, version),
        )
        conn.execute(
            """INSERT INTO test_results(run_id,concurrency_level,input_tokens_target,ttft,tps,
               error,prompt_text,token_source,extra_metrics)
               VALUES (1,1,512,0.21,42,NULL,'confidential prompt','=SUM(1,2)',?)""",
            (version,),
        )
    assert store.sync_result_run(job_id, "test-worker") == 1
    store.finish(job_id, "test-worker", outcome=RunStatus.COMPLETED)
    token = "b" * 40
    settings = Settings(
        api_token=token,
        db_path=db_path,
        artifact_root=tmp_path / "artifacts",
        endpoints={
            "lab": Endpoint(
                "lab", "Lab", "OpenAI", "http://127.0.0.1:9010/v1", "test-model", "LAB_KEY"
            )
        },
    )
    client = TestClient(create_app(settings, store))
    headers = {"Authorization": f"Bearer {token}"}
    summary = client.get(f"/api/v1/jobs/{job_id}/summary", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["overall"]["metrics"]["ttft"]["median"] == 0.21
    assert summary.json()["integrity"]["verified"] is True
    html = client.get(f"/api/v1/jobs/{job_id}/report?format=html", headers=headers)
    assert html.status_code == 200
    assert "test-model" in html.text
    assert "confidential prompt" not in html.text
    assert "完整性核验通过" in html.text
    markdown = client.get(f"/api/v1/jobs/{job_id}/report?format=markdown", headers=headers)
    assert markdown.status_code == 200
    assert "完整性" in markdown.text and METRIC_CONTRACT_VERSION in markdown.text
    assert "confidential prompt" not in markdown.text
    results = client.get(f"/api/v1/jobs/{job_id}/results", headers=headers)
    assert "confidential prompt" not in results.text
    assert summary.json()["group_axis"] == "输入长度"
    csv_export = client.get(f"/api/v1/jobs/{job_id}/export.csv", headers=headers)
    assert csv_export.status_code == 200
    assert "confidential prompt" not in csv_export.text
    assert "'=SUM(1,2)" in csv_export.text
    assert METRIC_CONTRACT_VERSION in csv_export.text

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE test_results SET extra_metrics = NULL WHERE run_id = 1")
    assert client.get(f"/api/v1/jobs/{job_id}/summary", headers=headers).status_code == 409
    assert client.get(f"/api/v1/jobs/{job_id}/report", headers=headers).status_code == 409
