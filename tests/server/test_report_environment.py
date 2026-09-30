"""User-entered report context survives admission and execution without changing host identity."""

import json
import sqlite3
from dataclasses import make_dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from core.benchmark.metrics import METRIC_CONTRACT_VERSION
from core.run_lifecycle import RunStatus
from server.api import create_app
from server.runner_adapter import execute_job
from server.settings import Endpoint, Settings
from server.specs import ReportEnvironment, describe_report_environment
from server.store import JobStore


@pytest.fixture
def platform(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_KEY", "synthetic-key")
    store = JobStore(tmp_path / "platform.db")
    endpoint = Endpoint("lab", "Lab", "OpenAI", "http://127.0.0.1:9010/v1", "m", "LAB_KEY")
    settings = Settings(
        api_token="a" * 40,
        db_path=store.path,
        artifact_root=tmp_path / "artifacts",
        endpoints={"lab": endpoint},
    )
    client = TestClient(create_app(settings, store))
    return client, store, settings, endpoint, {"Authorization": "Bearer " + "a" * 40}


@pytest.fixture
def environment():
    return {
        "scope": "model_server",
        "processor": "  Remote CPU  ",
        "mainboard": "Remote board",
        "memory": "512 GB",
        "gpu": "2 × H100",
        "system": "Linux",
        "engine_name": "vLLM FP8 TP=2",
    }


@pytest.mark.parametrize(
    "invalid",
    [
        {"gpu": "x" * 241},
        {"gpu": 8},
        {"processor": "a\nb"},
        {"system": "a\x7fb"},
        {"scope": "auto_verified"},
        {"api_key": "unsupported"},  # pragma: allowlist secret
    ],
)
def test_report_environment_rejects_unbounded_or_ambiguous_values(invalid):
    with pytest.raises(ValidationError):
        ReportEnvironment.model_validate(invalid)


def test_blank_environment_is_absent_and_values_are_user_reported(environment):
    assert describe_report_environment({"gpu": "   "}) is None
    assert describe_report_environment(None) is None
    described = describe_report_environment(environment)
    assert described["source"] == "user_reported"
    assert described["scope"] == "model_server"
    assert described["fields"]["processor"] == "Remote CPU"


def test_environment_roundtrips_through_preset_single_job_and_batch(platform, environment):
    client, store, _, _, headers = platform
    payload = {
        "endpoint_id": "lab",
        "test_type": "concurrency",
        "parameters": {"selected_concurrencies": [1], "rounds_per_level": 1, "max_tokens": 8},
        "run_config": {"report_environment": environment},
    }
    saved = client.post(
        "/api/v1/presets", json={**payload, "name": "Report context"}, headers=headers
    )
    assert saved.status_code == 201
    config = saved.json()["run_config"]
    assert config["report_environment"]["processor"] == "Remote CPU"
    assert client.get("/api/v1/presets", headers=headers).json()["items"][0]["run_config"] == config
    planned = client.post("/api/v1/jobs/plan", json=payload, headers=headers)
    assert planned.status_code == 200
    assert store.list()[1] == 0
    created = client.post("/api/v1/jobs", json=payload, headers=headers)
    assert created.status_code == 201
    assert created.json()["parameters"]["_run_config"] == config
    batched = client.post(
        "/api/v1/jobs/batch", json={"endpoint_id": "lab", "items": [payload]}, headers=headers
    )
    assert batched.status_code == 201
    assert batched.json()["items"][0]["parameters"]["_run_config"] == config
    rejected = client.post(
        "/api/v1/jobs",
        json={**payload, "run_config": {"report_environment": {"gpu": 2}}},
        headers=headers,
    )
    assert rejected.status_code == 422


def _performance_result(store, job_id, environment):
    version = {"metric_contract_version": METRIC_CONTRACT_VERSION}
    with sqlite3.connect(store.path) as conn:
        run_id = conn.execute(
            """INSERT INTO test_runs(test_id,test_type,status,model_id,config_json,system_info_json)
               VALUES (?, 'concurrency', 'completed', 'm', ?, ?)""",
            (
                job_id,
                json.dumps({**version, "report_environment": environment}),
                json.dumps({"processor": "Actual worker CPU", "machine_id": "worker-id"}),
            ),
        ).lastrowid
        conn.executemany(
            """INSERT INTO test_results(run_id,concurrency_level,ttft,tps,error,extra_metrics)
               VALUES (?,1,?,10,?,?)""",
            [
                (run_id, value, error, json.dumps(version))
                for value, error in [(0.2, None), (0.4, None), (99, "timeout"), (0, None)]
            ],
        )
    return run_id


@pytest.mark.parametrize("test_type", ["concurrency", "quality", "robustness"])
def test_environment_in_json_html_and_markdown_is_escaped(platform, environment, test_type):
    client, store, settings, _, headers = platform
    environment["processor"] = "<script>alert(1)</script> | [click](javascript:alert(2))"
    job = store.submit(
        test_type=test_type, endpoint_id="lab", model_id="m", parameters={}, progress_total=4
    )
    store.claim("w")
    if test_type == "concurrency":
        run_id = _performance_result(store, job["job_id"], environment)
        store.finish(job["job_id"], "w", outcome=RunStatus.COMPLETED, result_run_id=run_id)
    else:
        settings.artifact_root.mkdir()
        artifact = settings.artifact_root / "report.json"
        artifact.write_text(
            json.dumps(
                {
                    "report_environment": describe_report_environment(environment),
                    **(
                        {
                            "datasets": {
                                "gsm8k": {"accuracy": 0.5, "correct_samples": 1, "total_samples": 2}
                            }
                        }
                        if test_type == "quality"
                        else {"robustness": {"total_samples": 2}}
                    ),
                }
            )
        )
        store.finish(job["job_id"], "w", outcome=RunStatus.COMPLETED, result_artifact="report.json")
    root = f"/api/v1/jobs/{job['job_id']}/report"
    data = client.get(root, headers=headers).json()
    report = data["summary"] if test_type == "concurrency" else data
    assert report["report_environment"]["fields"]["gpu"] == "2 × H100"
    if test_type == "concurrency":
        auto = json.loads(report["provenance"]["system_info_json"])
        assert auto["processor"] == "Actual worker CPU" and auto["machine_id"] == "worker-id"
    for format in ["html", "markdown"]:
        response = client.get(root + f"?format={format}", headers=headers)
        assert response.status_code == 200
        assert "用户填写，未经自动核验" in response.text
        assert "受测模型服务器" in response.text
        assert "2 × H100" in response.text
        assert "&lt;script&gt;" in response.text and "<script>" not in response.text
        if format == "markdown":
            assert "\\[click\\]" in response.text and "\\|" in response.text


def test_png_figure_uses_report_statistics_and_enforces_contract(platform, environment):
    client, store, _, _, headers = platform
    job = store.submit(
        test_type="concurrency", endpoint_id="lab", model_id="m", parameters={}, progress_total=4
    )
    store.claim("w")
    run_id = _performance_result(store, job["job_id"], environment)
    store.finish(job["job_id"], "w", outcome=RunStatus.COMPLETED, result_run_id=run_id)
    root = f"/api/v1/jobs/{job['job_id']}"
    assert client.get(root + "/figure").status_code == 401
    response = client.get(root + "/figure", headers=headers)
    assert response.status_code == 200
    policy = response.headers["Content-Security-Policy"]
    assert "img-src 'self' data: blob:;" in policy
    assert "script-src 'self';" in policy and "connect-src 'self';" in policy
    figure = response.json()["figure"]
    summary = client.get(root + "/summary", headers=headers).json()
    assert figure["data"][0]["y"] == [summary["groups"][0]["metrics"]["ttft"]["median"]]
    assert figure["data"][1]["y"] == [summary["groups"][0]["metrics"]["ttft"]["p95"]]
    assert figure["data"][1]["text"] == ["n=2"]
    assert figure["layout"]["yaxis"]["title"]["text"] == "TTFT (s)"
    caption = figure["layout"]["annotations"][0]["text"]
    assert "正式请求 4 · 失败 1" in caption and "用户填写" in caption
    assert METRIC_CONTRACT_VERSION in caption
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE test_results SET ttft=0")
    assert client.get(root + "/figure", headers=headers).status_code == 409
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE test_results SET ttft=0.2, extra_metrics=NULL")
    assert client.get(root + "/figure", headers=headers).status_code == 409
    assert client.get("/api/v1/jobs/missing/figure", headers=headers).status_code == 404
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE test_results SET extra_metrics=?",
            (json.dumps({"metric_contract_version": METRIC_CONTRACT_VERSION}),),
        )
        conn.execute(
            "UPDATE test_runs SET config_json=?",
            (
                json.dumps(
                    {
                        "metric_contract_version": METRIC_CONTRACT_VERSION,
                        "report_environment": {"gpu": 2},
                    }
                ),
            ),
        )
    invalid = client.get(root + "/figure", headers=headers)
    assert invalid.status_code == 409 and "Invalid user-reported" in invalid.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("test_type", ["quality", "robustness"])
async def test_adapter_persists_environment_in_report_artifacts(
    platform, environment, monkeypatch, test_type
):
    _, store, settings, endpoint, _ = platform
    params = {"_run_config": {"report_environment": environment}}
    if test_type == "quality":
        params.update(datasets=["gsm8k"], max_samples=1)
        evaluator = MagicMock()
        evaluator.run_evaluation = AsyncMock(
            return_value={
                "gsm8k": SimpleNamespace(
                    total_samples=1, to_dict=lambda: {"accuracy": 1, "total_samples": 1}
                ),
            }
        )
        monkeypatch.setattr(
            "core.quality_evaluator.QualityEvaluator", MagicMock(return_value=evaluator)
        )
    elif test_type == "robustness":
        params.update(samples=[{"question": "1+1?", "correct_answer": "2"}])
        report_type = make_dataclass("Report", [("model_id", str), ("results", list)])
        tester = MagicMock()
        tester.test_batch = AsyncMock(return_value=report_type("", []))
        monkeypatch.setattr(
            "core.robustness_tester.RobustnessTester", MagicMock(return_value=tester)
        )
    job = store.submit(test_type=test_type, endpoint_id="lab", model_id="m", parameters=params)
    job = store.claim("w")
    output = await execute_job(job, endpoint, settings, store, "w")
    artifact = json.loads((settings.artifact_root / output.result_artifact).read_text())
    assert artifact["report_environment"] == describe_report_environment(environment)


@pytest.mark.asyncio
async def test_adapter_passes_environment_to_performance_runner(platform, environment, monkeypatch):
    _, store, settings, endpoint, _ = platform
    runner = MagicMock(last_run_id=1, results_list=[{}], total_requests=1, _db_run=None)
    runner.run_concurrency_test = AsyncMock()
    constructor = MagicMock(return_value=runner)
    monkeypatch.setattr("core.benchmark_runner.BenchmarkRunner", constructor)
    job = store.submit(
        test_type="concurrency",
        endpoint_id="lab",
        model_id="m",
        parameters={"_run_config": {"report_environment": environment}},
    )
    store.claim("w")
    await execute_job(job, endpoint, settings, store, "w")
    assert constructor.call_args.kwargs["report_environment"] == environment
    runner.run_concurrency_test.assert_awaited_once_with()
