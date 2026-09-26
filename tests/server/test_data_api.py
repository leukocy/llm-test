"""Phase 1+2 新增数据/图表/管理路由的行为测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.api import create_app
from server.settings import Endpoint, Settings
from server.store import JobStore

TOKEN = "b" * 48


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """带种子数据库的 TestClient；core 单例按测试隔离。"""
    from core.database.connection import Database
    from core.database.manager import DatabaseManager

    Database._instance = None
    DatabaseManager._instance = None
    monkeypatch.setenv("LAB_API_KEY", "placeholder")
    db_path = tmp_path / "data.db"
    settings = Settings(
        api_token=TOKEN,
        db_path=db_path,
        artifact_root=tmp_path / "artifacts",
        endpoints={
            "lab": Endpoint(
                id="lab",
                label="Lab",
                provider="OpenAI",
                api_base_url="http://127.0.0.1:9010/v1",
                model_id="test-model",
                api_key_env="LAB_API_KEY",  # pragma: allowlist secret
            )
        },
    )
    # store 与业务库同文件(与生产默认一致), 测试可直接 UPDATE control_jobs
    client = TestClient(create_app(settings, JobStore(db_path)))
    manager = DatabaseManager(str(db_path))  # 单例: 与 app 内同一个
    yield client, manager, settings
    Database._instance = None
    DatabaseManager._instance = None


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _seed_run(manager, test_id: str, *, tester="alice", level="internal", machine="host-1"):
    """造一条已完成运行 + 2 条请求结果 + 引擎指标。"""
    run = manager.start_test_run(
        "concurrency",
        "test-model",
        system_info={"hardware_fingerprint": {"machine_id": machine}},
        test_id=test_id,
    )
    # 先存请求结果再完成运行——完成后聚合统计(avg_tps 等)才有数据
    manager.save_result(
        run,
        {
            "session_id": 1,
            "concurrency_level": 4,
            "ttft": 0.1,
            "tps": 50.0,
            "tpot": 0.02,
            "total_time": 1.0,
        },
    )
    manager.save_result(
        run,
        {
            "session_id": 2,
            "concurrency_level": 8,
            "ttft": 0.2,
            "tps": 45.0,
            "tpot": 0.022,
            "total_time": 1.2,
        },
    )
    manager.complete_test_run(
        run,
        success=True,
        extra_fields={
            "machine_id": machine,
            "tester": tester,
            "external_level": level,
            "engine_metrics_json": json.dumps(
                {
                    "engine_family": "vllm",
                    "sample_count": 2,
                    "preemption_total": 1,
                    "cache_config": {"kv_capacity_tokens": 1000},
                    "timeline": [
                        {"t": 1.0, "gpu_cache_usage_perc": 0.4, "num_requests_running": 4},
                        {"t": 2.0, "gpu_cache_usage_perc": 0.6, "num_requests_running": 8},
                    ],
                }
            ),
        },
    )
    return run


# ---------- specs / run_config / since_id ----------


def test_specs_endpoint_lists_nine_types_with_schemas(env):
    client, _, _ = env
    r = client.get("/api/v1/specs", headers=auth())
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 9
    assert items["concurrency"]["schema"]["type"] == "object"
    assert "dataset" in items
    assert "run_config_schema" in r.json()


def test_submit_persists_run_config_under_reserved_key(env):
    client, _, _ = env
    body = {
        "endpoint_id": "lab",
        "test_type": "concurrency",
        "parameters": {"selected_concurrencies": [1], "rounds_per_level": 1, "max_tokens": 8},
        "run_config": {"temperature": 0.3, "random_seed": 7, "skip_first_token_for_tps": True},
    }
    r = client.post("/api/v1/jobs", json=body, headers=auth())
    assert r.status_code == 201, r.text
    job = r.json()
    stored = job["parameters"]
    assert stored["_run_config"]["temperature"] == 0.3
    assert stored["_run_config"]["random_seed"] == 7
    # spec 参数字段未被污染
    assert "temperature" not in {k: v for k, v in stored.items() if not k.startswith("_")}


def test_results_since_id_returns_incremental_rows(env):
    client, manager, _ = env
    run = _seed_run(manager, "seed-res-1")
    job = client.post(
        "/api/v1/jobs",
        json={
            "endpoint_id": "lab",
            "test_type": "concurrency",
            "parameters": {"selected_concurrencies": [1], "rounds_per_level": 1, "max_tokens": 8},
        },
        headers=auth(),
    ).json()
    # 手工把 job 指到种子的 run(results 路由按 result_run_id 查)
    manager.db.execute(
        "UPDATE control_jobs SET result_run_id = ? WHERE job_id = ?",
        (run.id, job["job_id"]),
    )
    r = client.get(f"/api/v1/jobs/{job['job_id']}/results", headers=auth())
    assert r.status_code == 200
    all_items = r.json()["items"]
    assert len(all_items) == 2
    first_id = all_items[0]["id"]
    r2 = client.get(f"/api/v1/jobs/{job['job_id']}/results?since_id={first_id}", headers=auth())
    incremental = r2.json()["items"]
    assert len(incremental) == 1
    assert incremental[0]["id"] > first_id


# ---------- figures ----------


def test_trend_figure_returns_plotly_json(env):
    client, manager, _ = env
    _seed_run(manager, "seed-fig-1")
    r = client.get("/api/v1/warehouse/figures/trend?metric=decode_tps", headers=auth())
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["figure"]["data"], "趋势图无数据轨迹"
    r_bad = client.get("/api/v1/warehouse/figures/trend?metric=nope", headers=auth())
    assert r_bad.status_code == 422


def test_compare_figures_table_and_bars(env):
    client, manager, _ = env
    _seed_run(manager, "seed-cmp-1")
    _seed_run(manager, "seed-cmp-2", machine="host-2")
    r = client.post(
        "/api/v1/warehouse/figures/compare",
        json={"test_ids": ["seed-cmp-1", "seed-cmp-2"]},
        headers=auth(),
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["table"], "对比表为空"
    assert payload["missing_test_ids"] == []


def test_run_detail_figures_distributions_and_engine(env):
    client, manager, _ = env
    _seed_run(manager, "seed-det-1")
    r = client.get("/api/v1/warehouse/runs/seed-det-1/figures", headers=auth())
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["found"] is True
    assert payload["distributions"]["ttft"]["histogram"] is not None
    assert payload["engine"]["figure"] is not None
    assert payload["engine"]["summary"]["engine_family"] == "vllm"
    assert client.get("/api/v1/warehouse/runs/nope/figures", headers=auth()).status_code == 404


# ---------- 写操作 ----------


def test_runs_delete_and_unknown_ids(env):
    client, manager, _ = env
    run = _seed_run(manager, "seed-del-1")
    r = client.post(
        "/api/v1/warehouse/runs/delete",
        json={"test_ids": ["seed-del-1", "ghost"]},
        headers=auth(),
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["deleted"] == [run.id]
    assert payload["unknown_test_ids"] == ["ghost"]
    assert manager.db.fetch_one("SELECT id FROM test_runs WHERE id = ?", (run.id,)) is None


def test_metadata_patch_and_gate_review(env):
    client, manager, _ = env
    _seed_run(manager, "seed-meta-1")
    r = client.patch(
        "/api/v1/warehouse/runs/seed-meta-1/metadata",
        json={"fields": {"tester": "carol", "external_level": "review", "hacker_field": "x"}},
        headers=auth(),
    )
    assert r.status_code == 200, r.text
    row = manager.db.fetch_one(
        "SELECT tester, external_level FROM test_runs WHERE test_id = 'seed-meta-1'"
    )
    assert row["tester"] == "carol" and row["external_level"] == "review"

    r_gate = client.post("/api/v1/warehouse/runs/seed-meta-1/gate", headers=auth())
    assert r_gate.status_code == 200, r_gate.text
    gate = r_gate.json()
    assert gate["level"] in {"internal", "review", "publishable"}
    assert "config_complete" in gate["gates"]

    r_bad = client.patch(
        "/api/v1/warehouse/runs/seed-meta-1/metadata",
        json={"fields": {"hacker_field": "x"}},
        headers=auth(),
    )
    assert r_bad.status_code == 422


# ---------- cases / capability ----------


def test_cases_crud_and_capability(env):
    client, _, _ = env
    body = {
        "scenario": "客服问答",
        "model_name": "m1",
        "machine_id": "host-1",
        "quality_score": 88.0,
        "external_level": "review",
    }
    r = client.post("/api/v1/cases", json=body, headers=auth())
    assert r.status_code == 201, r.text
    case_id = r.json()["case_id"]

    listed = client.get("/api/v1/cases?scenario=客服问答", headers=auth())
    assert listed.status_code == 200
    assert any(c["case_id"] == case_id for c in listed.json()["items"])

    cap = client.get("/api/v1/cases/capability?min_level=review", headers=auth())
    assert cap.status_code == 200
    assert "markdown" in cap.json()

    assert client.delete(f"/api/v1/cases/{case_id}", headers=auth()).status_code == 204
    assert client.delete(f"/api/v1/cases/{case_id}", headers=auth()).status_code == 404


# ---------- 导出 ----------


def test_ma_test_export_and_zip(env):
    client, _, _ = env
    client.post(
        "/api/v1/cases",
        json={"scenario": "s", "model_name": "m"},
        headers=auth(),
    )
    r = client.get("/api/v1/warehouse/export?template=maTest&format=csv", headers=auth())
    assert r.status_code == 200
    assert "scenario" in r.text

    rz = client.get("/api/v1/warehouse/export.zip?format=csv", headers=auth())
    assert rz.status_code == 200
    assert rz.headers["content-type"] == "application/zip"
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(rz.content)) as zf:
        assert len(zf.namelist()) == 3


# ---------- admin ----------


def test_admin_import_csv_validation_and_success(env):
    client, _, _ = env
    bad = client.post(
        "/api/v1/admin/import/csv",
        files={"file": ("note.txt", b"hello", "text/plain")},
        headers=auth(),
    )
    assert bad.status_code == 422

    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["session_id", "tps", "metric_contract_version"])
    writer.writeheader()
    writer.writerow({"session_id": 1, "tps": 42.0, "metric_contract_version": "decode-interval-v2"})
    ok = client.post(
        "/api/v1/admin/import/csv?test_type=concurrency",
        files={"file": ("results_concurrency.csv", buf.getvalue().encode(), "text/csv")},
        headers=auth(),
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["imported"] == 1


def test_admin_db_health_and_backups(env):
    client, _, settings = env
    health = client.get("/api/v1/admin/db/health", headers=auth())
    assert health.status_code == 200
    payload = health.json()
    assert payload["tables"]["test_runs"] == 0
    assert payload["integrity"] == "ok"

    created = client.post("/api/v1/admin/backups", headers=auth())
    assert created.status_code == 201, created.text
    listed = client.get("/api/v1/admin/backups", headers=auth())
    assert any(b["path"] == created.json()["path"] for b in listed.json()["items"])

    # 目录外路径必须被拒
    outside = client.post(
        "/api/v1/admin/backups/restore",
        json={"path": str(settings.artifact_root / "nope.db")},
        headers=auth(),
    )
    assert outside.status_code == 422


# ---------- 鉴权抽查 ----------


def test_new_routes_require_auth(env):
    client, _, _ = env
    assert client.get("/api/v1/specs").status_code == 401
    assert client.get("/api/v1/warehouse/figures/trend").status_code == 401
    assert client.post("/api/v1/warehouse/runs/delete", json={"test_ids": ["x"]}).status_code == 401
    assert client.get("/api/v1/admin/db/health").status_code == 401


# ---------- dataset spec / batch / logs / environment ----------


def test_dataset_spec_requires_exactly_one_source(env):
    client, _, _ = env
    base = {"concurrency": 2, "max_tokens": 8}
    both = {**base, "dataset": "x.json", "rows": [{"prompt": "q1"}]}
    neither = dict(base)
    for params, want in ((both, 422), (neither, 422)):
        r = client.post(
            "/api/v1/jobs",
            json={"endpoint_id": "lab", "test_type": "dataset", "parameters": params},
            headers=auth(),
        )
        assert r.status_code == want, (params, r.text)
    ok = client.post(
        "/api/v1/jobs",
        json={
            "endpoint_id": "lab",
            "test_type": "dataset",
            "parameters": {**base, "rows": [{"prompt": "q1"}, {"prompt": "q2"}]},
        },
        headers=auth(),
    )
    assert ok.status_code == 201, ok.text
    job = ok.json()
    assert job["test_type"] == "dataset"
    assert job["progress_total"] == 2


def test_batch_submission_creates_children_with_parent_link(env):
    client, _, _ = env
    body = {
        "endpoint_id": "lab",
        "items": [
            {
                "test_type": "concurrency",
                "parameters": {
                    "selected_concurrencies": [1],
                    "rounds_per_level": 1,
                    "max_tokens": 8,
                },
            },
            {
                "test_type": "prefill",
                "parameters": {"token_levels": [8], "requests_per_level": 1, "max_tokens": 8},
            },
        ],
    }
    r = client.post(
        "/api/v1/jobs/batch",
        json=body,
        headers={**auth(), "Idempotency-Key": "campaign-x"},
    )
    assert r.status_code == 201, r.text
    payload = r.json()
    assert payload["batch_id"] == "campaign-x"
    assert len(payload["items"]) == 2
    assert all(j["parent_job_id"] == "campaign-x" for j in payload["items"])

    # 重放同键: 返回原任务不重复创建
    again = client.post(
        "/api/v1/jobs/batch",
        json=body,
        headers={**auth(), "Idempotency-Key": "campaign-x"},
    )
    assert [j["job_id"] for j in again.json()["items"]] == [j["job_id"] for j in payload["items"]]

    # parent_job_id 过滤
    listed = client.get("/api/v1/jobs?parent_job_id=campaign-x", headers=auth())
    assert listed.status_code == 200
    assert listed.json()["total"] == 2


def test_batch_rejects_bad_item_and_unknown_endpoint(env):
    client, _, _ = env
    bad_item = client.post(
        "/api/v1/jobs/batch",
        json={
            "endpoint_id": "lab",
            "items": [{"test_type": "concurrency", "parameters": {"wrong": 1}}],
        },
        headers=auth(),
    )
    assert bad_item.status_code == 422
    bad_endpoint = client.post(
        "/api/v1/jobs/batch",
        json={
            "endpoint_id": "ghost",
            "items": [
                {
                    "test_type": "concurrency",
                    "parameters": {
                        "selected_concurrencies": [1],
                        "rounds_per_level": 1,
                        "max_tokens": 8,
                    },
                }
            ],
        },
        headers=auth(),
    )
    assert bad_endpoint.status_code == 422


def test_job_logs_cursor_reads_jsonl(env):
    client, _, settings = env
    job = client.post(
        "/api/v1/jobs",
        json={
            "endpoint_id": "lab",
            "test_type": "concurrency",
            "parameters": {"selected_concurrencies": [1], "rounds_per_level": 1, "max_tokens": 8},
        },
        headers=auth(),
    ).json()
    job_dir = settings.artifact_root / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs.jsonl").write_text(
        '{"timestamp": 1.0, "level": "INFO", "message": "第一条"}\n'
        '{"timestamp": 2.0, "level": "WARNING", "message": "第二条"}\n',
        encoding="utf-8",
    )
    r = client.get(f"/api/v1/jobs/{job['job_id']}/logs", headers=auth())
    assert r.status_code == 200
    items = r.json()["items"]
    assert [i["message"] for i in items] == ["第一条", "第二条"]
    assert items[1]["id"] == 2

    incremental = client.get(f"/api/v1/jobs/{job['job_id']}/logs?since_id=1", headers=auth())
    assert [i["message"] for i in incremental.json()["items"]] == ["第二条"]


def test_environment_and_datasets_endpoints(env):
    client, _, _ = env
    env_r = client.get("/api/v1/environment", headers=auth())
    assert env_r.status_code == 200
    assert isinstance(env_r.json(), dict)

    ds = client.get("/api/v1/datasets", headers=auth())
    assert ds.status_code == 200
    payload = ds.json()
    assert "builtin" in payload and "custom" in payload
