"""Saved-file parity: scientific statistics, safe snapshots and complete browser API workflow."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from server.api import create_app
from server.history import HistoryError, HistoryNotFound, SavedCsvHistory
from server.settings import Settings
from server.store import JobStore

CSV = b"concurrency,ttft,tps,prefill_tokens,token_calc_method,error\n1,1,10,100,API usage,\n1,3,30,100,API usage,\n2,100,100,100,API usage,timeout\n2,NaN,-1,100,API usage,\n"
REDACTION_MARKER = "fixture-redaction-marker"
META = {
    "test_type": "Concurrency Test",
    "model_id": "historical-model",
    "provider": "Local",
    "duration": 3,
    "test_config": {"Concurrency Levels": [1, 2], "api_key": REDACTION_MARKER},
    "system_info": {"engine_name": "vLLM"},
}


@pytest.fixture
def environment(tmp_path):
    from core.database.connection import Database
    from core.database.manager import DatabaseManager

    Database._instance = None
    DatabaseManager._instance = None
    root = tmp_path / "history"
    path = root / "model-a" / "benchmark_results_model-a_Concurrency_Test_20260528_120000.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(CSV)
    path.with_suffix(".csv.meta.json").write_text(json.dumps(META))
    settings = Settings(
        api_token="h" * 48,
        db_path=tmp_path / "platform.db",
        artifact_root=tmp_path / "artifacts",
        endpoints={},
        history_root=root,
    )
    store = JobStore(settings.db_path)
    client = TestClient(create_app(settings, store))
    history = SavedCsvHistory(root)
    headers = {"Authorization": "Bearer " + "h" * 48}
    yield client, history, store, headers, path
    Database._instance = None
    DatabaseManager._instance = None


def test_catalog_and_reports_restore_legacy_metadata_without_creating_jobs(environment):
    client, history, store, headers, _ = environment
    catalog = client.get("/api/v1/history", headers=headers)
    assert catalog.status_code == 200
    entry = catalog.json()["items"][0]
    assert entry["model_id"] == "historical-model"
    assert entry["test_type"] == "concurrency"
    result = client.get(f"/api/v1/history/{entry['id']}?limit=2", headers=headers)
    assert result.status_code == 200
    data = result.json()
    summary = data["summary"]
    assert summary["metric_contract_version"] == "legacy-unversioned"
    assert summary["integrity"]["verified"] is False
    assert summary["overall"]["requests"] == 4
    assert summary["overall"]["successes"] == 3
    assert summary["overall"]["failures"] == 1
    assert summary["overall"]["metrics"]["ttft"]["count"] == 2
    assert summary["overall"]["metrics"]["ttft"]["median"] == 2
    assert summary["overall"]["metrics"]["ttft"]["p95"] == pytest.approx(2.9)
    assert summary["overall"]["metrics"]["tps"]["median"] == 20
    assert summary["groups"][1]["metrics"]["ttft"]["median"] is None
    assert summary["origin"]["csv_sha256"] == hashlib.sha256(CSV).hexdigest()
    assert REDACTION_MARKER not in json.dumps(summary)
    assert data["preview"]["total"] == 4 and len(data["preview"]["items"]) == 2
    again = SavedCsvHistory(history.root).load(entry["id"])
    assert again["summary"] == summary
    assert store.list()[0] == []
    for kind in ["html", "markdown", "json", "csv"]:
        report = client.get(f"/api/v1/history/{entry['id']}/report?format={kind}", headers=headers)
        assert report.status_code == 200
        assert REDACTION_MARKER not in report.text
        if kind != "csv":
            assert summary["origin"]["csv_sha256"] in report.text
    figure = client.get(f"/api/v1/history/{entry['id']}/figure?metric=tps", headers=headers)
    assert figure.status_code == 200
    chart = figure.json()["figure"]
    assert chart["layout"]["yaxis"]["title"]["text"] == "TPS (token/s)"
    assert chart["data"][0]["y"] == [20, None]
    assert summary["origin"]["csv_sha256"] in json.dumps(chart)


def test_missing_outcomes_are_neither_successes_nor_failures(environment):
    _, history, _, _, _ = environment
    entry = history.upload(b"ttft,tps\n1,10\n2,20\n", "unknown.csv")
    summary = history.load(entry["id"])["summary"]
    assert summary["overall"]["requests"] == summary["overall"]["unknown_outcomes"] == 2
    assert summary["overall"]["failures"] == summary["overall"]["successes"] == 0
    assert summary["overall"]["success_rate"] is None
    assert summary["overall"]["metrics"]["ttft"]["count"] == 0
    assert entry["test_type"] == "unknown"
    assert entry["model_id"] == "未知模型"


def test_current_export_success_column_and_contract_are_respected(environment):
    _, history, _, _, _ = environment
    raw = b"concurrency_level,ttft,tps,success,metric_contract_version\n1,0.2,10,1,v2\n1,999,999,0,v2\n"
    entry = history.upload(raw, "current.csv", b'{"test_type":"concurrency"}')
    summary = history.load(entry["id"])["summary"]
    assert summary["metric_contract_version"] == "v2"
    assert summary["overall"]["success_rate"] == 0.5
    assert summary["overall"]["metrics"]["ttft"]["median"] == 0.2
    assert summary["integrity"]["verified"] is False


@pytest.mark.parametrize(
    "raw",
    [
        b"ttft,error\n",
        b"ttft_ms,error\n1,\n",
        b"ttft,ttft,error\n1,2,\n",
        b"ttft,error\n1,,extra\n",
        b"ttft,error\n1\n",
        b"ttft,success,error\n1,true,timeout\n",
        b"ttft,success\n1,perhaps\n",
        b"ttft,error,metric_contract_version\n1,,v1\n2,,v2\n",
        b"ttft,error,extra_metrics\n1,,not-json\n",
        b"ttft,error\n\xff,\n",
        b"ttft,error,concurrency,concurrency_level\n1,,1,2\n",
    ],
)
def test_invalid_upload_is_rejected_without_partial_files(environment, raw):
    _, history, _, _, _ = environment
    with pytest.raises(HistoryError):
        history.upload(raw, "bad.csv")
    assert len(history.catalog()["items"]) == 1
    assert not (history.root / "uploads").exists()


def test_bounds_are_enforced_without_truncated_statistics(environment, monkeypatch):
    _, history, _, _, _ = environment
    monkeypatch.setattr("server.history.MAX_ROWS", 2)
    with pytest.raises(HistoryError, match="20,000"):
        history.upload(CSV, "too-many.csv")
    monkeypatch.setattr("server.history.MAX_ROWS", 20000)
    monkeypatch.setattr("server.history.MAX_GROUPS", 1)
    with pytest.raises(HistoryError, match="128"):
        history.upload(CSV, "many-groups.csv")
    monkeypatch.setattr("server.history.MAX_ENTRIES", 1)
    catalog = history.catalog()
    assert catalog["truncated"] is True
    assert catalog["scan_limit"] == 1


def test_upload_deduplicates_concurrently_and_exports_safely(environment):
    client, history, _, headers, _ = environment
    raw = b"ttft,error,notes,api_key\n1,,=HYPERLINK(1),private-fixture\n"
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(lambda _: history.upload(raw, "../../safe.csv"), range(4)))
    assert len({entry["id"] for entry in entries}) == 1
    assert entries[0]["filename"] == "safe.csv"
    result = client.get(f"/api/v1/history/{entries[0]['id']}/report?format=csv", headers=headers)
    row = next(csv.DictReader(io.StringIO(result.text)))
    assert row["notes"].startswith("'=")
    assert row["api_key"] == "[已隐去]"
    assert (
        "private-fixture"
        not in client.get(f"/api/v1/history/{entries[0]['id']}", headers=headers).text
    )


def test_file_change_blocks_stale_delete_and_export(environment):
    client, history, _, headers, path = environment
    entry = history.catalog()["items"][0]
    path.write_bytes(CSV + b"1,4,40,100,API usage,\n")
    response = client.post(
        f"/api/v1/history/{entry['id']}/delete",
        headers=headers,
        json={"confirmed": True, "revision": entry["revision"]},
    )
    assert response.status_code == 409
    assert path.exists()
    exported = client.get(
        f"/api/v1/history/{entry['id']}/report?format=csv&revision={entry['revision']}",
        headers=headers,
    )
    assert exported.status_code == 409


def test_metadata_is_escaped_and_unknown_conditions_are_reported(environment):
    client, history, _, headers, path = environment
    unsafe = {**META, "model_id": "<script>alert(1)</script>"}
    unsafe["system_info"] = {"engine_name": "<img src=x onerror=alert(1)>"}
    path.with_suffix(".csv.meta.json").write_text(json.dumps(unsafe))
    original = list(csv.DictReader(io.StringIO(CSV.decode())))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=[*original[0], "token_source"])
    writer.writeheader()
    writer.writerows(
        {**row, "token_source": "<script>token-source</script> ![link](https://example.test)"}
        for row in original
    )
    path.write_text(buffer.getvalue())
    entry = history.catalog()["items"][0]
    report = client.get(f"/api/v1/history/{entry['id']}/report?format=html", headers=headers)
    assert report.status_code == 200
    assert "<script>" not in report.text and "<img src=x" not in report.text
    assert "&lt;script&gt;" in report.text and "&lt;img" in report.text
    assert "未经核验" in report.text
    markdown = client.get(f"/api/v1/history/{entry['id']}/report?format=markdown", headers=headers)
    assert "<script>token-source</script>" not in markdown.text
    assert "&lt;script&gt;token-source&lt;/script&gt;" in markdown.text
    assert r"!\[link\]" in markdown.text
    path.with_suffix(".csv.meta.json").write_bytes(b"malformed JSON")
    summary = history.load(entry["id"])["summary"]
    assert summary["origin"]["metadata"] == {}
    assert summary["origin"]["metadata_sha256"] == hashlib.sha256(b"malformed JSON").hexdigest()
    assert any("元数据无效" in note for note in summary["data_quality"]["warnings"])


def test_formatted_metadata_limit_does_not_leave_files(environment, monkeypatch):
    _, history, _, _, _ = environment
    monkeypatch.setattr("server.history.MAX_METADATA_BYTES", 48)
    with pytest.raises(HistoryError, match="整理后的元数据"):
        history.upload(CSV, "x" * 80 + ".csv", b"{}")
    assert not (history.root / "uploads").exists()


def test_delete_requires_confirmation_and_retains_both_files(environment):
    client, history, _, headers, path = environment
    entry = history.catalog()["items"][0]
    route = f"/api/v1/history/{entry['id']}/delete"
    refused = client.post(
        route, headers=headers, json={"confirmed": False, "revision": entry["revision"]}
    )
    assert refused.status_code == 422 and path.exists()
    deleted = client.post(
        route, headers=headers, json={"confirmed": True, "revision": entry["revision"]}
    )
    assert deleted.status_code == 200 and deleted.json()["backup_retained"] is True
    assert not path.exists() and not path.with_suffix(".csv.meta.json").exists()
    assert history.catalog()["items"] == []
    backup = next((history.root / ".trash").glob("*/result.csv"))
    assert backup.read_bytes() == CSV
    assert json.loads(backup.with_suffix(".csv.meta.json").read_text()) == META


def test_partial_delete_rolls_back_when_metadata_move_fails(environment, monkeypatch):
    _, history, _, _, path = environment
    entry = history.catalog()["items"][0]
    original = type(path).rename

    def rename(source, destination):
        if source.name.endswith(".meta.json"):
            raise OSError("simulated move failure")
        return original(source, destination)

    monkeypatch.setattr(type(path), "rename", rename)
    with pytest.raises(OSError):
        history.delete(entry["id"], entry["revision"])
    assert path.read_bytes() == CSV and path.with_suffix(".csv.meta.json").exists()


def test_symlink_and_sibling_directory_cannot_escape_history_root(environment):
    _, history, _, _, path = environment
    entry = history.catalog()["items"][0]
    outside = history.root.with_name(history.root.name + "-outside")
    outside.mkdir()
    external = outside / "other.csv"
    external.write_bytes(CSV)
    (history.root / "linked.csv").symlink_to(external)
    assert len(history.catalog()["items"]) == 1
    assert history.catalog()["skipped"] == 1
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(HistoryNotFound):
        history.load(entry["id"])
    with pytest.raises(HistoryError):
        history._safe(external)
    (history.root / "uploads").symlink_to(outside, target_is_directory=True)
    with pytest.raises(HistoryError):
        history.upload(CSV, "new.csv")


def test_routes_are_authenticated_and_limit_upload_before_parsing(environment):
    client, history, _, headers, _ = environment
    entry = history.catalog()["items"][0]
    for route in [
        "/api/v1/history",
        f"/api/v1/history/{entry['id']}",
        f"/api/v1/history/{entry['id']}/report",
        f"/api/v1/history/{entry['id']}/figure",
    ]:
        assert client.get(route).status_code == 401
    denied = client.post("/api/v1/history/uploads", files={"file": ("test.csv", CSV)})
    assert denied.status_code == 401
    excessive = client.post(
        "/api/v1/history/uploads",
        headers={**headers, "Content-Length": str(12 * 1024 * 1024)},
        content=b"ignored",
    )
    assert excessive.status_code == 413
    missing = client.post(
        "/api/v1/history/uploads",
        headers={**headers, "Transfer-Encoding": "chunked"},
        content=b"ignored",
    )
    assert missing.status_code == 411
    uploaded = client.post(
        "/api/v1/history/uploads",
        headers=headers,
        files={
            "file": ("legacy.csv", CSV),
            "metadata": ("legacy.meta.json", json.dumps(META).encode()),
        },
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["model_id"] == "historical-model"
    assert client.get("/api/v1/history/not-a-path", headers=headers).status_code == 404
