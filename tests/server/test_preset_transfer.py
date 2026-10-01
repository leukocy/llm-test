"""Preset labels and portable import cannot overwrite records or include credentials."""

import sqlite3

import pytest

from core.database.migrations import run_migrations
from server.store import JobStore
from tests.server.test_data_api import auth, env  # noqa: F401


def body(name="Tagged"):
    return {
        "name": name,
        "description": "two\nlines",
        "tags": [" math ", "perf", "math"],
        "endpoint_id": "lab",
        "test_type": "concurrency",
        "parameters": {
            "selected_concurrencies": [1, 2],
            "rounds_per_level": 1,
            "input_tokens_target": 64,
            "max_tokens": 8,
        },
        "run_config": {"custom_params": [{"name": "optional", "value": None}]},
    }


def test_tags_export_import_and_conflicts_are_atomic(env):  # noqa: F811
    client, _, settings = env
    store = JobStore(settings.db_path)
    saved = client.post("/api/v1/presets", json=body(), headers=auth())
    assert saved.status_code == 201
    original = saved.json()
    assert original["tags"] == ["math", "perf"]
    url = "/api/v1/presets/" + original["preset_id"] + "/export"
    assert client.get(url).status_code == 401
    exported = client.get(url, headers=auth()).json()
    assert exported["format"] == "llm-test-preset" and exported["version"] == 1
    assert "credential" not in str(exported) and "api_key" not in str(exported)
    assert exported["preset"]["run_config"]["custom_params"][0]["value"] is None
    assert client.post("/api/v1/presets/import", json=exported, headers=auth()).status_code == 409
    assert store.get_preset(original["preset_id"]) == original
    exported["preset"]["name"] = "Imported"
    imported = client.post("/api/v1/presets/import", json=exported, headers=auth())
    assert imported.status_code == 201
    assert imported.json()["preset_id"] != original["preset_id"]
    assert imported.json()["parameters"] == original["parameters"]
    exported["preset"]["tags"] = ["revised"]
    updated = client.put(
        "/api/v1/presets/" + imported.json()["preset_id"], json=exported["preset"], headers=auth()
    )
    assert updated.status_code == 200 and updated.json()["tags"] == ["revised"]


@pytest.mark.parametrize("tags", [[""], ["x" * 33], ["line\nbreak"], ["x"] * 21])
def test_bad_tags_rejected_without_saved_record(env, tags):  # noqa: F811
    client, _, settings = env
    store = JobStore(settings.db_path)
    content = body()
    content["tags"] = tags
    assert client.post("/api/v1/presets", json=content, headers=auth()).status_code == 422
    assert store.list_presets() == []


def test_import_rejects_unknown_version_credential_fields_and_bad_parameters(env):  # noqa: F811
    client, _, settings = env
    store = JobStore(settings.db_path)
    for changed in (
        {"version": 2},
        {"preset": {**body(), "api_key": "synthetic"}},  # pragma: allowlist secret
        {"preset": {**body(), "parameters": {"selected_concurrencies": []}}},
    ):
        imported = {"format": "llm-test-preset", "version": 1, "preset": body(), **changed}
        assert (
            client.post("/api/v1/presets/import", json=imported, headers=auth()).status_code == 422
        )
    assert store.list_presets() == []


def test_upgrade_preserves_existing_plan_and_empty_tags(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE db_meta (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        conn.execute("INSERT INTO db_meta (key,value) VALUES ('schema_version','1.12.0')")
        conn.execute(
            "CREATE TABLE control_presets (preset_id TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE,description TEXT,endpoint_id TEXT,test_type TEXT,parameters_json TEXT,run_config_json TEXT,created_at REAL,updated_at REAL)"
        )
        conn.execute(
            "INSERT INTO control_presets VALUES ('old','Existing','keep','lab','concurrency','{}','{}',1,1)"
        )
        run_migrations(conn)
        run_migrations(conn)
    saved = JobStore(path).get_preset("old")
    assert saved["tags"] == [] and saved["description"] == "keep" and saved["parameters"] == {}
