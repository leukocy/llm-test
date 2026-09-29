"""Real offline tokenizer validation with bounded, synthetic Hub responses."""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

from config.tokenizer_paths import MANIFEST_NAME, registered_tokenizer_path
from core.benchmark.metrics import METRIC_CONTRACT_VERSION
from core.run_lifecycle import RunStatus
from core.tokenizer_utils import get_cached_tokenizer, tokenizer_provenance
from server.analytics import MetricContractConflict, run_summary
from server.api import create_app
from server.reports import render_html, render_markdown
from server.settings import Endpoint, Settings
from server.store import JobStore, LeaseLost
from server.tokenizer_install import InstallFailure, _selected_files, execute_install
from server.tokenizer_queue import InstallCancelled, TokenizerInstallQueue, staging_path
from server.tokenizer_tools import count_text

NAME = "DeepSeek-V3.2"
SECOND = "Qwen3-Next-80B-A3B-Instruct"
REVISION = "a" * 40


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_TEST_TOKENIZER_DOWNLOAD_ROOT", str(tmp_path / "cache"))
    store = JobStore(tmp_path / "platform.db")
    queue = TokenizerInstallQueue(store)
    return store, queue, tmp_path / "cache"


@pytest.fixture
def fake_hub(monkeypatch):
    # Load the third-party package at fixture time. A local tokenizers/ data
    # directory must not change how CI classifies the top-level import block.
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    files = {
        "tokenizer.json": tokenizer.to_str().encode(),
        "tokenizer_config.json": json.dumps(
            {"tokenizer_class": "PreTrainedTokenizerFast"}
        ).encode(),
    }
    metadata = {
        "sha": REVISION,
        "siblings": [
            {
                "rfilename": name,
                "size": len(data),
                "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
            }
            for name, data in files.items()
        ]
        + [
            {"rfilename": "model.safetensors", "size": 500000000000},
            {"rfilename": "tokenization_bad.py", "size": 100},
            {"rfilename": "../tokenizer.json", "size": 100},
            {"rfilename": "nested/tokenizer.json", "size": 100},
        ],
    }
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.host == "huggingface.co"
        assert "authorization" not in request.headers
        if "/api/models/" in request.url.path:
            return httpx.Response(200, json=metadata)
        assert f"/resolve/{REVISION}/" in request.url.path
        return httpx.Response(200, content=files[request.url.path.rsplit("/", 1)[-1]])

    monkeypatch.setattr(
        "server.tokenizer_install._hub_client",
        lambda: httpx.Client(transport=httpx.MockTransport(handle)),
    )
    return files, metadata, requests


def _claim(queue, name=NAME):
    queue.enqueue([name])
    return queue.claim("installer")


def test_install_validates_atomically_and_is_shared_with_measurement(setup, fake_hub):
    _, queue, root = setup
    files, _, requests = fake_hub
    task = _claim(queue)
    assert registered_tokenizer_path(NAME) is None
    execute_install(task, queue, "installer")
    result = queue.get(task["install_id"])
    assert result["status"] == "completed"
    assert result["downloaded_bytes"] == result["total_bytes"] == sum(map(len, files.values()))
    assert result["completed_files"] == result["total_files"] == 2
    assert {path.name for path in (root / NAME).iterdir()} == {*files, MANIFEST_NAME}
    assert not staging_path(task["install_id"]).exists()
    manifest = tokenizer_provenance(f"./tokenizers/{NAME}")
    assert manifest["revision"] == REVISION
    assert manifest["trust_remote_code"] is False
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    assert count_text("hello world", "local", NAME)["count"] == 2
    assert len(get_cached_tokenizer(f"./tokenizers/{NAME}").encode("hello world")) == 2
    assert len(requests) == 3
    config = {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "tokenizer_installation": manifest,
    }
    with sqlite3.connect(queue.store.path) as conn:
        run_id = conn.execute(
            "INSERT INTO test_runs(test_id,test_type,status,model_id,config_json) VALUES ('observed','concurrency','completed','m',?)",
            (json.dumps(config),),
        ).lastrowid
        conn.execute(
            "INSERT INTO test_results(run_id,ttft,tps,extra_metrics) VALUES (?,0.2,12,?)",
            (run_id, json.dumps({"metric_contract_version": METRIC_CONTRACT_VERSION})),
        )
    summary = run_summary(str(queue.store.path), run_id)
    assert summary["tokenizer_installation"] == manifest
    job = {"job_id": "observed", "model_id": "m", "test_type": "concurrency", "status": "completed"}
    for report in [render_html(job, summary), render_markdown(job, summary)]:
        assert REVISION in report and manifest["files"][0]["sha256"] in report
        assert "最终 Token 指标" in report
    with sqlite3.connect(queue.store.path) as conn:
        conn.execute(
            "UPDATE test_runs SET config_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "metric_contract_version": METRIC_CONTRACT_VERSION,
                        "tokenizer_installation": {"revision": "main"},
                    }
                ),
                run_id,
            ),
        )
    with pytest.raises(MetricContractConflict, match="tokenizer installation"):
        run_summary(str(queue.store.path), run_id)
    # A new queue / process reads the installation from persistent storage.
    assert (
        TokenizerInstallQueue(JobStore(queue.store.path)).get(task["install_id"])["status"]
        == "completed"
    )
    assert queue.enqueue([NAME]) == {"items": [], "skipped_names": [NAME]}


def test_enqueue_deduplicates_across_process_connections(setup):
    store, queue, _ = setup
    with ThreadPoolExecutor(max_workers=4) as pool:
        submitted = list(
            pool.map(
                lambda _: TokenizerInstallQueue(JobStore(store.path)).enqueue([NAME]), range(8)
            )
        )
    assert len({item["items"][0]["install_id"] for item in submitted}) == 1
    assert queue.claim("one") is not None
    assert queue.claim("two") is None


def test_installs_and_measurements_never_overlap_and_measurements_take_priority(setup):
    store, queue, _ = setup
    pending = queue.enqueue([NAME])["items"][0]
    job = store.submit(test_type="concurrency", endpoint_id="lab", model_id="m", parameters={})
    assert queue.claim("installer") is None
    assert store.claim("measurement")["job_id"] == job["job_id"]
    assert queue.claim("installer") is None
    store.finish(job["job_id"], "measurement", outcome=RunStatus.COMPLETED)
    task = queue.claim("installer")
    assert task["install_id"] == pending["install_id"]
    second = store.submit(test_type="concurrency", endpoint_id="lab", model_id="m", parameters={})
    assert store.claim("measurement") is None
    queue.cancel(task["install_id"])
    assert store.claim("measurement") is None
    execute_install(task, queue, "installer")
    assert queue.get(task["install_id"])["status"] == "cancelled"
    assert store.claim("measurement")["job_id"] == second["job_id"]


def test_cancel_before_publish_preserves_absence_of_partial_install(setup, fake_hub, monkeypatch):
    _, queue, root = setup
    task = _claim(queue)

    class CancellingTokenizer:
        def encode(self, *_args, **_kwargs):
            queue.cancel(task["install_id"])
            return [0]

    monkeypatch.setattr(
        "server.tokenizer_install.get_cached_tokenizer", lambda _: CancellingTokenizer()
    )
    execute_install(task, queue, "installer")
    assert queue.get(task["install_id"])["status"] == "cancelled"
    assert not (root / NAME).exists()
    assert not staging_path(task["install_id"]).exists()


@pytest.mark.parametrize("kind", ["hash", "size", "unsafe_code"])
def test_invalid_download_is_not_published_and_can_be_retried(setup, fake_hub, kind):
    _, queue, root = setup
    files, metadata, _ = fake_hub
    task = _claim(queue)
    if kind == "hash":
        metadata["siblings"][0]["lfs"]["sha256"] = "0" * 64
    elif kind == "size":
        metadata["siblings"][0]["size"] = 1
    else:
        files["tokenizer_config.json"] = json.dumps(
            {
                "tokenizer_class": "UnknownRemoteTokenizer",
                "auto_map": {"AutoTokenizer": ["tokenization_bad.UnknownRemoteTokenizer", None]},
            }
        ).encode()
        metadata["siblings"][1].update(size=len(files["tokenizer_config.json"]), lfs={})
    execute_install(task, queue, "installer")
    assert queue.get(task["install_id"])["status"] == "failed"
    assert not (root / NAME).exists()
    assert not staging_path(task["install_id"]).exists()
    assert queue.enqueue([NAME])["items"][0]["install_id"] != task["install_id"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"sha": "main", "siblings": []},
        {
            "sha": REVISION,
            "siblings": [{"rfilename": "tokenizer.json", "size": 64 * 1024 * 1024 + 1}],
        },
        {"sha": REVISION, "siblings": [{"rfilename": "tokenizer.json"}]},
        {
            "sha": REVISION,
            "siblings": [
                {"rfilename": name, "size": 64 * 1024 * 1024}
                for name in ["tokenizer.json", "vocab.json", "config.json"]
            ],
        },
        {"sha": REVISION, "siblings": [{"rfilename": "tokenizer.json", "size": 1}] * 2},
    ],
)
def test_manifest_requires_immutable_version_and_bounded_file_sizes(metadata):
    with pytest.raises(InstallFailure):
        _selected_files(metadata)


def test_restart_reaps_expired_lease_and_stale_worker_cannot_publish(setup):
    store, queue, root = setup
    task = _claim(queue)
    stage = staging_path(task["install_id"])
    stage.mkdir(parents=True)
    (stage / "tokenizer.json").write_text("partial")
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE tokenizer_installs SET lease_until = 0, status = 'validating'")
    with pytest.raises(LeaseLost):
        queue.publish(task["install_id"], "installer")
    assert queue.reap_expired() == 1
    assert queue.get(task["install_id"])["status"] == "failed"
    assert not stage.exists()
    assert not (root / NAME).exists()
    assert _claim(queue)["install_id"] != task["install_id"]


def test_cancellation_and_ownership_are_checked_before_publication(setup):
    _, queue, _ = setup
    task = _claim(queue)
    with pytest.raises(LeaseLost):
        queue.publish(task["install_id"], "other-worker")
    queue.cancel(task["install_id"])
    with pytest.raises(InstallCancelled):
        queue.publish(task["install_id"], "installer")


@pytest.mark.parametrize("status", [401, 502])
def test_download_failure_is_actionable_without_copying_source_response(setup, monkeypatch, status):
    _, queue, root = setup
    task = _claim(queue)
    monkeypatch.setattr(
        "server.tokenizer_install._hub_client",
        lambda: httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(status, text="private response payload")
            )
        ),
    )
    execute_install(task, queue, "installer")
    result = queue.get(task["install_id"])
    assert result["status"] == "failed"
    assert "授权" in result["message"] if status == 401 else "502" in result["message"]
    assert "private response payload" not in json.dumps(result)
    assert not (root / NAME).exists()


def test_download_requires_authentication_registered_names_and_returns_safe_status(
    setup, monkeypatch
):
    store, queue, _ = setup
    monkeypatch.setenv("LAB_KEY", "synthetic-key")
    endpoint = Endpoint("lab", "Lab", "OpenAI", "http://127.0.0.1:9010/v1", "m", "LAB_KEY")
    settings = Settings(
        api_token="a" * 40,
        db_path=store.path,
        artifact_root=store.path.parent / "artifacts",
        endpoints={"lab": endpoint},
    )
    client = TestClient(create_app(settings, store))
    route = "/api/v1/tokenizers/installations"
    headers = {"Authorization": "Bearer " + "a" * 40}
    assert client.post(route, json={"names": [NAME]}).status_code == 401
    assert client.get("/api/v1/tokenizers").status_code == 401
    for body in [
        {"names": ["../private"]},
        {"names": [NAME, NAME]},
        {"names": [NAME], "repo_id": "attacker/repo"},
        {"names": []},
    ]:
        assert client.post(route, json=body, headers=headers).status_code == 422
    created = client.post(route, json={"names": [NAME, SECOND]}, headers=headers)
    assert created.status_code == 202
    first, second = created.json()["items"]
    assert "lease_owner" not in first and "lease_until" not in first
    assert client.get(f"{route}/missing", headers=headers).status_code == 404
    assert client.post(f"{route}/{first['install_id']}/cancel").status_code == 401
    assert (
        client.post(f"{route}/{first['install_id']}/cancel", headers=headers).json()["status"]
        == "cancelled"
    )
    assert queue.claim("installer")["install_id"] == second["install_id"]
    catalog = client.get("/api/v1/tokenizers?model_id=DeepSeek-V3.2", headers=headers).json()
    item = next(item for item in catalog["items"] if item["name"] == SECOND)
    assert item["installation"]["status"] == "downloading"
    assert "lease_owner" not in item["installation"]


def test_presets_keep_description_and_upgrade_existing_database(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE db_meta (key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            INSERT INTO db_meta(key,value) VALUES ('schema_version', '1.10.0');
            CREATE TABLE control_presets (preset_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
              endpoint_id TEXT NOT NULL, test_type TEXT NOT NULL, parameters_json TEXT NOT NULL,
              run_config_json TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
            INSERT INTO control_presets VALUES ('old', 'Old preset', 'lab', 'concurrency', '{}', '{}', 1, 1);
        """)
    store = JobStore(path)
    assert store.get_preset("old")["description"] == ""
    assert JobStore(path).get_preset("old")["name"] == "Old preset"
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute("SELECT value FROM db_meta WHERE key='schema_version'").fetchone()[0]
            == "1.11.0"
        )
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    monkeypatch.setenv("LAB_KEY", "synthetic-key")
    settings = Settings(
        api_token="a" * 40,
        db_path=path,
        artifact_root=tmp_path / "artifacts",
        endpoints={
            "lab": Endpoint("lab", "Lab", "OpenAI", "http://127.0.0.1:9010/v1", "m", "LAB_KEY")
        },
    )
    client = TestClient(create_app(settings, store))
    headers = {"Authorization": "Bearer " + "a" * 40}
    payload = {
        "name": "Scientific plan",
        "description": "  Purpose\nConditions: GPU / model version  ",
        "endpoint_id": "lab",
        "test_type": "concurrency",
        "parameters": {"selected_concurrencies": [1], "rounds_per_level": 1, "max_tokens": 8},
    }
    saved = client.post("/api/v1/presets", json=payload, headers=headers)
    assert saved.status_code == 201
    assert saved.json()["description"] == payload["description"].strip()
    identifier = saved.json()["preset_id"]
    assert (
        client.put(
            f"/api/v1/presets/{identifier}",
            json={**payload, "description": "Updated"},
            headers=headers,
        ).json()["description"]
        == "Updated"
    )
    assert store.get_preset(identifier)["description"] == "Updated"
    for description in ["x" * 501, "invalid\x00description"]:
        assert (
            client.put(
                f"/api/v1/presets/{identifier}",
                json={**payload, "description": description},
                headers=headers,
            ).status_code
            == 422
        )
    assert client.delete(f"/api/v1/presets/{identifier}", headers=headers).status_code == 204
    assert [item["preset_id"] for item in store.list_presets()] == ["old"]
