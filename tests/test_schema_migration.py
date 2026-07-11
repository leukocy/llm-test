"""Phase 4: schema 迁移 + TestRun 模型 + 持久化测试。

迁移机制用原始 sqlite3 连接验证（绕开单例）；extra_fields / 元数据写入用临时 DB
（重置单例并在 finally 还原，避免污染其它测试）。
"""

from __future__ import annotations

import sqlite3

import pytest

from core.database.migrations import MIGRATIONS, run_migrations
from core.database.schema import create_tables
from core.models.test_run import TestRun

# 1.2.0 + 1.3.0 新增的列（迁移 + DDL 都应包含）
WAREHOUSE_COLUMNS = [
    "machine_id",
    "tester",
    "external_level",
    "bottleneck",
    "next_action",
    "supersedes_test_id",
    "comparison_group",
    "mtp_enabled",
    "effective_bandwidth_gbps",
    "bandwidth_utilization_pct",
    "gpu_vram_peak_gb",
    "system_memory_peak_gb",
    "model_spec_json",
    "serving_config_json",
    "resource_monitor_json",
    "status_detail",
    # 1.3.0 推理引擎运行时
    "engine_metrics_json",
    "gpu_kv_cache_usage_peak_pct",
    "num_preemption_total",
    "engine_running_requests_peak",
    "kv_cache_capacity_tokens",
]

ENGINE_COLUMNS = [
    "engine_metrics_json",
    "gpu_kv_cache_usage_peak_pct",
    "num_preemption_total",
    "engine_running_requests_peak",
    "kv_cache_capacity_tokens",
]


def _columns(conn, table="test_runs") -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


# ---------- DDL ----------


def test_create_tables_has_warehouse_columns(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "fresh.db"))
    try:
        create_tables(conn)
        cols = _columns(conn)
        for c in WAREHOUSE_COLUMNS:
            assert c in cols, f"缺少列 {c}"
        # external_level 有默认值
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='test_runs'"
        ).fetchone()
        assert "external_level TEXT DEFAULT 'internal'" in row[0]
    finally:
        conn.close()


# ---------- 迁移 ----------


def test_migration_1_1_0_to_latest_adds_all_warehouse_columns(tmp_path):
    """模拟一个 1.1.0 老库，迁移到最新版本应补齐 1.2.0 + 1.3.0 全部列。"""
    conn = sqlite3.connect(str(tmp_path / "legacy.db"))
    try:
        # 用一份不含仓库列的精简 DDL 建表，模拟 1.1.0 schema
        legacy_ddl = """
        CREATE TABLE test_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT UNIQUE NOT NULL,
            test_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            model_id TEXT NOT NULL,
            csv_path TEXT
        );
        CREATE TABLE db_meta (key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE test_results (id INTEGER PRIMARY KEY);
        CREATE TABLE api_logs (id INTEGER PRIMARY KEY);
        CREATE TABLE execution_logs (id INTEGER PRIMARY KEY);
        CREATE TABLE reports (id INTEGER PRIMARY KEY);
        """
        conn.executescript(legacy_ddl)
        conn.execute(
            "INSERT INTO db_meta (key, value) VALUES ('schema_version', '1.1.0')"
        )
        conn.commit()

        assert "machine_id" not in _columns(conn)

        from core.database.schema import SCHEMA_VERSION

        run_migrations(conn, SCHEMA_VERSION)

        cols = _columns(conn)
        for c in WAREHOUSE_COLUMNS:
            assert c in cols, f"迁移后仍缺列 {c}"
        # 1.3.0 引擎列必须在
        for c in ENGINE_COLUMNS:
            assert c in cols
        ver = conn.execute(
            "SELECT value FROM db_meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert ver == SCHEMA_VERSION
    finally:
        conn.close()


def test_migration_is_idempotent(tmp_path):
    """重复迁移不报错（_add_column 吞 duplicate column name）。"""
    conn = sqlite3.connect(str(tmp_path / "idem.db"))
    try:
        create_tables(conn)
        # 再次手动执行 1.2.0 迁移里的 ADD COLUMN（列已存在）应被吞掉
        for m in MIGRATIONS["1.2.0"]:
            m(conn)  # 不抛异常
        conn.commit()
        assert "machine_id" in _columns(conn)
    finally:
        conn.close()


def test_migration_skips_when_already_at_target(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "skip.db"))
    try:
        create_tables(conn)
        # create_tables 已把 version 设为 1.2.0，再 run_migrations 应直接跳过
        run_migrations(conn)
        assert "machine_id" in _columns(conn)
    finally:
        conn.close()


# ---------- TestRun 模型往返 ----------


def test_testrun_roundtrip_warehouse_fields():
    run = TestRun.create(test_type="concurrency", model_id="DeepSeek-V3.1")
    run.machine_id = "abc123"
    run.tester = "alice"
    run.external_level = "publishable"
    run.bottleneck = "memory_bandwidth"
    run.mtp_enabled = True
    run.effective_bandwidth_gbps = 1850.0
    run.bandwidth_utilization_pct = 55.2
    run.gpu_vram_peak_gb = 70.5
    run.model_spec = {"active_params_b": 37, "weight_dtype": "fp8"}
    run.serving_config = {"engine": "vllm", "tp_size": 8}
    run.resource_monitor = {"peaks": {"gpu_vram_gb": 70.5}}
    run.status_detail = "passed"

    d = run.to_dict()
    assert d["machine_id"] == "abc123"
    assert d["external_level"] == "publishable"
    assert d["mtp_enabled"] == 1  # bool → int
    assert '"active_params_b": 37' in d["model_spec_json"]
    assert d["effective_bandwidth_gbps"] == 1850.0
    # 原始 dict 字段已被 pop
    assert "model_spec" not in d
    assert "serving_config" not in d

    # 反序列化
    restored = TestRun.from_row(d)
    assert restored.machine_id == "abc123"
    assert restored.external_level == "publishable"
    assert restored.mtp_enabled is True  # int → bool
    assert restored.model_spec["active_params_b"] == 37
    assert restored.serving_config["tp_size"] == 8
    assert restored.status_detail == "passed"


def test_testrun_external_level_defaults_internal():
    d = TestRun.create(test_type="x", model_id="m").to_dict()
    assert d["external_level"] == "internal"


# ---------- extra_fields / 元数据写入（临时 DB） ----------


def _fresh_db(tmp_path):
    from core.database.connection import Database

    Database._instance = None
    return Database(str(tmp_path / "wh.db"))


def test_repo_complete_writes_warehouse_columns(tmp_path):
    from core.repositories.test_run import TestRunRepository

    db = _fresh_db(tmp_path)
    try:
        repo = TestRunRepository(db)
        run = TestRun.create(test_type="concurrency", model_id="m")
        rid = repo.insert(run)
        repo.complete(
            rid,
            success=True,
            stats={
                "machine_id": "abc123",
                "gpu_vram_peak_gb": 70.5,
                "effective_bandwidth_gbps": 1850.0,
                "mtp_enabled": 1,
                "bottleneck": "memory_bandwidth",
                "model_spec_json": '{"active_params_b": 37}',
            },
        )
        row = db.fetch_one("SELECT * FROM test_runs WHERE id = ?", (rid,))
        assert row["machine_id"] == "abc123"
        assert row["gpu_vram_peak_gb"] == pytest.approx(70.5)
        assert row["effective_bandwidth_gbps"] == pytest.approx(1850.0)
        assert row["mtp_enabled"] == 1
        assert "memory_bandwidth" in (row["bottleneck"] or "")
        assert "active_params_b" in (row["model_spec_json"] or "")
    finally:
        from core.database.connection import Database

        Database._instance = None


def test_manager_complete_test_run_with_extra_fields(tmp_path):
    from core.database.connection import Database
    from core.database.manager import DatabaseManager

    Database._instance = None
    DatabaseManager._instance = None
    try:
        mgr = DatabaseManager(str(tmp_path / "mgr.db"))
        run = mgr.start_test_run("concurrency", "DeepSeek-V3.1")
        mgr.complete_test_run(
            run,
            success=True,
            extra_fields={
                "machine_id": "host1",
                "gpu_vram_peak_gb": 70.5,
                "effective_bandwidth_gbps": 1850.0,
                "bandwidth_utilization_pct": 55.2,
                "mtp_enabled": 1,
            },
        )
        row = mgr.db.fetch_one("SELECT * FROM test_runs WHERE id = ?", (run.id,))
        assert row["machine_id"] == "host1"
        assert row["effective_bandwidth_gbps"] == pytest.approx(1850.0)
        assert row["bandwidth_utilization_pct"] == pytest.approx(55.2)
        assert row["mtp_enabled"] == 1
    finally:
        Database._instance = None
        DatabaseManager._instance = None


def test_manager_update_publish_metadata(tmp_path):
    from core.database.connection import Database
    from core.database.manager import DatabaseManager

    Database._instance = None
    DatabaseManager._instance = None
    try:
        mgr = DatabaseManager(str(tmp_path / "meta.db"))
        run = mgr.start_test_run("concurrency", "m")
        mgr.complete_test_run(run, success=True)
        ok = mgr.update_publish_metadata(
            run.id,
            {
                "tester": "bob",
                "external_level": "review",
                "next_action": "需复测长上下文",
                "bottleneck": "kv_cache",
                "bogus_field": "ignored",  # 不在 allowlist，应被忽略
            },
        )
        assert ok is True
        row = mgr.db.fetch_one("SELECT * FROM test_runs WHERE id = ?", (run.id,))
        assert row["tester"] == "bob"
        assert row["external_level"] == "review"
        assert row["next_action"] == "需复测长上下文"
    finally:
        Database._instance = None
        DatabaseManager._instance = None
