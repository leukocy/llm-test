"""
Database Migrations Module

管理 Schema VersionandMigration。
"""

import logging
import sqlite3
from typing import Any, Callable

from .schema import SCHEMA_VERSION

logger = logging.getLogger(__name__)


# Migration函数类型
MigrationFunc = Callable[[sqlite3.Connection], None]


def _add_column(table: str, column: str, type_spec: str) -> MigrationFunc:
    """生成幂等的 ADD COLUMN 迁移（重复列名视为已迁移，吞掉异常）。"""

    def _migrate(conn: sqlite3.Connection) -> None:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_spec}")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise

    return _migrate


def _exec(sql: str) -> MigrationFunc:
    """生成执行 DDL（CREATE INDEX/TABLE 等）的迁移；忽略返回的 Cursor。"""

    def _migrate(conn: sqlite3.Connection) -> None:
        conn.execute(sql)

    return _migrate


# MigrationRegister表：Version号 -> Migration函数列表
MIGRATIONS: dict[str, list[MigrationFunc]] = {
    # 1.0.0 -> 1.1.0: Add prompt_text and output_text 字段（幂等，新库已含则跳过）
    "1.1.0": [
        _add_column("test_results", "prompt_text", "TEXT"),
        _add_column("test_results", "output_text", "TEXT"),
    ],
    # 1.1.0 -> 1.2.0: 数据仓库扩展 —— 把 test_runs 升级为富记录（手册：报告是切片，仓库是全集）
    "1.2.0": [
        # 筛选/分组/对外口径字段（一等列）
        _add_column("test_runs", "machine_id", "TEXT"),
        _add_column("test_runs", "tester", "TEXT"),
        _add_column("test_runs", "external_level", "TEXT DEFAULT 'internal'"),
        _add_column("test_runs", "bottleneck", "TEXT"),
        _add_column("test_runs", "next_action", "TEXT"),
        _add_column("test_runs", "supersedes_test_id", "TEXT"),
        _add_column("test_runs", "comparison_group", "TEXT"),
        _add_column("test_runs", "mtp_enabled", "INTEGER"),
        # 资源监控 / 等效带宽 头条指标
        _add_column("test_runs", "effective_bandwidth_gbps", "REAL"),
        _add_column("test_runs", "bandwidth_utilization_pct", "REAL"),
        _add_column("test_runs", "gpu_vram_peak_gb", "REAL"),
        _add_column("test_runs", "system_memory_peak_gb", "REAL"),
        # 变长 JSON 字段
        _add_column("test_runs", "model_spec_json", "TEXT"),
        _add_column("test_runs", "serving_config_json", "TEXT"),
        _add_column("test_runs", "resource_monitor_json", "TEXT"),
        _add_column("test_runs", "status_detail", "TEXT"),
        # 索引
        _exec("CREATE INDEX IF NOT EXISTS idx_test_runs_machine ON test_runs(machine_id)"),
        _exec(
            "CREATE INDEX IF NOT EXISTS idx_test_runs_external_level ON test_runs(external_level)"
        ),
        _exec("CREATE INDEX IF NOT EXISTS idx_test_runs_comparison ON test_runs(comparison_group)"),
    ],
    # 1.2.0 -> 1.3.0: 推理引擎运行时（/metrics 轮询 + KV 实况）
    "1.3.0": [
        _add_column("test_runs", "engine_metrics_json", "TEXT"),
        _add_column("test_runs", "gpu_kv_cache_usage_peak_pct", "REAL"),
        _add_column("test_runs", "num_preemption_total", "INTEGER"),
        _add_column("test_runs", "engine_running_requests_peak", "INTEGER"),
        _add_column("test_runs", "kv_cache_capacity_tokens", "INTEGER"),
    ],
    # 1.3.0 -> 1.4.0: 模型×应用质量评估表（手册 maTest 模板采集层）
    # 新表，CREATE TABLE IF NOT EXISTS 幂等（create_tables 对新老库都已建；此处双保险）。
    "1.4.0": [
        _exec(
            """CREATE TABLE IF NOT EXISTS application_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL UNIQUE,
            run_id INTEGER,
            source TEXT DEFAULT 'manual',
            evaluator_name TEXT DEFAULT '',
            sample_id TEXT DEFAULT '',
            date TEXT DEFAULT '',
            tester TEXT DEFAULT '',
            scenario TEXT DEFAULT '',
            task_name TEXT DEFAULT '',
            customer_type TEXT DEFAULT '',
            model_name TEXT DEFAULT '',
            machine_id TEXT DEFAULT '',
            engine TEXT DEFAULT '',
            usecase_set_version TEXT DEFAULT '',
            input_tokens INTEGER,
            output_tokens INTEGER,
            context_length INTEGER,
            concurrency INTEGER,
            ttft_s REAL,
            retrieval_latency_s REAL,
            prefill_latency_s REAL,
            total_latency_s REAL,
            decode_tps REAL,
            quality_score REAL,
            success INTEGER,
            citation_score REAL,
            tool_success_rate REAL,
            privacy_requirement TEXT DEFAULT '',
            cost_note TEXT DEFAULT '',
            recommended_config TEXT DEFAULT '',
            sales_summary TEXT DEFAULT '',
            external_level TEXT DEFAULT 'internal',
            failure_reason TEXT DEFAULT '',
            evidence_path TEXT DEFAULT '',
            next_action TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            extra_json TEXT
        )"""
        ),
        _exec(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_app_cases_case_id ON application_cases(case_id)"
        ),
        _exec("CREATE INDEX IF NOT EXISTS idx_app_cases_run ON application_cases(run_id)"),
        _exec("CREATE INDEX IF NOT EXISTS idx_app_cases_scenario ON application_cases(scenario)"),
        _exec("CREATE INDEX IF NOT EXISTS idx_app_cases_model ON application_cases(model_name)"),
        _exec(
            "CREATE INDEX IF NOT EXISTS idx_app_cases_external ON application_cases(external_level)"
        ),
        _exec("CREATE INDEX IF NOT EXISTS idx_app_cases_machine ON application_cases(machine_id)"),
    ],
}


def get_current_version(conn: sqlite3.Connection) -> str:
    """Get当前DatabaseVersion"""
    try:
        cursor = conn.execute("SELECT value FROM db_meta WHERE key = 'schema_version'")
        row = cursor.fetchone()
        return str(row[0]) if row else "0.0.0"
    except sqlite3.OperationalError:
        # db_meta 表not存in
        return "0.0.0"


def set_version(conn: sqlite3.Connection, version: str):
    """SetDatabaseVersion"""
    conn.execute(
        """
        INSERT OR REPLACE INTO db_meta (key, value, updated_at)
        VALUES ('schema_version', ?, CURRENT_TIMESTAMP)
    """,
        (version,),
    )


def compare_versions(v1: str, v2: str) -> int:
    """
    比较Version号

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """

    def parse(v):
        return [int(x) for x in v.split(".")]

    p1, p2 = parse(v1), parse(v2)

    # 补齐长度
    while len(p1) < len(p2):
        p1.append(0)
    while len(p2) < len(p1):
        p2.append(0)

    for a, b in zip(p1, p2, strict=True):
        if a < b:
            return -1
        if a > b:
            return 1
    return 0


def run_migrations(conn: sqlite3.Connection, target_version: str = SCHEMA_VERSION):
    """
    执行Migration

    Args:
        conn: DatabaseConnect
        target_version: 目标Version
    """
    current_version = get_current_version(conn)

    if compare_versions(current_version, target_version) >= 0:
        logger.info(f"Database已is最新Version: {current_version}")
        return

    logger.info(f"开始Migration: {current_version} -> {target_version}")

    # Getneed执行Migration
    migrations_to_run = []
    for version, migrations in MIGRATIONS.items():
        if (
            compare_versions(version, current_version) > 0
            and compare_versions(version, target_version) <= 0
        ):
            migrations_to_run.append((version, migrations))

    # 按VersionSort
    migrations_to_run.sort(key=lambda x: [int(v) for v in x[0].split(".")])

    # 执行Migration
    for version, migrations in migrations_to_run:
        logger.info(f"执行Migration: {version}")
        try:
            for migration in migrations:
                migration(conn)
            set_version(conn, version)
            conn.commit()
            logger.info(f"Migration完成: {version}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Migration失败: {version}, Error: {e}")
            raise

    logger.info(f"所hasMigration完成，当前Version: {target_version}")


def register_migration(version: str):
    """
    Decorator：RegisterMigration函数

    Usage:
        @register_migration("1.1.0")
        def add_new_field(conn):
            conn.execute("ALTER TABLE test_runs ADD COLUMN new_field TEXT")
    """

    def decorator(func: MigrationFunc) -> MigrationFunc:
        if version not in MIGRATIONS:
            MIGRATIONS[version] = []
        MIGRATIONS[version].append(func)
        return func

    return decorator


def check_database_health(conn: sqlite3.Connection) -> dict:
    """
    CheckDatabase健康Status

    Returns:
        健康Status字典
    """
    issues: list[str] = []
    tables_info: dict[str, dict[str, Any]] = {}
    health = {
        "version": get_current_version(conn),
        "tables": tables_info,
        "integrity": "unknown",
        "issues": issues,
    }

    try:
        # Check完整性
        cursor = conn.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        health["integrity"] = result

        if result != "ok":
            issues.append(f"完整性Check失败: {result}")

        # Check表
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            tables_info[table] = {"count": count}

        # Check外键
        cursor = conn.execute("PRAGMA foreign_key_check")
        fk_issues = cursor.fetchall()
        if fk_issues:
            issues.append(f"外键问题: {len(fk_issues)} ")

    except Exception as e:
        issues.append(f"Check失败: {e}")

    return health


def optimize_database(conn: sqlite3.Connection):
    """优化Database"""
    logger.info("开始优化Database...")

    # 分析表
    conn.execute("ANALYZE")

    # 重建Index
    conn.execute("REINDEX")

    # CleanupIdle页
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    conn.commit()
    logger.info("Database优化完成")
