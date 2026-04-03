"""
Database Migrations Module

管理 Schema VersionandMigration。
"""

import sqlite3
import logging
from typing import List, Callable, Dict

from .schema import SCHEMA_VERSION

logger = logging.getLogger(__name__)


# Migration函数类型
MigrationFunc = Callable[[sqlite3.Connection], None]


# MigrationRegister表：Version号 -> Migration函数列表
MIGRATIONS: Dict[str, List[MigrationFunc]] = {
    # 1.0.0 -> 1.1.0: Add prompt_text and output_text 字段
    "1.1.0": [
        lambda conn: conn.execute("ALTER TABLE test_results ADD COLUMN prompt_text TEXT"),
        lambda conn: conn.execute("ALTER TABLE test_results ADD COLUMN output_text TEXT"),
    ],
}


def get_current_version(conn: sqlite3.Connection) -> str:
    """Get当前DatabaseVersion"""
    try:
        cursor = conn.execute("SELECT value FROM db_meta WHERE key = 'schema_version'")
        row = cursor.fetchone()
        return row[0] if row else "0.0.0"
    except sqlite3.OperationalError:
        # db_meta 表not存in
        return "0.0.0"


def set_version(conn: sqlite3.Connection, version: str):
    """SetDatabaseVersion"""
    conn.execute("""
        INSERT OR REPLACE INTO db_meta (key, value, updated_at)
        VALUES ('schema_version', ?, CURRENT_TIMESTAMP)
    """, (version,))


def compare_versions(v1: str, v2: str) -> int:
    """
    比较Version号

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    def parse(v):
        return [int(x) for x in v.split('.')]

    p1, p2 = parse(v1), parse(v2)

    # 补齐长度
    while len(p1) < len(p2):
        p1.append(0)
    while len(p2) < len(p1):
        p2.append(0)

    for a, b in zip(p1, p2):
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
        if compare_versions(version, current_version) > 0 and compare_versions(version, target_version) <= 0:
            migrations_to_run.append((version, migrations))

    # 按VersionSort
    migrations_to_run.sort(key=lambda x: [int(v) for v in x[0].split('.')])

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
    health = {
        "version": get_current_version(conn),
        "tables": {},
        "integrity": "unknown",
        "issues": [],
    }

    try:
        # Check完整性
        cursor = conn.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        health["integrity"] = result

        if result != "ok":
            health["issues"].append(f"完整性Check失败: {result}")

        # Check表
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            health["tables"][table] = {"count": count}

        # Check外键
        cursor = conn.execute("PRAGMA foreign_key_check")
        fk_issues = cursor.fetchall()
        if fk_issues:
            health["issues"].append(f"外键问题: {len(fk_issues)} ")

    except Exception as e:
        health["issues"].append(f"Check失败: {e}")

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
