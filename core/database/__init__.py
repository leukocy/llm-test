"""
Database模块

提供 SQLite DatabaseConnect、Migrationand管理功能。
"""

from .backup import DatabaseBackup
from .connection import Database, db
from .manager import DatabaseManager, db_manager
from .migrations import check_database_health, optimize_database, run_migrations
from .schema import SCHEMA_VERSION, create_tables, get_schema_sql

__all__ = [
    "Database",
    "db",
    "SCHEMA_VERSION",
    "create_tables",
    "get_schema_sql",
    "DatabaseBackup",
    "run_migrations",
    "check_database_health",
    "optimize_database",
    "DatabaseManager",
    "db_manager",
]
