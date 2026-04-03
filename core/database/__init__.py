"""
Database模块

提供 SQLite DatabaseConnect、Migrationand管理功能。
"""

from .connection import Database, db
from .schema import SCHEMA_VERSION, create_tables, get_schema_sql
from .backup import DatabaseBackup
from .migrations import run_migrations, check_database_health, optimize_database
from .manager import DatabaseManager, db_manager

__all__ = [
    'Database',
    'db',
    'SCHEMA_VERSION',
    'create_tables',
    'get_schema_sql',
    'DatabaseBackup',
    'run_migrations',
    'check_database_health',
    'optimize_database',
    'DatabaseManager',
    'db_manager',
]
