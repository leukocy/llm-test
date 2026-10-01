"""
DatabaseConnect管理模块

提供Thread安全 SQLite Connection Pooland常用操作封装。
"""

import os
import re
import sqlite3
import threading
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Optional, cast

from .schema import create_tables

# 标识符白名单: 表名/列名/排序字段拼接进 SQL 前必须通过校验(安全审查 #8)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
# order_by 允许 "col ASC/DESC" 形式
_ORDER_BY_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?( (ASC|DESC))?"
    r"(, [A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?( (ASC|DESC))?)*$"
)


def _validate_identifier(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Illegal SQL identifier: {name!r}")
    return name


def _validate_order_by(order_by: str) -> str:
    if not _ORDER_BY_RE.match(order_by):
        raise ValueError(f"Illegal ORDER BY expression: {order_by!r}")
    return order_by


def _validate_where(where: str) -> str:
    """WHERE 子句仅做结构性黑名单校验(参数值已 ? 参数化)。"""
    lowered = where.lower()
    for token in (
        ";",
        "--",
        "/*",
        "drop ",
        "delete ",
        "insert ",
        "update ",
        "attach ",
        "pragma ",
        "union select",
    ):
        if token in lowered:
            raise ValueError(f"Illegal WHERE clause fragment: {where!r}")
    return where


class Database:
    """
    DatabaseConnect管理器

    特性：
    - Singleton模式，全局共享
    - Thread安全（每Thread独立Connect）
    - WAL 模式支持Concurrency读写
    - 自动Initialize Schema
    """

    _instance: Optional["Database"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, db_path: str = "data/benchmark.db"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init(db_path)
        return cls._instance

    def _init(self, db_path: str):
        """InitializeDatabase"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialized = False

        # 确保Database文件and Schema 存in
        self._ensure_schema()

    def _ensure_schema(self):
        """确保 Schema 已Create"""
        if self._initialized:
            return

        with closing(self._get_raw_connection()) as conn, conn:
            create_tables(conn)
            # 执行迁移：补齐历史从未运行的 1.1.0，以及本次 1.2.0（老库才能拿到新列）。
            # 此前 run_migrations 是死代码（无调用方），现在补上。
            from .migrations import run_migrations

            run_migrations(conn)

        self._initialized = True

    def _get_raw_connection(self) -> sqlite3.Connection:
        """Get原始Connect（用于Initialize）"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        except BaseException:
            conn.close()
            raise
        return conn

    @contextmanager
    def get_connection(self):
        """
        GetThread安全Connect

        Yields:
            sqlite3.Connection: DatabaseConnect
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._get_raw_connection()
            self._local.conn.row_factory = sqlite3.Row

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    @property
    def path(self) -> Path:
        """DatabaseFile path"""
        return self.db_path

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        执行单条 SQL（自动提交）

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            Cursor 对象
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cast(sqlite3.Cursor, cursor)

    def execute_many(self, sql: str, params_list: list[tuple]) -> int:
        """
        Batch Execution SQL

        Args:
            sql: SQL 语句
            params_list: 参数列表

        Returns:
            影响行数
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(sql, params_list)
            conn.commit()
            return cast(int, cursor.rowcount)

    def fetch_one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        """
        Query单条记录

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            字典or None
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """
        Query多条记录

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            字典列表
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def fetch_value(self, sql: str, params: tuple = ()) -> Any:
        """
        Query单值

        Args:
            sql: SQL 语句
            params: 参数元组

        Returns:
            单值or None
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            return row[0] if row else None

    def insert(self, table: str, data: dict[str, Any]) -> int:
        """
        Insert记录

        Args:
            table: 表名
            data: Data字典

        Returns:
            新记录 ID
        """
        _validate_identifier(table)
        columns = ", ".join(_validate_identifier(k) for k in data)
        placeholders = ", ".join(["?" for _ in data])
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor = self.execute(sql, tuple(data.values()))
        return cast(int, cursor.lastrowid)

    def update(self, table: str, data: dict[str, Any], where: str, where_params: tuple = ()) -> int:
        """
        Update记录

        Args:
            table: 表名
            data: UpdateData字典
            where: WHERE 子句
            where_params: WHERE 参数

        Returns:
            影响行数
        """
        _validate_identifier(table)
        _validate_where(where)
        set_clause = ", ".join(f"{_validate_identifier(k)} = ?" for k in data)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        cursor = self.execute(sql, tuple(data.values()) + where_params)
        return cursor.rowcount

    def delete(self, table: str, where: str, where_params: tuple = ()) -> int:
        """
        Delete记录

        Args:
            table: 表名
            where: WHERE 子句
            where_params: WHERE 参数

        Returns:
            影响行数
        """
        _validate_identifier(table)
        _validate_where(where)
        sql = f"DELETE FROM {table} WHERE {where}"
        cursor = self.execute(sql, where_params)
        return cursor.rowcount

    def count(self, table: str, where: str = "", where_params: tuple = ()) -> int:
        """
        计数

        Args:
            table: 表名
            where: WHERE 子句（optional）
            params: 参数元组

        Returns:
            记录数
        """
        _validate_identifier(table)
        if where:
            _validate_where(where)
        sql = f"SELECT COUNT(*) as cnt FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return self.fetch_value(sql, where_params) or 0

    def table_exists(self, table_name: str) -> bool:
        """Check表is否存in"""
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        return self.fetch_one(sql, (table_name,)) is not None

    def get_database_size(self) -> int:
        """GetDatabase文件大小（字节）"""
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0

    def close(self):
        """Close当前ThreadConnect"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def close_all(self):
        """Close所hasConnect（仅用于Test）"""
        self.close()
        Database._instance = None


# 全局Database实例
db = Database(os.getenv("LLM_TEST_DB_PATH", "data/benchmark.db"))
