"""
DatabaseConnect管理模块

提供Thread安全 SQLite Connection Pooland常用操作封装。
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .schema import create_tables, SCHEMA_VERSION


class Database:
    """
    DatabaseConnect管理器

    特性：
    - Singleton模式，全局共享
    - Thread安全（每Thread独立Connect）
    - WAL 模式支持Concurrency读写
    - 自动Initialize Schema
    """

    _instance: Optional['Database'] = None
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

        with self._get_raw_connection() as conn:
            create_tables(conn)

        self._initialized = True

    def _get_raw_connection(self) -> sqlite3.Connection:
        """Get原始Connect（用于Initialize）"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        return conn

    @contextmanager
    def get_connection(self):
        """
        GetThread安全Connect

        Yields:
            sqlite3.Connection: DatabaseConnect
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
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

    def execute(self, sql: str, params: Tuple = ()) -> sqlite3.Cursor:
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
            return cursor

    def execute_many(self, sql: str, params_list: List[Tuple]) -> int:
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
            return cursor.rowcount

    def execute_script(self, script: str) -> None:
        """
        执行 SQL 脚本

        Args:
            script: SQL 脚本
        """
        with self.get_connection() as conn:
            conn.executescript(script)
            conn.commit()

    def fetch_one(self, sql: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
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

    def fetch_all(self, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
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

    def fetch_value(self, sql: str, params: Tuple = ()) -> Any:
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

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """
        Insert记录

        Args:
            table: 表名
            data: Data字典

        Returns:
            新记录 ID
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor = self.execute(sql, tuple(data.values()))
        return cursor.lastrowid

    def update(self, table: str, data: Dict[str, Any], where: str, where_params: Tuple = ()) -> int:
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
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        cursor = self.execute(sql, tuple(data.values()) + where_params)
        return cursor.rowcount

    def delete(self, table: str, where: str, where_params: Tuple = ()) -> int:
        """
        Delete记录

        Args:
            table: 表名
            where: WHERE 子句
            where_params: WHERE 参数

        Returns:
            影响行数
        """
        sql = f"DELETE FROM {table} WHERE {where}"
        cursor = self.execute(sql, where_params)
        return cursor.rowcount

    def count(self, table: str, where: str = "", where_params: Tuple = ()) -> int:
        """
        计数

        Args:
            table: 表名
            where: WHERE 子句（optional）
            where_params: WHERE 参数

        Returns:
            记录数
        """
        sql = f"SELECT COUNT(*) as cnt FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return self.fetch_value(sql, where_params) or 0

    def table_exists(self, table_name: str) -> bool:
        """Check表is否存in"""
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        return self.fetch_one(sql, (table_name,)) is not None

    def get_schema_version(self) -> str:
        """Get Schema Version"""
        result = self.fetch_one("SELECT value FROM db_meta WHERE key = 'schema_version'")
        return result['value'] if result else "unknown"

    def vacuum(self):
        """执行 VACUUM 优化Database"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()

    def get_database_size(self) -> int:
        """GetDatabase文件大小（字节）"""
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0

    def close(self):
        """Close当前ThreadConnect"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def close_all(self):
        """Close所hasConnect（仅用于Test）"""
        self.close()
        Database._instance = None


# 全局Database实例
db = Database()
