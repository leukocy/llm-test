"""
执行Log Repository
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from core.database.connection import Database, db
from core.repositories.base import BaseRepository
from core.models.exec_log import ExecLog, LogLevel


class ExecLogRepository(BaseRepository[ExecLog]):
    """执行Log Repository"""

    def __init__(self, database: Database = None):
        super().__init__(database)
        self._table_name = "execution_logs"

    def _from_row(self, row: Dict[str, Any]) -> ExecLog:
        return ExecLog.from_row(row)

    def insert(self, log: ExecLog) -> int:
        """InsertLog"""
        data = log.to_dict()
        columns = [k for k, v in data.items() if v is not None and k != 'id']
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)
        values = [v for k, v in data.items() if v is not None and k != 'id']

        sql = f"INSERT INTO execution_logs ({columns_str}) VALUES ({placeholders})"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.lastrowid

    def insert_batch(self, logs: List[ExecLog]) -> int:
        """批量InsertLog"""
        if not logs:
            return 0

        sql = """
            INSERT INTO execution_logs (
                run_id, level, message, session_id, metrics_json, error, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        params_list = []
        for log in logs:
            data = log.to_dict()
            params_list.append((
                data.get('run_id'),
                data.get('level'),
                data.get('message'),
                data.get('session_id'),
                data.get('metrics_json'),
                data.get('error'),
                data.get('timestamp'),
            ))

        return self.db.execute_many(sql, params_list)

    def find_by_run_id(self, run_id: int, limit: int = 1000) -> List[ExecLog]:
        """based on运行 ID 查找"""
        return self.find_by("run_id = ?", (run_id,), limit, order_by="timestamp ASC")

    def find_by_level(self, level: str, run_id: int = None, limit: int = 100) -> List[ExecLog]:
        """based on级别查找"""
        if run_id:
            return self.find_by("level = ? AND run_id = ?", (level, run_id), limit)
        return self.find_by("level = ?", (level,), limit)

    def find_errors(self, run_id: int = None, limit: int = 100) -> List[ExecLog]:
        """查找Error Logs"""
        return self.find_by_level(LogLevel.ERROR.value, run_id, limit)

    def find_warnings(self, run_id: int = None, limit: int = 100) -> List[ExecLog]:
        """查找WarningLog"""
        return self.find_by_level(LogLevel.WARNING.value, run_id, limit)

    def get_level_counts(self, run_id: int = None) -> Dict[str, int]:
        """Get各级别Log数量"""
        where = "run_id = ?" if run_id else "1=1"
        params = (run_id,) if run_id else ()

        sql = f"""
            SELECT level, COUNT(*) as count
            FROM execution_logs
            WHERE {where}
            GROUP BY level
        """
        rows = self.db.fetch_all(sql, params)
        return {row['level']: row['count'] for row in rows}

    def search(self, query: str, run_id: int = None, limit: int = 50) -> List[ExecLog]:
        """搜索Log"""
        pattern = f"%{query}%"
        if run_id:
            return self.find_by(
                "run_id = ? AND message LIKE ?",
                (run_id, pattern),
                limit
            )
        return self.find_by("message LIKE ?", (pattern,), limit)

    def cleanup_old_logs(self, days: int = 30) -> int:
        """Cleanup旧Log"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        return self.delete_by("timestamp < ?", (datetime.fromtimestamp(cutoff).isoformat(),))
