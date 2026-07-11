"""
API Log Repository
"""

from datetime import datetime
from typing import Any

from core.database.connection import Database
from core.models.api_log import ApiLog, ApiLogStatus
from core.repositories.base import BaseRepository


class ApiLogRepository(BaseRepository[ApiLog]):
    """API Log Repository"""

    def __init__(self, database: Database | None = None):
        super().__init__(database)
        self._table_name = "api_logs"

    def _from_row(self, row: dict[str, Any]) -> ApiLog:
        return ApiLog.from_row(row)

    def insert(self, log: ApiLog) -> int | None:
        """InsertLog"""
        data = log.to_dict()
        columns = [k for k, v in data.items() if v is not None and k != "id"]
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)
        values = [v for k, v in data.items() if v is not None and k != "id"]

        sql = f"INSERT INTO api_logs ({columns_str}) VALUES ({placeholders})"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.lastrowid

    def find_by_run_id(self, run_id: int, limit: int = 1000) -> list[ApiLog]:
        """based on运行 ID 查找"""
        return self.find_by("run_id = ?", (run_id,), limit)

    def find_errors(self, limit: int = 100) -> list[ApiLog]:
        """查找Error Logs"""
        return self.find_by("status = ?", (ApiLogStatus.ERROR.value,), limit)

    def find_by_session(self, session_id: str) -> list[ApiLog]:
        """based on session_id 查找"""
        return self.find_by("session_id = ?", (session_id,))

    def find_by_date_range(
        self, start: datetime, end: datetime, limit: int = 1000
    ) -> list[ApiLog]:
        """based on日期范围查找"""
        return self.find_by(
            "created_at BETWEEN ? AND ?", (start.isoformat(), end.isoformat()), limit
        )

    def get_statistics(self, run_id: int | None = None) -> dict[str, Any]:
        """GetStatistics信息"""
        where = "run_id = ?" if run_id else "1=1"
        params = (run_id,) if run_id else ()

        sql = f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error,
                AVG(ttft) as avg_ttft,
                AVG(total_time) as avg_total_time
            FROM api_logs
            WHERE {where}
        """
        row = self.db.fetch_one(sql, params)
        return row if row else {}

    def cleanup_old_logs(self, days: int = 30) -> int:
        """Cleanup旧Log"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        return self.delete_by(
            "created_at < ?", (datetime.fromtimestamp(cutoff).isoformat(),)
        )

    def get_error_summary(self, limit: int = 20) -> list[dict[str, Any]]:
        """GetError摘要"""
        sql = """
            SELECT
                error,
                COUNT(*) as count,
                model_id,
                provider
            FROM api_logs
            WHERE status = 'error' AND error IS NOT NULL
            GROUP BY error, model_id, provider
            ORDER BY count DESC
            LIMIT ?
        """
        return self.db.fetch_all(sql, (limit,))
