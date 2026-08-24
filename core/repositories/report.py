"""
报告 Repository
"""

from datetime import datetime
from typing import Any

from core.database.connection import Database
from core.models.report import Report
from core.repositories.base import BaseRepository


class ReportRepository(BaseRepository[Report]):
    """报告 Repository"""

    def __init__(self, database: Database | None = None):
        super().__init__(database)
        self._table_name = "reports"

    def _from_row(self, row: dict[str, Any]) -> Report:
        return Report.from_row(row)

    def insert(self, report: Report) -> int | None:
        """Insert报告"""
        data = report.to_dict()
        columns = [k for k, v in data.items() if v is not None and k != "id"]
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)
        values = [v for k, v in data.items() if v is not None and k != "id"]

        sql = f"INSERT INTO reports ({columns_str}) VALUES ({placeholders})"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.lastrowid

    def update(self, report: Report) -> bool:
        """Update报告"""
        if report.id is None:
            return False

        data = report.to_dict()
        set_clause = ", ".join([f"{k} = ?" for k in data if k != "id"])
        values = [v for k, v in data.items() if k != "id"] + [report.id]

        sql = f"UPDATE reports SET {set_clause} WHERE id = ?"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.rowcount > 0


    def find_by_run_id(self, run_id: int) -> list[Report]:
        """based on运行 ID 查找"""
        return self.find_by("run_id = ?", (run_id,))

    def find_by_model(self, model_id: str, limit: int = 100) -> list[Report]:
        """based onModel查找"""
        return self.find_by("model_id = ?", (model_id,), limit)

    def find_by_type(self, report_type: str, limit: int = 100) -> list[Report]:
        """based on报告类型查找"""
        return self.find_by("report_type = ?", (report_type,), limit)

    def find_by_date_range(
        self, start: datetime, end: datetime, limit: int = 100
    ) -> list[Report]:
        """based on日期范围查找"""
        return self.find_by(
            "created_at BETWEEN ? AND ?", (start.isoformat(), end.isoformat()), limit
        )

    def search(self, query: str, limit: int = 50) -> list[Report]:
        """搜索报告"""
        pattern = f"%{query}%"
        return self.find_by(
            "model_id LIKE ? OR notes LIKE ? OR tags LIKE ?",
            (pattern, pattern, pattern),
            limit,
        )



    def delete_old_reports(self, days: int = 90) -> int:
        """Delete旧报告"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        return self.delete_by(
            "created_at < ?", (datetime.fromtimestamp(cutoff).isoformat(),)
        )
