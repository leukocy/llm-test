"""
报告 Repository
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from core.database.connection import Database, db
from core.repositories.base import BaseRepository
from core.models.report import Report, ReportType


class ReportRepository(BaseRepository[Report]):
    """报告 Repository"""

    def __init__(self, database: Database = None):
        super().__init__(database)
        self._table_name = "reports"

    def _from_row(self, row: Dict[str, Any]) -> Report:
        return Report.from_row(row)

    def insert(self, report: Report) -> int:
        """Insert报告"""
        data = report.to_dict()
        columns = [k for k, v in data.items() if v is not None and k != 'id']
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)
        values = [v for k, v in data.items() if v is not None and k != 'id']

        sql = f"INSERT INTO reports ({columns_str}) VALUES ({placeholders})"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.lastrowid

    def update(self, report: Report) -> bool:
        """Update报告"""
        if report.id is None:
            return False

        data = report.to_dict()
        set_clause = ", ".join([f"{k} = ?" for k in data.keys() if k != 'id'])
        values = [v for k, v in data.items() if k != 'id'] + [report.id]

        sql = f"UPDATE reports SET {set_clause} WHERE id = ?"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.rowcount > 0

    def find_by_report_id(self, report_id: str) -> Optional[Report]:
        """based on report_id 查找"""
        return self.find_one_by("report_id = ?", (report_id,))

    def find_by_run_id(self, run_id: int) -> List[Report]:
        """based on运行 ID 查找"""
        return self.find_by("run_id = ?", (run_id,))

    def find_by_model(self, model_id: str, limit: int = 100) -> List[Report]:
        """based onModel查找"""
        return self.find_by("model_id = ?", (model_id,), limit)

    def find_by_type(self, report_type: str, limit: int = 100) -> List[Report]:
        """based on报告类型查找"""
        return self.find_by("report_type = ?", (report_type,), limit)

    def find_by_date_range(
        self,
        start: datetime,
        end: datetime,
        limit: int = 100
    ) -> List[Report]:
        """based on日期范围查找"""
        return self.find_by(
            "created_at BETWEEN ? AND ?",
            (start.isoformat(), end.isoformat()),
            limit
        )

    def search(self, query: str, limit: int = 50) -> List[Report]:
        """搜索报告"""
        pattern = f"%{query}%"
        return self.find_by(
            "model_id LIKE ? OR notes LIKE ? OR tags LIKE ?",
            (pattern, pattern, pattern),
            limit
        )

    def update_export_paths(
        self,
        report_id: int,
        json_path: str = None,
        html_path: str = None,
        markdown_path: str = None,
        excel_path: str = None
    ) -> bool:
        """UpdateExport路径"""
        data = {}
        if json_path:
            data['json_path'] = json_path
        if html_path:
            data['html_path'] = html_path
        if markdown_path:
            data['markdown_path'] = markdown_path
        if excel_path:
            data['excel_path'] = excel_path

        if not data:
            return False

        return self.update_by(data, "id = ?", (report_id,)) > 0

    def get_summary(self, model_id: str = None) -> Dict[str, Any]:
        """Get报告摘要"""
        where = "model_id = ?" if model_id else "1=1"
        params = (model_id,) if model_id else ()

        sql = f"""
            SELECT
                COUNT(*) as total_reports,
                COUNT(DISTINCT model_id) as unique_models,
                COUNT(DISTINCT report_type) as report_types,
                MAX(created_at) as latest_report
            FROM reports
            WHERE {where}
        """
        row = self.db.fetch_one(sql, params)
        return row if row else {}

    def delete_old_reports(self, days: int = 90) -> int:
        """Delete旧报告"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        return self.delete_by("created_at < ?", (datetime.fromtimestamp(cutoff).isoformat(),))
