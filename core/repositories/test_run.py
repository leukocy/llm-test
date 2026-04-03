"""
Test运行 Repository
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from core.database.connection import Database, db
from core.repositories.base import BaseRepository
from core.models.test_run import TestRun, TestRunStatus


class TestRunRepository(BaseRepository[TestRun]):
    """Test运行 Repository"""

    def __init__(self, database: Database = None):
        super().__init__(database)
        self._table_name = "test_runs"

    def _from_row(self, row: Dict[str, Any]) -> TestRun:
        return TestRun.from_row(row)

    def insert(self, run: TestRun) -> int:
        """
        InsertTest运行

        Args:
            run: TestRun 实例

        Returns:
            新记录 ID
        """
        data = run.to_dict()
        columns = [k for k, v in data.items() if v is not None and k != 'id']
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)
        values = [v for k, v in data.items() if v is not None and k != 'id']

        sql = f"INSERT INTO test_runs ({columns_str}) VALUES ({placeholders})"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.lastrowid

    def update(self, run: TestRun) -> bool:
        """
        UpdateTest运行

        Args:
            run: TestRun 实例

        Returns:
            is否succeeded
        """
        if run.id is None:
            return False

        data = run.to_dict()
        set_clause = ", ".join([f"{k} = ?" for k in data.keys() if k != 'id'])
        values = [v for k, v in data.items() if k != 'id'] + [run.id]

        sql = f"UPDATE test_runs SET {set_clause} WHERE id = ?"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.rowcount > 0

    def find_by_test_id(self, test_id: str) -> Optional[TestRun]:
        """based on test_id 查找"""
        return self.find_one_by("test_id = ?", (test_id,))

    def find_by_status(self, status: str, limit: int = 100) -> List[TestRun]:
        """based onStatus查找"""
        return self.find_by("status = ?", (status,), limit)

    def find_by_model(self, model_id: str, limit: int = 100) -> List[TestRun]:
        """based onModel查找"""
        return self.find_by("model_id = ?", (model_id,), limit)

    def find_by_type(self, test_type: str, limit: int = 100) -> List[TestRun]:
        """based onTest Type查找"""
        return self.find_by("test_type = ?", (test_type,), limit)

    def find_by_date_range(
        self,
        start: datetime,
        end: datetime,
        limit: int = 100
    ) -> List[TestRun]:
        """based on日期范围查找"""
        return self.find_by(
            "created_at BETWEEN ? AND ?",
            (start.isoformat(), end.isoformat()),
            limit
        )

    def find_running(self) -> List[TestRun]:
        """查找所hasRunningTest"""
        return self.find_by_status(TestRunStatus.RUNNING.value)

    def find_paused(self) -> List[TestRun]:
        """查找所has暂停Test"""
        return self.find_by_status(TestRunStatus.PAUSED.value)

    def find_recent(self, limit: int = 20) -> List[TestRun]:
        """查找最近Test"""
        return self.find_all(limit=limit, order_by="created_at DESC")

    def update_status(
        self,
        run_id: int,
        status: str,
        progress: float = None
    ) -> bool:
        """
        UpdateStatus（轻量级Update）

        Args:
            run_id: 运行 ID
            status: 新Status
            progress: 进度百分比（optional）

        Returns:
            is否succeeded
        """
        if progress is not None:
            data = {"status": status, "progress_percent": progress}
        else:
            data = {"status": status}

        return self.update_by(data, "id = ?", (run_id,)) > 0

    def update_progress(
        self,
        run_id: int,
        completed: int,
        total: int,
        failed: int = 0
    ) -> bool:
        """
        Update进度

        Args:
            run_id: 运行 ID
            completed: Completed数量
            total: 总数量
            failed: 失败数量

        Returns:
            is否succeeded
        """
        progress = (completed / total * 100) if total > 0 else 0
        success_rate = ((completed - failed) / completed * 100) if completed > 0 else 0

        data = {
            "completed_requests": completed,
            "total_requests": total,
            "failed_requests": failed,
            "progress_percent": progress,
            "success_rate": success_rate,
        }

        return self.update_by(data, "id = ?", (run_id,)) > 0

    def update_statistics(self, run_id: int, stats: Dict[str, Any]) -> bool:
        """
        UpdateStatistics信息

        Args:
            run_id: 运行 ID
            stats: Statistics信息字典

        Returns:
            is否succeeded
        """
        allowed_fields = [
            'avg_ttft', 'avg_tps', 'avg_tpot',
            'p50_ttft', 'p95_ttft', 'p99_ttft',
            'total_tokens', 'duration_seconds',
            'success_rate', 'completed_requests', 'failed_requests'
        ]

        data = {k: v for k, v in stats.items() if k in allowed_fields}

        if not data:
            return False

        return self.update_by(data, "id = ?", (run_id,)) > 0

    def complete(self, run_id: int, success: bool = True, stats: Dict = None) -> bool:
        """
        标记完成

        Args:
            run_id: 运行 ID
            success: is否succeeded
            stats: Statistics信息（optional）

        Returns:
            is否succeeded
        """
        status = TestRunStatus.COMPLETED.value if success else TestRunStatus.FAILED.value
        data = {
            "status": status,
            "completed_at": datetime.now().isoformat(),
        }

        if stats:
            data.update(stats)

        return self.update_by(data, "id = ?", (run_id,)) > 0

    def search(self, query: str, limit: int = 50) -> List[TestRun]:
        """
        搜索

        Args:
            query: 搜索关键词
            limit: 限制数量

        Returns:
            匹配Test运行列表
        """
        pattern = f"%{query}%"
        return self.find_by(
            "model_id LIKE ? OR tags LIKE ? OR notes LIKE ?",
            (pattern, pattern, pattern),
            limit
        )

    def get_statistics_summary(self, model_id: str = None) -> Dict[str, Any]:
        """
        GetStatistics摘要

        Args:
            model_id: Model ID（optional，not指定则Statistics所has）

        Returns:
            Statistics摘要字典
        """
        where = "model_id = ?" if model_id else "1=1"
        params = (model_id,) if model_id else ()

        sql = f"""
            SELECT
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_runs,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
                SUM(total_requests) as total_requests,
                SUM(completed_requests) as completed_requests,
                AVG(avg_ttft) as avg_ttft,
                AVG(avg_tps) as avg_tps
            FROM test_runs
            WHERE {where}
        """

        row = self.db.fetch_one(sql, params)
        return row if row else {}
