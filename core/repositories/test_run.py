"""
Test运行 Repository
"""

from datetime import datetime
from typing import Any

from core.database.connection import Database
from core.models.test_run import TestRun, TestRunStatus
from core.repositories.base import BaseRepository
from core.run_lifecycle import (
    InvalidRunTransition,
    RunEvent,
    RunStatus,
    advance_run,
    event_for_transition,
)


class TestRunRepository(BaseRepository[TestRun]):
    """Test运行 Repository"""

    def __init__(self, database: Database | None = None):
        super().__init__(database)
        self._table_name = "test_runs"

    def _from_row(self, row: dict[str, Any]) -> TestRun:
        return TestRun.from_row(row)

    def insert(self, run: TestRun) -> int | None:
        """
        InsertTest运行

        Args:
            run: TestRun 实例

        Returns:
            新记录 ID
        """
        data = run.to_dict()
        columns = [k for k, v in data.items() if v is not None and k != "id"]
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)
        values = [v for k, v in data.items() if v is not None and k != "id"]

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
        stored = self.db.fetch_one("SELECT status FROM test_runs WHERE id = ?", (run.id,))
        if stored is None:
            return False
        previous = stored["status"]
        if previous != run.status:
            event_for_transition(previous, run.status)

        data = run.to_dict()
        set_clause = ", ".join([f"{k} = ?" for k in data if k != "id"])
        values = [v for k, v in data.items() if k != "id"] + [run.id, previous]

        sql = f"UPDATE test_runs SET {set_clause} WHERE id = ? AND status = ?"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.rowcount > 0


    def find_by_status(self, status: str, limit: int = 100) -> list[TestRun]:
        """based onStatus查找"""
        return self.find_by("status = ?", (status,), limit)

    def find_by_model(self, model_id: str, limit: int = 100) -> list[TestRun]:
        """based onModel查找"""
        return self.find_by("model_id = ?", (model_id,), limit)

    def find_by_type(self, test_type: str, limit: int = 100) -> list[TestRun]:
        """based onTest Type查找"""
        return self.find_by("test_type = ?", (test_type,), limit)

    def find_by_date_range(
        self, start: datetime, end: datetime, limit: int = 100
    ) -> list[TestRun]:
        """based on日期范围查找"""
        return self.find_by(
            "created_at BETWEEN ? AND ?", (start.isoformat(), end.isoformat()), limit
        )



    def find_recent(self, limit: int = 20) -> list[TestRun]:
        """查找最近Test"""
        return self.find_all(limit=limit, order_by="created_at DESC")

    def update_status(
        self, run_id: int, status: str, progress: float | None = None
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
        row = self.db.fetch_one("SELECT status FROM test_runs WHERE id = ?", (run_id,))
        if row is None:
            return False
        current = row["status"]
        if current == status:
            if progress is None:
                return True
            if current in {
                RunStatus.CANCELLED.value,
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
            }:
                raise InvalidRunTransition(f"Cannot update progress of a {current} run")
            return self.update_by(
                {"progress_percent": progress}, "id = ? AND status = ?", (run_id, current)
            ) > 0
        event = event_for_transition(current, status)
        fields = {"progress_percent": progress} if progress is not None else None
        return self.advance(run_id, event, fields, expected_status=current)

    def advance(
        self,
        run_id: int,
        event: RunEvent,
        fields: dict[str, Any] | None = None,
        expected_status: str | None = None,
    ) -> bool:
        """Persist one legal transition if no other worker changed the state."""
        row = self.db.fetch_one("SELECT status FROM test_runs WHERE id = ?", (run_id,))
        if row is None:
            return False
        previous = row["status"]
        if expected_status is not None and previous != expected_status:
            return False
        next_status = advance_run(previous, event)
        data = {**(fields or {}), "status": next_status.value}
        return self.update_by(data, "id = ? AND status = ?", (run_id, previous)) > 0

    def update_progress(
        self, run_id: int, completed: int, total: int, failed: int = 0
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

        return self.update_by(
            data,
            "id = ? AND status IN (?, ?, ?)",
            (
                run_id,
                TestRunStatus.RUNNING.value,
                TestRunStatus.PAUSING.value,
                TestRunStatus.CANCELLING.value,
            ),
        ) > 0

    def update_statistics(self, run_id: int, stats: dict[str, Any]) -> bool:
        """
        UpdateStatistics信息

        Args:
            run_id: 运行 ID
            stats: Statistics信息字典

        Returns:
            is否succeeded
        """
        allowed_fields = [
            "avg_ttft",
            "avg_tps",
            "avg_tpot",
            "p50_ttft",
            "p95_ttft",
            "p99_ttft",
            "total_tokens",
            "duration_seconds",
            "success_rate",
            "completed_requests",
            "failed_requests",
        ]

        data = {k: v for k, v in stats.items() if k in allowed_fields}

        if not data:
            return False

        return self.update_by(data, "id = ?", (run_id,)) > 0

    def complete(
        self, run_id: int, success: bool = True, stats: dict | None = None
    ) -> bool:
        """
        标记完成

        Args:
            run_id: 运行 ID
            success: is否succeeded
            stats: Statistics信息（optional）

        Returns:
            is否succeeded
        """
        event = RunEvent.COMPLETE if success else RunEvent.FAIL
        data = {
            "completed_at": datetime.now().isoformat(),
        }

        if stats:
            data.update(stats)

        return self.advance(run_id, event, data)

    def search(self, query: str, limit: int = 50) -> list[TestRun]:
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
            limit,
        )
