"""
应用用例 Repository（application_cases 表的 CRUD）。

镜像 TestRunRepository 的模式：insert 跳过 None（让 DB 默认值生效），
upsert 按 case_id 去重（自动采集重跑同一样本时覆盖，而非堆叠）。
"""

from __future__ import annotations

from typing import Any

from core.database.connection import Database
from core.models.application_case import ApplicationCase
from core.repositories.base import BaseRepository


class ApplicationCaseRepository(BaseRepository[ApplicationCase]):
    """application_cases 表的 Repository。"""

    def __init__(self, database: Database | None = None):
        super().__init__(database)
        self._table_name = "application_cases"

    def _from_row(self, row: dict[str, Any]) -> ApplicationCase:
        return ApplicationCase.from_row(row)

    def insert(self, case: ApplicationCase) -> int | None:
        """插入一条用例（跳过 None，让 DB 默认值/NULL 生效）。返回新记录 id。"""
        data = case.to_dict()
        columns = [k for k, v in data.items() if v is not None and k != "id"]
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)
        values = [v for k, v in data.items() if v is not None and k != "id"]
        sql = f"INSERT INTO application_cases ({columns_str}) VALUES ({placeholders})"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.lastrowid

    def upsert(self, case: ApplicationCase) -> int | None:
        """按 case_id 去重写入：存在则更新，否则插入。返回记录 id。"""
        existing = self.find_one_by("case_id = ?", (case.case_id,))
        if existing is None:
            return self.insert(case)
        # 更新：只写非 None 的列，不动 id / case_id / created_at
        data = case.to_dict()
        update_cols = {
            k: v
            for k, v in data.items()
            if k not in ("id", "case_id", "created_at") and v is not None
        }
        if update_cols:
            self.update_by(update_cols, "case_id = ?", (case.case_id,))
        return existing.id

    def find_recent(self, limit: int = 200) -> list[ApplicationCase]:
        """最近用例（按 created_at 倒序）。"""
        return self.find_by("1=1", (), limit=limit, order_by="created_at DESC")

    def find_by_filter(
        self,
        scenario: str | None = None,
        model_name: str | None = None,
        machine_id: str | None = None,
        external_level: str | None = None,
        source: str | None = None,
        limit: int = 500,
    ) -> list[ApplicationCase]:
        """按 maTest 维度筛选。任一参数 None = 不过滤。"""
        clauses: list[str] = []
        params: list[Any] = []
        if scenario:
            clauses.append("scenario = ?")
            params.append(scenario)
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if machine_id:
            clauses.append("machine_id = ?")
            params.append(machine_id)
        if external_level:
            clauses.append("external_level = ?")
            params.append(external_level)
        if source:
            clauses.append("source = ?")
            params.append(source)
        where = " AND ".join(clauses) if clauses else "1=1"
        return self.find_by(
            where, tuple(params), limit=limit, order_by="created_at DESC"
        )
