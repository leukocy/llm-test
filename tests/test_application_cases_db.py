"""application_cases 表 Repository CRUD + upsert（临时 sqlite，不碰单例库）。"""

from __future__ import annotations

import pytest

from core.database.connection import Database
from core.models import ApplicationCase
from core.repositories.application_case import ApplicationCaseRepository


@pytest.fixture(autouse=True)
def _reset_db_singleton():
    """每个测试重置 Database 全局单例，让 Database(tmp_path) 指向临时库。"""
    Database._instance = None
    yield
    Database._instance = None


def _fresh_repo(tmp_path) -> ApplicationCaseRepository:
    """临时 DB（_ensure_schema 自动建 application_cases），返回 Repository。"""
    db_path = str(tmp_path / "test_app_cases.db")
    db = Database(db_path)  # 单例已重置 → 指向 tmp
    return ApplicationCaseRepository(db)


def test_insert_and_find(tmp_path):
    repo = _fresh_repo(tmp_path)
    c = ApplicationCase(case_id="t1", scenario="coding", model_name="M1", success=True)
    rid = repo.insert(c)
    assert rid > 0
    got = repo.find_one_by("case_id = ?", ("t1",))
    assert got is not None
    assert got.scenario == "coding"
    assert got.success is True


def test_upsert_dedups_by_case_id(tmp_path):
    repo = _fresh_repo(tmp_path)
    repo.upsert(ApplicationCase(case_id="dup1", scenario="coding", model_name="M1", success=True))
    repo.upsert(ApplicationCase(case_id="dup1", scenario="coding", model_name="M1",
                                failure_reason="oom"))
    assert repo.count("case_id = 'dup1'") == 1
    got = repo.find_one_by("case_id = ?", ("dup1",))
    assert got.failure_reason == "oom"  # 第二次覆盖


def test_find_by_filter(tmp_path):
    repo = _fresh_repo(tmp_path)
    repo.insert(ApplicationCase(case_id="a", scenario="coding", model_name="M1", external_level="internal"))
    repo.insert(ApplicationCase(case_id="b", scenario="retrieval", model_name="M2", external_level="publishable"))
    repo.insert(ApplicationCase(case_id="c", scenario="coding", model_name="M1", external_level="review"))

    coding_m1 = repo.find_by_filter(scenario="coding", model_name="M1")
    assert {c.case_id for c in coding_m1} == {"a", "c"}

    publishable = repo.find_by_filter(external_level="publishable")
    assert {c.case_id for c in publishable} == {"b"}

    all_rows = repo.find_by_filter()
    assert len(all_rows) == 3


def test_delete_by_case_id(tmp_path):
    repo = _fresh_repo(tmp_path)
    repo.insert(ApplicationCase(case_id="del1", scenario="coding"))
    assert repo.delete_by("case_id = ?", ("del1",)) == 1
    assert repo.find_one_by("case_id = ?", ("del1",)) is None
