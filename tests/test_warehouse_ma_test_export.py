"""build_ma_test_rows_from_cases 单元测试（行键集 == MA_TEST_FIELDS）。"""

from __future__ import annotations

import pytest

from core.database.connection import Database
from core.models import ApplicationCase
from core.repositories.application_case import ApplicationCaseRepository
from core.warehouse import build_ma_test_rows_from_cases
from core.warehouse.templates import MA_TEST_FIELDS


@pytest.fixture(autouse=True)
def _reset_db_singleton():
    Database._instance = None
    yield
    Database._instance = None


class _FakeDB:
    """暴露 list_application_cases 的假 DB（包装 repository）。"""

    def __init__(self, repo):
        self._repo = repo

    def list_application_cases(self, **kw):
        return self._repo.find_by_filter(**kw)


def _fresh(tmp_path):
    db_path = str(tmp_path / "ma_test.db")
    return ApplicationCaseRepository(Database(db_path))  # 单例已重置 → tmp


def test_rows_match_ma_test_field_set(tmp_path):
    repo = _fresh(tmp_path)
    repo.insert(
        ApplicationCase(
            case_id="c1",
            scenario="coding",
            model_name="M1",
            success=True,
            quality_score=8.0,
        )
    )
    rows = build_ma_test_rows_from_cases(_FakeDB(repo))
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(MA_TEST_FIELDS)
    assert rows[0]["case_id"] == "c1"
    assert rows[0]["scenario"] == "coding"
    assert rows[0]["quality_score"] == 8.0


def test_filter_by_scenario(tmp_path):
    repo = _fresh(tmp_path)
    repo.insert(ApplicationCase(case_id="c1", scenario="coding", model_name="M1"))
    repo.insert(ApplicationCase(case_id="c2", scenario="retrieval", model_name="M1"))
    rows = build_ma_test_rows_from_cases(_FakeDB(repo), scenario="coding")
    assert len(rows) == 1
    assert rows[0]["scenario"] == "coding"


def test_empty_when_no_db_method():
    class _NoMethod:
        pass

    assert build_ma_test_rows_from_cases(_NoMethod()) == []


def test_missing_fields_default_none(tmp_path):
    repo = _fresh(tmp_path)
    repo.insert(ApplicationCase(case_id="c1", scenario="coding"))  # 大多字段缺测
    rows = build_ma_test_rows_from_cases(_FakeDB(repo))
    assert rows[0]["citation_score"] is None
    assert rows[0]["tool_success_rate"] is None
    assert rows[0]["retrieval_latency_s"] is None
