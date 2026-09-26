"""数据管理后端操作测试：delete_runs 级联 / count_runs / get_run_count。"""

from __future__ import annotations

from datetime import datetime

import pytest

from core.models import TestRun
from core.warehouse.query import WarehouseFilter, count_runs, query_runs


class _FakeDB:
    """最小数据库替身：get_recent_runs + get_run_count。"""

    def __init__(self, runs):
        self._runs = runs

    def get_recent_runs(self, limit: int = 500):
        return list(self._runs[:limit])

    def get_run_count(self) -> int:
        return len(self._runs)


def _run(test_id="t1", model_id="m", machine_id="host1", created=None) -> TestRun:
    return TestRun(
        test_id=test_id,
        model_id=model_id,
        test_type="concurrency",
        machine_id=machine_id,
        created_at=created or datetime(2026, 6, 1, 12, 0, 0),
        system_info={"hardware_fingerprint": {"machine_id": machine_id}},
    )


# ---------- count_runs ----------


def test_count_runs_ignores_limit_but_keeps_filter():
    runs = [
        _run("t1", machine_id="host1", created=datetime(2026, 6, 1)),
        _run("t2", machine_id="host1", created=datetime(2026, 6, 2)),
        _run("t3", machine_id="host2", created=datetime(2026, 6, 3)),
    ]
    db = _FakeDB(runs)
    # limit=1 会截断 query_runs 的候选, 但 count_runs 必须给全量匹配数
    flt = WarehouseFilter(machine_id="host1", limit=1)
    assert len(query_runs(db, flt)) == 1
    assert count_runs(db, flt) == 2
    assert count_runs(db, WarehouseFilter(limit=1)) == 3


# ---------- get_run_count / delete_runs（真实 SQLite）----------


@pytest.fixture()
def mgr(tmp_path):
    from core.database.connection import Database
    from core.database.manager import DatabaseManager

    Database._instance = None
    DatabaseManager._instance = None
    try:
        yield DatabaseManager(str(tmp_path / "dm.db"))
    finally:
        Database._instance = None
        DatabaseManager._instance = None


def test_get_run_count(mgr):
    assert mgr.get_run_count() == 0
    mgr.start_test_run("concurrency", "m1")
    mgr.start_test_run("prefill", "m2")
    assert mgr.get_run_count() == 2


def test_delete_runs_cascades_and_cleans_links(mgr):
    run = mgr.start_test_run("concurrency", "m")
    mgr.complete_test_run(run, success=True)
    mgr.save_result(run, {"ttft": 0.1, "tps": 50.0, "total_time": 1.0})

    # api_logs / reports 的 FK 是 SET NULL —— delete_runs 应显式清掉, 不留孤儿
    mgr.db.execute(
        "INSERT INTO api_logs (log_id, run_id, session_id, test_type, status, provider,"
        " model_id, api_base_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("log1", run.id, "s1", "concurrency", "success", "p", "m", "http://x"),
    )
    mgr.db.execute(
        "INSERT INTO reports (report_id, run_id, report_type, version, model_id, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("rep1", run.id, "concurrency", "1.0", "m", "2026-01-01"),
    )

    result = mgr.delete_runs([run.id])
    assert result == {"deleted": [run.id], "failed": {}}
    assert mgr.get_run_count() == 0
    for table in ("test_results", "api_logs", "reports", "execution_logs"):
        n = mgr.db.fetch_value(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run.id,))
        assert n == 0, f"{table} 残留 {n} 行"

    # 再删一次 → 记录不存在
    result2 = mgr.delete_runs([run.id])
    assert result2["deleted"] == []
    assert run.id in result2["failed"]


def test_delete_runs_partial_failure_does_not_abort(mgr):
    r1 = mgr.start_test_run("concurrency", "m1")
    r2 = mgr.start_test_run("concurrency", "m2")
    result = mgr.delete_runs([r1.id, 999999, r2.id])
    assert sorted(result["deleted"]) == sorted([r1.id, r2.id])
    assert 999999 in result["failed"]
    assert mgr.get_run_count() == 0


def test_update_publish_metadata_supports_tags(mgr):
    run = mgr.start_test_run("concurrency", "m")
    mgr.complete_test_run(run, success=True)
    ok = mgr.update_publish_metadata(
        run.id, {"tags": "baseline,weekly", "external_level": "review"}
    )
    assert ok is True
    row = mgr.db.fetch_one("SELECT tags, external_level FROM test_runs WHERE id = ?", (run.id,))
    assert row["tags"] == "baseline,weekly"
    assert row["external_level"] == "review"
