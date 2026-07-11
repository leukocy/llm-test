"""core.warehouse.query 单元测试（纯函数，无 Streamlit）。"""

from __future__ import annotations

from datetime import datetime

from core.models import TestRun
from core.warehouse.query import (
    WarehouseFilter,
    build_cross_matrix,
    build_hardware_inventory_rows,
    build_hm_test_rows,
    project_run,
    query_runs,
)


class _FakeDB:
    """最小数据库替身：只需 get_recent_runs。"""

    def __init__(self, runs):
        self._runs = runs

    def get_recent_runs(self, limit: int = 500):
        return list(self._runs[:limit])


def _run(
    test_id="t1",
    model_id="DeepSeek-V3.1",
    machine_id="abc123",
    tester="alice",
    external_level="internal",
    status_detail="passed",
    engine="vllm",
    avg_tps=55.0,
    created=None,
    supersedes_test_id=None,
    test_type="concurrency",
) -> TestRun:
    return TestRun(
        test_id=test_id,
        model_id=model_id,
        test_type=test_type,
        machine_id=machine_id,
        tester=tester,
        external_level=external_level,
        status_detail=status_detail,
        avg_tps=avg_tps,
        created_at=created or datetime(2026, 6, 1, 12, 0, 0),
        supersedes_test_id=supersedes_test_id,
        serving_config={"engine": engine, "engine_version": "0.6.3", "tp_size": 8},
        model_spec={
            "architecture": "moe",
            "total_params_b": 671,
            "active_params_b": 37,
            "weight_dtype": "bf16",
            "max_position_embeddings": 128000,
        },
        system_info={
            "hardware_fingerprint": {
                "machine_id": machine_id,
                "cpu": {
                    "model_name": "EPYC 9355",
                    "sockets": 1,
                    "cores_per_socket": 32,
                },
                "memory": {
                    "type": "DDR5",
                    "total_gb": 1133,
                    "channels": 24,
                    "ecc": True,
                },
                "gpus": [
                    {
                        "name": "RTX PRO 6000",
                        "vram_gb": 96,
                        "nominal_bandwidth_gbps": 1792,
                    }
                ],
                "cuda": {"driver": "580.159.03", "cuda_version": "13.0"},
            }
        },
        resource_monitor={"peaks": {"gpu_vram_gb": 70.5, "gpu_util_percent": 95.0}},
        effective_bandwidth_gbps=1850.0,
        bottleneck="memory_bandwidth",
    )


# --------------------------------------------------------------------------
# project_run
# --------------------------------------------------------------------------


def test_project_run_populates_all_three_template_fields():
    row = project_run(_run())
    # 模板 #2
    assert row["machine_id"] == "abc123"
    assert row["engine"] == "vllm"
    assert row["parallel_strategy"] == "tp8"
    assert row["model_type"] == "moe"
    assert row["total_params"] == 671.0
    assert row["decode_tps"] == 55.0
    assert row["external_level"] == "internal"
    # 模板 #1（硬件盘点）
    assert row["cpu_model"] == "EPYC 9355"
    assert row["gpu_model"] == "RTX PRO 6000"
    assert row["gpu_bandwidth_gbps"] == 1792
    assert row["cuda_or_rocm"] == "13.0"
    assert row["gpu_count"] == 1
    # 模板 #3（应用）缺测为 None
    assert row["quality_score"] is None
    assert row["citation_score"] is None


def test_project_run_handles_empty_dicts():
    row = project_run(TestRun(test_id="x", model_id="m"))
    assert row["machine_id"] == ""
    assert row["engine"] == ""
    assert row["external_level"] == "internal"  # 默认
    assert row["gpu_model"] == ""


def test_project_run_falls_back_to_avg_tps_for_decode():
    row = project_run(TestRun(test_id="x", model_id="m", avg_tps=42.0))
    assert row["decode_tps"] == 42.0


# --------------------------------------------------------------------------
# WarehouseFilter.matches / query_runs
# --------------------------------------------------------------------------


def test_filter_by_machine_and_external_level():
    db = _FakeDB(
        [
            _run(test_id="t1", machine_id="m1", external_level="internal"),
            _run(test_id="t2", machine_id="m2", external_level="publishable"),
        ]
    )
    flt = WarehouseFilter(machine_id="m2")
    runs = query_runs(db, flt)
    assert [r.test_id for r in runs] == ["t2"]

    flt2 = WarehouseFilter(external_level="publishable")
    assert [r.test_id for r in query_runs(db, flt2)] == ["t2"]


def test_filter_by_engine_fuzzy_and_test_type():
    db = _FakeDB(
        [
            _run(test_id="t1", engine="vllm", test_type="concurrency"),
            _run(test_id="t2", engine="sglang", test_type="prefill"),
        ]
    )
    assert {r.test_id for r in query_runs(db, WarehouseFilter(engine="sglang"))} == {
        "t2"
    }
    assert {
        r.test_id for r in query_runs(db, WarehouseFilter(test_type="prefill"))
    } == {"t2"}


def test_filter_search_across_fields():
    db = _FakeDB(
        [
            _run(test_id="t1", model_id="DeepSeek-V3.1", tester="alice"),
            _run(test_id="t2", model_id="Qwen3", tester="bob"),
        ]
    )
    # search 命中 model_name
    assert {r.test_id for r in query_runs(db, WarehouseFilter(search="deepseek"))} == {
        "t1"
    }
    # search 命中 tester
    assert {r.test_id for r in query_runs(db, WarehouseFilter(search="bob"))} == {"t2"}


def test_supersedes_collapse_keeps_latest_only():
    db = _FakeDB(
        [
            _run(test_id="old", created=datetime(2026, 6, 1)),
            _run(test_id="new", created=datetime(2026, 6, 2), supersedes_test_id="old"),
        ]
    )
    runs = query_runs(db, WarehouseFilter(include_superseded=False))
    assert {r.test_id for r in runs} == {"new"}

    runs_all = query_runs(db, WarehouseFilter(include_superseded=True))
    assert {r.test_id for r in runs_all} == {"old", "new"}


# --------------------------------------------------------------------------
# build_hardware_inventory_rows / build_hm_test_rows
# --------------------------------------------------------------------------


def test_hardware_inventory_dedups_by_machine_id():
    runs = [
        _run(test_id="t1", machine_id="m1"),
        _run(test_id="t2", machine_id="m1", created=datetime(2026, 7, 1)),
        _run(test_id="t3", machine_id="m2"),
    ]
    rows = build_hardware_inventory_rows(runs)
    assert len(rows) == 2  # m1 去重
    assert {r["machine_id"] for r in rows} == {"m1", "m2"}
    # m1 取最新一次（7月）
    m1 = next(r for r in rows if r["machine_id"] == "m1")
    assert "cpu_model" in m1 and m1["cpu_model"] == "EPYC 9355"


def test_hm_test_rows_match_template_field_set():
    from core.warehouse.templates import HM_TEST_FIELDS

    rows = build_hm_test_rows([_run()])
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(HM_TEST_FIELDS)


# --------------------------------------------------------------------------
# build_cross_matrix
# --------------------------------------------------------------------------


def test_cross_matrix_latest_aggregation():
    runs = [
        _run(
            test_id="t1",
            machine_id="m1",
            model_id="A",
            avg_tps=40.0,
            created=datetime(2026, 6, 1),
        ),
        _run(
            test_id="t2",
            machine_id="m1",
            model_id="A",
            avg_tps=60.0,
            created=datetime(2026, 6, 2),
        ),
        _run(
            test_id="t3",
            machine_id="m2",
            model_id="A",
            avg_tps=30.0,
            created=datetime(2026, 6, 3),
        ),
    ]
    mx = build_cross_matrix(runs, metric="decode_tps", agg="latest")
    assert mx.row_labels == ["m1", "m2"]
    assert mx.col_labels == ["A"]
    assert mx.cells["m1"]["A"] == 60.0  # 最新
    assert mx.cells["m2"]["A"] == 30.0


def test_cross_matrix_best_aggregation():
    runs = [
        _run(
            test_id="t1",
            machine_id="m1",
            model_id="A",
            avg_tps=40.0,
            created=datetime(2026, 6, 1),
        ),
        _run(
            test_id="t2",
            machine_id="m1",
            model_id="A",
            avg_tps=70.0,
            created=datetime(2026, 6, 2),
        ),
    ]
    mx = build_cross_matrix(runs, metric="decode_tps", agg="best")
    assert mx.cells["m1"]["A"] == 70.0


def test_cross_matrix_empty_when_no_matching_metric():
    # metric 全缺测（avg_tps=None 且 best 模式）
    runs = [_run(test_id="t1", machine_id="m1", model_id="A", avg_tps=None)]
    mx = build_cross_matrix(runs, metric="decode_tps", agg="best")
    assert mx.cells == {}  # best 模式忽略缺测


def test_query_runs_no_get_recent_runs_returns_empty():
    class _NoMethod:
        pass

    assert query_runs(_NoMethod(), WarehouseFilter()) == []
