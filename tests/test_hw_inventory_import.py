"""core.services.hw_inventory_import 单元测试。

覆盖：
- snapshot_to_run 字段映射（system_info/serving_config/model_spec/config/machine_id/
  tester/notes/status_detail/external_level）。
- 磁盘兜底：manual 未给 SSD 时从 fingerprint.disks[] 取最大 SSD 填 config。
- manual 给了 SSD 时不被兜底覆盖。
- import_snapshot 写入 DB（fake db）；dedupe 同机同指纹跳过、不同指纹导入。
- 缺 machine_id 拒绝。
- build_hm_test_rows 跳过 hardware_inventory 行（守卫）。
- build_hardware_inventory_rows 按 machine_id 去重取最新。
- project_run 的磁盘兜底投影（disks[] → ssd_model/ssd_capacity_tb）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.hw_inventory_import import (
    HW_INVENTORY_STATUS_DETAIL,
    HW_INVENTORY_TEST_TYPE,
    _largest_ssd,
    import_snapshot,
    import_snapshots,
    snapshot_to_run,
)
from core.models import TestRun
from core.warehouse.query import build_hardware_inventory_rows, build_hm_test_rows, project_run

# ---------- 固定快照 ----------


def _snap(
    machine_id="mid_aaaa1111bbbb2222",
    disks=None,
    manual=None,
    engine=None,
    model_spec=None,
) -> dict[str, Any]:
    fp = {
        "machine_id": machine_id,
        "os": {
            "hostname": "box01",
            "name": "Linux",
            "release": "6.8",
            "machine": "x86_64",
        },
        "cpu": {"model_name": "AMD EPYC 9355", "sockets": 2, "cores_per_socket": 32},
        "memory": {"type": "DDR5", "total_gb": 1133.35},
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA RTX PRO 6000",
                "vram_gb": 95.59,
                "nominal_bandwidth_gbps": 1792.0,
                "pcie_gen": 5,
                "pcie_width": 16,
            }
        ]
        * 8,
        "cuda": {"driver": "580.159.03", "cuda_version": "13.0"},
        "disks": (
            disks
            if disks is not None
            else [
                {
                    "name": "/dev/nvme0",
                    "model": "Samsung 990 Pro",
                    "size_tb": 3.5,
                    "is_ssd": True,
                },
                {
                    "name": "/dev/nvme1",
                    "model": "WD SN850X",
                    "size_tb": 7.68,
                    "is_ssd": True,
                },
                {"name": "/dev/sda", "model": "HDD", "size_tb": 8.0, "is_ssd": False},
            ]
        ),
        "captured_at": "2026-06-24T01:00:00",
    }
    return {
        "schema": "hw-snapshot/v1",
        "collector_version": "1.0.0",
        "collected_at": "2026-06-24T01:00:00",
        "machine_id": machine_id,
        "hostname": "box01",
        "hardware_fingerprint": fp,
        "system_info": {
            "python_version": "3.13.0",
            "hostname": "box01",
            "hardware_fingerprint": fp,
            "machine_id": machine_id,
        },
        "manual": manual or {},
        **({"engine_capture": engine} if engine is not None else {}),
        **({"model_spec": model_spec} if model_spec is not None else {}),
    }


# ---------- _largest_ssd ----------


def test_largest_ssd_picks_biggest_ssd_ignores_hdd():
    disks = [
        {"model": "Small SSD", "size_tb": 1.0, "is_ssd": True},
        {"model": "Big SSD", "size_tb": 7.68, "is_ssd": True},
        {"model": "Big HDD", "size_tb": 16.0, "is_ssd": False},
    ]
    model, size = _largest_ssd(disks)
    assert model == "Big SSD"
    assert size == 7.68


def test_largest_ssd_none_when_no_ssd():
    assert _largest_ssd([{"model": "HDD", "size_tb": 8.0, "is_ssd": False}]) == (
        None,
        None,
    )
    assert _largest_ssd([]) == (None, None)


# ---------- snapshot_to_run 字段映射 ----------


def test_snapshot_to_run_maps_core_fields():
    snap = _snap(
        manual={"owner": "张三", "remark": "测试机", "product_line": "数据中心"}
    )
    run = snapshot_to_run(snap)
    assert isinstance(run, TestRun)
    assert run.test_type == HW_INVENTORY_TEST_TYPE
    assert run.status_detail == HW_INVENTORY_STATUS_DETAIL
    assert run.external_level == "internal"
    assert run.machine_id == "mid_aaaa1111bbbb2222"
    assert run.tester == "张三"
    assert run.notes == "测试机"
    assert run.status == "completed"
    assert run.total_requests == 0
    # 持久化结构：system_info 内嵌 fingerprint
    assert (
        run.system_info["hardware_fingerprint"]["machine_id"] == "mid_aaaa1111bbbb2222"
    )
    assert run.system_info["machine_id"] == "mid_aaaa1111bbbb2222"


def test_snapshot_to_run_engine_and_model_spec_propagate():
    snap = _snap(
        engine={"engine": "vllm"}, model_spec={"name": "glm-5", "architecture": "moe"}
    )
    run = snapshot_to_run(snap)
    assert run.serving_config == {"engine": "vllm"}
    assert run.model_spec == {"name": "glm-5", "architecture": "moe"}


def test_snapshot_to_run_disk_fallback_when_manual_missing():
    """manual 未给 SSD → 从 disks[] 取最大 SSD 填 config。"""
    snap = _snap()  # disks 含 7.68TB WD SN850X（最大 SSD）
    run = snapshot_to_run(snap)
    assert run.config["ssd_model"] == "WD SN850X"
    assert run.config["ssd_capacity_tb"] == 7.68


def test_snapshot_to_run_manual_ssd_not_overridden_by_fallback():
    """manual 给了 SSD → 不被兜底覆盖。"""
    snap = _snap(manual={"ssd_model": "用户指定SSD", "ssd_capacity_tb": 2.0})
    run = snapshot_to_run(snap)
    assert run.config["ssd_model"] == "用户指定SSD"
    assert run.config["ssd_capacity_tb"] == 2.0


def test_snapshot_to_run_manual_ssd_model_only_fills_capacity():
    """manual 只给了 ssd_model（无 capacity）→ capacity 仍从最大 SSD 兜底。"""
    snap = _snap(manual={"ssd_model": "用户指定SSD"})
    run = snapshot_to_run(snap)
    assert run.config["ssd_model"] == "用户指定SSD"
    assert run.config["ssd_capacity_tb"] == 7.68  # 从最大 SSD 兜底


def test_snapshot_to_run_stores_fingerprint_hash_for_dedupe():
    snap = _snap()
    run = snapshot_to_run(snap)
    assert run.config["_fingerprint_hash"]  # dedupe 读这个，免得每次重算
    assert run.config["snapshot_schema"] == "hw-snapshot/v1"


def test_snapshot_to_run_manual_none_values_dropped_from_config():
    snap = _snap(manual={"owner": "x"})  # 其余人工列为空
    run = snapshot_to_run(snap)
    assert "power_supply_w" not in run.config  # None 不进 config
    assert "product_line" not in run.config


# ---------- import_snapshot（fake DB） ----------


class _FakeRunsRepo:
    def __init__(self):
        self.inserted: list[TestRun] = []
        self.completed: list[tuple[int, bool, dict | None]] = []
        self._next_id = 1

    def insert(self, run: TestRun) -> int:
        run.id = self._next_id
        self._next_id += 1
        self.inserted.append(run)
        return run.id

    def complete(self, run_id: int, success: bool = True, stats: dict = None) -> bool:
        self.completed.append((run_id, success, stats))
        return True


class _FakeDB:
    """最小 DB 替身：实现 import_snapshot 用到的接口。"""

    def __init__(self):
        self.runs = _FakeRunsRepo()
        self._recent: list[TestRun] = []

    def complete_test_run(
        self, run, success=True, calculate_stats=True, extra_fields=None
    ):
        self.runs.complete(run.id, success, extra_fields)
        return True

    def get_recent_runs(self, limit=20):
        return list(self._recent)


def test_import_snapshot_inserts_run_and_completes():
    db = _FakeDB()
    r = import_snapshot(db, _snap(manual={"owner": "张三"}))
    assert r["ok"] is True
    assert r["machine_id"] == "mid_aaaa1111bbbb2222"
    assert r["run_id"] == 1
    assert len(db.runs.inserted) == 1
    # complete 写了一等列 machine_id / status_detail
    assert len(db.runs.completed) == 1
    _, _, extra = db.runs.completed[0]
    assert extra["machine_id"] == "mid_aaaa1111bbbb2222"
    assert extra["status_detail"] == HW_INVENTORY_STATUS_DETAIL


def test_import_snapshot_rejects_missing_machine_id():
    db = _FakeDB()
    snap = _snap()
    snap["machine_id"] = None
    snap["hardware_fingerprint"]["machine_id"] = None
    r = import_snapshot(db, snap)
    assert r["ok"] is False
    assert "machine_id" in r["reason"]
    assert len(db.runs.inserted) == 0


def test_import_snapshot_dedupe_skips_same_fingerprint():
    db = _FakeDB()
    # 先导入一次
    import_snapshot(db, _snap())
    # 把刚插入的 run 放进 recent，模拟 DB 已有该机记录
    db._recent = list(db.runs.inserted)
    # 同指纹再导入 → 跳过
    r = import_snapshot(db, _snap(), dedupe=True)
    assert r["ok"] is False
    assert r["skipped"] is True
    assert len(db.runs.inserted) == 1  # 没新增


def test_import_snapshot_dedupe_imports_when_fingerprint_changed():
    db = _FakeDB()
    import_snapshot(db, _snap())
    db._recent = list(db.runs.inserted)
    # GPU 数变了 → 指纹变 → 应导入新行
    snap2 = _snap()
    snap2["hardware_fingerprint"]["gpus"] = snap2["hardware_fingerprint"]["gpus"][:4]
    r = import_snapshot(db, snap2, dedupe=True)
    assert r["ok"] is True
    assert len(db.runs.inserted) == 2


def test_import_snapshot_no_dedupe_always_inserts():
    db = _FakeDB()
    import_snapshot(db, _snap())
    db._recent = list(db.runs.inserted)
    import_snapshot(db, _snap(), dedupe=False)
    assert len(db.runs.inserted) == 2


# ---------- import_snapshots 批量 ----------


def test_import_snapshots_summary_counts(tmp_path):
    from core.hw_snapshot import write_snapshot

    s1 = _snap(machine_id="mid_11110000")
    s2 = _snap(machine_id="mid_22220000")
    # 第三份：坏 schema
    p1 = write_snapshot(tmp_path / "s1.json", s1)
    p2 = write_snapshot(tmp_path / "s2.json", s2)
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "wrong"}', encoding="utf-8")

    db = _FakeDB()
    summary = import_snapshots(db, [p1, p2, bad])
    assert summary["total"] == 3
    assert summary["imported"] == 2
    assert summary["failed"] == 1
    assert summary["skipped"] == 0
    assert len(summary["results"]) == 3


# ---------- query 守卫：hmTest 跳过 hardware_inventory ----------


def _hw_run(machine_id="mid_aaaa1111bbbb2222", created=None) -> TestRun:
    snap = _snap(machine_id=machine_id)
    run = snapshot_to_run(snap)
    run.created_at = created or datetime(2026, 6, 24)
    return run


def _normal_run(machine_id="mid_aaaa1111bbbb2222") -> TestRun:
    run = TestRun.create(test_type="concurrency", model_id="glm-5")
    run.machine_id = machine_id
    run.system_info = {
        "hardware_fingerprint": _snap(machine_id)["hardware_fingerprint"],
        "machine_id": machine_id,
    }
    run.status_detail = "completed"  # 非 hardware_inventory
    run.created_at = datetime(2026, 6, 23)
    return run


def test_build_hm_test_rows_skips_hardware_inventory():
    runs = [_hw_run(), _normal_run()]
    rows = build_hm_test_rows(runs)
    # 只剩 normal run，hardware_inventory 被跳过
    assert len(rows) == 1
    assert rows[0].get("model_name") == "glm-5"


def test_build_hardware_inventory_rows_dedupes_by_machine_id_keeps_latest():
    """同 machine_id 多条 hardware_inventory run → 只留最新一条。"""
    older = _hw_run(created=datetime(2026, 6, 20))
    newer = _hw_run(created=datetime(2026, 6, 24))
    other_machine = _hw_run(machine_id="mid_zzzz0000", created=datetime(2026, 6, 23))
    rows = build_hardware_inventory_rows([older, newer, other_machine])
    mids = {r["machine_id"] for r in rows}
    assert mids == {"mid_aaaa1111bbbb2222", "mid_zzzz0000"}
    # 两条机器各一行
    assert len(rows) == 2


# ---------- project_run 磁盘兜底投影 ----------


def test_project_run_falls_back_to_disks_for_ssd_when_config_empty():
    """config 无 ssd_model → project_run 从 fingerprint.disks[] 取最大 SSD。"""
    run = (
        _hw_run()
    )  # snapshot_to_run 已用兜底填了 config；这里手动清空验证 project_run 兜底
    run.config = {}  # 清掉 config 里的 ssd_*
    row = project_run(run)
    assert row["ssd_model"] == "WD SN850X"  # disks[] 最大 SSD
    assert row["ssd_capacity_tb"] == 7.68


def test_project_run_uses_config_ssd_when_present():
    run = _hw_run()
    run.config = {"ssd_model": "手动SSD", "ssd_capacity_tb": 4.0}
    row = project_run(run)
    assert row["ssd_model"] == "手动SSD"
    assert row["ssd_capacity_tb"] == 4.0


def test_project_run_hw_inventory_row_has_hardware_fields():
    run = _hw_run()
    row = project_run(run)
    assert row["machine_id"] == "mid_aaaa1111bbbb2222"
    assert row["cpu_model"] == "AMD EPYC 9355"
    assert row["gpu_model"] == "NVIDIA RTX PRO 6000"
    assert row["gpu_count"] == 8
    assert row["gpu_vram_gb"] == 95.59
    assert row["cuda_or_rocm"] == "13.0"
    assert row["driver"] == "580.159.03"
