"""硬件盘点快照导入服务——把 hw_snapshot 产出的 JSON 快照导入数据仓库。

设计动机：不是每台机器都部署完整 llm-test。采集端用 hw_snapshot.py(或零安装的
hw-snapshot.pyz)产出单文件快照，本模块在中心机把快照转成一条 TestRun 写入 DB，
从而汇入现有数据仓库的「硬件盘点」口径(warehouse.templates.HARDWARE_INVENTORY_FIELDS
+ query.build_hardware_inventory_rows)，无需新表、零 schema 迁移。

快照转 TestRun 的字段映射（与 live_bench / historical_import 的持久化结构一致）：
- system_info_json ← 快照的 hardware_fingerprint + system_info（A/B 维）
- serving_config_json ← 快照的 engine_capture（C 维，可选）
- model_spec_json ← 快照的 model_spec（D 维，可选）
- machine_id ← 快照 machine_id（一等列，hwInventory 去重键）
- tester ← manual.owner；notes ← manual.remark
- config_json ← manual 的人工列（product_line/location/电源/散热/engine_ready/
  ssd_model/ssd_capacity_tb），供 project_run() 投影到 hwInventory 模板
- status_detail='hardware_inventory'（导入端标识；query.build_hm_test_rows 据此跳过）
- external_level='internal'（机器盘点不直接对外）

磁盘兜底：采集器采了 disks[](lsblk)，但 project_run() 的 ssd_model/ssd_capacity_tb
原本只读 config。导入时若 manual 未给 SSD，从 disks[] 取最大 SSD 填进 config，
让磁盘数据能在 hwInventory 导出里浮现。

dedupe：同 machine_id 且硬件指纹哈希一致 → 跳过，避免机器重启一次就重复导入。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.hw_snapshot import fingerprint_hash, load_snapshot
from core.models import TestRun, TestRunStatus

logger = logging.getLogger(__name__)

# 快照导入专用的 test_type / status_detail 标识
HW_INVENTORY_TEST_TYPE = "hardware_inventory"
HW_INVENTORY_STATUS_DETAIL = "hardware_inventory"


def _largest_ssd(disks: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    """从 hardware_fingerprint.disks[] 取最大 SSD 的 (model, size_tb)。无 SSD 返回 (None, None)。"""
    ssds = [d for d in disks if d.get("is_ssd") and d.get("size_tb")]
    if not ssds:
        return None, None
    biggest = max(ssds, key=lambda d: d.get("size_tb") or 0)
    return biggest.get("model"), biggest.get("size_tb")


def snapshot_to_run(snapshot: dict[str, Any]) -> TestRun:
    """把一份快照转成一条 TestRun（纯函数，不碰 DB）。

    不设 id / created_at 由 TestRun.create 填默认；status 设为 completed。
    """
    fp: dict[str, Any] = snapshot.get("hardware_fingerprint") or {}
    sys_info: dict[str, Any] = snapshot.get("system_info") or {}
    manual: dict[str, Any] = snapshot.get("manual") or {}
    engine: dict[str, Any] = snapshot.get("engine_capture") or {}
    model_spec: dict[str, Any] = snapshot.get("model_spec") or {}

    machine_id = snapshot.get("machine_id") or fp.get("machine_id")

    # 持久化结构：system_info 内嵌 hardware_fingerprint（与 _start_db_run 一致）
    persisted_sys_info = dict(sys_info)
    persisted_sys_info["hardware_fingerprint"] = fp
    persisted_sys_info["machine_id"] = machine_id

    # config 人工列：直接进 run.config，供 project_run() 投影到 hwInventory 模板
    config: dict[str, Any] = {
        "product_line": manual.get("product_line"),
        "location": manual.get("location"),
        "power_supply_w": manual.get("power_supply_w"),
        "cooling_note": manual.get("cooling_note"),
        "engine_ready": manual.get("engine_ready"),
    }
    # 磁盘兜底：manual 未给的字段从采集到的 disks[] 取最大 SSD 补齐。
    # model 与 capacity 独立兜底——manual 只给型号时 capacity 仍应从最大 SSD 取。
    ssd_model = manual.get("ssd_model")
    ssd_cap = manual.get("ssd_capacity_tb")
    d_model, d_cap = _largest_ssd(fp.get("disks") or [])
    if not ssd_model:
        ssd_model = d_model
    if ssd_cap is None and d_cap is not None:
        ssd_cap = d_cap
    config["ssd_model"] = ssd_model
    config["ssd_capacity_tb"] = ssd_cap
    # 留一份采集来源指纹哈希，dedupe 时直接读，免得每次重算
    config["_fingerprint_hash"] = fingerprint_hash(snapshot)
    config["snapshot_schema"] = snapshot.get("schema")
    config["collector_version"] = snapshot.get("collector_version")

    run = TestRun.create(
        test_type=HW_INVENTORY_TEST_TYPE,
        model_id=manual.get("product_line") or "hardware_inventory",
    )
    run.system_info = persisted_sys_info
    run.serving_config = engine  # C 维引擎配置（可能为空）
    run.model_spec = model_spec  # D 维模型架构（可能为空）
    run.config = {k: v for k, v in config.items() if v is not None}
    run.machine_id = machine_id
    run.tester = manual.get("owner")
    run.notes = manual.get("remark") or "硬件盘点快照导入"
    run.status_detail = HW_INVENTORY_STATUS_DETAIL
    run.external_level = "internal"
    run.total_requests = 0
    run.completed_requests = 0
    run.status = TestRunStatus.COMPLETED.value
    return run


def import_snapshot(
    db,
    snapshot: dict[str, Any],
    dedupe: bool = False,
) -> dict[str, Any]:
    """导入一份快照到 DB。返回统计 {ok, run_id, machine_id, skipped, reason}。

    Args:
        db: 数据库管理器（需 runs.insert + complete_test_run + get_recent_runs）。
        snapshot: hw_snapshot.build_snapshot() 产出的字典。
        dedupe: True 时同 machine_id 且指纹哈希一致 → 跳过不导入。
    """
    machine_id = snapshot.get("machine_id")
    if not machine_id:
        return {"ok": False, "skipped": False, "reason": "快照缺 machine_id（无法归并）"}

    new_hash = fingerprint_hash(snapshot)

    if dedupe and hasattr(db, "get_recent_runs"):
        try:
            for r in db.get_recent_runs(limit=2000):
                if r.test_type != HW_INVENTORY_TEST_TYPE:
                    continue
                if r.machine_id != machine_id:
                    continue
                prev_hash = (r.config or {}).get("_fingerprint_hash")
                if prev_hash and prev_hash == new_hash:
                    return {
                        "ok": False,
                        "skipped": True,
                        "reason": "同 machine_id 指纹未变，跳过",
                        "machine_id": machine_id,
                        "existing_run_id": r.id,
                    }
        except Exception as e:  # noqa: BLE001  dedupe 失败不阻塞导入
            logger.warning(f"dedupe 查询失败，按非 dedupe 导入: {e}")

    run = snapshot_to_run(snapshot)
    try:
        run_id = db.runs.insert(run)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "skipped": False,
            "reason": f"写入 DB 失败: {e}",
            "machine_id": machine_id,
        }
    run.id = run_id

    # 写 1.2.0 一等列（machine_id / status_detail），calculate_stats=False（无 per-request 结果）
    extra = {"machine_id": machine_id, "status_detail": HW_INVENTORY_STATUS_DETAIL}
    try:
        if hasattr(db, "complete_test_run"):
            db.complete_test_run(run, success=True, calculate_stats=False, extra_fields=extra)
        elif hasattr(db, "runs") and hasattr(db.runs, "complete"):
            db.runs.complete(run_id, True, extra)
    except Exception as e:  # noqa: BLE001  complete 失败不抹掉已插入的 run
        logger.warning(f"complete_test_run 失败 run_id={run_id}: {e}")

    return {
        "ok": True,
        "skipped": False,
        "run_id": run_id,
        "machine_id": machine_id,
        "fingerprint_hash": new_hash,
    }


def import_snapshot_file(db, path: str | Path, dedupe: bool = False) -> dict[str, Any]:
    """读取一份快照 JSON 文件并导入。schema 不兼容时记为失败，不抛异常。"""
    p = Path(path)
    try:
        snapshot = load_snapshot(p)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return {"ok": False, "skipped": False, "reason": f"读取快照失败 {p}: {e}"}
    result = import_snapshot(db, snapshot, dedupe=dedupe)
    result["file"] = str(p)
    return result


def import_snapshots(db, paths: list[str | Path], dedupe: bool = False) -> dict[str, Any]:
    """批量导入多份快照。返回汇总 {total, imported, skipped, failed, results}。"""
    summary: dict[str, Any] = {
        "total": len(paths),
        "imported": 0,
        "skipped": 0,
        "failed": 0,
        "results": [],
    }
    for path in paths:
        r = import_snapshot_file(db, path, dedupe=dedupe)
        summary["results"].append(r)
        if r.get("ok"):
            summary["imported"] += 1
        elif r.get("skipped"):
            summary["skipped"] += 1
        else:
            summary["failed"] += 1
    return summary
