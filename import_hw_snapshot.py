#!/usr/bin/env python3
"""硬件盘点快照导入 CLI（中心机用）。

把各台机器用 hw_snapshot.py(或零安装 hw-snapshot.pyz)采到的 JSON 快照导入
数据仓库，汇入「硬件盘点」口径；可选直接导出整合后的硬件盘点 CSV。

用法：
    # 导入单份
    python import_hw_snapshot.py snap.json --db data/benchmark.db

    # 批量导入（支持 glob），同机同指纹去重
    python import_hw_snapshot.py snapshots/*.json --dedupe --db data/benchmark.db

    # 导入并导出整合后的硬件盘点 CSV（复用 warehouse 导出）
    python import_hw_snapshot.py snapshots/*.json --dedupe --export-hwInventory inv.csv

导入后这些行 status_detail='hardware_inventory'，仓库的 build_hm_test_rows 会跳过
（硬件盘点不是「硬件×模型测试」），但 build_hardware_inventory_rows 会按 machine_id
收纳去重，导出硬件盘点表。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.hw_inventory_import import import_snapshots
from core.warehouse.export import export_template_csv
from core.warehouse.query import WarehouseFilter, build_hardware_inventory_rows, query_runs

# db_manager 延迟到 main() 里按 --db 路径实例化（单例首次创建即锁定路径），
# 故不在模块顶层 import，避免用默认 data/benchmark.db 抢先初始化。


def _expand_paths(args_paths: list[str]) -> list[str]:
    """展开 CLI 路径参数（支持 shell 未展开的 *.json glob）。"""
    paths: list[str] = []
    for p in args_paths:
        pp = Path(p)
        # 含通配符且 shell 未展开 → 手动 glob
        if any(ch in p for ch in "*?[") and not pp.exists():
            for hit in sorted(Path(".").glob(p)):
                if hit.is_file():
                    paths.append(str(hit))
        elif pp.is_file():
            paths.append(p)
        elif pp.is_dir():
            paths.extend(str(h) for h in sorted(pp.rglob("*.json")))
        else:
            print(f"[warn] 路径不存在，跳过: {p}", file=sys.stderr)
    return paths


def _export_hw_inventory(db, out_path: str, limit: int = 5000) -> int:
    """导出整合后的硬件盘点 CSV（每个 machine_id 一行）。返回行数。"""
    runs = query_runs(db, WarehouseFilter(limit=limit, include_superseded=True))
    rows = build_hardware_inventory_rows(runs)
    csv_text = export_template_csv("hwInventory", rows)
    Path(out_path).write_text(csv_text, encoding="utf-8")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="import_hw_snapshot",
        description="导入硬件盘点快照 JSON 到数据仓库（中心机用）。",
    )
    parser.add_argument("paths", nargs="+", help="快照 JSON 文件/目录/glob（如 snapshots/*.json）")
    parser.add_argument("--db", default="data/benchmark.db", help="SQLite DB 路径")
    parser.add_argument("--dedupe", action="store_true", help="同 machine_id 且指纹未变则跳过")
    parser.add_argument(
        "--export-hwInventory",
        help="导入后导出整合后的硬件盘点 CSV 到此路径（复用 warehouse 模板）",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=5000,
        help="导出时查询的最近 run 上限（默认 5000）",
    )
    args = parser.parse_args(argv)

    # 获取 DB 管理器单例。--db 非默认路径时必须先实例化以锁定路径
    # （单例首次创建即锁定，之后再实例化也改不了路径）。
    from core.database.manager import DatabaseManager

    DatabaseManager(args.db)
    from core.database.manager import db_manager

    paths = _expand_paths(args.paths)
    if not paths:
        print("[error] 没有找到可导入的快照 JSON", file=sys.stderr)
        return 2

    print(f"准备导入 {len(paths)} 份快照（dedupe={args.dedupe}）…", file=sys.stderr)
    summary = import_snapshots(db_manager, paths, dedupe=args.dedupe)

    print(
        f"完成：导入 {summary['imported']}，跳过 {summary['skipped']}，"
        f"失败 {summary['failed']}（共 {summary['total']}）",
        file=sys.stderr,
    )
    for r in summary["results"]:
        if r.get("ok"):
            print(
                f"  [OK]   {r.get('file')}  machine_id={r.get('machine_id')}  run_id={r.get('run_id')}",
                file=sys.stderr,
            )
        elif r.get("skipped"):
            print(f"  [SKIP] {r.get('file')}  {r.get('reason')}", file=sys.stderr)
        else:
            print(f"  [FAIL] {r.get('file')}  {r.get('reason')}", file=sys.stderr)

    if args.export_hwInventory:
        n = _export_hw_inventory(db_manager, args.export_hwInventory, limit=args.query_limit)
        print(
            f"已导出硬件盘点: {args.export_hwInventory}（{n} 台机器）",
            file=sys.stderr,
        )

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
