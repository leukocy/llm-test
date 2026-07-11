#!/usr/bin/env python3
"""硬件 / 环境 / 引擎 / 模型架构信息快照采集 CLI。

无需部署完整 llm-test 即可在目标机器上采集环境信息，导出单文件 JSON 快照，
再由中心机的 import_hw_snapshot.py 导入整合到数据仓库。

零安装用法（pyz，见 tools/make_hw_snapshot_bundle.py）：
    scp dist/hw-snapshot.pyz host:
    ssh host 'python hw-snapshot.pyz -o snap.json'
    # 机器上正在跑推理引擎时，顺带采引擎态：
    ssh host 'python hw-snapshot.pyz --probe-engine http://127.0.0.1:8000/v1 -o snap.json'
    # 有模型 config.json 时，顺带采模型架构：
    ssh host 'python hw-snapshot.pyz --model-config /data/glm5/config.json -o snap.json'

仓库内用法（开发机，已 pip install -e .）：
    python hw_snapshot.py -o snap.json --owner 张三 --location 机房A-机柜3

采集范围 = A 维(硬件) + B 维(系统)，可选 C 维(引擎) + D 维(模型架构)。
F/G 维(per-test 资源监控 / 引擎运行时指标)不属机器盘点，仍由 live_bench.py 测试时采。
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from core.hw_snapshot import build_snapshot, snapshot_filename, summarize, write_snapshot

# sudo 密码的环境变量名（避免进 argv / shell history）。设了就自动用，无需交互。
SUDO_PASSWORD_ENV = "HW_SNAPSHOT_SUDO_PASSWORD"


def _build_manual(args: argparse.Namespace) -> dict:
    """从 CLI 参数组装人工补字段（仅非空项）。"""
    mapping = {
        "product_line": args.product_line,
        "owner": args.owner,
        "location": args.location,
        "power_supply_w": args.psu_w,
        "cooling_note": args.cooling,
        "engine_ready": "yes" if args.engine_ready else None,
        "ssd_model": args.ssd_model,
        "ssd_capacity_tb": args.ssd_capacity_tb,
        "remark": args.remark,
    }
    return {k: v for k, v in mapping.items() if v not in (None, "")}


def _resolve_sudo_password(args: argparse.Namespace) -> str | None:
    """安全解析 sudo 密码。优先级：--sudo-password-file > 环境变量 > 交互输入。

    绝不从命令行值取（--sudo-password 不存在）——那会进 ps aux / shell history。
    返回的密码只在内存里用一次（透传给 dmidecode 的 sudo -S stdin），不落快照。
    """
    # 1. 文件（权限 0600 由用户负责；脚本只读首行，去尾换行）
    if args.sudo_password_file:
        p = Path(args.sudo_password_file)
        try:
            return p.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError):
            print(f"[warn] 读取 sudo 密码文件失败: {p}，跳过 sudo", file=sys.stderr)
            return None
    # 2. 环境变量（CI / 脚本场景；进程结束即随 env 消失）
    env_pw = os.environ.get(SUDO_PASSWORD_ENV)
    if env_pw:
        return env_pw
    # 3. 交互输入（getpass 不回显，不进 history）
    if args.ask_sudo_password:
        try:
            return getpass.getpass("sudo 密码（用于 dmidecode 采内存细节，不回显）: ")
        except (EOFError, KeyboardInterrupt):
            print(
                "\n[info] 未输入 sudo 密码，内存 DIMM 细节将留空（其余不受影响）", file=sys.stderr
            )
            return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hw_snapshot",
        description="采集硬件/环境/引擎/模型架构信息快照（单文件 JSON）。",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="输出 JSON 路径；省略时自动命名为 hw-snapshot_<host>_<mid>_<ts>.json",
    )
    parser.add_argument(
        "--probe-engine",
        help="可选：在跑的推理引擎 OpenAI 兼容端点（如 http://127.0.0.1:8000/v1），"
        "采 C 维引擎配置（docker inspect + /v1/models + 引擎适配器）",
    )
    parser.add_argument(
        "--model-config",
        help="可选：模型 config.json 路径，采 D 维模型架构（架构/参数量/quant/MTP）",
    )
    # sudo 密码（采内存 DIMM 细节：类型/通道/频率/ECC，仅 dmidecode 需要 root）
    # 注意：无 --sudo-password 值参数——那会进 ps aux / shell history，不安全。
    sudo_grp = parser.add_argument_group("sudo 密码（采内存 DIMM 细节，可选）")
    sudo_grp.add_argument(
        "--ask-sudo-password",
        action="store_true",
        help="交互输入 sudo 密码（getpass 不回显，不进 history）。最安全。",
    )
    sudo_grp.add_argument(
        "--sudo-password-file",
        help="从文件读 sudo 密码（读首行）。建议文件权限 0600，用完删除。",
    )
    parser.add_argument("--owner", help="机器负责人/测试员")
    parser.add_argument("--location", help="机房/机柜位置")
    parser.add_argument("--product-line", help="产品线（如 端侧/数据中心）")
    parser.add_argument("--psu-w", help="电源功率(W)，如 2000")
    parser.add_argument("--cooling", help="散热说明（如 风冷/液冷）")
    parser.add_argument("--ssd-model", help="SSD 型号（未给则自动从 lsblk 取最大 SSD）")
    parser.add_argument("--ssd-capacity-tb", help="SSD 容量(TB)")
    parser.add_argument("--engine-ready", action="store_true", help="标记引擎已就绪可测")
    parser.add_argument("--remark", help="备注")
    args = parser.parse_args(argv)

    manual = _build_manual(args)
    sudo_password = _resolve_sudo_password(args)

    print("正在采集环境信息…", file=sys.stderr)
    snapshot = build_snapshot(
        manual=manual,
        engine_url=args.probe_engine,
        model_config_path=args.model_config,
        sudo_password=sudo_password,
    )
    # 密码用完即弃——del 后即便快照意外引用也只拿到 None
    del sudo_password

    out = Path(args.output) if args.output else Path(snapshot_filename(snapshot))
    write_snapshot(out, snapshot)

    print(summarize(snapshot), file=sys.stderr)
    print(f"已写入: {out}", file=sys.stderr)
    # stdout 仅打印快照路径，便于脚本管道取用
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
