#!/usr/bin/env python3
"""把硬件/环境采集器打包成单文件 zipapp（hw-snapshot.pyz），供裸机零安装使用。

为什么需要：不是每台机器都该部署完整 llm-test（streamlit/pandas/transformers/
datasets/fastapi 等重依赖）。本脚本从仓库 canonical 的 core/ 源码挑出纯 stdlib
的采集模块 + 生成 __main__.py，用 zipapp 打成单个 .pyz——scp 过去 `python
hw-snapshot.pyz` 即可采，无需 pip install。

纳入的模块（全部纯 stdlib，httpx/psutil/pynvml/torch 均懒加载降级）：
  core/__init__.py            （空包标识）
  core/hardware_fingerprint.py（A 维 硬件）
  core/system_info.py         （B 维 系统）
  core/engine_capture.py      （C 维 引擎，可选）
  core/model_spec.py          （D 维 模型架构，可选）
  core/hw_snapshot.py         （快照组装）
+ 生成的 __main__.py          （CLI 入口，镜像 hw_snapshot.py）

用法：
    python tools/make_hw_snapshot_bundle.py            # → dist/hw-snapshot.pyz
    python tools/make_hw_snapshot_bundle.py -o /tmp/x.pyz

从 canonical 源生成，无代码漂移：改 core/ 后重新跑本脚本即可刷新 .pyz。
"""

from __future__ import annotations

import argparse
import sys
import zipapp
from pathlib import Path

# 仓库根（本脚本在 tools/ 下）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 纳入 pyz 的 core 模块（相对仓库根的路径）。全是纯 stdlib + 懒加载可选依赖。
CORE_MODULES = [
    "core/__init__.py",
    "core/hardware_fingerprint.py",
    "core/system_info.py",
    "core/engine_capture.py",
    "core/model_spec.py",
    "core/hw_snapshot.py",
]

# __main__.py 模板：镜像顶层 hw_snapshot.py 的 CLI，但用相对包内 import（pyz 里
# core 仍是顶层包，故 from core.xxx 与仓库内一致，无需改写）。
_MAIN_TEMPLATE = '''\
#!/usr/bin/env python3
"""hw-snapshot.pyz —— 零安装硬件/环境采集器（zipapp，由 tools/make_hw_snapshot_bundle.py 生成）。

裸机用法：
    python hw-snapshot.pyz -o snap.json
    python hw-snapshot.pyz --probe-engine http://127.0.0.1:8000/v1 -o snap.json
    python hw-snapshot.pyz --model-config /data/glm5/config.json -o snap.json
    python hw-snapshot.pyz --owner 张三 --location 机房A -o snap.json
    # 采内存 DIMM 细节（类型/通道/频率/ECC，dmidecode 需 root）：
    python hw-snapshot.pyz --ask-sudo-password -o snap.json        # 交互输入，不回显
    HW_SNAPSHOT_SUDO_PASSWORD=xxx python hw-snapshot.pyz -o snap.json  # 环境变量
    python hw-snapshot.pyz --sudo-password-file /tmp/pw -o snap.json   # 文件（建议 0600）

采集 A 维(硬件) + B 维(系统)，可选 C 维(引擎) + D 维(模型架构)。
psutil/pynvml/torch/httpx 缺失自动降级，不阻塞采集。
sudo 密码只经 stdin 透传给 dmidecode，绝不进 argv/ps、不落快照。
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from core.hw_snapshot import build_snapshot, snapshot_filename, summarize, write_snapshot

SUDO_PASSWORD_ENV = "HW_SNAPSHOT_SUDO_PASSWORD"


def _build_manual(args):
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


def _resolve_sudo_password(args):
    """安全解析 sudo 密码：文件 > 环境变量 > 交互。绝不从命令行值取。"""
    if args.sudo_password_file:
        p = Path(args.sudo_password_file)
        try:
            return p.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError):
            print(f"[warn] 读取 sudo 密码文件失败: {p}，跳过 sudo", file=sys.stderr)
            return None
    env_pw = os.environ.get(SUDO_PASSWORD_ENV)
    if env_pw:
        return env_pw
    if args.ask_sudo_password:
        try:
            return getpass.getpass("sudo 密码（用于 dmidecode 采内存细节，不回显）: ")
        except (EOFError, KeyboardInterrupt):
            print("\\n[info] 未输入 sudo 密码，内存 DIMM 细节将留空（其余不受影响）", file=sys.stderr)
            return None
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="hw-snapshot",
        description="采集硬件/环境/引擎/模型架构信息快照（单文件 JSON，零安装）。",
    )
    parser.add_argument("-o", "--output", help="输出 JSON 路径；省略时自动命名")
    parser.add_argument("--probe-engine", help="可选：在跑的推理引擎端点(http://host:port/v1)")
    parser.add_argument("--model-config", help="可选：模型 config.json 路径")
    sudo_grp = parser.add_argument_group("sudo 密码（采内存 DIMM 细节，可选）")
    sudo_grp.add_argument("--ask-sudo-password", action="store_true",
                          help="交互输入 sudo 密码（getpass 不回显，不进 history）")
    sudo_grp.add_argument("--sudo-password-file", help="从文件读 sudo 密码（读首行，建议 0600）")
    parser.add_argument("--owner", help="机器负责人/测试员")
    parser.add_argument("--location", help="机房/机柜位置")
    parser.add_argument("--product-line", help="产品线")
    parser.add_argument("--psu-w", help="电源功率(W)")
    parser.add_argument("--cooling", help="散热说明")
    parser.add_argument("--ssd-model", help="SSD 型号")
    parser.add_argument("--ssd-capacity-tb", help="SSD 容量(TB)")
    parser.add_argument("--engine-ready", action="store_true", help="标记引擎已就绪")
    parser.add_argument("--remark", help="备注")
    args = parser.parse_args(argv)

    sudo_password = _resolve_sudo_password(args)
    print("正在采集环境信息…", file=sys.stderr)
    snapshot = build_snapshot(
        manual=_build_manual(args),
        engine_url=args.probe_engine,
        model_config_path=args.model_config,
        sudo_password=sudo_password,
    )
    del sudo_password
    out = Path(args.output) if args.output else Path(snapshot_filename(snapshot))
    write_snapshot(out, snapshot)
    print(summarize(snapshot), file=sys.stderr)
    print(f"已写入: {out}", file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def build_pyz(output: Path) -> Path:
    """组装 pyz：把 CORE_MODULES + 生成的 __main__.py 打成单文件。"""
    missing = [m for m in CORE_MODULES if not (REPO_ROOT / m).exists()]
    if missing:
        print(f"[error] 缺少源文件: {missing}", file=sys.stderr)
        sys.exit(1)

    # 构建源映射 {arcname: bytes}，arcname 用相对仓库根的路径（保留 core/ 包结构）
    sources: dict[str, bytes] = {}
    for mod in CORE_MODULES:
        sources[mod] = (REPO_ROOT / mod).read_bytes()
    sources["__main__.py"] = _MAIN_TEMPLATE.encode("utf-8")

    # zipapp.create_archive 接受一个 bytes IO + source 的方式不太直观，
    # 这里先写进临时目录再打包，保证 arcname 干净。
    import tempfile

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for arcname, data in sources.items():
            target = td_path / arcname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        # source 已含 __main__.py，zipapp 会自动用它做入口，不能再传 main=
        zipapp.create_archive(
            source=td_path,
            target=str(output),
            interpreter="/usr/bin/env python3",
            compressed=True,
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_hw_snapshot_bundle",
        description="把硬件采集器打包成单文件 zipapp（hw-snapshot.pyz）。",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dist/hw-snapshot.pyz",
        help="输出 .pyz 路径（默认 dist/hw-snapshot.pyz）",
    )
    args = parser.parse_args(argv)

    out = Path(args.output).resolve()
    build_pyz(out)
    size_kb = out.stat().st_size / 1024
    print(f"已生成: {out}（{size_kb:.1f} KB）", file=sys.stderr)
    print(
        f"裸机用法: scp {out} host: && ssh host 'python {out.name} -o snap.json'", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
