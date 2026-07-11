#!/usr/bin/env python3
"""在宿主机上一次性生成 data/hw_memory.json，供容器内硬件指纹采集兜底用。

容器内通常无 dmidecode / 无 DMI 访问权限，采不到内存通道/频率/类型/ECC。
本脚本在**宿主机**执行 `sudo dmidecode -t memory`，提取关键字段写成 JSON，
容器通过挂载的 ./data 目录即可读到（无需容器特权）。

用法（在宿主机 llm-test 目录下）：
    python tools/make_hw_memory_json.py          # 默认 sudo -n（需 NOPASSWD）
    python tools/make_hw_memory_json.py --password '密码'  # sudo -S 传密码

输出：data/hw_memory.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run_dmidecode(password: str | None) -> str | None:
    """运行 dmidecode -t memory，返回 stdout；失败返回 None。"""
    # 先试 sudo -n（NOPASSWD）
    for args, stdin in (
        (["sudo", "-n", "dmidecode", "-t", "memory"], None),
        (["dmidecode", "-t", "memory"], None),  # 已是 root
    ):
        try:
            r = subprocess.run(
                args, capture_output=True, text=True, timeout=10, input=stdin
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout
        except (OSError, subprocess.SubprocessError):
            continue
    # sudo -S + 密码
    if password:
        try:
            r = subprocess.run(
                ["sudo", "-S", "-k", "dmidecode", "-t", "memory"],
                capture_output=True,
                text=True,
                timeout=10,
                input=password + "\n",
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def parse_memory(raw: str) -> dict:
    """从 dmidecode -t memory 输出提取 type/channels/speed/ecc。"""
    info: dict = {"type": None, "channels": None, "speed_mt_s": None, "ecc": None}
    dimms = [b for b in raw.split("\n\n") if "Memory Device" in b]
    populated = [
        d
        for d in dimms
        if "Size:" in d and "No Module" not in d and "No Module Installed" not in d
    ]

    def field(block: str, name: str) -> str | None:
        for line in block.splitlines():
            if name in line:
                return line.split(":", 1)[-1].strip() or None
        return None

    types = {field(d, "Type:") for d in populated} - {None, "Unknown"}
    if types:
        info["type"] = "/".join(sorted(types))
    if populated:
        info["channels"] = len(populated)
    for d in populated:
        speed = field(d, "Speed:")
        if speed and "Unknown" not in speed:
            m = re.search(r"\d+", speed)
            if m:
                info["speed_mt_s"] = int(m.group())
                break
    if "Error Correction" in raw:
        for line in raw.splitlines():
            if "Error Correction" in line:
                info["ecc"] = line.split(":", 1)[-1].strip() or None
                break
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 data/hw_memory.json 供容器采集兜底")
    ap.add_argument("--password", default=None, help="sudo 密码（默认用 sudo -n）")
    ap.add_argument("--out", default="data/hw_memory.json", help="输出路径")
    args = ap.parse_args()

    raw = run_dmidecode(args.password)
    if not raw:
        print(
            "[ERROR] 无法运行 dmidecode（需 sudo 权限或 --password）", file=sys.stderr
        )
        return 1

    info = parse_memory(raw)
    if not any(info.values()):
        print("[ERROR] dmidecode 输出解析不到内存字段", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已写入 {out}: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
