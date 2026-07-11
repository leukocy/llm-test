"""
硬件 / 环境 / 引擎 / 模型架构信息快照——单文件采集器。

本模块是「不部署完整 llm-test」的那台机器上唯一需要跑的采集逻辑：把
《环境信息采集指南》里的 A 维(硬件) + B 维(系统) 一次性采全，并可选地
附带 C 维(推理引擎) + D 维(模型架构) 的「快照态」——即只采当前在跑的
引擎配置与模型 config.json，而非 per-test 的持续资源监控(F/G 维仍由
live_bench.py 在测试时采，不属于机器盘点)。

设计原则（与 hardware_fingerprint / engine_capture 一脉相承）：
- 纯函数、零 Streamlit / DB 依赖，可被 CLI、pyz、测试独立调用。
- 优雅降级：每个子采集独立 try/except，缺 GPU / 无 root / 无 docker /
  无 httpx 都降级为空值，绝不抛异常（采集失败比采集不到更糟）。
- 单文件 JSON 快照，schema 版本化，可被中心机的 hw_inventory_import 导入。

pyz 打包(tools/make_hw_snapshot_bundle.py)只纳入本模块 + hardware_fingerprint
+ system_info + engine_capture + model_spec——全是纯 stdlib（httpx/torch/
pynvml 均懒加载），故裸机 `python hw-snapshot.pyz` 零安装即可采。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.engine_capture import capture_engine_config
from core.hardware_fingerprint import capture_hardware_fingerprint
from core.model_spec import from_local_config
from core.system_info import capture_system_info

# 快照 schema 版本：导入端按此校验，不兼容时拒绝而非静默吞字段。
SCHEMA_VERSION = "hw-snapshot/v1"

# 采集器版本（pyproject 版本取不到时回退，便于导入端追溯采集工具新旧）。
try:
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    _PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if _PYPROJECT.exists():
        with _PYPROJECT.open("rb") as _f:
            _DATA = tomllib.load(_f)
            COLLECTOR_VERSION = _DATA.get("project", {}).get("version", "unknown")
    else:
        COLLECTOR_VERSION = "unknown"
except Exception:  # noqa: BLE001  采集兜底，版本号不可断
    COLLECTOR_VERSION = "unknown"

# hwInventory 模板里采集不到、需人工填的字段（导入端写进 run.config / 一等列）。
# 键名与 warehouse.templates.HARDWARE_INVENTORY_FIELDS 的人工列逐字对齐。
MANUAL_FIELDS: tuple[str, ...] = (
    "product_line",
    "owner",
    "location",
    "power_supply_w",
    "cooling_note",
    "engine_ready",
    "ssd_model",
    "ssd_capacity_tb",
    "remark",
)


def build_snapshot(
    manual: dict[str, Any] | None = None,
    engine_url: str | None = None,
    model_config_path: str | Path | None = None,
    sudo_password: str | None = None,
) -> dict[str, Any]:
    """采集一份机器环境快照。

    Args:
        manual: 人工补字段（owner/location/power_supply_w/cooling_note/product_line/
            engine_ready/ssd_model/ssd_capacity_tb/remark）。未知键被忽略。
        engine_url: 可选，在跑的推理引擎 OpenAI 兼容端点（如
            http://127.0.0.1:8000/v1）。给定时采 C 维（docker inspect +
            /v1/models + 引擎适配器）；缺 httpx/docker 时降级，不阻塞。
        model_config_path: 可选，模型 config.json 路径。给定时采 D 维
            （架构/参数量/quant/attention 类型/MTP）。读取失败降级为空。
        sudo_password: 可选 sudo 密码，仅透传给 dmidecode 的 sudo -S 通道（采
            内存 DIMM 细节：类型/通道/频率/ECC）。**密码只经 stdin 透传给子进程，
            绝不写入返回的快照、不进 argv、不落任何持久化**；用完即弃。

    Returns:
        hw-snapshot/v1 字典：{schema, collector_version, collected_at,
        machine_id, hostname, hardware_fingerprint, system_info,
        engine_capture?, model_spec?, manual}。绝不包含 sudo_password。
    """
    manual = {k: v for k, v in (manual or {}).items() if k in MANUAL_FIELDS and v not in (None, "")}

    # A 维 —— 硬件指纹（CPU/内存/GPU/CUDA/拓扑/磁盘 + machine_id）
    fingerprint = capture_hardware_fingerprint(sudo_password)

    # B 维 —— 系统 / 运行时库（Python/OS/git/库版本）
    system_info = capture_system_info(sudo_password)
    # 把富指纹并入 system_info（与 _start_db_run 的持久化结构一致，导入端原样取用）
    system_info["hardware_fingerprint"] = fingerprint
    system_info["machine_id"] = fingerprint.get("machine_id")

    snapshot: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "collected_at": datetime.now().isoformat(),
        "machine_id": fingerprint.get("machine_id"),
        "hostname": fingerprint.get("os", {}).get("hostname") or system_info.get("hostname"),
        "hardware_fingerprint": fingerprint,
        "system_info": system_info,
        "manual": manual,
    }

    # C 维 —— 引擎配置（可选快照态）。capture_engine_config 永不抛异常。
    if engine_url:
        try:
            snapshot["engine_capture"] = capture_engine_config(engine_url)
        except Exception:  # noqa: BLE001  采集兜底
            snapshot["engine_capture"] = {"capture_source": ["error"], "api_base": engine_url}

    # D 维 —— 模型架构（可选快照态）。from_local_config 只读 config.json。
    if model_config_path:
        try:
            spec = from_local_config(model_config_path)
            if spec is not None:
                snapshot["model_spec"] = spec.to_dict()
        except Exception:  # noqa: BLE001  采集兜底
            snapshot["model_spec"] = None

    return snapshot


def snapshot_filename(snapshot: dict[str, Any]) -> str:
    """生成稳定可读的快照文件名：hw-snapshot_<host>_<machine_id>_<ts>.json。"""
    host = (snapshot.get("hostname") or "host").replace(" ", "_") or "host"
    mid = snapshot.get("machine_id") or "noid"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # hostname / machine_id 都已限制字符集，仍兜底替换文件系统不友好字符
    safe_host = "".join(c if c.isalnum() or c in "-_." else "_" for c in host)
    safe_mid = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(mid))
    return f"hw-snapshot_{safe_host}_{safe_mid}_{ts}.json"


def write_snapshot(path: str | Path, snapshot: dict[str, Any]) -> Path:
    """把快照写成 UTF-8 JSON（ensure_ascii=False，缩进 2）。返回写入路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return p


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """读取并校验快照。schema 不匹配时抛 ValueError（导入端应在调用前捕获）。"""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"快照不是 JSON 对象: {path}")
    schema = data.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"快照 schema 不兼容: 期望 {SCHEMA_VERSION}，实际 {schema!r}（{path}）")
    if "hardware_fingerprint" not in data:
        raise ValueError(f"快照缺少 hardware_fingerprint 字段: {path}")
    return data


def fingerprint_hash(snapshot: dict[str, Any]) -> str:
    """快照硬件指纹的稳定哈希（用于 dedupe：同机同指纹视为重复，不重复导入）。

    只纳入硬件本体属性（与 compute_machine_id 同口径），忽略时间戳/可用内存/
    引擎态等易变值——否则机器重启一次就判定为"新机器"会无限增长。
    """
    fp = snapshot.get("hardware_fingerprint") or {}
    gpus = fp.get("gpus") or []
    gpu_sig = sorted(f"{g.get('name')}|{g.get('vram_gb')}" for g in gpus if g.get("name"))
    cpu = fp.get("cpu") or {}
    mem = fp.get("memory") or {}
    disks = fp.get("disks") or []
    disk_sig = sorted(f"{d.get('name')}|{d.get('size_tb')}" for d in disks if d.get("name"))
    stable = {
        "cpu_model": cpu.get("model_name"),
        "sockets": cpu.get("sockets"),
        "cores_per_socket": cpu.get("cores_per_socket"),
        "memory_total_gb": mem.get("total_gb"),
        "gpus": gpu_sig,
        "disks": disk_sig,
    }
    payload = json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def summarize(snapshot: dict[str, Any]) -> str:
    """生成供 CLI 打印的单行人类可读摘要。"""
    fp = snapshot.get("hardware_fingerprint") or {}
    cpu = fp.get("cpu") or {}
    gpus = fp.get("gpus") or []
    cuda = fp.get("cuda") or {}
    mem = fp.get("memory") or {}
    gpu0 = gpus[0] if gpus else {}
    gpu_desc = f"{gpu0.get('name', '?')} ×{len(gpus)}" if gpus else "(无 GPU)"
    vram = gpu0.get("vram_gb")
    vram_desc = f", {vram}GB/卡" if vram else ""
    parts = [
        f"machine_id={snapshot.get('machine_id') or 'noid'}",
        f"host={snapshot.get('hostname') or '?'}",
        f"CPU={cpu.get('model_name', '?')}",
        f"sockets={cpu.get('sockets', '?')}",
        f"cores/socket={cpu.get('cores_per_socket', '?')}",
        f"mem={mem.get('total_gb', '?')}GB",
        f"GPU={gpu_desc}{vram_desc}",
        f"CUDA={cuda.get('cuda_version') or '?'}",
        f"driver={cuda.get('driver') or '?'}",
    ]
    eng = (snapshot.get("engine_capture") or {}).get("engine")
    if eng:
        parts.append(f"engine={eng}")
    mspec = snapshot.get("model_spec") or {}
    if mspec.get("name"):
        parts.append(f"model={mspec['name']}")
    manual = snapshot.get("manual") or {}
    if manual.get("owner"):
        parts.append(f"owner={manual['owner']}")
    return "  ".join(parts)
