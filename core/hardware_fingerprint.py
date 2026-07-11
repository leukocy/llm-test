"""
结构化硬件指纹捕获模块

把原来 system_info 里“GPU 只是一个名字字符串”的采集，升级为结构化的硬件指纹：
CPU 拓扑(socket / 核 / 线程 / NUMA)、内存(类型 / 容量 / 通道 / 频率 / ECC)、
GPU(数量 / 显存 / 显存类型 / 标称带宽 / PCIe Gen×宽)、CUDA / 驱动版本，以及一个
跨重启稳定、可分组的 machine_id 哈希。

设计原则（来自《端侧 AI 硬件与模型》手册 #hardware / #redlines）：
- 配置冻结：任何对外结论都必须绑定硬件指纹，缺字段不可对外。
- 优雅降级：每个字段独立 try/except，缺 GPU / 无 dmidecode 权限时降级为空值，
  永不让整次采集抛异常（采集失败比采集不到更糟）。
- 纯函数、零 Streamlit / DB 依赖：可被 runner、UI、测试独立调用。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# GPU 标称显存带宽查表（峰值理论值，单位 GB/s）
# 对 LLM decode 而言，真正的瓶颈是显存带宽(HBM/GDDR)而非 PCIe，因此命中查表时
# 查表值优先于 PCIe 公式。键为 GPU 名称子串，匹配时按长度降序避免短键误命中。
# 数值为公开规格的典型峰值带宽。
# ---------------------------------------------------------------------------
_GPU_BANDWIDTH_GBPS: list[tuple[str, float]] = [
    # AMD Instinct (HBM)
    ("mi300x", 5300.0),
    ("mi300a", 5300.0),
    ("mi325x", 5300.0),
    ("mi250x", 3277.0),
    ("mi250", 3277.0),
    ("mi210", 1633.0),
    ("mi300", 5300.0),
    # NVIDIA Hopper / Blackwell (HBM)
    ("b200", 8000.0),
    ("b100", 8000.0),
    ("gb200", 8000.0),
    ("h200", 4800.0),
    ("h100", 3350.0),
    ("gh200", 3350.0),
    # NVIDIA Ampere (HBM2e / GDDR6)
    ("a100", 2039.0),  # 80GB；40GB 约 1555
    ("a30", 933.0),
    ("a10", 600.0),
    ("a10g", 600.0),
    # NVIDIA Ada / Hopper 消费与专业卡 (GDDR6X / GDDR7)
    ("l40s", 864.0),
    ("l40", 864.0),
    ("l4", 192.0),
    ("rtx pro 6000 blackwell", 1792.0),  # RTX PRO 6000 Blackwell (GDDR7, 96GB)
    ("rtx 5090", 1792.0),
    ("rtx 5080", 960.0),
    ("rtx 5070", 512.0),
    ("rtx 4090", 1008.0),
    ("rtx 4080", 717.0),
    ("rtx 4070", 504.0),
    ("rtx pro 6000", 1792.0),  # 默认按 Blackwell 版计
    ("pro 6000", 960.0),  # RTX 6000 Ada
    ("rtx 6000 ada", 960.0),
    ("rtx 3090", 936.0),
    ("rtx 3080", 760.0),
    # 其余兜底
    ("tesla t4", 320.0),
    ("tesla v100", 900.0),
    ("tesla p100", 720.0),
]


def _run_cmd(
    args: list[str], timeout: float = 8.0, env: dict[str, str] | None = None
) -> str | None:
    """运行外部命令，返回 stdout 文本；失败/超时返回 None（绝不抛异常）。
    env: 追加/覆盖的环境变量（与 os.environ 合并），如 {"LC_ALL": "C"} 强制英文 locale。
    """
    try:
        full_env = None
        if env:
            full_env = dict(os.environ)
            full_env.update(env)
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=full_env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _lookup_gpu_bandwidth(gpu_name: str, vram_gb: float | None) -> float | None:
    """按 GPU 名称子串查标称显存带宽；查不到返回 None（交由 PCIe 公式兜底）。"""
    name = (gpu_name or "").lower()
    # 按子串长度降序匹配，避免 "A100" 命中在 "A100 80GB" 之前之类的问题
    for key, bw in sorted(_GPU_BANDWIDTH_GBPS, key=lambda kv: len(kv[0]), reverse=True):
        if key in name:
            return bw
    return None


def _pcie_bandwidth_gbps(pcie_gen: int | None, pcie_width: int | None) -> float | None:
    """由 PCIe Gen × 宽度估算单向峰值带宽(GB/s)。
    PCIe 每通道每方向：Gen4≈1.97 GB/s、Gen5≈3.94 GB/s、Gen3≈0.985 GB/s。
    """
    if not pcie_gen or not pcie_width:
        return None
    per_lane = {1: 0.250, 2: 0.500, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.56}.get(pcie_gen)
    if per_lane is None:
        return None
    return round(per_lane * pcie_width, 1)


def _query_gpus() -> list[dict[str, Any]]:
    """通过 nvidia-smi 一次性查询所有 GPU（含 PCIe / 显存类型），torch 兜底。
    每张卡返回 {index, name, vram_gb, memory_type, nominal_bandwidth_gbps,
               pcie_gen, pcie_width}。
    """
    gpus: list[dict[str, Any]] = []
    out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ]
    )
    if out:
        for line in out.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                index = int(parts[0])
            except (ValueError, IndexError):
                index = len(gpus)
            name = parts[1]
            vram_mb = _to_float(parts[2])
            # nvidia-smi 无有效字段报显存类型（memory.type 无效）→ 留空
            memory_type = None
            pcie_gen = _to_int(parts[3]) if len(parts) > 3 else None
            pcie_width = _to_int(parts[4]) if len(parts) > 4 else None
            vram_gb = round(vram_mb / 1024.0, 2) if vram_mb is not None else None
            # 标称带宽优先查显存(HBM/GDDR)峰值表；查不到再退化为 PCIe 带宽
            nominal = _lookup_gpu_bandwidth(name, vram_gb)
            if nominal is None:
                nominal = _pcie_bandwidth_gbps(pcie_gen, pcie_width)
            gpus.append(
                {
                    "index": index,
                    "name": name,
                    "vram_gb": vram_gb,
                    "memory_type": memory_type,
                    "nominal_bandwidth_gbps": nominal,
                    "pcie_gen": pcie_gen,
                    "pcie_width": pcie_width,
                }
            )

    if not gpus:
        # torch.cuda 兜底（至少拿到名字与显存）
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    vram_gb = round(props.total_memory / 1024**3, 2)
                    name = props.name
                    gpus.append(
                        {
                            "index": i,
                            "name": name,
                            "vram_gb": vram_gb,
                            "memory_type": None,
                            "nominal_bandwidth_gbps": _lookup_gpu_bandwidth(
                                name, vram_gb
                            ),
                            "pcie_gen": None,
                            "pcie_width": None,
                        }
                    )
        except Exception:  # noqa: BLE001  采集兜底，任何失败都不应中断
            pass

    return gpus


def _query_cuda_versions() -> dict[str, str | None]:
    """查询驱动版本与 CUDA 版本：NVML 优先，nvidia-smi 兜底。"""
    result: dict[str, str | None] = {"driver": None, "cuda_version": None}

    # NVML（nvidia-ml-py / pynvml 接口一致）
    try:
        try:
            import pynvml
        except ImportError:
            from nvidia_ml_py import pynvml  # noqa: F401
        pynvml.nvmlInit()
        try:
            result["driver"] = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(result["driver"], bytes):
                result["driver"] = result["driver"].decode("utf-8", "ignore")
            cv = pynvml.nvmlSystemGetCudaDriverVersion()
            # 返回形如 12000 表示 12.0
            if isinstance(cv, int):
                result["cuda_version"] = f"{cv // 1000}.{(cv % 1000) // 10}"
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
        if result["driver"]:
            return result
    except Exception:  # noqa: BLE001
        pass

    # nvidia-smi 兜底
    out = _run_cmd(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if out:
        result["driver"] = out.strip().splitlines()[0].strip() or None
    out = _run_cmd(["nvidia-smi"])
    if out:
        for line in out.splitlines():
            low = line.lower()
            if "cuda version" in low:
                # 形如 "| CUDA Version: 12.4   |"
                after = line.split(":", 1)[-1].strip()
                # 去掉表格竖线
                after = after.split("|")[0].strip()
                if after:
                    result["cuda_version"] = after
                    break
    return result


def _query_cpu_topology() -> dict[str, Any]:
    """CPU 拓扑：型号 / socket 数 / 每 socket 核数 / 每核线程数 / NUMA 节点数。"""
    info: dict[str, Any] = {
        "model_name": None,
        "sockets": None,
        "cores_per_socket": None,
        "threads_per_core": None,
        "numa_nodes": None,
    }

    # lscpu(Linux 主路径,强制 C locale 避免 zh_CN 下键名"套接字/核"导致解析 miss)
    out = _run_cmd(["lscpu"], env={"LC_ALL": "C", "LANG": "C"})
    parsed: dict[str, str] = {}
    if out:
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                parsed[k.strip()] = v.strip()
    if parsed:
        info["model_name"] = parsed.get("Model name") or None
        info["sockets"] = _to_int(parsed.get("Socket(s)"))
        info["cores_per_socket"] = _to_int(parsed.get("Core(s) per socket"))
        info["threads_per_core"] = _to_int(parsed.get("Thread(s) per core"))
        info["numa_nodes"] = _to_int(parsed.get("NUMA node(s)"))

    # /proc/cpuinfo 兜底交叉验证 socket 数(locale 无关,唯一 physical id 数 = socket 数)
    proc_sockets = _proc_socket_count()
    if proc_sockets:
        info["sockets"] = proc_sockets

    # psutil / platform 兜底(lscpu + /proc 都没给时,psutil 至少给核数/线程数,socket 默认 1)
    if info["model_name"] is None:
        info["model_name"] = _platform_cpu_model()
    if info["cores_per_socket"] is None or info["sockets"] is None:
        phys = _psutil_cpu_count(logical=False)
        logical = _psutil_cpu_count(logical=True)
        if info["sockets"] is None:
            info["sockets"] = 1
        if phys is not None:
            info["cores_per_socket"] = (
                phys // info["sockets"] if info["sockets"] else phys
            )
        if phys and logical:
            info["threads_per_core"] = max(1, logical // phys)

    return info


def _proc_socket_count() -> int | None:
    """从 /proc/cpuinfo 数唯一 physical id(= socket 数)。locale 无关。失败返回 None。"""
    try:
        pids: set[str] = set()
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("physical id"):
                    pids.add(line.split(":", 1)[-1].strip())
        return len(pids) or None
    except OSError:
        return None


def _platform_cpu_model() -> str | None:
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[-1].strip()
        except OSError:
            pass
    return None


def _psutil_cpu_count(logical: bool) -> int | None:
    try:
        import psutil

        count = psutil.cpu_count(logical=logical)
        return int(count) if count is not None else None
    except Exception:  # noqa: BLE001
        return None


def _query_memory_details() -> dict[str, Any]:
    """内存细节：类型 / 总量 / 通道数 / 频率 / ECC。dmidecode 需 root，失败退回 psutil。"""
    info: dict[str, Any] = {
        "type": None,
        "total_gb": None,
        "available_gb": None,
        "channels": None,
        "speed_mt_s": None,
        "ecc": None,
    }

    # psutil：总量 / 可用（总能拿到）
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["total_gb"] = round(mem.total / 1024**3, 2)
        info["available_gb"] = round(mem.available / 1024**3, 2)
    except Exception:  # noqa: BLE001
        pass

    # dmidecode：DIMM 级细节（需 root / 可执行）
    if shutil.which("dmidecode"):
        out = _run_cmd(["dmidecode", "-t", "memory"], timeout=10)
        if not out:
            # dmidecode 需 root；尝试 sudo -n（需 sudoers 配 NOPASSWD: dmidecode）
            out = _run_cmd(["sudo", "-n", "dmidecode", "-t", "memory"], timeout=10)
        if out:
            # dmidecode 每条记录：第一行 "Handle 0x.. type 17"，第二行才是 "Memory Device"，
            # 故整块匹配（不能只看第一行）。
            dimms = [b for b in out.split("\n\n") if "Memory Device" in b]
            populated = [
                d
                for d in dimms
                if "Size:" in d
                and "No Module" not in d
                and "No Module Installed" not in d
            ]
            types = {
                t
                for d in populated
                if (t := _dmidecode_field(d, "Type:")) and t != "Unknown"
            }
            speeds = {_dmidecode_field(d, "Speed:") for d in populated} - {None}
            if types:
                info["type"] = "/".join(sorted(types))
            if populated:
                info["channels"] = len(populated)
            # 取第一个有效速率（去掉 "Configured" 前缀干扰）
            for d in populated:
                speed = _dmidecode_field(d, "Speed:")
                if speed and "Unknown" not in speed:
                    mt = _extract_first_int(speed)
                    if mt:
                        info["speed_mt_s"] = mt
                        break
            if "Error Correction" in out:
                ecc_line = next(
                    (l for l in out.splitlines() if "Error Correction" in l), ""
                )
                info["ecc"] = ecc_line.split(":", 1)[-1].strip() or None

    return info


def _dmidecode_field(block: str, field: str) -> str | None:
    for line in block.splitlines():
        if field in line:
            return line.split(":", 1)[-1].strip() or None
    return None


def _extract_first_int(text: str) -> int | None:
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else None


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def compute_machine_id(fingerprint: dict[str, Any]) -> str:
    """由稳定字段计算跨重启稳定的 machine_id（sha1 前 16 位）。
    只纳入硬件本体属性，忽略可用内存、pstate、时间戳等易变值。
    """
    gpus = fingerprint.get("gpus") or []
    gpu_sig = sorted(
        f"{g.get('name')}|{g.get('vram_gb')}" for g in gpus if g.get("name")
    )
    cpu = fingerprint.get("cpu") or {}
    mem = fingerprint.get("memory") or {}
    stable = {
        "cpu_model": cpu.get("model_name"),
        "sockets": cpu.get("sockets"),
        "cores_per_socket": cpu.get("cores_per_socket"),
        "memory_total_gb": mem.get("total_gb"),
        "gpus": gpu_sig,
    }
    payload = json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def capture_hardware_fingerprint() -> dict[str, Any]:
    """采集结构化硬件指纹。任何子项失败都降级为空值，绝不抛异常。"""
    fingerprint: dict[str, Any] = {}

    try:
        fingerprint["cpu"] = _query_cpu_topology()
    except Exception:  # noqa: BLE001
        fingerprint["cpu"] = {}

    try:
        fingerprint["memory"] = _query_memory_details()
    except Exception:  # noqa: BLE001
        fingerprint["memory"] = {}

    try:
        fingerprint["gpus"] = _query_gpus()
    except Exception:  # noqa: BLE001
        fingerprint["gpus"] = []

    try:
        fingerprint["cuda"] = _query_cuda_versions()
    except Exception:  # noqa: BLE001
        fingerprint["cuda"] = {"driver": None, "cuda_version": None}

    fingerprint["os"] = {
        "name": platform.system() or None,
        "release": platform.release() or None,
        "machine": platform.machine() or None,
        "hostname": (platform.node()[:50] or None),
    }

    # 可用处理器数（廉价补充）
    fingerprint["cpu"]["logical_cores"] = os.cpu_count()

    fingerprint["captured_at"] = datetime.now().isoformat()
    fingerprint["machine_id"] = compute_machine_id(fingerprint)
    return fingerprint
