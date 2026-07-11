"""
系统信息自动捕获模块

自动捕获运行环境信息，确保Test可复现性。
"""

import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def capture_system_info(sudo_password: str | None = None) -> dict[str, Any]:
    """
    自动捕获系统信息

    Args:
        sudo_password: 可选 sudo 密码，仅透传给 hardware_fingerprint 的 dmidecode
            sudo -S 通道（采内存 DIMM 细节）。不写入返回值、不落持久化。

    Returns:
        包含系统信息字典
    """
    info: dict[str, Any] = {}

    # Python 信息
    info["python_version"] = platform.python_version()
    info["python_implementation"] = platform.python_implementation()
    info["python_compiler"] = platform.python_compiler()

    # 操作系统
    info["os_name"] = platform.system()
    info["os_version"] = platform.release()
    info["os_arch"] = platform.machine()
    info["hostname"] = _safe_get(lambda: platform.node()[:50], "")

    # 硬件 - CPU
    info["cpu_count"] = os.cpu_count()
    try:
        info["cpu_model"] = _get_cpu_model()
    except Exception:
        info["cpu_model"] = ""

    # 硬件 - 内存 (MB)
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["memory_total_mb"] = round(mem.total / 1024 / 1024)
        info["memory_available_mb"] = round(mem.available / 1024 / 1024)
    except ImportError:
        info["memory_total_mb"] = None
        info["memory_available_mb"] = None

    # 硬件 - 结构化指纹（CPU 拓扑 / 内存 / GPU / PCIe / CUDA）+ 向后兼容的扁平 gpu 字符串
    # 富指纹放入 auto_sys_info，user_sys_info（sidebar 自定义）仍可在合并时覆盖。
    from core.hardware_fingerprint import capture_hardware_fingerprint

    try:
        fingerprint = capture_hardware_fingerprint(sudo_password)
        info["hardware_fingerprint"] = fingerprint
        info["machine_id"] = fingerprint.get("machine_id")
        gpu_names = [
            g.get("name") for g in fingerprint.get("gpus", []) if g.get("name")
        ]
        info["gpu"] = "; ".join(gpu_names) if gpu_names else _get_gpu_info()
    except Exception:
        info["hardware_fingerprint"] = {}
        info["machine_id"] = None
        info["gpu"] = _get_gpu_info()

    # Git 信息
    info["git_hash"] = get_git_hash()
    info["git_branch"] = get_git_branch()

    #  items目Version
    info["project_version"] = get_project_version()

    # 关键库版本(torch/transformers/datasets/tiktoken/psutil/numpy/pandas/streamlit/...)
    info["library_versions"] = get_library_versions()

    # 时间戳
    info["captured_at"] = datetime.now().isoformat()

    return info


def _safe_get(func, default=""):
    """安全Get值，异常时Returndefault值"""
    try:
        return func()
    except Exception:
        return default


def _get_cpu_model() -> str:
    """Get CPU 型号"""
    system = platform.system()

    if system == "Windows":
        try:
            import wmi

            c = wmi.WMI()
            return str(c.Win32_Processor()[0].Name)
        except ImportError:
            # 回退到环境变量
            return os.environ.get("PROCESSOR_IDENTIFIER", "")
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    elif system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

    return ""


def _get_gpu_info() -> str:
    """Get GPU 信息"""
    gpus = []

    # 尝试 nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    gpus.append(line.strip())
    except Exception:
        pass

    # 尝试 PyTorch
    if not gpus:
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append(torch.cuda.get_device_name(i))
        except ImportError:
            pass

    return "; ".join(gpus) if gpus else ""


def get_git_hash(repo_path: str | Path | None = None) -> str:
    """
    Get Git commit hash

    Args:
        repo_path: 仓库路径，defaultis items目根目录

    Returns:
        Git hash (前8位)
    """
    if repo_path is None:
        repo_path = Path(__file__).parent.parent

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_path,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass

    return ""


def get_git_branch(repo_path: str | Path | None = None) -> str:
    """
    Get Git 分支名

    Args:
        repo_path: 仓库路径

    Returns:
        分支名
    """
    if repo_path is None:
        repo_path = Path(__file__).parent.parent

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_path,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return ""


def get_project_version() -> str:
    """
    Get items目Version (从 pyproject.toml)

    Returns:
        Version字符串
    """
    try:
        # 尝试use tomli (Python 3.11+ use tomllib)
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                # 支持 [project] and [tool.poetry] 两种格式
                version = data.get("project", {}).get("version", "")
                if not version:
                    version = data.get("tool", {}).get("poetry", {}).get("version", "")
                return str(version)
    except Exception:
        pass

    return ""


def get_library_versions() -> dict[str, str]:
    """
    Get关键库Version

    Returns:
        库名 -> Version 字典
    """
    libs = {}

    # 标准库
    libs["python"] = platform.python_version()

    # optional依赖
    optional_libs = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("tiktoken", "Tiktoken"),
        ("psutil", "PSUtil"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("streamlit", "Streamlit"),
        ("requests", "Requests"),
        ("aiohttp", "AIOHTTP"),
    ]

    for module_name, display_name in optional_libs:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            libs[display_name] = version
        except ImportError:
            pass

    return libs


def format_system_info(info: dict[str, Any]) -> str:
    """
    Format系统信息is可读字符串

    Args:
        info: 系统信息字典

    Returns:
        Format字符串
    """
    lines = []

    # 基础信息
    lines.append(f"Python: {info.get('python_version', 'N/A')}")
    lines.append(f"OS: {info.get('os_name', 'N/A')} {info.get('os_version', '')}")
    lines.append(
        f"CPU: {info.get('cpu_model', 'N/A')} ({info.get('cpu_count', 'N/A')} cores)"
    )

    # 内存
    mem = info.get("memory_total_mb")
    if mem:
        lines.append(f"Memory: {mem:,} MB")

    # GPU
    gpu = info.get("gpu")
    if gpu:
        lines.append(f"GPU: {gpu}")

    # Git
    git_hash = info.get("git_hash")
    if git_hash:
        lines.append(f"Git: {git_hash}")

    #  items目Version
    version = info.get("project_version")
    if version:
        lines.append(f"Version: {version}")

    return "\n".join(lines)


def compare_system_info(info1: dict, info2: dict) -> dict[str, Any]:
    """
    比较两系统信息

    Args:
        info1: 一系统信息
        info2: 二系统信息

    Returns:
        比较Result字典
    """
    key_fields = [
        "python_version",
        "os_name",
        "cpu_count",
        "memory_total_mb",
        "git_hash",
        "project_version",
    ]

    result: dict[str, Any] = {"identical": True, "differences": []}

    for key in key_fields:
        v1 = info1.get(key)
        v2 = info2.get(key)
        if v1 != v2:
            result["identical"] = False
            result["differences"].append({"field": key, "value1": v1, "value2": v2})

    return result
