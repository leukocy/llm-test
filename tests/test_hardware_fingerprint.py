"""core.hardware_fingerprint 单元测试。

覆盖目标：
- nvidia-smi / lscpu / dmidecode 输出的解析（mock subprocess）。
- 标称带宽查表（HBM/GDDR）优先于 PCIe 公式。
- 无 GPU 路径：gpus 为空但 machine_id 仍可计算且稳定。
- compute_machine_id 对字段重排稳定。
- capture_hardware_fingerprint 整体不抛异常（优雅降级）。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import core.hardware_fingerprint as hf


# ---------- 辅助 ----------

def _fake_subprocess(commands_map: dict[tuple, str | None]):
    """根据 (args tuple) -> stdout 映射返回结果。未命中返回 None。"""

    def fake_run(args, capture_output=True, text=True, timeout=None, check=False):
        class R:
            returncode = 0

        r = R()
        r.stdout = commands_map.get(tuple(args))
        if r.stdout is None:
            r.returncode = 1
            r.stdout = ""
        return r

    return fake_run


NVIDIA_SMI_GPU = (
    "0, NVIDIA H100 80GB HBM3, 81920, HBM3, 5, 16\n"
    "1, NVIDIA RTX 4090, 24564, GDDR6X, 4, 16\n"
)

LSCPU_OUT = (
    "Architecture:        x86_64\n"
    "Model name:          AMD EPYC 9654 96-Core Processor\n"
    "Socket(s):           2\n"
    "Core(s) per socket:  96\n"
    "Thread(s) per core:  2\n"
    "NUMA node(s):        2\n"
)


# ---------- _query_gpus ----------

def test_query_gpus_parses_nvidia_smi_and_uses_bandwidth_table():
    cmds = {("nvidia-smi", "--query-gpu=index,name,memory.total,memory.type,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader,nounits"): NVIDIA_SMI_GPU}
    with patch("core.hardware_fingerprint.subprocess.run", side_effect=_fake_subprocess(cmds)), \
         patch("core.hardware_fingerprint.shutil.which", return_value=None):
        gpus = hf._query_gpus()
    assert len(gpus) == 2
    h100 = gpus[0]
    assert h100["name"] == "NVIDIA H100 80GB HBM3"
    assert h100["vram_gb"] == pytest.approx(80.0, abs=0.1)
    assert h100["memory_type"] == "HBM3"
    assert h100["pcie_gen"] == 5
    assert h100["pcie_width"] == 16
    # H100 命中查表 → 3350 GB/s（而非 PCIe 公式）
    assert h100["nominal_bandwidth_gbps"] == 3350.0
    # 4090 命中查表 → 1008
    assert gpus[1]["nominal_bandwidth_gbps"] == 1008.0


def test_query_gpus_empty_when_no_nvidia_smi_and_no_torch():
    with patch("core.hardware_fingerprint.subprocess.run", side_effect=_fake_subprocess({})), \
         patch("core.hardware_fingerprint.shutil.which", return_value=None), \
         patch.dict("sys.modules", {"torch": None}):
        gpus = hf._query_gpus()
    assert gpus == []


# ---------- bandwidth lookup / pcie fallback ----------

def test_lookup_gpu_bandwidth_longest_substring_wins():
    assert hf._lookup_gpu_bandwidth("NVIDIA A100-SXM4-80GB", 80) == 2039.0
    assert hf._lookup_gpu_bandwidth("RTX 4090", 24) == 1008.0
    assert hf._lookup_gpu_bandwidth("Unknown Weird Card", 16) is None


def test_pcie_bandwidth_formula_fallback():
    # Gen4 x16 ≈ 1.969 * 16 ≈ 31.5
    assert hf._pcie_bandwidth_gbps(4, 16) == pytest.approx(31.5, abs=0.2)
    assert hf._pcie_bandwidth_gbps(None, 16) is None


# ---------- _query_cpu_topology ----------

def test_query_cpu_topology_parses_lscpu():
    cmds = {("lscpu",): LSCPU_OUT}
    with patch("core.hardware_fingerprint.subprocess.run", side_effect=_fake_subprocess(cmds)):
        cpu = hf._query_cpu_topology()
    assert cpu["model_name"] == "AMD EPYC 9654 96-Core Processor"
    assert cpu["sockets"] == 2
    assert cpu["cores_per_socket"] == 96
    assert cpu["threads_per_core"] == 2
    assert cpu["numa_nodes"] == 2


def test_query_cpu_topology_psutil_fallback():
    cmds = {("lscpu",): None}
    with patch("core.hardware_fingerprint.subprocess.run", side_effect=_fake_subprocess(cmds)), \
         patch("core.hardware_fingerprint._platform_cpu_model", return_value="fallback cpu"), \
         patch("core.hardware_fingerprint._psutil_cpu_count", side_effect=[64, 128]):
        cpu = hf._query_cpu_topology()
    assert cpu["model_name"] == "fallback cpu"
    assert cpu["sockets"] == 1
    assert cpu["cores_per_socket"] == 64
    assert cpu["threads_per_core"] == 2


# ---------- compute_machine_id ----------

def test_compute_machine_id_stable_under_reorder():
    fp_a = {
        "cpu": {"model_name": "X", "sockets": 2, "cores_per_socket": 32},
        "memory": {"total_gb": 256.0},
        "gpus": [
            {"name": "A", "vram_gb": 80},
            {"name": "B", "vram_gb": 24},
        ],
    }
    fp_b = {
        "cpu": {"model_name": "X", "sockets": 2, "cores_per_socket": 32},
        "memory": {"total_gb": 256.0},
        # GPU 顺序打乱
        "gpus": [
            {"name": "B", "vram_gb": 24},
            {"name": "A", "vram_gb": 80},
        ],
    }
    assert hf.compute_machine_id(fp_a) == hf.compute_machine_id(fp_b)


def test_compute_machine_id_changes_when_hardware_differs():
    fp_a = {"cpu": {"model_name": "X", "sockets": 1, "cores_per_socket": 32},
            "memory": {"total_gb": 128.0}, "gpus": []}
    fp_b = {"cpu": {"model_name": "Y", "sockets": 1, "cores_per_socket": 32},
            "memory": {"total_gb": 128.0}, "gpus": []}
    assert hf.compute_machine_id(fp_a) != hf.compute_machine_id(fp_b)


def test_compute_machine_id_works_with_empty_gpus():
    fp = {"cpu": {"model_name": "X", "sockets": 1, "cores_per_socket": 8},
          "memory": {"total_gb": 16.0}, "gpus": []}
    mid = hf.compute_machine_id(fp)
    assert isinstance(mid, str)
    assert len(mid) == 16


# ---------- capture_hardware_fingerprint (整体优雅降级) ----------

def test_capture_hardware_fingerprint_never_raises_and_has_machine_id():
    with patch("core.hardware_fingerprint._query_gpus", side_effect=RuntimeError("boom")), \
         patch("core.hardware_fingerprint._query_cpu_topology", return_value={"model_name": "CPU"}), \
         patch("core.hardware_fingerprint._query_memory_details", return_value={"total_gb": 64.0}), \
         patch("core.hardware_fingerprint._query_cuda_versions", return_value={"driver": "535.0"}):
        fp = hf.capture_hardware_fingerprint()
    # 即使 _query_gpus 抛异常，整体仍返回（gpus 被降级为 []）
    assert fp["gpus"] == []
    assert "machine_id" in fp
    assert isinstance(fp["machine_id"], str)
    assert len(fp["machine_id"]) == 16
    assert fp["cuda"]["driver"] == "535.0"


def test_capture_hardware_fingerprint_full_shape():
    cmds = {
        ("nvidia-smi", "--query-gpu=index,name,memory.total,memory.type,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader,nounits"): "0, NVIDIA H100, 81920, HBM3, 5, 16\n",
        ("lscpu",): LSCPU_OUT,
    }
    with patch("core.hardware_fingerprint.subprocess.run", side_effect=_fake_subprocess(cmds)), \
         patch("core.hardware_fingerprint._query_cuda_versions", return_value={"driver": "535", "cuda_version": "12.2"}), \
         patch("core.hardware_fingerprint.shutil.which", return_value=None):
        fp = hf.capture_hardware_fingerprint()
    assert fp["cpu"]["sockets"] == 2
    assert fp["gpus"][0]["nominal_bandwidth_gbps"] == 3350.0
    assert fp["cuda"]["cuda_version"] == "12.2"
    assert fp["os"]["name"]  # 非 None
    assert fp["captured_at"]


# ---------- system_info 集成 ----------

def test_capture_system_info_includes_fingerprint_and_machine_id():
    from core.system_info import capture_system_info
    with patch("core.hardware_fingerprint._query_gpus", return_value=[
        {"index": 0, "name": "NVIDIA RTX 4090", "vram_gb": 24.0,
         "memory_type": "GDDR6X", "nominal_bandwidth_gbps": 1008.0,
         "pcie_gen": 4, "pcie_width": 16}
    ]), patch("core.hardware_fingerprint._query_cpu_topology",
              return_value={"model_name": "X", "sockets": 1, "cores_per_socket": 8}), \
         patch("core.hardware_fingerprint._query_memory_details", return_value={"total_gb": 32.0}), \
         patch("core.hardware_fingerprint._query_cuda_versions",
               return_value={"driver": "535", "cuda_version": "12.2"}):
        info = capture_system_info()
    assert "hardware_fingerprint" in info
    assert info["machine_id"]
    # 向后兼容：扁平 gpu 字符串仍存在
    assert "RTX 4090" in info["gpu"]
    assert info["hardware_fingerprint"]["gpus"][0]["nominal_bandwidth_gbps"] == 1008.0
