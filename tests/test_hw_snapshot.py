"""core.hw_snapshot 单元测试。

覆盖：
- build_snapshot 结构与必填字段（A/B 维始终在；C/D 维可选）。
- machine_id 来自 fingerprint 且稳定。
- manual 人工字段过滤（未知键丢弃、空值丢弃）。
- 无 GPU / 无 httpx 降级：快照仍产出，不抛异常。
- 磁盘兜底：disks[] 存在时 snapshot 内含。
- fingerprint_hash 稳定性（同硬件同哈希；易变值不影响）。
- load_snapshot schema 校验。
- snapshot_filename 稳定可读。
- summarize 人类可读。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import core.hw_snapshot as hws

# ---------- 固定指纹 / 系统信息（mock 采集器，不依赖真实硬件） ----------

_FAKE_FP = {
    "machine_id": "abc123def4567890",
    "os": {"hostname": "testbox01", "name": "Linux", "release": "6.8", "machine": "x86_64"},
    "cpu": {
        "model_name": "AMD EPYC 9355",
        "sockets": 2,
        "cores_per_socket": 32,
        "threads_per_core": 2,
        "numa_nodes": 2,
        "logical_cores": 128,
    },
    "memory": {
        "type": "DDR5",
        "total_gb": 1133.35,
        "channels": 16,
        "speed_mt_s": 6400,
        "ecc": "Multi-bit ECC",
    },
    "gpus": [
        {
            "index": 0,
            "name": "NVIDIA RTX PRO 6000 Blackwell",
            "vram_gb": 95.59,
            "nominal_bandwidth_gbps": 1792.0,
            "pcie_gen": 5,
            "pcie_width": 16,
        }
    ]
    * 8,
    "cuda": {"driver": "580.159.03", "cuda_version": "13.0"},
    "disks": [
        {"name": "/dev/nvme0", "model": "Samsung 990 Pro", "size_tb": 3.5, "is_ssd": True},
        {"name": "/dev/sda", "model": "WD Blue HDD", "size_tb": 8.0, "is_ssd": False},
    ],
    "gpu_topology": {"has_nvlink": False, "matrix": []},
    "captured_at": "2026-06-24T01:00:00",
}

_FAKE_SYSINFO = {
    "python_version": "3.13.0",
    "os_name": "Linux",
    "os_version": "6.8",
    "hostname": "testbox01",
    "cpu_count": 128,
    "memory_total_mb": 1160576,
    "git_hash": "abc12345",
    "library_versions": {"python": "3.13.0"},
    "captured_at": "2026-06-24T01:00:00",
}

_FAKE_ENGINE = {
    "engine": "vllm",
    "engine_version": "0.22.0",
    "capture_source": ["api", "docker_inspect"],
}

_FAKE_MODELSPEC = {
    "name": "glm-5",
    "architecture": "moe",
    "total_params_b": 744,
    "active_params_b": 40,
}


@pytest.fixture
def patched_collectors():
    """mock 四个采集器，返回固定值，避免依赖真实硬件/docker/config.json。"""
    with (
        patch.object(hws, "capture_hardware_fingerprint", return_value=_FAKE_FP),
        patch.object(hws, "capture_system_info", return_value=_FAKE_SYSINFO),
        patch.object(hws, "capture_engine_config", return_value=_FAKE_ENGINE),
        patch.object(
            hws,
            "from_local_config",
            return_value=type("S", (), {"to_dict": staticmethod(lambda: _FAKE_MODELSPEC)})(),
        ),
    ):
        yield


# ---------- build_snapshot 结构 ----------


def test_build_snapshot_ab_dimensions_always_present(patched_collectors):
    s = hws.build_snapshot()
    assert s["schema"] == hws.SCHEMA_VERSION
    assert s["machine_id"] == _FAKE_FP["machine_id"]
    assert s["hostname"] == "testbox01"
    assert s["hardware_fingerprint"] is _FAKE_FP
    assert "hardware_fingerprint" in s["system_info"]  # 内嵌指纹
    assert s["system_info"]["machine_id"] == _FAKE_FP["machine_id"]
    # C/D 维未给参数 → 不出现
    assert "engine_capture" not in s
    assert "model_spec" not in s
    assert s["manual"] == {}


def test_build_snapshot_with_engine_and_model(patched_collectors):
    s = hws.build_snapshot(
        engine_url="http://127.0.0.1:8000/v1", model_config_path="/data/cfg.json"
    )
    assert s["engine_capture"]["engine"] == "vllm"
    assert s["model_spec"]["name"] == "glm-5"


def test_build_snapshot_manual_filters_unknown_and_empty(patched_collectors):
    manual = {
        "owner": "张三",
        "location": "机房A",  # 已知 + 非空
        "power_supply_w": "",  # 已知但空 → 丢
        "bogus_field": "xxx",  # 未知键 → 丢
    }
    s = hws.build_snapshot(manual=manual)
    assert s["manual"] == {"owner": "张三", "location": "机房A"}


# ---------- 降级 ----------


def test_engine_capture_failure_degrades_gracefully():
    """capture_engine_config 抛异常时，快照仍产出，engine_capture 标 error。"""
    with (
        patch.object(hws, "capture_hardware_fingerprint", return_value=_FAKE_FP),
        patch.object(hws, "capture_system_info", return_value=_FAKE_SYSINFO),
        patch.object(hws, "capture_engine_config", side_effect=RuntimeError("boom")),
    ):
        s = hws.build_snapshot(engine_url="http://x/v1")
    assert s["machine_id"] == _FAKE_FP["machine_id"]  # A/B 维不受影响
    assert s["engine_capture"]["capture_source"] == ["error"]


def test_model_config_failure_degrades_to_none():
    with (
        patch.object(hws, "capture_hardware_fingerprint", return_value=_FAKE_FP),
        patch.object(hws, "capture_system_info", return_value=_FAKE_SYSINFO),
        patch.object(hws, "from_local_config", side_effect=FileNotFoundError("nope")),
    ):
        s = hws.build_snapshot(model_config_path="/nope/config.json")
    assert s["model_spec"] is None
    assert s["machine_id"]  # 仍产出


def test_no_gpu_no_disks_still_produces_snapshot():
    """无 GPU / 无磁盘的机器：fingerprint 字段为空，快照仍完整。"""
    bare_fp = {**_FAKE_FP, "gpus": [], "disks": [], "machine_id": "baremachine0001"}
    with (
        patch.object(hws, "capture_hardware_fingerprint", return_value=bare_fp),
        patch.object(hws, "capture_system_info", return_value=_FAKE_SYSINFO),
    ):
        s = hws.build_snapshot()
    assert s["machine_id"] == "baremachine0001"
    assert s["hardware_fingerprint"]["gpus"] == []


# ---------- fingerprint_hash 稳定性 ----------


def test_fingerprint_hash_stable_for_same_hardware():
    snap = {"hardware_fingerprint": _FAKE_FP}
    assert hws.fingerprint_hash(snap) == hws.fingerprint_hash(snap)


def test_fingerprint_hash_ignores_volatile_fields():
    """易变值（captured_at / available memory）变化不应改变哈希。"""
    base = {"hardware_fingerprint": _FAKE_FP}
    # captured_at 变
    fp2 = {**_FAKE_FP, "captured_at": "2099-01-01"}
    snap2 = {"hardware_fingerprint": fp2}
    assert hws.fingerprint_hash(base) == hws.fingerprint_hash(snap2)
    # 可用内存变（不参与 stable 字段集）
    fp3 = {**_FAKE_FP, "memory": {**_FAKE_FP["memory"], "available_gb": 999.0}}
    snap3 = {"hardware_fingerprint": fp3}
    assert hws.fingerprint_hash(base) == hws.fingerprint_hash(snap3)


def test_fingerprint_hash_changes_when_gpu_changes():
    fp = {**_FAKE_FP, "gpus": _FAKE_FP["gpus"][:4]}  # 8 卡 → 4 卡
    assert hws.fingerprint_hash({"hardware_fingerprint": fp}) != hws.fingerprint_hash(
        {"hardware_fingerprint": _FAKE_FP}
    )


# ---------- load_snapshot / write_snapshot ----------


def test_write_then_load_roundtrip(tmp_path: Path, patched_collectors):
    s = hws.build_snapshot(manual={"owner": "李四"})
    p = hws.write_snapshot(tmp_path / "snap.json", s)
    assert p.exists()
    loaded = hws.load_snapshot(p)
    assert loaded["machine_id"] == s["machine_id"]
    assert loaded["manual"]["owner"] == "李四"


def test_load_snapshot_rejects_wrong_schema(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"schema": "hw-snapshot/v999", "hardware_fingerprint": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema 不兼容"):
        hws.load_snapshot(bad)


def test_load_snapshot_rejects_missing_fingerprint(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": hws.SCHEMA_VERSION}), encoding="utf-8")
    with pytest.raises(ValueError, match="hardware_fingerprint"):
        hws.load_snapshot(bad)


# ---------- snapshot_filename ----------


def test_snapshot_filename_readable_and_stable(patched_collectors):
    s = hws.build_snapshot()
    name = hws.snapshot_filename(s)
    assert name.startswith("hw-snapshot_testbox01_abc123def4567890_")
    assert name.endswith(".json")


def test_snapshot_filename_sanitizes_unsafe_chars():
    s = {"hostname": "bad host!", "machine_id": "mid with space"}
    name = hws.snapshot_filename(s)
    assert " " not in name
    assert "!" not in name


# ---------- summarize ----------


def test_summarize_includes_key_fields(patched_collectors):
    s = hws.build_snapshot(manual={"owner": "王五"}, engine_url="http://x/v1")
    line = hws.summarize(s)
    assert "machine_id=abc123def4567890" in line
    assert "GPU=" in line and "×8" in line
    assert "CUDA=13.0" in line
    assert "engine=vllm" in line
    assert "owner=王五" in line
