"""sudo -S 密码通道安全测试（core.hardware_fingerprint + core.hw_snapshot）。

安全契约（必须成立）：
1. sudo 密码绝不进 subprocess argv（ps aux 看不到）——只经 stdin 透传。
2. sudo 密码绝不落快照 JSON（snapshot 里无密码字段、无残留字符串）。
3. 降级链正确：dmidecode(无权限) → sudo -n(NOPASSWD) → sudo -S(密码) → 留空。
4. 无密码时优雅降级，内存 DIMM 细节留空但不抛异常、machine_id 不受影响。
5. build_snapshot 透传 sudo_password 但不写入返回的快照。
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

import core.hardware_fingerprint as hf
import core.hw_snapshot as hws

# ---------- 通用 mock：记录所有 subprocess.run 的 argv ----------


def _spy_subprocess(
    sudo_s_stdout: str = "", sudo_n_fails: bool = True, dmidecode_fails: bool = True
):
    """返回 (fake_run, seen_argv) 。seen_argv 记录每次调用的 argv（模拟 ps 可见内容）。"""
    seen_argv: list[list[str]] = []

    def fake_run(args, **kwargs):
        seen_argv.append(list(args))
        r = MagicMock()
        r.returncode = 0
        r.stdout = ""
        a = list(args)
        # 直跑 dmidecode（无 root）
        if a[0:1] == ["dmidecode"]:
            r.returncode = 0 if not dmidecode_fails else 1
            r.stdout = "" if dmidecode_fails else sudo_s_stdout
        # sudo -n（NOPASSWD）
        elif "sudo" in a and "-n" in a:
            r.returncode = 1 if sudo_n_fails else 0
            r.stdout = sudo_s_stdout if not sudo_n_fails else ""
        # sudo -S（密码经 stdin）
        elif "sudo" in a and "-S" in a:
            r.returncode = 0
            r.stdout = sudo_s_stdout
        return r

    return fake_run, seen_argv


_DMIDECODE_MEMORY = (
    "Handle 0x0019, DMI type 16\nPhysical Memory Array\n"
    "\tError Correction Type: Multi-bit ECC\n\n"
    "Handle 0x001E, DMI type 17\nMemory Device\n"
    "\tSize: 48 GB\n\tType: DDR5\n\tSpeed: 6400 MT/s\n"
)


@pytest.fixture(autouse=True)
def _psutil_available():
    """让 _query_memory_details 的 psutil 路径不崩（注入 fake psutil）。"""
    fake = MagicMock()
    fake.virtual_memory.return_value = MagicMock(total=1024**3, available=512 * 1024**2)
    sys.modules["psutil"] = fake
    yield
    sys.modules.pop("psutil", None)


# ---------- 契约 1：密码不进 argv ----------


def test_password_never_in_argv():
    """sudo -S 的 argv 只有 ['sudo','-S','-k','dmidecode','-t','memory']，密码经 stdin。"""
    fake_run, seen_argv = _spy_subprocess(sudo_s_stdout=_DMIDECODE_MEMORY)
    with (
        patch("core.hardware_fingerprint.subprocess.run", side_effect=fake_run),
        patch("core.hardware_fingerprint.shutil.which", return_value="/usr/sbin/dmidecode"),
    ):
        hf._query_memory_details(sudo_password="SECRET_PASS_xyz789")

    leaked = [a for a in seen_argv if "SECRET_PASS_xyz789" in a]
    assert not leaked, f"密码泄漏到 argv! {leaked}"
    # 确认 sudo -S 调用确实发生
    sudo_s_calls = [a for a in seen_argv if "sudo" in a and "-S" in a]
    assert sudo_s_calls, "应有 sudo -S 调用"
    assert sudo_s_calls[0] == ["sudo", "-S", "-k", "dmidecode", "-t", "memory"]


def test_password_passed_via_stdin_not_argv():
    """stdin_input 参数携带密码，但 args 不含。"""
    captured_inputs: list[str | None] = []
    fake_run, _ = _spy_subprocess(sudo_s_stdout=_DMIDECODE_MEMORY)

    def recording_run(args, **kwargs):
        captured_inputs.append(kwargs.get("input"))
        return fake_run(args, **kwargs)

    with (
        patch("core.hardware_fingerprint.subprocess.run", side_effect=recording_run),
        patch("core.hardware_fingerprint.shutil.which", return_value="/usr/sbin/dmidecode"),
    ):
        hf._query_memory_details(sudo_password="mypw")

    # sudo -S 那次调用的 input 应含密码（带换行）
    assert any(i and "mypw" in i for i in captured_inputs), "密码应经 stdin 传入"


# ---------- 契约 2：密码不落快照 ----------


def test_password_not_in_snapshot_json(tmp_path):
    """build_snapshot 透传密码，但产出的快照 JSON 不含密码。"""
    fake_run, _ = _spy_subprocess(sudo_s_stdout=_DMIDECODE_MEMORY)
    fake_fp = {
        "machine_id": "mid_test",
        "os": {"hostname": "h"},
        "cpu": {},
        "memory": {},
        "gpus": [],
        "cuda": {},
        "disks": [],
        "captured_at": "t",
    }
    with (
        patch("core.hardware_fingerprint.subprocess.run", side_effect=fake_run),
        patch("core.hardware_fingerprint.shutil.which", return_value="/usr/sbin/dmidecode"),
        patch.object(hws, "capture_hardware_fingerprint", return_value=fake_fp),
        patch.object(hws, "capture_system_info", return_value={"hostname": "h"}),
    ):
        snap = hws.build_snapshot(sudo_password="LEAK_CHECK_pw")
    blob = json.dumps(snap, ensure_ascii=False)
    assert "LEAK_CHECK_pw" not in blob
    assert "sudo_password" not in blob.lower()


# ---------- 契约 3：降级链 ----------


def test_fallback_chain_dmidecode_then_sudo_n_then_sudo_s():
    """dmidecode 失败 → sudo -n 失败 → sudo -S 成功。"""
    fake_run, seen_argv = _spy_subprocess(
        sudo_s_stdout=_DMIDECODE_MEMORY, sudo_n_fails=True, dmidecode_fails=True
    )
    with (
        patch("core.hardware_fingerprint.subprocess.run", side_effect=fake_run),
        patch("core.hardware_fingerprint.shutil.which", return_value="/usr/sbin/dmidecode"),
    ):
        info = hf._query_memory_details(sudo_password="pw")
    assert info["type"] == "DDR5"
    assert info["ecc"] is not None
    # 调用顺序：dmidecode → sudo -n → sudo -S
    dmidecode_calls = [a for a in seen_argv if "dmidecode" in " ".join(a)]
    assert dmidecode_calls[0] == ["dmidecode", "-t", "memory"]
    assert dmidecode_calls[1] == ["sudo", "-n", "dmidecode", "-t", "memory"]
    assert "-S" in dmidecode_calls[2]


def test_sudo_n_success_skips_sudo_s():
    """sudo -n 成功（NOPASSWD 已配）时不再走 sudo -S（不浪费密码）。"""
    fake_run, seen_argv = _spy_subprocess(
        sudo_s_stdout=_DMIDECODE_MEMORY, sudo_n_fails=False, dmidecode_fails=True
    )
    with (
        patch("core.hardware_fingerprint.subprocess.run", side_effect=fake_run),
        patch("core.hardware_fingerprint.shutil.which", return_value="/usr/sbin/dmidecode"),
    ):
        hf._query_memory_details(sudo_password="pw")
    sudo_s_calls = [a for a in seen_argv if "-S" in a]
    assert not sudo_s_calls, "sudo -n 已成功，不应再调 sudo -S"


# ---------- 契约 4：无密码优雅降级 ----------


def test_no_password_degrades_gracefully():
    """无密码且 sudo -n 也失败时，内存细节留空但不抛异常。"""
    fake_run, seen_argv = _spy_subprocess(
        sudo_s_stdout=_DMIDECODE_MEMORY, sudo_n_fails=True, dmidecode_fails=True
    )
    with (
        patch("core.hardware_fingerprint.subprocess.run", side_effect=fake_run),
        patch("core.hardware_fingerprint.shutil.which", return_value="/usr/sbin/dmidecode"),
        patch("core.hardware_fingerprint._merge_hw_memory_file"),
    ):
        info = hf._query_memory_details(sudo_password=None)
    # type/channels/speed/ecc 留空（total_gb 仍由 psutil 拿到）
    assert info["type"] is None
    assert info["channels"] is None
    assert info["total_gb"] is not None  # psutil 总能拿到
    # 无密码 → 不应出现 sudo -S 调用
    assert not any("-S" in a for a in seen_argv)


def test_capture_hardware_fingerprint_accepts_sudo_password():
    """capture_hardware_fingerprint 透传 sudo_password 不抛异常。"""
    fake_run, _ = _spy_subprocess(sudo_s_stdout=_DMIDECODE_MEMORY)
    with (
        patch("core.hardware_fingerprint.subprocess.run", side_effect=fake_run),
        patch("core.hardware_fingerprint.shutil.which", return_value="/usr/sbin/dmidecode"),
        patch("core.hardware_fingerprint._query_cpu_topology", return_value={}),
        patch("core.hardware_fingerprint._query_gpus", return_value=[]),
        patch("core.hardware_fingerprint._query_cuda_versions", return_value={}),
        patch("core.hardware_fingerprint._query_gpu_topology", return_value={}),
        patch("core.hardware_fingerprint._query_disk_info", return_value=[]),
    ):
        fp = hf.capture_hardware_fingerprint(sudo_password="pw")
    assert "memory" in fp
    assert fp["memory"]["type"] == "DDR5"
