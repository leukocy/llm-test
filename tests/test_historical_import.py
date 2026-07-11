"""core.historical_import 解析逻辑单元测试（纯函数，不碰 DB）。"""

from __future__ import annotations

from core.historical_import import (
    build_legacy_fingerprint,
    parse_engine_name,
    parse_gpu_str,
    parse_memory_str,
)

# ---- parse_engine_name ----


def test_parse_engine_vllm_with_mtp():
    r = parse_engine_name("vLLM-v0.22.0-MTP3")
    assert r["engine"] == "vllm"
    assert r["engine_version"] == "0.22.0"
    assert r["mtp_enabled"] is True
    assert r["num_speculative_tokens"] == 3


def test_parse_engine_sglang():
    r = parse_engine_name("SGLang-v0.4.5")
    assert r["engine"] == "sglang"
    assert r["engine_version"] == "0.4.5"
    assert "mtp_enabled" not in r


def test_parse_engine_none():
    assert parse_engine_name(None) == {}
    assert parse_engine_name("") == {}


def test_parse_engine_no_version():
    r = parse_engine_name("vllm")
    assert r.get("engine") == "vllm"
    assert "engine_version" not in r


# ---- parse_gpu_str ----


def test_parse_gpu_with_count():
    gpus = parse_gpu_str("4*H100")
    assert len(gpus) == 1
    assert gpus[0]["name"] == "H100"
    assert gpus[0]["count"] == 4
    assert gpus[0]["nominal_bandwidth_gbps"] == 3350  # H100 查表


def test_parse_gpu_pro6000():
    gpus = parse_gpu_str("1* pro 6000")
    assert gpus[0]["name"] == "pro 6000"
    assert gpus[0]["count"] == 1
    assert gpus[0]["nominal_bandwidth_gbps"] is not None  # RTX PRO 6000 查表


def test_parse_gpu_none():
    assert parse_gpu_str(None) == []
    assert parse_gpu_str("") == []


# ---- parse_memory_str ----


def test_parse_memory_ddr5():
    m = parse_memory_str("4*48G DDR5 6400")
    assert m["sticks"] == 4
    assert m["capacity_gb_per_stick"] == 48
    assert m["total_gb"] == 192
    assert m["type"] == "DDR5"
    assert m["speed_mt_s"] == 6400


def test_parse_memory_no_speed():
    m = parse_memory_str("8*32G DDR5")
    assert m["total_gb"] == 256
    assert m["speed_mt_s"] is None


def test_parse_memory_none():
    assert parse_memory_str(None) == {}
    assert parse_memory_str("") == {}


# ---- build_legacy_fingerprint ----


def test_fingerprint_assembles_and_machine_id_stable():
    sys_info = {
        "gpu": "1* pro 6000",
        "memory": "4*48G DDR5 6400",
        "engine_name": "vLLM-v0.22.0-MTP3",
    }
    fp = build_legacy_fingerprint(sys_info)
    assert fp["machine_id"] is not None
    assert len(fp["machine_id"]) == 16
    assert fp["gpus"][0]["name"] == "pro 6000"
    assert fp["memory"]["total_gb"] == 192
    assert fp["source"] == "legacy_meta"
    assert fp["cpu"] == {}  # legacy 缺
    # 同输入 → 同 machine_id
    fp2 = build_legacy_fingerprint(sys_info)
    assert fp["machine_id"] == fp2["machine_id"]


def test_fingerprint_machine_id_independent_of_gpu_count():
    # count 不可靠，machine_id 不依赖它（只依赖型号 + 内存总量）
    a = build_legacy_fingerprint({"gpu": "1* pro 6000", "memory": "4*48G DDR5 6400"})
    b = build_legacy_fingerprint({"gpu": "4* pro 6000", "memory": "4*48G DDR5 6400"})
    assert a["machine_id"] == b["machine_id"]


def test_fingerprint_empty_sys_info():
    fp = build_legacy_fingerprint({})
    assert fp["machine_id"] is None
    assert fp["gpus"] == []
