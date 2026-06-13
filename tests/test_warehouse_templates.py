"""core.warehouse.templates 单元测试——字段清单与手册 #templates 一致。"""

from __future__ import annotations

from core.warehouse.templates import (
    HARDWARE_INVENTORY_FIELDS,
    HM_TEST_FIELDS,
    MA_TEST_FIELDS,
    TEMPLATE_FIELDS,
    TEMPLATE_TITLES,
    all_template_names,
    template_fields,
)


def test_hardware_inventory_fields_match_manual():
    # 手册 <pre id="hwInventory"> 的首尾字段 + 总字段数
    assert HARDWARE_INVENTORY_FIELDS[0] == "machine_id"
    assert HARDWARE_INVENTORY_FIELDS[-1] == "remark"
    assert "gpu_bandwidth_gbps" in HARDWARE_INVENTORY_FIELDS
    assert "cuda_or_rocm" in HARDWARE_INVENTORY_FIELDS
    assert len(HARDWARE_INVENTORY_FIELDS) == 30


def test_hm_test_fields_match_manual():
    # 手册 <pre id="hmTest">
    assert HM_TEST_FIELDS[0] == "test_id"
    assert HM_TEST_FIELDS[-1] == "supersedes_test_id"
    for key in (
        "effective_bandwidth_gbps",
        "bandwidth_utilization_pct",
        "external_level",
        "next_action",
        "supersedes_test_id",
        "bottleneck",
        "error_type",
    ):
        assert key in HM_TEST_FIELDS


def test_ma_test_fields_match_manual():
    # 手册 <pre id="maTest">（应用质量维度）
    assert MA_TEST_FIELDS[0] == "case_id"
    assert MA_TEST_FIELDS[-1] == "next_action"
    for key in (
        "quality_score",
        "citation_score",
        "tool_success_rate",
        "retrieval_latency_s",
        "scenario",
        "sales_summary",
    ):
        assert key in MA_TEST_FIELDS


def test_no_duplicate_fields_within_template():
    for name, fields in TEMPLATE_FIELDS.items():
        assert len(fields) == len(set(fields)), f"模板 {name} 有重复字段"


def test_template_registry_and_titles():
    assert all_template_names() == ["hwInventory", "hmTest", "maTest"]
    assert template_fields("hwInventory") is not HARDWARE_INVENTORY_FIELDS  # 返回拷贝
    assert template_fields("hwInventory") == HARDWARE_INVENTORY_FIELDS
    assert template_fields("nonexistent") == []
    assert set(TEMPLATE_TITLES) == set(all_template_names())
