"""core.warehouse.capability_sheet 单元测试。"""

from __future__ import annotations

from core.models import ApplicationCase
from core.warehouse.capability_sheet import (
    CAPABILITY_COLUMNS,
    build_capability_markdown,
    build_capability_sheet,
)


def _case(**kw):
    base = {
        "customer_type": "金融",
        "scenario": "coding",
        "model_name": "M1",
        "success": True,
        "quality_score": 8.0,
        "decode_tps": 40.0,
        "external_level": "review",
        "sales_summary": "适合代码补全",
        "task_name": "bugfix",
    }
    base.update(kw)
    return ApplicationCase(**base)


def test_groups_by_customer_scenario_model():
    cases = [
        _case(customer_type="金融", scenario="coding", model_name="M1", success=True),
        _case(customer_type="金融", scenario="coding", model_name="M1", success=False),
        _case(
            customer_type="金融", scenario="retrieval", model_name="M1", success=True
        ),
        _case(customer_type="制造", scenario="coding", model_name="M1", success=True),
    ]
    sheet = build_capability_sheet(cases)
    keys = {(r["customer_type"], r["scenario"], r["model_name"]) for r in sheet}
    assert keys == {
        ("金融", "coding", "M1"),
        ("金融", "retrieval", "M1"),
        ("制造", "coding", "M1"),
    }


def test_success_rate_aggregation():
    cases = [
        _case(success=True),
        _case(success=True),
        _case(success=False),
        _case(success=None),
    ]
    sheet = build_capability_sheet(cases)
    assert len(sheet) == 1
    # success 非空的 3 个里 2 个成功 → 0.667
    assert sheet[0]["success_rate"] == round(2 / 3, 3)
    assert sheet[0]["case_count"] == 4


def test_avg_quality_and_decode_tps():
    cases = [
        _case(quality_score=8.0, decode_tps=40.0),
        _case(quality_score=6.0, decode_tps=20.0),
    ]
    sheet = build_capability_sheet(cases)
    assert sheet[0]["avg_quality_score"] == 7.0
    assert sheet[0]["avg_decode_tps"] == 30.0


def test_max_external_level():
    cases = [
        _case(external_level="internal"),
        _case(external_level="review"),
        _case(external_level="publishable"),
    ]
    sheet = build_capability_sheet(cases)
    assert sheet[0]["external_level"] == "publishable"


def test_min_external_level_filter():
    cases = [
        _case(external_level="internal"),
        _case(external_level="review"),
    ]
    # 下限 publishable → internal/review 组都被过滤
    sheet = build_capability_sheet(cases, min_external_level="publishable")
    assert sheet == []
    # 下限 review → internal 不够（但 max level=review 达标）
    sheet2 = build_capability_sheet(cases, min_external_level="review")
    assert len(sheet2) == 1


def test_sales_summary_and_evidence():
    cases = [
        _case(sales_summary="", evidence_path=""),
        _case(sales_summary="适合代码", evidence_path="a.png"),
    ]
    sheet = build_capability_sheet(cases)
    assert sheet[0]["sales_summary"] == "适合代码"
    assert sheet[0]["evidence_count"] == 1


def test_empty_cases():
    assert build_capability_sheet([]) == []


def test_markdown_contains_sections():
    cases = [
        _case(customer_type="金融", scenario="coding"),
        _case(customer_type="制造", scenario="retrieval"),
    ]
    md = build_capability_markdown(build_capability_sheet(cases))
    assert "# 客户能力表" in md
    assert "## 金融" in md
    assert "## 制造" in md
    assert "coding" in md and "retrieval" in md


def test_markdown_empty_message():
    md = build_capability_markdown([])
    assert "暂无可用数据" in md


def test_capability_columns_complete():
    for col in (
        "customer_type",
        "scenario",
        "model_name",
        "success_rate",
        "avg_quality_score",
        "external_level",
        "sales_summary",
    ):
        assert col in CAPABILITY_COLUMNS
