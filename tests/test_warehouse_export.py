"""core.warehouse.export 单元测试（CSV/JSON/ZIP，纯函数）。"""

from __future__ import annotations

import csv
import io
import json
import zipfile

from core.warehouse.export import (
    export_all_templates_zip,
    export_template_csv,
    export_template_json,
    template_header_note,
)


def test_csv_header_matches_template_fields():
    rows = [{"test_id": "t1", "machine_id": "m1"}, {"test_id": "t2", "machine_id": "m2"}]
    out = export_template_csv("hmTest", rows)
    # BOM（﻿）是 Excel 友好的 UTF-8 标记，读表头前剥掉
    reader = csv.reader(io.StringIO(out.lstrip("﻿")))
    header = next(reader)
    from core.warehouse.templates import HM_TEST_FIELDS
    assert header == HM_TEST_FIELDS
    data = list(reader)
    assert len(data) == 2
    assert data[0][0] == "t1"  # test_id
    # None → ""（缺测字段）
    blank_idx = HM_TEST_FIELDS.index("tester")
    assert data[0][blank_idx] == ""


def test_csv_has_utf8_bom():
    out = export_template_csv("hwInventory", [{}])
    assert out.startswith("﻿")


def test_json_preserves_none():
    rows = [{"case_id": "c1", "quality_score": None, "citation_score": 0.9}]
    out = export_template_json("maTest", rows)
    data = json.loads(out)
    assert isinstance(data, list)
    assert data[0]["case_id"] == "c1"
    assert data[0]["quality_score"] is None  # 保留 None 区分缺测
    assert data[0]["citation_score"] == 0.9
    # JSON 字段顺序遵循模板
    assert list(data[0].keys())[0] == "case_id"


def test_zip_contains_all_templates():
    template_rows = {
        "hwInventory": [{"machine_id": "m1"}],
        "hmTest": [{"test_id": "t1"}],
        "maTest": [{"case_id": "c1"}],
    }
    data = export_all_templates_zip(template_rows, fmt="csv")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert len(names) == 3
        for n in names:
            assert n.endswith(".csv")


def test_zip_json_format():
    data = export_all_templates_zip({"hmTest": [{"test_id": "t1"}]}, fmt="json")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert any(n.endswith(".json") for n in zf.namelist())


def test_zip_rejects_unknown_format():
    try:
        export_all_templates_zip({"hmTest": []}, fmt="xml")
        raise AssertionError("应抛 ValueError")
    except ValueError:
        pass


def test_header_note_contains_template_title_and_count():
    note = template_header_note("hmTest", 7)
    assert "硬件 × 模型测试字段" in note
    assert "7 行" in note
