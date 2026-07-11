"""
数据仓库模板导出（CSV / JSON / ZIP，纯函数，无 Streamlit 依赖）。

按手册 #templates 的字段顺序把投影行导出成机器可读文件——这是"报告是切片，
仓库是全集"的兑现：导出的不是某次测试的图，而是可筛选、可追溯、可对外口径的
全集行。缺测字段落 ""（手册："缺测本身就是决策信息"）。
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from typing import Any

from core.warehouse.templates import TEMPLATE_DESCRIPTIONS, TEMPLATE_FIELDS, TEMPLATE_TITLES


def _blank(value: Any) -> Any:
    """None → ""（CSV 视觉友好），其余原样。JSON 用 _keep_none 保留 None。"""
    return "" if value is None else value


def export_template_csv(template: str, rows: list[dict[str, Any]]) -> str:
    """把行按模板字段顺序导出为 CSV 字符串（UTF-8 with BOM，Excel 友好）。

    Args:
        template: hwInventory / hmTest / maTest。
        rows: 已投影并选字后的行（键应覆盖模板字段；缺键落 ""）。
    """
    fields = TEMPLATE_FIELDS.get(template, [])
    buf = io.StringIO()
    buf.write("﻿")  # UTF-8 BOM
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        writer.writerow([_blank(row.get(f)) for f in fields])
    return buf.getvalue()


def export_template_json(template: str, rows: list[dict[str, Any]]) -> str:
    """按模板字段顺序导出为 JSON（保留 None 以区分"缺测"与"零值"）。"""
    fields = TEMPLATE_FIELDS.get(template, [])
    out = [{f: row.get(f) for f in fields} for row in rows]
    return json.dumps(out, ensure_ascii=False, indent=2, default=str)


def export_all_templates_zip(
    template_rows: dict[str, list[dict[str, Any]]],
    fmt: str = "csv",
) -> bytes:
    """把多套模板打包成一个 ZIP（一个文件一套模板）。

    Args:
        template_rows: {template_name: rows}。
        fmt: "csv" 或 "json"。
    """
    fmt = fmt.lower()
    if fmt not in ("csv", "json"):
        raise ValueError(f"不支持的导出格式: {fmt}（仅 csv/json）")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, rows in template_rows.items():
            if name not in TEMPLATE_FIELDS:
                continue
            content = (
                export_template_csv(name, rows)
                if fmt == "csv"
                else export_template_json(name, rows)
            )
            # 文件名：模板中文名 + 格式
            title = TEMPLATE_TITLES.get(name, name)
            zf.writestr(f"{title}_{name}.{fmt}", content)
    return buf.getvalue()


def template_header_note(template: str, row_count: int) -> str:
    """生成导出文件头部注释（说明口径 + 行数）。"""
    return (
        f"# {TEMPLATE_TITLES.get(template, template)}（{row_count} 行）\n"
        f"# {TEMPLATE_DESCRIPTIONS.get(template, '')}\n"
        "# 字段口径：test-standard/端侧AI硬件与模型.html #templates\n"
    )
