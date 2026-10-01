"""Descriptive multi-run quality matrix; statistical pairing stays a separate operation."""

import csv
import io
import json
from html import escape
from typing import Any

from server.paired_quality import CONDITIONS
from server.quality_analysis import figure
from utils.spreadsheet import safe_spreadsheet_text


def quality_matrix(entries: list[tuple[dict, dict]], basis: str) -> dict:
    datasets = sorted({name for _, payload in entries for name in payload["datasets"]})
    if not datasets:
        raise ValueError("Selected quality reports contain no datasets")
    rows: list[dict[str, Any]] = []
    for job, payload in entries:
        cells: dict[str, dict[str, Any] | None] = {}
        for name in datasets:
            result = payload["datasets"].get(name)
            if result is None:
                cells[name] = None
                continue
            n, correct = result.get("total_samples"), result.get("correct_samples")
            details = result.get("details") or []
            warnings = []
            valid = type(n) is int and type(correct) is int and 0 <= correct <= n
            complete = (
                valid
                and len(details) == n
                and all(
                    type(row.get("is_correct")) is bool
                    and type(row.get("is_judge_corrected", False)) is bool
                    and not (row.get("is_judge_corrected") and not row["is_correct"])
                    for row in details
                )
            )
            if basis == "standard":
                correct = (
                    sum(row["is_correct"] and not row.get("is_judge_corrected") for row in details)
                    if complete
                    else None
                )
            if not complete:
                warnings.append("逐样本记录不完整；规则成绩未推算。")
            elif sum(row["is_correct"] for row in details) != result.get("correct_samples"):
                warnings.append("汇总与逐样本正确数不一致。")
            if any(row.get("error") for row in details):
                warnings.append("包含请求错误；只作原始成绩描述。")
            config = result.get("config") or {}
            source = config.get("dataset_provenance") or {}
            cells[name] = {
                "accuracy": correct / n if valid and n and correct is not None else None,
                "correct": correct if valid else None,
                "total": n if valid else None,
                "sample_sha256": source.get("sample_sha256"),
                "few_shot_sha256": source.get("few_shot_sha256"),
                "scoring_contract": config.get("scoring_contract"),
                "conditions": {
                    field: (config.get("dataset_overrides") or {})
                    .get(name, {})
                    .get(field, config.get(field))
                    for field in CONDITIONS
                },
                "warnings": warnings,
            }
        rows.append(
            {
                "job_id": job["job_id"],
                "model_id": job["model_id"],
                "endpoint_id": job["endpoint_id"],
                "cells": cells,
            }
        )
    matrix_warnings: dict[str, str] = {}
    for name in datasets:
        present = [row["cells"][name] for row in rows if row["cells"][name] is not None]
        identity = [
            (cell["sample_sha256"], cell["few_shot_sha256"], cell["scoring_contract"])
            for cell in present
        ]
        if any(not all(values) for values in identity) or len(set(identity)) > 1:
            matrix_warnings[name] = "样本/few-shot/评分来源缺失或不同，不能直接解释为模型差异。"
        if (
            any(any(value is None for value in cell["conditions"].values()) for cell in present)
            or len({json.dumps(cell["conditions"], sort_keys=True) for cell in present}) > 1
        ):
            matrix_warnings[name] = (
                matrix_warnings.get(name, "") + " 生成/评分条件缺失或不同；只作描述性对照。"
            )
    chart = figure(
        "多模型数据集成绩（描述性）",
        [
            {
                "type": "bar",
                "name": escape(row["model_id"]) + " · " + row["job_id"][:8],
                "x": [escape(name) for name in datasets],
                "y": [
                    row["cells"][name]["accuracy"] * 100
                    if row["cells"][name] and row["cells"][name]["accuracy"] is not None
                    else None
                    for name in datasets
                ],
            }
            for row in rows
        ],
        barmode="group",
        yaxis={"range": [0, 105], "title": {"text": "准确率 / %"}},
    )
    result = {
        "version": "quality-matrix-v1",
        "score_basis": basis,
        "datasets": datasets,
        "rows": rows,
        "figure": chart,
        "warnings": matrix_warnings,
        "note": "各单元使用自己的样本数；缺测不填零，不计算跨任务总分。此处为描述性对照，显著性检验使用两作业共同样本模块。",
    }
    result["exports"] = matrix_exports(result)
    return result


def matrix_svg(matrix: dict) -> str:
    width = max(680, len(matrix["datasets"]) * 130)
    colors = [
        "#169d83",
        "#3185b5",
        "#d5963c",
        "#8a66b8",
        "#c96879",
        "#66a38c",
        "#687cb4",
        "#a88763",
    ]
    pieces = [
        f'<svg role="img" viewBox="0 0 {width} 420" xmlns="http://www.w3.org/2000/svg"><title>多模型数据集分组柱图</title>'
    ]
    for percent in range(0, 101, 25):
        y = 300 - percent * 2.4
        pieces.append(
            f'<line x1="50" y1="{y}" x2="{width - 50}" y2="{y}" stroke="#dfe7ec"/>'
            f'<text x="42" y="{y + 4}" text-anchor="end" font-size="11">{percent}%</text>'
        )
    group_width = (width - 100) / max(1, len(matrix["datasets"]))
    bar_width = group_width / (len(matrix["rows"]) + 1)
    for dataset_index, name in enumerate(matrix["datasets"]):
        start = 50 + dataset_index * group_width
        pieces.append(
            f'<text x="{start}" y="325" font-size="12"><title>{escape(name)}</title>'
            f"{escape(name[:16])}{'…' if len(name) > 16 else ''}</text>"
        )
        for model_index, row in enumerate(matrix["rows"]):
            cell = row["cells"][name]
            if cell and cell["accuracy"] is not None:
                height = cell["accuracy"] * 240
                x = start + model_index * bar_width
                pieces.append(
                    f'<rect x="{x}" y="{300 - height}" width="{bar_width * 0.85}" height="{height}" fill="{colors[model_index]}"><title>{escape(row["model_id"])}: {cell["accuracy"]:.2%} ({cell["correct"]}/{cell["total"]})</title></rect>'
                )
    for index, row in enumerate(matrix["rows"]):
        x = 30 + (index % 4) * (width / 4)
        y = 365 + (index // 4) * 25
        pieces.append(
            f'<rect x="{x}" y="{y - 10}" width="12" height="12" fill="{colors[index]}"/><text x="{x + 18}" y="{y}" font-size="12"><title>{escape(row["model_id"])} · {row["job_id"][:8]}</title>{escape(row["model_id"][:12])}{"…" if len(row["model_id"]) > 12 else ""} · {row["job_id"][:8]}</text>'
        )
    pieces.append("</svg>")
    return "".join(pieces)


def matrix_exports(matrix: dict) -> dict[str, str]:
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "job_id",
            "model_id",
            "endpoint_id",
            "dataset",
            "score_basis",
            "accuracy",
            "correct",
            "total",
            "sample_sha256",
            "few_shot_sha256",
            "scoring_contract",
        ]
    )
    basis_label = (
        "规则成绩（排除 Judge 改判）"
        if matrix["score_basis"] == "standard"
        else "最终成绩（含 Judge 改判）"
    )
    markdown = [
        "# 多模型质量对照",
        "",
        matrix["note"],
        "评分口径：" + basis_label,
        "",
        "| 作业 / 模型 | "
        + " | ".join(escape(name).replace("|", "\\|") for name in matrix["datasets"])
        + " |",
        "|---|" + "---:|" * len(matrix["datasets"]),
    ]
    html_rows = []
    for row in matrix["rows"]:
        values = []
        for name in matrix["datasets"]:
            cell = row["cells"][name]
            value = (
                f"{cell['accuracy']:.2%} ({cell['correct']}/{cell['total']})"
                if cell and cell["accuracy"] is not None
                else "未记录"
            )
            values.append(value)
            data = {
                "job_id": row["job_id"],
                "model_id": row["model_id"],
                "endpoint_id": row["endpoint_id"],
                "dataset": name,
                "score_basis": matrix["score_basis"],
                **(cell or {}),
            }
            writer.writerow(
                [
                    safe_spreadsheet_text(data.get(field) if data.get(field) is not None else "")
                    for field in (
                        "job_id",
                        "model_id",
                        "endpoint_id",
                        "dataset",
                        "score_basis",
                        "accuracy",
                        "correct",
                        "total",
                        "sample_sha256",
                        "few_shot_sha256",
                        "scoring_contract",
                    )
                ]
            )
        label = row["model_id"] + " · " + row["job_id"][:8]
        markdown.append(
            "| " + escape(label).replace("|", "\\|") + " | " + " | ".join(values) + " |"
        )
        html_rows.append(
            "<tr><td>"
            + escape(label)
            + "</td>"
            + "".join("<td>" + escape(value) + "</td>" for value in values)
            + "</tr>"
        )
    notes = [name + "：" + warning for name, warning in matrix["warnings"].items()]
    notes.extend(
        row["model_id"] + " / " + name + "：" + warning
        for row in matrix["rows"]
        for name, cell in row["cells"].items()
        if cell
        for warning in cell["warnings"]
    )
    markdown.extend(["", *["- " + escape(note).replace("|", "\\|") for note in notes]])
    html = (
        '<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多模型质量对照</title><style>body{font:14px system-ui;color:#23384b;max-width:1100px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #dfe7ec;white-space:nowrap}th{background:#edf7f3}p{line-height:1.6}</style></head><body><h1>多模型质量对照</h1><p>'
        + escape(matrix["note"])
        + " · 评分口径："
        + basis_label
        + '</p><div style="overflow:auto"><table><thead><tr><th>作业 / 模型</th>'
        + "".join("<th>" + escape(name) + "</th>" for name in matrix["datasets"])
        + "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table></div>"
        + matrix_svg(matrix)
        + "".join("<p>" + escape(note) + "</p>" for note in notes)
        + "</body></html>"
    )
    return {"csv": output.getvalue(), "markdown": "\n".join(markdown) + "\n", "html": html}
