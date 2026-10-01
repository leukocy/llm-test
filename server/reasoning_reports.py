"""Summaries use persisted diagnostics, never rescore historical reasoning."""

import math
from html import escape
from typing import Any

from core.reasoning_assessment import DIMENSIONS, VERSION

LABELS = dict(zip(DIMENSIONS, ("连贯性", "完整性", "相关性", "正确性代理", "效率"), strict=True))
NOTE = "本地规则启发式，不是经校准的能力评分；正确性依据规则答案判定，不验证推理步骤。雷达图仅使用五维均有记录的共同样本；分维度表保留各自 n，不绘制缺失值。"


def reasoning_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {dimension: [] for dimension in DIMENSIONS}
    complete: list[dict[str, float]] = []
    ignored = 0
    for row in rows:
        record = row.get("reasoning_assessment")
        if (
            row.get("error")
            or not isinstance(record, dict)
            or record.get("version") != VERSION
            or record.get("source") != "local_rule_heuristics"
        ):
            ignored += 1
            continue
        dimensions = record.get("dimensions") or {}
        if not isinstance(dimensions, dict):
            ignored += 1
            continue
        valid = {}
        for dimension in DIMENSIONS:
            value = dimensions.get(dimension)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and 0 <= value <= 10
            ):
                valid[dimension] = float(value)
                values[dimension].append(float(value))
        if len(valid) == 5:
            complete.append(valid)
    stats = {
        dimension: {
            "label": LABELS[dimension],
            "n": len(observations),
            "mean": sum(observations) / len(observations) if observations else None,
        }
        for dimension, observations in values.items()
    }
    means = (
        [sum(row[dimension] for row in complete) / len(complete) for dimension in DIMENSIONS]
        if complete
        else []
    )
    return {
        "version": VERSION,
        "source": "local_rule_heuristics",
        "total": len(rows),
        "ignored": ignored,
        "dimensions": stats,
        "radar_n": len(complete),
        "note": NOTE,
        "figure": {
            "data": [
                {
                    "type": "scatterpolar",
                    "r": means + [means[0]],
                    "theta": list(LABELS.values()) + [LABELS[DIMENSIONS[0]]],
                    "fill": "toself",
                    "name": "本地规则诊断",
                    "line": {"color": "#169d83"},
                    "hovertemplate": "%{theta}<br>%{r:.2f} / 10<extra></extra>",
                }
            ],
            "layout": {
                "height": 340,
                "polar": {"radialaxis": {"range": [0, 10], "visible": True}},
                "showlegend": False,
                "margin": {"l": 55, "r": 55, "t": 35, "b": 35},
            },
        }
        if complete
        else None,
    }


def reasoning_markdown(name: str, summary: dict) -> str:
    title = escape(name).replace("|", "\\|")
    rows = [
        f"\n### {title} · 五维推理规则诊断",
        "",
        f"来源：{VERSION}；雷达共同样本 n={summary['radar_n']}。{NOTE}",
        "",
        "| 维度 | 有记录 n | 均值 / 10 |",
        "|---|---:|---:|",
    ]
    for stat in summary["dimensions"].values():
        mean = "未记录" if stat["mean"] is None else f"{stat['mean']:.2f}"
        rows.append(f"| {stat['label']} | {stat['n']} | {mean} |")
    return "\n".join(rows) + "\n"


def reasoning_html(name: str, summary: dict) -> str:
    chart = ""
    if summary["figure"]:
        values = summary["figure"]["data"][0]["r"][:5]
        points = []
        labels = []
        for index, (label, value) in enumerate(zip(LABELS.values(), values, strict=True)):
            angle = -math.pi / 2 + 2 * math.pi * index / 5
            points.append(
                f"{200 + math.cos(angle) * 120 * value / 10:.2f},{160 + math.sin(angle) * 120 * value / 10:.2f}"
            )
            labels.append(
                f'<text x="{200 + math.cos(angle) * 145:.2f}" y="{160 + math.sin(angle) * 145:.2f}" text-anchor="middle">{label}</text>'
            )
        chart = (
            '<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="五维推理规则雷达图" viewBox="0 0 400 330" style="width:100%;max-width:600px;font:12px system-ui"><title>五维推理规则诊断 · 范围 0–10</title><circle cx="200" cy="160" r="120" fill="none" stroke="#d7e0e5"/><circle cx="200" cy="160" r="60" fill="none" stroke="#d7e0e5"/><polygon points="'
            + " ".join(points)
            + '" fill="#169d83" fill-opacity=".2" stroke="#169d83"/>'
            + "".join(labels)
            + "</svg>"
        )
    table = "".join(
        f"<tr><td>{stat['label']}</td><td>{stat['n']}</td><td>{'未记录' if stat['mean'] is None else format(stat['mean'], '.2f')}</td></tr>"
        for stat in summary["dimensions"].values()
    )
    return f"<section><h3>{escape(name)} · 五维推理规则诊断</h3><p>来源：{VERSION}；雷达共同样本 n={summary['radar_n']}。{NOTE}</p>{chart}<table><tr><th>维度</th><th>有记录 n</th><th>均值 / 10</th></tr>{table}</table></section>"
