"""Shared descriptive quality report data, rebuilt from persisted observations."""

import math
from collections import Counter
from html import escape
from typing import Any

from server.analytics import wilson_interval
from server.failure_reports import failure_html, failure_markdown, failure_summary
from server.observations import describe_values
from server.reasoning_reports import reasoning_html, reasoning_markdown, reasoning_summary

METRICS = {
    "ttft_ms": "首内容延迟 / ms",
    "tps": "单次响应生成速度 / tokens/s",
    "total_time_ms": "响应总时间 / ms",
    "latency_ms": "样本调用耗时 / ms",
    "input_tokens": "输入 Token",
    "output_tokens": "输出 Token",
}


def number(value: Any, *, zero: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        measured = float(value)
    except OverflowError:
        return None
    if not math.isfinite(measured) or measured < 0 or (not zero and measured == 0):
        return None
    return measured


def figure(title: str, traces: list[dict], **layout) -> dict:
    return {
        "data": traces,
        "layout": {
            "title": {"text": title, "font": {"size": 14}},
            "height": 360,
            "margin": {"l": 65, "r": 25, "t": 55, "b": 80},
            "paper_bgcolor": "#fff",
            "plot_bgcolor": "#fff",
            "font": {"color": "#26394c"},
            **layout,
        },
    }


def quality_analysis(report: dict) -> dict:
    datasets: dict[str, dict] = {}
    for name, result in report.get("datasets", {}).items():
        if not isinstance(result, dict):
            raise ValueError("Invalid quality dataset result")
        rows = [row for row in result.get("details", []) if isinstance(row, dict)]
        n, correct = result.get("total_samples"), result.get("correct_samples")
        valid_counts = type(n) is int and type(correct) is int and 0 <= correct <= n
        complete = (
            valid_counts
            and len(rows) == n
            and all(
                type(row.get("is_correct")) is bool
                and type(row.get("is_judge_corrected", False)) is bool
                and not (row.get("is_judge_corrected") and not row["is_correct"])
                for row in rows
            )
        )
        warnings = []
        if not complete:
            warnings.append("逐样本记录不完整；分布、方式与诊断只覆盖已记录样本。")
        elif sum(row["is_correct"] for row in rows) != correct:
            warnings.append("逐样本成绩与汇总正确数不一致；不能据此推断模型差异。")
        errors = sum(bool(row.get("error")) for row in rows)
        grade_verified = (
            complete and sum(row["is_correct"] for row in rows) == correct and not errors
        )
        if errors:
            warnings.append(f"{errors} 条记录有请求错误标记；成绩只作描述，不生成准确率区间。")
        usable = [
            row
            for row in rows
            if not row.get("error")
            and (row.get("measurement_provenance") or {}).get("from_cache") is not True
        ]
        cache_count = sum(
            (row.get("measurement_provenance") or {}).get("from_cache") is True for row in rows
        )
        unknown_cache = sum(
            type((row.get("measurement_provenance") or {}).get("from_cache")) is not bool
            for row in rows
        )
        if unknown_cache:
            warnings.append(
                f"{unknown_cache} 条记录未保存本地缓存来源；正值性能统计可能含缓存读取。"
            )
        stats = {}
        if any(
            (row.get("measurement_provenance") or {}).get("output_token_scope")
            == "response_candidates_excluding_thoughts"
            for row in usable
        ):
            warnings.append(
                "Gemini 原生测量：首内容延迟从请求到首个答案文本，输出 token/速度只含答案；"
                "思考 token 与缓存 token 保存在逐样本用量记录，不能把该速度当作含思考的生成速度。"
            )
        for key in METRICS:
            values = []
            sources: Counter[str] = Counter()
            for row in usable:
                provenance = row.get("measurement_provenance") or {}
                source = provenance.get(
                    "input_token_source" if key == "input_tokens" else "output_token_source",
                    "unknown",
                )
                value = number(
                    row.get(key), zero=key.endswith("tokens") and source in {"api", "tokenizer"}
                )
                if value is not None:
                    values.append(value)
                    sources[str(source)] += 1
            stats[key] = {
                **describe_values(values),
                "total": sum(values) if values and math.isfinite(sum(values)) else None,
                "eligible": len(usable),
                "sources": dict(sources) if key.endswith("tokens") else {},
            }
        methods = Counter(str(row.get("evaluation_method") or "未记录") for row in rows)
        parsers = Counter(str(row.get("answer_parse_method") or "未记录") for row in rows)
        confidence = [
            float(row["answer_parse_confidence"])
            for row in rows
            if number(row.get("answer_parse_confidence")) is not None
            and row["answer_parse_confidence"] <= 1
        ]
        # Legacy default zero is not proof that a parser measured zero confidence.
        categories: dict[str, list[bool]] = {}
        for row in rows:
            if type(row.get("is_correct")) is bool:
                categories.setdefault(str(row.get("category") or "未分类"), []).append(
                    row["is_correct"]
                )
        category_rows: list[dict[str, Any]] = [
            {
                "name": category,
                "count": len(grades),
                "correct": sum(grades),
                "accuracy": sum(grades) / len(grades),
            }
            for category, grades in categories.items()
        ]
        category_rows.sort(key=lambda row: (-row["accuracy"], row["name"]))
        shown = category_rows[:20]
        category_figure = figure(
            "类别准确率 · 前 20 类（完整类别见表）",
            [
                {
                    "type": "bar",
                    "orientation": "h",
                    "y": [escape(row["name"]) for row in shown],
                    "x": [row["accuracy"] * 100 for row in shown],
                    "text": [f"n={row['count']}" for row in shown],
                    "marker": {"color": "#169d83"},
                }
            ],
            xaxis={"range": [0, 105], "title": {"text": "准确率 / %"}},
            yaxis={"autorange": "reversed"},
        )
        datasets[name] = {
            "failure_analysis": failure_summary(result),
            "reasoning_analysis": reasoning_summary(rows),
            "name": name,
            "total": n if valid_counts else None,
            "model": str(result.get("model_id") or report.get("model_id") or "未记录"),
            "timestamp": str(result.get("timestamp") or "未记录"),
            "correct": correct if valid_counts else None,
            "accuracy": correct / n if valid_counts and n else None,
            "ci95": wilson_interval(int(correct or 0), int(n or 0))
            if grade_verified and n
            else None,
            "standard_correct": sum(
                row["is_correct"] and not row.get("is_judge_corrected") for row in rows
            )
            if complete
            else None,
            "judge_corrected": sum(row.get("is_judge_corrected") is True for row in rows)
            if complete
            else None,
            "duration_seconds": number(result.get("duration_seconds"), zero=True),
            "detail_count": len(rows),
            "complete": complete,
            "metrics": stats,
            "methods": dict(methods.most_common()),
            "parsers": dict(parsers.most_common()),
            "confidence": {
                "count": len(confidence),
                "mean": sum(confidence) / len(confidence) if confidence else None,
                "high": sum(value >= 0.8 for value in confidence),
                "low": sum(value < 0.5 for value in confidence),
            },
            "categories": category_rows,
            "cache_count": cache_count,
            "unknown_cache": unknown_cache,
            "category_figure": category_figure,
            "request_errors": sum(bool(row.get("error")) for row in rows),
            "warnings": warnings,
            "non_error_fraction": sum(not row.get("error") for row in rows) / len(rows)
            if rows
            else None,
        }
    names = list(datasets)
    labels = [escape(name) for name in names]
    accuracies = [datasets[name]["accuracy"] for name in names]
    figures = {
        "accuracy": figure(
            "数据集准确率 · 95% Wilson",
            [
                {
                    "type": "bar",
                    "x": labels,
                    "y": [value * 100 if value is not None else None for value in accuracies],
                    "marker": {"color": "#169d83"},
                    "error_y": {
                        "type": "data",
                        "symmetric": False,
                        "array": [
                            (datasets[name]["ci95"][1] - datasets[name]["accuracy"]) * 100
                            if datasets[name]["ci95"]
                            else None
                            for name in names
                        ],
                        "arrayminus": [
                            (datasets[name]["accuracy"] - datasets[name]["ci95"][0]) * 100
                            if datasets[name]["ci95"]
                            else None
                            for name in names
                        ],
                    },
                }
            ],
            yaxis={"title": {"text": "准确率 / %"}, "range": [0, 105]},
        )
    }
    valid_names = [name for name in names if datasets[name]["accuracy"] is not None]
    if len(valid_names) >= 3:
        closed = valid_names + valid_names[:1]
        figures["radar"] = figure(
            "数据集成绩雷达（描述性）",
            [
                {
                    "type": "scatterpolar",
                    "theta": [escape(name) for name in closed],
                    "r": [datasets[name]["accuracy"] * 100 for name in closed],
                    "fill": "toself",
                    "line": {"color": "#3185b5"},
                }
            ],
            polar={"radialaxis": {"range": [0, 100]}},
            showlegend=False,
        )
    for key in ("ttft_ms", "tps"):
        figures[key] = figure(
            (
                "首内容延迟 · 正值均值 / ms"
                if key == "ttft_ms"
                else "生成速度 · 正值均值 / tokens/s"
            ),
            [
                {
                    "type": "bar",
                    "x": labels,
                    "y": [datasets[name]["metrics"][key]["mean"] for name in names],
                    "marker": {"color": "#3185b5"},
                }
            ],
            yaxis={"title": {"text": METRICS[key]}, "rangemode": "tozero"},
        )
    return {
        "version": "quality-report-v1",
        "datasets": datasets,
        "figures": figures,
        "notes": [
            "Wilson 区间采用独立二元样本假设；不覆盖评分误差、数据污染或固定全量题库的重复运行误差。",
            "不同数据集难度和样本规模不同；不计算跨任务总分或以雷达面积排名。",
            "性能以逐样本正值记录复算，排除请求错误和已知本地响应缓存；生成速度不等于系统吞吐。",
            "Token 来源分别记录 API 用量与 tokenizer 估计；缺失值不补零。",
            "Token 合计只覆盖已记录的无错误主响应，不含 Judge 额外调用和失败重试，不是账单总额。来源旁数量为记录条数。",
            "分位数按 (n−1)p 位置线性插值；均值按已记录主响应等权。无错误标记比例不等同于模型正确率或独立核验的网络成功率。",
        ],
    }


def quality_analysis_markdown(analysis: dict) -> str:
    def text(value):
        if isinstance(value, float):
            value = f"{value:.6g}"
        return (
            escape(str(value if value is not None else "—")).replace("|", "\\|").replace("\n", " ")
        )

    lines = [
        "\n## 性能复算与解析来源\n",
        "| 数据集 | 指标 | 观测/可用 | 均值 | 中位数 | P95 | P99 | 最小 | 最大 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    method_lines: list[str] = []
    for name, dataset in analysis["datasets"].items():
        for key, label in METRICS.items():
            stat = dataset["metrics"][key]
            cells = [
                name,
                label,
                f"{stat['count']}/{stat['eligible']}",
                *[stat[field] for field in ("mean", "median", "p95", "p99", "min", "max")],
            ]
            lines.append("| " + " | ".join(text(cell) for cell in cells) + " |")
        method_lines.extend(
            [
                "",
                f"### {text(name)} · 评分与解析",
                f"- 评分方式：{text(dataset['methods'])}",
                f"- 解析方式：{text(dataset['parsers'])}",
                f"- 正值解析置信度：{text(dataset['confidence'])}",
                f"- 输入 Token 来源：{text(dataset['metrics']['input_tokens']['sources'])}",
                f"- 输出 Token 来源：{text(dataset['metrics']['output_tokens']['sources'])}",
                *[f"- {text(warning)}" for warning in dataset["warnings"]],
            ]
        )
    lines.extend(method_lines)
    lines.extend(
        reasoning_markdown(name, dataset["reasoning_analysis"])
        for name, dataset in analysis["datasets"].items()
    )
    lines.extend(
        failure_markdown(name, dataset["failure_analysis"])
        for name, dataset in analysis["datasets"].items()
    )
    lines.extend(["", *[f"- {text(note)}" for note in analysis["notes"]]])
    return "\n".join(lines) + "\n"


def quality_chart_svg(analysis: dict, key: str) -> str:
    """Print-friendly charts from the same numeric series as the interactive view."""
    chart = analysis["figures"].get(key)
    if not chart:
        return ""
    trace = chart["data"][0]
    parts = [
        '<svg role="img" viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg">',
        f"<title>{escape(chart['layout']['title']['text'])}</title>",
    ]
    if key == "radar":
        values, labels = trace["r"][:-1], trace["theta"][:-1]
        coordinates = []
        for index, (value, label) in enumerate(zip(values, labels, strict=True)):
            angle = -math.pi / 2 + index * 2 * math.pi / len(values)
            x, y = 340 + 120 * math.cos(angle), 180 + 120 * math.sin(angle)
            parts.append(f'<line x1="340" y1="180" x2="{x}" y2="{y}" stroke="#c6d6dc"/>')
            parts.append(
                f'<text x="{340 + 145 * math.cos(angle)}" y="{180 + 145 * math.sin(angle)}" text-anchor="middle" font-size="12">{label}</text>'
            )
            coordinates.append(
                f"{340 + value * 1.2 * math.cos(angle)},{180 + value * 1.2 * math.sin(angle)}"
            )
        parts.append(
            f'<polygon points="{" ".join(coordinates)}" fill="#3185b533" stroke="#3185b5"/>'
        )
    else:
        values, labels = trace["y"], trace["x"]
        maximum = (
            105
            if key == "accuracy"
            else max([value for value in values if value is not None] or [1]) * 1.1 or 1
        )
        width = 600 / max(1, len(values))
        parts.append('<line x1="50" y1="255" x2="650" y2="255" stroke="#9cacb4"/>')
        for index, (value, label) in enumerate(zip(values, labels, strict=True)):
            x = 50 + width * index + width * 0.12
            height = value / maximum * 200 if value is not None else 0
            parts.append(
                f'<rect x="{x}" y="{255 - height}" width="{width * 0.65}" height="{height}" fill="#169d83"/>'
            )
            parts.append(
                f'<text x="{x}" y="{245 - height}" font-size="11">{value:.2f}</text>'
                if value is not None
                else f'<text x="{x}" y="245" font-size="11">未记录</text>'
            )
            parts.append(
                f'<text x="{x}" y="280" transform="rotate(25 {x} 280)" font-size="11">{label}</text>'
            )
            if key == "accuracy" and value is not None:
                interval = analysis["datasets"][list(analysis["datasets"])[index]]["ci95"]
                if interval:
                    center = x + width * 0.325
                    parts.append(
                        f'<line x1="{center}" y1="{255 - interval[1] * 100 / maximum * 200}" x2="{center}" y2="{255 - interval[0] * 100 / maximum * 200}" stroke="#26394c"/>'
                    )
    parts.append("</svg>")
    return "".join(parts)


def quality_analysis_html(analysis: dict) -> str:
    def text(value):
        if isinstance(value, float):
            value = f"{value:.6g}"
        return escape(str(value if value is not None else "—"))

    rows, methods = [], []
    for name, dataset in analysis["datasets"].items():
        for key, label in METRICS.items():
            stat = dataset["metrics"][key]
            values = [
                name,
                label,
                f"{stat['count']}/{stat['eligible']}",
                *[stat[field] for field in ("mean", "median", "p95", "p99", "min", "max")],
            ]
            rows.append("<tr>" + "".join(f"<td>{text(value)}</td>" for value in values) + "</tr>")
        methods.append(
            f'<section class="card"><h3>{text(name)} · 评分与解析来源</h3><p>评分方式：{text(dataset["methods"])}</p><p>解析方式：{text(dataset["parsers"])}</p><p>正值解析置信度：{text(dataset["confidence"])}</p><p>输入/输出 Token 来源：{text(dataset["metrics"]["input_tokens"]["sources"])} / {text(dataset["metrics"]["output_tokens"]["sources"])}</p>'
            + "".join(f"<p>{text(warning)}</p>" for warning in dataset["warnings"])
            + "</section>"
        )
    charts = "".join(
        f'<section class="card"><h3>{text(analysis["figures"][key]["layout"]["title"]["text"])}</h3>{quality_chart_svg(analysis, key)}</section>'
        for key in ("accuracy", "radar", "ttft_ms", "tps")
        if key in analysis["figures"]
    )
    return (
        charts
        + "".join(
            reasoning_html(name, dataset["reasoning_analysis"])
            for name, dataset in analysis["datasets"].items()
        )
        + "".join(
            failure_html(name, dataset["failure_analysis"])
            for name, dataset in analysis["datasets"].items()
        )
        + '<section class="card"><h2>逐样本性能复算</h2><div style="overflow-x:auto"><table><thead><tr>'
        + "".join(
            f"<th>{label}</th>"
            for label in (
                "数据集",
                "指标",
                "观测/可用",
                "均值",
                "中位数",
                "P95",
                "P99",
                "最小",
                "最大",
            )
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
        + "".join(methods)
        + "".join(f'<p class="note">{text(note)}</p>' for note in analysis["notes"])
    )
