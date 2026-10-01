"""Project persisted heuristic diagnostics without rerunning newer rules on old data."""

from html import escape
from typing import Any


def failure_summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("details") or []
    wrong = [row for row in rows if row.get("is_correct") is False and not row.get("error")]
    errors = sum(bool(row.get("error")) for row in rows)
    total = result.get("total_samples")
    complete = (
        type(total) is int
        and len(rows) == total
        and all(type(row.get("is_correct")) is bool for row in rows)
    )
    denominator = len(rows) - errors
    stored = (result.get("extended_metrics") or {}).get("failure_analysis")
    warnings = []
    status = "none" if complete and not wrong else "unavailable"
    distribution: dict[str, int] = {}
    issues: list[str] = []
    suggestions: list[str] = []
    source = "未记录"
    if isinstance(stored, dict):
        candidate = stored.get("category_distribution")
        if (
            isinstance(candidate, dict)
            and all(
                isinstance(name, str) and type(count) is int and count >= 0
                for name, count in candidate.items()
            )
            and sum(candidate.values()) == len(wrong)
        ):
            distribution = candidate
            status = "recorded" if stored.get("version") == "heuristic-failure-v1" else "legacy"
        else:
            warnings.append("历史分类计数与现有错误样本不一致，分类图暂不展示。")
        issues = [item for item in stored.get("top_issues", []) if isinstance(item, str)]
        suggestions = [item for item in stored.get("suggestions", []) if isinstance(item, str)]
        source = str(stored.get("version") or "历史规则摘要（版本未记录）")
    if not complete:
        warnings.append("逐样本覆盖不完整，不推算总体失败率。")
    return {
        "status": status,
        "source": source,
        "failed_samples": len(wrong),
        "figure": {
            "data": [
                {
                    "type": "pie",
                    "labels": [escape(name) for name in distribution],
                    "values": list(distribution.values()),
                    "hole": 0.35,
                }
            ],
            "layout": {
                "title": {"text": "规则分类分布 · 错误案例", "font": {"size": 14}},
                "height": 320,
                "margin": {"l": 20, "r": 20, "t": 50, "b": 40},
            },
        }
        if distribution
        else None,
        "response_samples": denominator,
        "request_errors": errors,
        "failure_rate": len(wrong) / denominator if complete and denominator else None,
        "distribution": distribution,
        "top_issues": issues,
        "suggestions": suggestions,
        "warnings": warnings,
        "note": "分类与建议来自规则启发式，尚未核验错误根因；分类置信度不是模型校准概率。失败率使用有判定且无请求错误的最终成绩，Judge 改判另见复核样本。",
    }


def failure_markdown(name: str, summary: dict) -> str:
    def safe(value):
        return escape(str(value)).replace("|", "\\|").replace("\n", " ")

    rate = f"{summary['failure_rate']:.1%}" if summary["failure_rate"] is not None else "未计算"
    lines = [
        f"\n### {safe(name)} · 自动失败分析",
        "",
        f"- 来源：{safe(summary['source'])}",
        f"- 最终错误：{summary['failed_samples']} / {summary['response_samples']}；失败率 {rate}；请求错误另列 {summary['request_errors']}",
        f"- {safe(summary['note'])}",
        "",
        "| 规则分类 | 案例数 |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {safe(category)} | {count} |" for category, count in summary["distribution"].items()
    )
    lines.extend(
        [
            "",
            "#### 主要问题",
            *[f"- {safe(item)}" for item in summary["top_issues"]],
            "",
            "#### 改进建议",
            *[f"- {safe(item)}" for item in summary["suggestions"]],
            *[f"- {safe(item)}" for item in summary["warnings"]],
        ]
    )
    return "\n".join(lines) + "\n"


def failure_html(name: str, summary: dict) -> str:
    def safe(value):
        return escape(str(value))

    rows = "".join(
        f"<tr><td>{safe(category)}</td><td>{count}</td></tr>"
        for category, count in summary["distribution"].items()
    )
    issues = "".join(f"<li>{safe(item)}</li>" for item in summary["top_issues"])
    suggestions = "".join(f"<li>{safe(item)}</li>" for item in summary["suggestions"])
    warnings = "".join(f"<p>{safe(item)}</p>" for item in summary["warnings"])
    chart = []
    highest = max(summary["distribution"].values(), default=1) or 1
    for index, (category, count) in enumerate(list(summary["distribution"].items())[:20]):
        y = 22 + index * 25
        chart.append(
            f'<text x="5" y="{y}" font-size="11">{safe(category)}</text><rect x="160" y="{y - 12}" width="{count / highest * 350}" height="15" fill="#169d83"/><text x="{170 + count / highest * 350}" y="{y}" font-size="11">{count}</text>'
        )
    chart_html = (
        f'<svg role="img" viewBox="0 0 560 {max(50, len(chart) * 25 + 20)}" xmlns="http://www.w3.org/2000/svg"><title>规则分类分布 · 错误案例</title>{"".join(chart)}</svg>'
        if chart
        else ""
    )
    rate = f"{summary['failure_rate']:.1%}" if summary["failure_rate"] is not None else "未计算"
    return f'<section class="card"><h3>{safe(name)} · 自动失败分析</h3><p>来源：{safe(summary["source"])}</p><p>最终错误 {summary["failed_samples"]}/{summary["response_samples"]} · 失败率 {rate} · 请求错误另列 {summary["request_errors"]}</p><p>{safe(summary["note"])}</p>{chart_html}<table><thead><tr><th>规则分类</th><th>案例数</th></tr></thead><tbody>{rows}</tbody></table><h4>主要问题</h4><ul>{issues}</ul><h4>改进建议</h4><ul>{suggestions}</ul>{warnings}</section>'
