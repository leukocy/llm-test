"""Scientific projections of one persisted sample, preserving missing observations."""

import math
from html import escape
from typing import Any


def count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 10**12 else None


def sample_visuals(sample: dict[str, Any]) -> dict[str, Any]:
    provenance = sample.get("measurement_provenance") or {}
    usage = provenance.get("provider_usage") or {}
    notes = []
    latencies = []
    scope = provenance.get("ttft_scope") or "历史口径未记录"
    cache_or_error = bool(provenance.get("from_cache") or sample.get("error"))
    for key, label, domain in (
        ("ttft_ms", "TTFT", scope),
        ("ttut_ms", "TTUT", "独立首个非思考文本计时；未记录时留空"),
        ("total_time_ms", "响应总耗时", "响应调用计时，不含诊断评分"),
    ):
        raw = sample.get(key)
        value = (
            float(raw)
            if isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(raw)
            and raw > 0
            and not cache_or_error
            else None
        )
        maximum = max(
            2000 if key == "ttft_ms" else 5000 if key == "ttut_ms" else 10000, (value or 0) * 1.15
        )
        figure = (
            {
                "data": [
                    {
                        "type": "indicator",
                        "mode": "gauge+number",
                        "value": value,
                        "number": {"suffix": " ms", "valueformat": ",.1f"},
                        "title": {"text": label},
                        "gauge": {"axis": {"range": [0, maximum]}, "bar": {"color": "#169d83"}},
                    }
                ],
                "layout": {"height": 260, "margin": {"l": 35, "r": 35, "t": 50, "b": 25}},
            }
            if value is not None
            else None
        )
        latencies.append(
            {"key": key, "label": label, "value": value, "scope": domain, "figure": figure}
        )
    if cache_or_error:
        notes.append("缓存命中或请求错误不作为实时延迟测量。")
    notes.append("仪表量程随数值扩展；颜色不代表合格阈值。历史默认零不视为计时观测。")
    source = "未记录"
    labels = ["输入", "思考", "答案"]
    tokens: list[int | None] = [None, None, None]
    if (
        isinstance(usage, dict)
        and provenance.get("output_token_scope") == "response_candidates_excluding_thoughts"
    ):
        tokens = [
            count(usage.get(key))
            for key in ("promptTokenCount", "thoughtsTokenCount", "candidatesTokenCount")
        ]
        source = "Gemini 原生 API 用量（答案排除思考）"
        total = count(usage.get("totalTokenCount"))
        if (
            total is not None
            and all(value is not None for value in tokens)
            and sum(value for value in tokens if value is not None) > total
        ):
            tokens = [None, None, None]
            notes.append("API 用量分项大于总数，分布暂不展示。")
    elif isinstance(usage, dict) and "completion_tokens" in usage:
        detail = usage.get("completion_tokens_details") or {}
        prompt, completion = (
            count(usage.get("prompt_tokens")),
            count(usage.get("completion_tokens")),
        )
        reasoning = count(detail.get("reasoning_tokens")) if isinstance(detail, dict) else None
        residual = (
            completion - reasoning
            if completion is not None and reasoning is not None and completion >= reasoning
            else None
        )
        tokens = [prompt, reasoning, residual]
        labels[2] = "非思考输出"
        source = "OpenAI API 用量（输出总数减思考）"
        notes.append(
            "非思考输出是总输出扣除思考的余量，可能包含音频或其他输出；不等同于纯文本答案。"
        )
        total = count(usage.get("total_tokens"))
        if (
            total is not None
            and prompt is not None
            and completion is not None
            and total < prompt + completion
        ):
            tokens = [None, None, None]
            notes.append("API 输入与输出合计大于总数，分布暂不展示。")
    known = all(value is not None for value in tokens)
    summed = sum(value or 0 for value in tokens)
    pie = (
        {
            "data": [
                {
                    "type": "pie",
                    "labels": labels,
                    "values": tokens,
                    "hole": 0.6,
                    "textinfo": "percent+label",
                    "marker": {"colors": ["#64748b", "#60a5fa", "#169d83"]},
                }
            ],
            "layout": {
                "height": 320,
                "margin": {"l": 25, "r": 25, "t": 40, "b": 40},
                "showlegend": True,
            },
        }
        if known and summed > 0 and not cache_or_error
        else None
    )
    notes.append(
        "仅完整且不重叠的 API 分区绘制比例；缓存 Token 是输入子集，不另加一块。缺失不补零，全部已知为零时只列数值。"
    )
    notes.append(
        "比例分母为上列分区计数之和，不代表接口报告总 Token 或账单；其他未归类用量不包含。"
    )
    return {
        "version": "sample-visuals-v1",
        "sample_id": str(sample.get("sample_id", "")),
        "latencies": latencies,
        "tokens": [
            {"label": label, "value": value} for label, value in zip(labels, tokens, strict=True)
        ],
        "token_source": source,
        "token_figure": pie,
        "notes": notes,
    }


def sample_visuals_html(visuals: dict[str, Any]) -> str:
    charts = []
    for row in visuals["latencies"]:
        if row["figure"]:
            maximum = row["figure"]["data"][0]["gauge"]["axis"]["range"][1]
            fraction = row["value"] / maximum * 100
            charts.append(
                f'<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 260 180" style="width:100%;max-width:320px"><title>{escape(row["label"])} 仪表 / ms</title><path d="M30 130 A100 100 0 0 1 230 130" pathLength="100" fill="none" stroke="#e2e8f0" stroke-width="16"/><path d="M30 130 A100 100 0 0 1 230 130" pathLength="100" fill="none" stroke="#169d83" stroke-width="16" stroke-dasharray="{fraction:.3f} 100"/><text x="130" y="110" text-anchor="middle">{row["value"]:.1f} ms</text><text x="130" y="165" text-anchor="middle">{escape(row["label"])} · 量程 0–{maximum:.1f} ms</text></svg>'
            )
    if visuals["token_figure"]:
        total = sum(row["value"] for row in visuals["tokens"])
        offset = 0.0
        slices = []
        for row, color in zip(visuals["tokens"], ("#64748b", "#60a5fa", "#169d83"), strict=True):
            share = row["value"] / total * 100
            slices.append(
                f'<circle cx="120" cy="120" r="80" pathLength="100" fill="none" stroke="{color}" stroke-width="35" stroke-dasharray="{share:.4f} {100 - share:.4f}" stroke-dashoffset="{-offset:.4f}" transform="rotate(-90 120 120)"><title>{escape(row["label"])}：{row["value"]}</title></circle>'
            )
            offset += share
        charts.append(
            '<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 240 240" style="width:100%;max-width:320px"><title>Token 分区比例</title>'
            + "".join(slices)
            + "</svg>"
        )
    chart_html = "".join(charts)
    rows = "".join(
        f"<tr><td>{escape(row['label'])}</td><td>{'未记录' if row['value'] is None else row['value']}</td><td>{escape(row['scope'])}</td></tr>"
        for row in visuals["latencies"]
    )
    tokens = "".join(
        f"<tr><td>{escape(row['label'])}</td><td>{'未记录' if row['value'] is None else row['value']}</td></tr>"
        for row in visuals["tokens"]
    )
    return (
        f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐样本计时与用量</title><style>body{{font:16px system-ui;margin:32px}}td,th{{padding:8px;border:1px solid #ddd}}table{{border-collapse:collapse}}</style><h1>样本 {escape(visuals["sample_id"])}</h1><div>{chart_html}</div><h2>计时 / ms</h2><table><tr><th>指标</th><th>值</th><th>口径</th></tr>{rows}</table><h2>Token 分区</h2><p>{escape(visuals["token_source"])}</p><table><tr><th>分区</th><th>计数</th></tr>{tokens}</table>'
        + "".join(f"<p>{escape(note)}</p>" for note in visuals["notes"])
        + "</html>"
    )
