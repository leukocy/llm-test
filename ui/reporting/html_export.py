"""Self-contained, escaped HTML reports for benchmark results."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from typing import Any, Iterable

import pandas as pd

from ui.reporting.presentation import build_quantile_figure
from ui.reporting.statistics import AnalysisReport, compact_summary_table
from ui.reporting.theme import AMBER, CANVAS, INDIGO, INK, MUTED, PAPER, RED, SUBTLE, TEAL

REPORT_CSS = f"""
:root {{ color-scheme: light; --ink:{INK}; --muted:{MUTED}; --line:{SUBTLE};
  --paper:{PAPER}; --canvas:{CANVAS}; --accent:{INDIGO}; --teal:{TEAL}; --amber:{AMBER}; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--canvas); color:var(--ink);
  font:15px/1.65 Inter,"Segoe UI","PingFang SC",Arial,sans-serif; }}
main {{ width:min(1320px,calc(100% - 48px)); margin:40px auto 72px; }}
header {{ padding:40px 44px; border-radius:22px; color:#fff;
  background:linear-gradient(130deg,#172438 0%,#273c5a 72%,#4f46e5 140%); }}
.eyebrow {{ margin:0 0 8px; color:#c7d2fe; font-size:11px; font-weight:800;
  letter-spacing:.16em; text-transform:uppercase; }}
h1 {{ margin:0; max-width:970px; font-size:clamp(28px,3vw,43px); line-height:1.15;
  letter-spacing:-.035em; }}
.deck {{ margin:14px 0 0; color:#dce5f3; max-width:760px; }}
.meta {{ margin-top:23px; color:#dce5f3; font-size:12px; }}
section {{ margin-top:25px; padding:28px 32px; background:var(--paper);
  border:1px solid var(--line); border-radius:18px; box-shadow:0 10px 34px rgba(23,36,56,.045); }}
.section-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:20px;
  border-bottom:1px solid var(--line); margin-bottom:23px; padding-bottom:13px; }}
h2 {{ margin:0; font-size:21px; letter-spacing:-.025em; }}
.section-subtitle {{ color:var(--muted); font-size:12px; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:13px; }}
.kpi {{ padding:17px 18px; border:1px solid var(--line); border-radius:12px;
  background:#fafbfe; min-width:0; }}
.kpi dt {{ color:var(--muted); font-size:11px; font-weight:750; text-transform:uppercase;
  letter-spacing:.07em; }}
.kpi dd {{ margin:5px 0 0; font-size:25px; font-weight:760; letter-spacing:-.04em; }}
.kpi small {{ display:block; margin-top:4px; color:var(--muted); font-size:11px; }}
.charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:17px; }}
.chart {{ min-width:0; padding:12px 12px 0; border:1px solid var(--line); border-radius:12px; }}
.chart h3 {{ margin:8px 10px 0; font-size:14px; }}
.table-wrap + h3 {{ margin:24px 0 10px; }}
.chart .plotly-graph-div {{ width:100% !important; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; }}
details {{ margin-top:15px; }}
summary {{ cursor:pointer; color:var(--accent); font-weight:650; margin-bottom:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; font-variant-numeric:tabular-nums; }}
th {{ background:#edf1fa; text-align:left; color:#263d5a; font-weight:750;
  position:sticky; top:0; white-space:nowrap; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); }}
tr:nth-child(even) td {{ background:#fafbfe; }}
tr:last-child td {{ border-bottom:0; }}
td {{ white-space:nowrap; }}
.note-list {{ margin:0; padding-left:20px; color:var(--muted); }}
.note-list li {{ margin:5px 0; }}
.insight-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
.insight {{ padding:16px 17px; border:1px solid var(--line); border-left:4px solid var(--accent);
  border-radius:10px; background:#fafbfe; }}
.insight.warning {{ border-left-color:var(--amber); }}
.insight.critical {{ border-left-color:{RED}; }}
.insight.positive {{ border-left-color:var(--teal); }}
.insight strong {{ display:block; font-size:13px; }}
.insight p {{ margin:5px 0 0; color:var(--muted); font-size:12px; }}
footer {{ margin:20px 3px; color:var(--muted); font-size:11px; }}
@media(max-width:900px) {{ main {{ width:calc(100% - 22px); margin:11px auto; }}
  header {{ padding:27px 24px; }} section {{ padding:21px 18px; }}
  .kpis,.charts,.insight-grid {{ grid-template-columns:1fr; }} }}
@media print {{ body {{ background:white; }} main {{ width:100%; margin:0; }}
  header {{ border-radius:0; print-color-adjust:exact; }} section {{ break-inside:avoid;
    box-shadow:none; border-radius:0; }} .chart {{ break-inside:avoid; }} }}
"""


def _table_html(table: Any) -> str:
    data = table if isinstance(table, pd.DataFrame) else getattr(table, "data", None)
    if isinstance(data, pd.DataFrame):
        display = data.copy()
        for column in display.columns:
            series = display[column]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            label = str(column)
            if label == "Accuracy":
                formatter = lambda value: f"{value:.1%}"
            elif (
                label in ("Requests", "Succeeded")
                or label.endswith(" valid n")
                or pd.api.types.is_integer_dtype(series)
            ):
                formatter = lambda value: f"{value:,.0f}"
            elif "TTFT" in label:
                formatter = lambda value: f"{value:,.3f}"
            elif "(%)" in label:
                formatter = lambda value: f"{value:,.1f}"
            else:
                formatter = lambda value: f"{value:,.2f}"
            display[column] = series.map(lambda value: "—" if pd.isna(value) else formatter(value))
        return str(display.to_html(index=False, escape=True, na_rep="—", border=0))
    return f"<pre>{escape(str(table))}</pre>"


def _insight_parts(insight: Any) -> tuple[str, str, str]:
    if isinstance(insight, str):
        title = None
        detail = None
    else:
        title = getattr(insight, "title", None)
        detail = getattr(insight, "detail", None)
    if not isinstance(title, str) or not title:
        raw = str(insight).replace("**", "")
        heading, separator, body = raw.partition(":")
        title = heading if separator else "Insight"
        detail = body.strip() if separator else raw
    severity = getattr(getattr(insight, "severity", None), "value", None) or "neutral"
    return (
        escape(str(title)),
        escape(str(detail)),
        str(severity) if severity in ("positive", "warning", "critical") else "",
    )


def build_html_report(
    figures: Iterable[Any],
    tables: Iterable[Any],
    insights: Iterable[Any] | None = None,
    *,
    title: str = "LLM Benchmark Report",
    analysis: AnalysisReport | None = None,
    overview_cards: Iterable[tuple[str, str, str]] | None = None,
    deck: str | None = None,
    table_heading: str | None = None,
    table_subtitle: str | None = None,
) -> str:
    """Build an offline report; model labels and tabular values remain plain text."""
    figure_list = [figure for figure in figures if figure is not None]
    if analysis is not None:
        analysis_figures = [
            build_quantile_figure(analysis, metric)
            for metric in ("TTFT (s)", "Decode TPS (tokens/s)")
        ]
        figure_list = [figure for figure in analysis_figures if figure is not None] + figure_list

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    safe_title = escape(str(title))
    description = deck or (
        "Request-level measurements, distribution summaries, and interpretation limits in one portable report."
        if analysis is not None
        else "Evaluation outcomes, sample coverage, and interpretation limits in one portable report."
    )
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{safe_title}</title><style>{REPORT_CSS}</style></head><body><main>",
        f"<header><p class='eyebrow'>LLM Benchmark · Evidence report</p><h1>{safe_title}</h1>",
        f"<p class='deck'>{escape(description)}</p>",
        f"<p class='meta'>Generated {generated} · Offline interactive charts · Units are shown beside each metric</p></header>",
    ]

    if analysis is not None:
        rate = "Unavailable" if analysis.success_rate is None else f"{analysis.success_rate:.1f}%"
        ci = (
            "Outcome data unavailable"
            if analysis.success_ci is None
            else f"95% Wilson: {analysis.success_ci[0] * 100:.1f}–{analysis.success_ci[1] * 100:.1f}%"
        )
        cards = (
            ("Recorded requests", f"{analysis.rows:,}", "Rows present in this result"),
            ("Configurations", str(len(analysis.groups)), "Distinct reported groups"),
            ("Success rate", rate, ci),
            (
                "Metric contract",
                analysis.metric_contract_version,
                "Historical definitions may differ",
            ),
        )
        parts.append(
            "<section><div class='section-head'><h2>Executive summary</h2><span class='section-subtitle'>Descriptive evidence</span></div><div class='kpis'>"
        )
        for label, value, detail in cards:
            parts.append(
                f"<dl class='kpi'><dt>{escape(label)}</dt><dd>{escape(value)}</dd><small>{escape(detail)}</small></dl>"
            )
        parts.append("</div></section>")

        status_subtitle = (
            "Success rate uses all recorded requests"
            if analysis.success_count is not None
            else "Request status unavailable"
        )
        parts.append(
            "<section><div class='section-head'><h2>Statistical breakdown</h2>"
            f"<span class='section-subtitle'>{status_subtitle}</span></div><div class='table-wrap'>"
        )
        parts.append(_table_html(compact_summary_table(analysis)))
        parts.append("</div>")
        parts.append(
            "<details><summary>All recorded metrics and grouping keys</summary><div class='table-wrap'>"
        )
        parts.append(_table_html(analysis.groups))
        parts.append("</div></details></section>")
        if analysis.findings:
            parts.append(
                "<section><div class='section-head'><h2>Observed patterns</h2><span class='section-subtitle'>Descriptive comparisons</span></div><div class='insight-grid'>"
            )
            for finding in analysis.findings:
                parts.append(f"<article class='insight'><p>{escape(finding)}</p></article>")
            parts.append("</div></section>")
        parts.append(
            "<section><div class='section-head'><h2>Methods and limits</h2></div><ul class='note-list'>"
        )
        if analysis.token_sources:
            parts.append(
                "<li>Token count source(s): " + escape(", ".join(analysis.token_sources)) + "</li>"
            )
        for note in analysis.notes:
            parts.append(f"<li>{escape(note)}</li>")
        parts.append("</ul></section>")

    if overview_cards is not None:
        parts.append(
            "<section><div class='section-head'><h2>Executive summary</h2>"
            "<span class='section-subtitle'>Recorded evidence</span></div><div class='kpis'>"
        )
        for label, value, detail in overview_cards:
            parts.append(
                f"<dl class='kpi'><dt>{escape(str(label))}</dt>"
                f"<dd>{escape(str(value))}</dd><small>{escape(str(detail))}</small></dl>"
            )
        parts.append("</div></section>")

    if insights:
        parts.append(
            "<section><div class='section-head'><h2>Interpretation</h2></div><div class='insight-grid'>"
        )
        for insight in insights:
            title_text, detail_text, severity = _insight_parts(insight)
            parts.append(
                f"<article class='insight {severity}'><strong>{title_text}</strong><p>{detail_text}</p></article>"
            )
        parts.append("</div></section>")

    if figure_list:
        parts.append(
            "<section><div class='section-head'><h2>Charts</h2><span class='section-subtitle'>Hover for exact values and sample counts</span></div><div class='charts'>"
        )
        for index, figure in enumerate(figure_list):
            parts.append("<article class='chart'>")
            parts.append(
                figure.to_html(
                    full_html=False,
                    include_plotlyjs=index == 0,
                    div_id=f"report-chart-{index + 1}",
                    config={"responsive": True, "displaylogo": False},
                )
            )
            parts.append("</article>")
        parts.append("</div></section>")

    table_list = [table for table in tables if table is not None]
    if table_list:
        heading = table_heading or (
            "Source summaries" if analysis is not None else "Evaluation summary"
        )
        subtitle = table_subtitle or (
            "Observed extrema from the existing test views"
            if analysis is not None
            else "Counts, accuracy, and uncertainty"
        )
        parts.append(
            f"<section><div class='section-head'><h2>{escape(heading)}</h2>"
            f"<span class='section-subtitle'>{escape(subtitle)}</span></div>"
        )
        for table in table_list:
            if isinstance(table, tuple) and len(table) == 2:
                table_title, table_data = table
                parts.append(f"<h3>{escape(str(table_title))}</h3>")
            else:
                table_data = table
            parts.append("<div class='table-wrap'>" + _table_html(table_data) + "</div>")
        parts.append("</section>")

    parts.append(
        "<footer>Generated by llm-test. Verify workload, token source, and run completeness before comparing systems.</footer></main></body></html>"
    )
    return "".join(parts)
