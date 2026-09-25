"""Scientific benchmark summary widgets shared by performance test pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.reporting.statistics import (
    GROUP_COLUMNS,
    GROUP_LABELS,
    AnalysisReport,
    compact_summary_table,
)
from ui.reporting.theme import AMBER, INDIGO, INK, PAPER, SUBTLE, TEAL


def build_quantile_figure(report: AnalysisReport, metric: str) -> go.Figure | None:
    """Plot the median and relevant tail without inventing zero values."""
    median = f"{metric} P50"
    tail = f"{metric} P95" if f"{metric} P95" in report.groups else f"{metric} P10"
    if median not in report.groups or tail not in report.groups:
        return None
    group_columns = [
        column
        for column in GROUP_COLUMNS.get(report.test_type, ())
        if column in report.groups.columns
    ]
    valid_count = f"{metric} valid n"
    if len(group_columns) > 1 and valid_count in report.groups:
        return _build_quantile_heatmap(report, metric, median, valid_count, group_columns)

    rows = report.groups[["Configuration", "Requests", median, tail]].copy()
    rows[median] = pd.to_numeric(rows[median], errors="coerce")
    rows[tail] = pd.to_numeric(rows[tail], errors="coerce")
    if rows[[median, tail]].notna().sum().sum() == 0:
        return None

    x_values = rows["Configuration"]
    numeric_x = False
    x_title = "Configuration"
    if group_columns:
        candidate = pd.to_numeric(report.groups[group_columns[0]], errors="coerce")
        if candidate.notna().all():
            x_values = candidate
            numeric_x = True
            x_title = GROUP_LABELS.get(group_columns[0], group_columns[0])

    figure = go.Figure()
    for name, color, symbol, dash in (
        (median, INDIGO, "circle", "solid"),
        (tail, TEAL if tail.endswith("P10") else AMBER, "diamond", "dot"),
    ):
        values = rows[name]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=values,
                mode="lines+markers" if numeric_x else "markers",
                name=name.rsplit(" ", 1)[-1],
                line={"color": color, "width": 2.5, "dash": dash},
                marker={"color": color, "size": 8, "symbol": symbol},
                customdata=rows[["Requests", "Configuration"]].to_numpy(),
                hovertemplate=(
                    "%{customdata[1]}<br>"
                    + name
                    + ": %{y:,.3f}<br>Recorded requests: %{customdata[0]}<extra></extra>"
                ),
                connectgaps=False,
            )
        )
    figure.update_layout(
        title={"text": metric, "x": 0, "font": {"size": 16, "color": INK}},
        font={"family": "Inter, Segoe UI, PingFang SC, sans-serif", "color": INK, "size": 12},
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        height=355,
        margin={"l": 54, "r": 18, "t": 56, "b": 80},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        xaxis={
            "title": x_title,
            "showgrid": False,
            "linecolor": SUBTLE,
            "tickangle": 0 if numeric_x else -30,
        },
        yaxis={"title": metric, "gridcolor": SUBTLE, "rangemode": "tozero", "zeroline": False},
    )
    return figure


def _build_quantile_heatmap(
    report: AnalysisReport,
    metric: str,
    median: str,
    valid_count: str,
    group_columns: list[str],
) -> go.Figure | None:
    """A two-dimensional sweep is a grid; connecting its rows would invent a path."""
    x_column, y_column = group_columns[:2]
    work = report.groups[[x_column, y_column, median, valid_count]].copy()

    def axis_label(value) -> str:
        if pd.isna(value):
            return "Missing"
        if isinstance(value, (bool, np.bool_)):
            return str(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
            return f"{int(value):,}"
        return str(value)

    work[x_column] = work[x_column].map(axis_label)
    work[y_column] = work[y_column].map(axis_label)
    x_order = work[x_column].drop_duplicates().tolist()
    y_order = work[y_column].drop_duplicates().tolist()
    values = work.pivot(index=y_column, columns=x_column, values=median).reindex(
        index=y_order, columns=x_order
    )
    if values.notna().sum().sum() == 0:
        return None
    counts = work.pivot(index=y_column, columns=x_column, values=valid_count).reindex(
        index=y_order, columns=x_order
    )
    color = TEAL if "TPS" in metric else INDIGO
    figure = go.Figure(
        go.Heatmap(
            x=x_order,
            y=y_order,
            z=values.to_numpy(),
            customdata=counts.to_numpy(),
            colorscale=[[0, "#edf2fa"], [1, color]],
            colorbar={"title": "P50"},
            hovertemplate=(
                f"{GROUP_LABELS.get(x_column, x_column)}: %{{x}}<br>"
                f"{GROUP_LABELS.get(y_column, y_column)}: %{{y}}<br>"
                f"{metric} P50: %{{z:,.3f}}<br>Valid n: %{{customdata}}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={"text": f"{metric} · P50", "x": 0, "font": {"size": 16, "color": INK}},
        font={"family": "Inter, Segoe UI, PingFang SC, sans-serif", "color": INK, "size": 12},
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        height=355,
        margin={"l": 60, "r": 40, "t": 55, "b": 75},
        xaxis={"title": GROUP_LABELS.get(x_column, x_column), "side": "bottom"},
        yaxis={"title": GROUP_LABELS.get(y_column, y_column)},
    )
    return figure


def render_scientific_panel(report: AnalysisReport) -> None:
    """Display statistical coverage, distributions, and interpretation limits."""
    st.markdown("<div class='report-eyebrow'>Measurement evidence</div>", unsafe_allow_html=True)
    st.subheader("Distribution and reliability")
    st.caption(
        "Observed request rows · empirical quantiles · no inferred values for missing measurements"
    )

    groups = report.groups
    status = "Unavailable" if report.success_rate is None else f"{report.success_rate:.1f}%"
    ci = (
        "Request status unavailable"
        if report.success_ci is None
        else f"95% Wilson {report.success_ci[0] * 100:.1f}–{report.success_ci[1] * 100:.1f}%"
    )
    ttft_valid = int(groups.get("TTFT (s) valid n", pd.Series(dtype=int)).sum())
    cards = st.columns(4)
    cards[0].metric("Recorded requests", f"{report.rows:,}")
    cards[1].metric("Configurations", f"{len(groups):,}")
    cards[2].metric("Success rate", status, help=ci)
    cards[3].metric("TTFT coverage", f"{ttft_valid}/{report.rows}")

    chart_specs = ["TTFT (s)", "Decode TPS (tokens/s)"]
    figures = [build_quantile_figure(report, metric) for metric in chart_specs]
    visible = [(metric, fig) for metric, fig in zip(chart_specs, figures, strict=True) if fig]
    if visible:
        columns = st.columns(len(visible))
        for column, (_, fig) in zip(columns, visible, strict=True):
            with column:
                st.plotly_chart(fig, use_container_width=True)

    display = compact_summary_table(report)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=min(520, 38 * (len(display) + 1) + 24),
    )
    with st.expander("All statistical metrics", expanded=False):
        from ui.export import safe_csv_bytes

        st.dataframe(groups, use_container_width=True, hide_index=True)
        st.download_button(
            "Download statistical table",
            data=safe_csv_bytes(groups),
            file_name="benchmark_statistical_summary.csv",
            mime="text/csv",
        )

    if report.findings:
        st.markdown("#### Observed patterns")
        for finding in report.findings:
            st.info(finding)

    prominent = [
        note for note in report.notes if "fewer than 20" in note or "status is missing" in note
    ]
    for note in prominent:
        st.warning(note)
    with st.expander("Methods and data quality", expanded=False):
        st.markdown(f"Metric contract: `{report.metric_contract_version}`")
        if report.token_sources:
            st.markdown("Token count source(s): " + ", ".join(report.token_sources))
        st.markdown("\n".join(f"- {note}" for note in report.notes))
