"""
Export functionality for benchmark results.

Supports export to Excel, HTML, enhanced Markdown formats, and static PNG charts.
"""

import base64
import re
from html import escape
from io import BytesIO

import pandas as pd
import streamlit as st

from ui.reporting.theme import AMBER, BLUE, SERIES, TEAL
from utils.spreadsheet import safe_csv_bytes as safe_csv_bytes
from utils.spreadsheet import safe_spreadsheet_text


def _excel_sheet_name(name, used):
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", str(name)).strip(" '\"") or "Results"
    base = cleaned[:31]
    candidate = base
    suffix = 2
    while candidate.casefold() in used:
        tail = f"_{suffix}"
        candidate = base[: 31 - len(tail)] + tail
        suffix += 1
    used.add(candidate.casefold())
    return candidate


def export_to_excel(df_dict, filename="benchmark_results.xlsx"):
    """Export typed values with readable formatting and safe text cells."""
    try:
        from ui.reporting.theme import INK, SUBTLE

        output = BytesIO()
        used_names: set[str] = set()
        prepared = []
        for sheet_name, df in df_dict.items():
            safe = df.copy()
            for column in safe.columns:
                if not pd.api.types.is_numeric_dtype(safe[column]):
                    safe[column] = safe[column].map(safe_spreadsheet_text)
            prepared.append((_excel_sheet_name(sheet_name, used_names), safe))
        if not prepared:
            raise ValueError("No worksheets to export")

        try:
            with pd.ExcelWriter(
                output,
                engine="xlsxwriter",
                engine_kwargs={"options": {"strings_to_formulas": False, "strings_to_urls": False}},
            ) as writer:
                workbook = writer.book
                header = workbook.add_format(
                    {
                        "bold": True,
                        "bg_color": INK,
                        "font_color": "#ffffff",
                        "font_size": 11,
                        "valign": "vcenter",
                        "bottom": 2,
                        "bottom_color": SUBTLE,
                    }
                )
                integer = workbook.add_format({"num_format": "#,##0", "font_color": INK})
                decimal = workbook.add_format({"num_format": "#,##0.00", "font_color": INK})
                for name, df in prepared:
                    df.to_excel(writer, sheet_name=name, index=False, startrow=1, header=False)
                    sheet = writer.sheets[name]
                    sheet.set_row(0, 34)
                    sheet.freeze_panes(1, 0)
                    for index, column in enumerate(df.columns):
                        sheet.write_string(
                            0, index, str(safe_spreadsheet_text(str(column))), header
                        )
                        samples = df[column].dropna().head(200).astype(str).map(len)
                        width = min(
                            44,
                            max(
                                12,
                                len(str(column)) + 2,
                                int(samples.max()) + 2 if len(samples) else 12,
                            ),
                        )
                        fmt = None
                        if pd.api.types.is_integer_dtype(df[column]):
                            fmt = integer
                        elif pd.api.types.is_float_dtype(df[column]):
                            fmt = decimal
                        sheet.set_column(index, index, width, fmt)
                    if len(df.columns):
                        sheet.autofilter(0, 0, len(df), len(df.columns) - 1)
        except ImportError:
            from openpyxl.styles import Alignment, Font, PatternFill

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for name, df in prepared:
                    df.to_excel(writer, sheet_name=name, index=False)
                    sheet = writer.sheets[name]
                    sheet.freeze_panes = "A2"
                    sheet.auto_filter.ref = sheet.dimensions
                    for cell in sheet[1]:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill("solid", fgColor=INK.removeprefix("#"))
                        cell.alignment = Alignment(vertical="center")
                    sheet.row_dimensions[1].height = 26
                    for index, column in enumerate(df.columns, 1):
                        letter = sheet.cell(1, index).column_letter
                        sheet.column_dimensions[letter].width = min(
                            44, max(12, len(str(column)) + 2)
                        )

        output.seek(0)
        return output
    except Exception as error:
        st.error(f"Excel Export failed: {error}")
        return None


def export_interactive_html(
    figures_list, tables_list, insights_list=None, title="LLM Benchmark Report", analysis=None
):
    """Export a self-contained, escaped HTML report with statistical evidence."""
    from ui.reporting.html_export import build_html_report

    return build_html_report(
        figures_list, tables_list, insights_list, title=title, analysis=analysis
    )


def create_html_download_link(
    html_content, filename="benchmark_report.html", link_text="Download HTML Report"
):
    """
    Create download link for HTML report.

    Args:
        html_content: HTML string
        filename: Download filename
        link_text: Link button text

    Returns:
        HTML download link
    """
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{escape(str(filename), quote=True)}" class="download-btn">{escape(str(link_text))}</a>'
    return href


# =====================================================================
# Static Chart Export (inspired by llm-performance-test.html style)
# =====================================================================


def create_static_chart_download_link(
    chart_bytes: BytesIO,
    filename: str = "chart.png",
    link_text: str = "Download Static Chart",
) -> str:
    """
    Create static chart download link

    Args:
        chart_bytes: Chart byte stream
        filename: Download filename
        link_text: Link button text

    Returns:
        HTML download link
    """
    if chart_bytes:
        chart_bytes.seek(0)
        b64 = base64.b64encode(chart_bytes.read()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-btn">{link_text}</a>'
        return href
    return ""


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first existing column name in DataFrame, or None if none exist"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_col_list(df: pd.DataFrame, col: str | None) -> list[float]:
    """Extract numeric chart values; leave missing measurements as gaps."""
    if col is None or col not in df.columns:
        return []
    import numpy as np

    numeric = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return [float(value) for value in numeric]


def export_benchmark_summary_chart(
    df: pd.DataFrame,
    test_type: str,
    model_id: str,
    provider: str,
    system_info: dict[str, str] | None = None,
) -> BytesIO | None:
    """
    Export unified 2×2 quad-panel summary chart (PNG) based on test type.

    Unified layout:
    ┌──────────────────┬──────────────────┐
    │  Top-left: TTFT (Latency) │  Top-right: Speed Metrics      │
    ├──────────────────┼──────────────────┤
    │  Bottom-left: Input Throughput  │  Bottom-right: Output Throughput    │
    └──────────────────┴──────────────────┘

    Args:
        df: Summary DataFrame
        test_type: Test Type
            'concurrency' / 'prefill' / 'long_context' / 'matrix' / 'segmented'
        model_id: ModelID
        provider: Provider
        system_info: Optional system info

    Returns:
        BytesIO object containing PNG image data; Returns None on failure
    """
    try:
        from ui.static_chart_generator import StaticChartGenerator

        generator = StaticChartGenerator(dpi=150)

        # ---- Common title logic ----
        type_titles = {
            "concurrency": "Concurrency Test",
            "prefill": "Prefill Stress Test",
            "long_context": "Long Context Test",
            "matrix": "Concurrency-Context Matrix Test",
            "segmented": "Segmented Context Test (Prefix Caching)",
        }
        base_title = type_titles.get(test_type, "Performance Test")

        # Calculate input/output token info and concurrency for title
        io_label = ""
        concurrency_label = ""

        if test_type == "concurrency":
            avg_in_col = _resolve_col(df, "Actual_Tokens_Mean", "Avg_Input_Tokens")
            max_out_col = _resolve_col(df, "Actual_Decode_Max", "Max_Output_Tokens")
            avg_in_mean = df[avg_in_col].mean() if avg_in_col and avg_in_col in df.columns else None
            max_out_max = (
                df[max_out_col].max() if max_out_col and max_out_col in df.columns else None
            )
            avg_in = int(avg_in_mean) if avg_in_mean is not None and pd.notna(avg_in_mean) else 0
            max_out = int(max_out_max) if max_out_max is not None and pd.notna(max_out_max) else 0
            if avg_in > 0 or max_out > 0:
                io_label = f" (In: ~{avg_in}, Out: ~{max_out})"

        elif test_type in ("long_context", "prefill", "segmented"):
            # For single-concurrency tests, show concurrency=1 if available
            if "concurrency" in df.columns:
                conc_values = df["concurrency"].unique()
                if len(conc_values) == 1:
                    concurrency_label = f" (Concurrency: {int(conc_values[0])})"

        # Combine labels for title
        full_label = io_label + concurrency_label

        if system_info:
            title = base_title + full_label
        else:
            title = f"{base_title}{full_label}\nModel: {model_id} | Provider: {provider}"

        # ---- matrix test → multi-line chart (special handling, has concurrency dimension) ----
        if test_type == "matrix":
            x_col = "context_length_target"
            y_col = _resolve_col(
                df,
                "Max_System_Output_Throughput",
                "Max_System_Output_Throughput (tokens/s)",
            )

            if x_col not in df.columns or y_col is None:
                return None

            unique_contexts = sorted(df[x_col].unique())

            if "concurrency" in df.columns:
                unique_concurrencies = sorted(df["concurrency"].unique())
                palette = SERIES

                datasets = []
                for i, conc in enumerate(unique_concurrencies):
                    subset = df[df["concurrency"] == conc]
                    data_points = []
                    for ctx in unique_contexts:
                        val = subset[subset[x_col] == ctx][y_col].values
                        data_points.append(float(val[0]) if len(val) > 0 else float("nan"))

                    datasets.append(
                        {
                            "label": f"Concurrency {int(conc)}",
                            "data": data_points,
                            "color": palette[i % len(palette)],
                        }
                    )

                fig = generator.draw_multi_line_chart(
                    unique_contexts,
                    datasets,
                    title,
                    "Context Length (tokens)",
                    "System Output Throughput (tokens/s)",
                    system_info=system_info,
                )
                return generator.save_figure_to_bytes(fig, "png")

            return None

        # ---- Other test types → unified 2×2 quad panel ----
        # Common column name resolution
        col_ttft = _resolve_col(
            df,
            "Uncached_TTFT (s)",
            "Uncached_TTFT",
            "Best_TTFT (s)",
            "Best_TTFT",
            "TTFT_Mean (s)",
        )
        col_prefill = _resolve_col(
            df,
            "Max_Prefill_Speed (tokens/s)",
            "Max_Prefill_Speed",
            "Best_Prefill_Speed",
        )
        col_sys_input = _resolve_col(
            df, "Max_System_Input_Throughput (tokens/s)", "Max_System_Input_Throughput"
        )
        col_sys_output = _resolve_col(
            df,
            "Max_System_Output_Throughput (tokens/s)",
            "Max_System_Output_Throughput",
            "Max_TPS (tokens/s)",
            "Max_TPS",
        )
        col_tpot = _resolve_col(df, "TPOT_Mean (ms)", "TPOT_Mean")
        col_cache_rate = _resolve_col(df, "Cache_Hit_Rate (%)", "Cache_Hit_Rate")

        # Determine X-axis
        if test_type == "concurrency":
            x_col = "concurrency"
            x_label = "Concurrency"
        elif test_type == "prefill":
            x_col = "input_tokens_target"
            x_label = "Input Length (tokens)"
        elif test_type in ("long_context", "segmented"):
            x_col = "context_length_target"
            x_label = "Context Length (tokens)"
        else:
            return None

        if x_col not in df.columns:
            return None

        df_sorted = df.sort_values(x_col)
        x_data = df_sorted[x_col].tolist()

        # Build 4-panel config dynamically
        chart_ttft = None
        if col_ttft:
            chart_ttft = {
                "y_data": _safe_col_list(df_sorted, col_ttft),
                "title": "Time To First Token (TTFT)",
                "y_label": "TTFT (s)",
                "color": TEAL,
            }

        chart_tpot = None
        if col_tpot:
            chart_tpot = {
                "y_data": _safe_col_list(df_sorted, col_tpot),
                "title": "Average Time Per Output Token (TPOT)",
                "y_label": "TPOT (ms)",
                "color": AMBER,
            }

        bottom_charts = []

        if test_type == "segmented" and col_cache_rate:
            bottom_charts.append(
                {
                    "y_data": _safe_col_list(df_sorted, col_cache_rate),
                    "title": "Cache Hit Rate",
                    "y_label": "Cache Hit Rate (%)",
                    "color": BLUE,
                }
            )

        # Add either Prefill Speed or System Input Throughput to avoid duplication
        if test_type in ("prefill", "segmented"):
            if col_prefill:
                bottom_charts.append(
                    {
                        "y_data": _safe_col_list(df_sorted, col_prefill),
                        "title": "Prefill Speed",
                        "y_label": "Prefill Speed (tokens/s)",
                        "color": TEAL,
                    }
                )
            elif col_sys_input:
                bottom_charts.append(
                    {
                        "y_data": _safe_col_list(df_sorted, col_sys_input),
                        "title": "System Input Throughput",
                        "y_label": "Input Throughput (tokens/s)",
                        "color": TEAL,
                    }
                )
        else:
            if col_sys_input:
                bottom_charts.append(
                    {
                        "y_data": _safe_col_list(df_sorted, col_sys_input),
                        "title": "System Input Throughput",
                        "y_label": "Input Throughput (tokens/s)",
                        "color": TEAL,
                    }
                )
            elif col_prefill:
                bottom_charts.append(
                    {
                        "y_data": _safe_col_list(df_sorted, col_prefill),
                        "title": "Prefill Speed",
                        "y_label": "Prefill Speed (tokens/s)",
                        "color": TEAL,
                    }
                )

        # Add Output Throughput
        if col_sys_output:
            bottom_charts.append(
                {
                    "y_data": _safe_col_list(df_sorted, col_sys_output),
                    "title": "System Output Throughput",
                    "y_label": "Output Throughput (tokens/s)",
                    "color": AMBER,
                }
            )

        charts = [c for c in [chart_ttft, chart_tpot] + bottom_charts if c is not None]

        # Max 4 panels
        charts = charts[:4]

        # Check if at least 1 valid panel exists
        if len(charts) == 0:
            return None

        fig = generator.draw_quad_chart(x_data, charts, title, x_label, system_info=system_info)
        return generator.save_figure_to_bytes(fig, "png")

    except Exception as e:
        st.error(f"Summary chart export failed: {e}")
        return None
