"""
Export functionality for benchmark results.

Supports export to Excel, HTML, enhanced Markdown formats, and static PNG charts.
"""
import base64
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def export_to_excel(df_dict, filename='benchmark_results.xlsx'):
    """
    Export DataFrames to Excel with professional formatting.

    Args:
        df_dict: Dictionary of {sheet_name: dataframe}
        filename: Output filename

    Returns:
        BytesIO object containing Excel file
    """
    try:
        output = BytesIO()

        # Try xlsxwriter first (better formatting)
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book

                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#007bff',
                    'font_color': 'white',
                    'align': 'center',
                    'valign': 'vcenter',
                    'border': 1,
                    'font_size': 11
                })

                workbook.add_format({
                    'align': 'center',
                    'valign': 'vcenter',
                    'border': 1,
                    'font_size': 10
                })

                number_format = workbook.add_format({
                    'align': 'center',
                    'valign': 'vcenter',
                    'border': 1,
                    'num_format': '0.00',
                    'font_size': 10
                })

                for sheet_name, df in df_dict.items():
                    # Write data
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False, startrow=1, header=False)

                    worksheet = writer.sheets[sheet_name[:31]]

                    # Write header
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)

                    # Set column widths
                    for i, col in enumerate(df.columns):
                        # Calculate max width
                        max_len = max(
                            df[col].astype(str).map(len).max() if len(df) > 0 else 10,
                            len(str(col))
                        ) + 2
                        worksheet.set_column(i, i, min(max_len, 30))

                    # Apply number format to numeric columns
                    for i, col in enumerate(df.columns):
                        if pd.api.types.is_numeric_dtype(df[col]):
                            worksheet.set_column(i, i, None, number_format)

        except ImportError:
            # Fallback to openpyxl
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in df_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        output.seek(0)
        return output

    except Exception as e:
        st.error(f"Excel Export failed: {e}")
        return None


def create_excel_download_link(df_dict, filename='benchmark_results.xlsx', link_text='📗 Download Excel Report'):
    """
    Create Streamlit download link for Excel file.

    Args:
        df_dict: Dictionary of {sheet_name: dataframe}
        filename: Download filename
        link_text: Link button text

    Returns:
        HTML download link
    """
    excel_file = export_to_excel(df_dict, filename)

    if excel_file:
        b64 = base64.b64encode(excel_file.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-btn">{link_text}</a>'
        return href
    return ''


def export_interactive_html(figures_list, tables_list, insights_list=None, title="LLM Benchmark Report"):
    """
    Export interactive HTML report with all charts and tables.

    Args:
        figures_list: List of plotly figures
        tables_list: List of styled DataFrames
        insights_list: Optional list of insight strings
        title: Report title

    Returns:
        HTML string
    """
    html_parts = []

    # HTML header
    html_parts.append(f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
                padding: 20px;
            }}

            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}

            h1 {{
                color: #007bff;
                border-bottom: 3px solid #007bff;
                padding-bottom: 15px;
                margin-bottom: 30px;
                font-size: 32px;
            }}

            h2 {{
                color: #333;
                margin-top: 40px;
                margin-bottom: 20px;
                font-size: 24px;
            }}

            .chart-container {{
                margin: 30px 0;
                padding: 20px;
                background-color: #fafafa;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}

            th {{
                background-color: #007bff;
                color: white;
                padding: 12px;
                text-align: center;
                font-weight: 600;
                font-size: 13px;
            }}

            td {{
                padding: 10px;
                border: 1px solid #e6e9ef;
                text-align: center;
                font-size: 12px;
            }}

            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}

            tr:hover {{
                background-color: #f1f8ff;
            }}

            .insights {{
                background-color: #e7f3ff;
                border-left: 4px solid #007bff;
                padding: 20px;
                margin: 30px 0;
                border-radius: 4px;
            }}

            .insights h3 {{
                color: #007bff;
                margin-bottom: 15px;
            }}

            .insights ul {{
                list-style-type: none;
                padding-left: 0;
            }}

            .insights li {{
                padding: 8px 0;
                border-bottom: 1px solid #cce5ff;
            }}

            .insights li:last-child {{
                border-bottom: none;
            }}

            .timestamp {{
                text-align: right;
                color: #666;
                font-size: 14px;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e6e9ef;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
    """)

    # Add insights if provided
    if insights_list and len(insights_list) > 0:
        html_parts.append("""
            <div class="insights">
                <h3>📊 Performance Insights</h3>
                <ul>
        """)
        for insight in insights_list:
            # Remove markdown formatting for HTML
            clean_insight = insight.replace('**', '').replace('*', '')
            html_parts.append(f"<li>{clean_insight}</li>")
        html_parts.append("</ul></div>")

    # Add charts
    for i, fig in enumerate(figures_list):
        if fig is not None:
            html_parts.append(f'<h2>Chart {i+1}</h2>')
            # Use 'cdn' for the first chart to include the library, 'False' for others to reuse it
            include_js = 'cdn' if i == 0 else False
            html_parts.append(fig.to_html(include_plotlyjs=include_js, full_html=False, div_id=f"chart{i}"))

    # Add tables
    for i, table in enumerate(tables_list):
        if table is not None:
            html_parts.append(f'<h2>Data Table {i+1}</h2>')
            if hasattr(table, 'to_html'):
                html_parts.append(table.to_html(index=False, escape=False))
            else:
                html_parts.append(table)

    # Add timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_parts.append(f'<div class="timestamp">Generated: {timestamp}</div>')

    # Close HTML
    html_parts.append("""
        </div>
    </body>
    </html>
    """)

    return ''.join(html_parts)


def create_html_download_link(html_content, filename='benchmark_report.html', link_text='🌐 Download HTML Report'):
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
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" class="download-btn">{link_text}</a>'
    return href


def export_enhanced_markdown(df, insights=None, charts_description=""):
    """
    Export enhanced markdown with tables and insights.

    Args:
        df: Results DataFrame
        insights: List of insights
        charts_description: Description of charts

    Returns:
        Markdown string
    """
    md_parts = []

    # Title
    md_parts.append("# LLM Performance Benchmark Report\n\n")

    # Insights
    if insights:
        md_parts.append("## 📊 Performance Insights\n\n")
        for insight in insights:
            md_parts.append(f"- {insight}\n")
        md_parts.append("\n")

    # Data table
    md_parts.append("## 📋 Results Summary\n\n")
    md_parts.append(df.to_markdown(index=False))
    md_parts.append("\n\n")

    # Charts description
    if charts_description:
        md_parts.append("## 📈 Charts\n\n")
        md_parts.append(charts_description)
        md_parts.append("\n")

    # Timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md_parts.append(f"\n---\n*Generated: {timestamp}*\n")

    return ''.join(md_parts)


# =====================================================================
# Static Chart Export (inspired by llm-performance-test.html style)
# =====================================================================

def export_static_performance_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str = '#4bc0c0',
    filename: str = 'performance_chart.png'
) -> BytesIO | None:
    """
    Export static performance chart (PNG format)

    Uses elegant line chart style with data point labels and shadow effects.

    Args:
        df: DataFrame containing the data
        x_col: X-axis column name
        y_col: Y-axis column name
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        color: Line color (Default teal)
        filename: Output filename

    Returns:
        BytesIO object containing PNG image data
    """
    try:
        from ui.static_chart_generator import StaticChartGenerator

        generator = StaticChartGenerator(dpi=150)
        x_data = df[x_col].tolist()
        y_data = df[y_col].tolist()

        fig = generator.draw_line_chart(x_data, y_data, title, x_label, y_label, color)
        return generator.save_figure_to_bytes(fig, 'png')

    except Exception as e:
        st.error(f"Static chart export failed: {e}")
        return None


def export_prefill_decode_report(
    test_results: list[dict[str, Any]],
    system_info: dict[str, str],
    test_time: datetime | None = None,
    filename: str = 'llm_performance_report.png'
) -> BytesIO | None:
    """
    Export prefill and output speed dual-chart report (similar to llm-performance-test.html style)

    Args:
        test_results: Test results list, each item contains:
            - input_length: Input length
            - prefill_speed: Prefill speed (token/s)
            - output_speed: Output speed (token/s)
        system_info: System info dict, containing:
            - processor: Processor
            - mainboard: Mainboard
            - memory: Memory
            - gpu: GPU
            - system: Operating System
            - engine_name: Inference engine name
            - model_name: Model name
        test_time: Test time (optional, Defaults to current time)
        filename: Output filename

    Returns:
        BytesIO object containing PNG image data
    """
    try:
        from ui.static_chart_generator import StaticChartGenerator

        generator = StaticChartGenerator(dpi=150)
        fig = generator.create_performance_report_image(test_results, system_info, test_time)
        return generator.save_figure_to_bytes(fig, 'png')

    except Exception as e:
        st.error(f"Performance report chart export failed: {e}")
        return None


def create_static_chart_download_link(
    chart_bytes: BytesIO,
    filename: str = 'chart.png',
    link_text: str = '📷 Download Static Chart'
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
    return ''


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first existing column name in DataFrame, or None if none exist"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_col_list(df: pd.DataFrame, col: str | None) -> list[float]:
    """Safely extract column data, convert to float list"""
    if col is None or col not in df.columns:
        return []
    return [float(v) if v is not None else 0.0 for v in df[col].tolist()]


def export_benchmark_summary_chart(
    df: pd.DataFrame,
    test_type: str,
    model_id: str,
    provider: str,
    system_info: dict[str, str] | None = None
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
            'concurrency': 'Concurrency Test',
            'prefill': 'Prefill Stress Test',
            'long_context': 'Long Context Test',
            'matrix': 'Concurrency-Context Matrix Test',
            'segmented': 'Segmented Context Test (Prefix Caching)',
        }
        base_title = type_titles.get(test_type, 'Performance Test')

        # Calculate input/output token info and concurrency for title
        io_label = ""
        concurrency_label = ""

        if test_type == 'concurrency':
            avg_in_col = _resolve_col(df, 'Actual_Tokens_Mean', 'Avg_Input_Tokens')
            max_out_col = _resolve_col(df, 'Actual_Decode_Max', 'Max_Output_Tokens')
            if avg_in_col and avg_in_col in df.columns:
                avg_in = int(df[avg_in_col].mean())
            else:
                avg_in = 0
            if max_out_col and max_out_col in df.columns:
                max_out = int(df[max_out_col].max())
            else:
                max_out = 0
            if avg_in > 0 or max_out > 0:
                io_label = f" (In: ~{avg_in}, Out: ~{max_out})"

        elif test_type in ('long_context', 'prefill', 'segmented'):
            # For single-concurrency tests, show concurrency=1 if available
            if 'concurrency' in df.columns:
                conc_values = df['concurrency'].unique()
                if len(conc_values) == 1:
                    concurrency_label = f" (Concurrency: {int(conc_values[0])})"

        # Combine labels for title
        full_label = io_label + concurrency_label

        if system_info:
            title = base_title + full_label
        else:
            title = f'{base_title}{full_label}\nModel: {model_id} | Provider: {provider}'

        # ---- matrix test → multi-line chart (special handling, has concurrency dimension) ----
        if test_type == 'matrix':
            x_col = 'context_length_target'
            y_col = _resolve_col(df, 'Max_System_Output_Throughput',
                                 'Max_System_Output_Throughput (tokens/s)')

            if x_col not in df.columns or y_col is None:
                return None

            unique_contexts = sorted(df[x_col].unique())

            if 'concurrency' in df.columns:
                unique_concurrencies = sorted(df['concurrency'].unique())
                palette = ['#4bc0c0', '#ff6384', '#36a2eb', '#ff9f40',
                           '#9966ff', '#4bc07a', '#ffcd56', '#c9cbcf']

                datasets = []
                for i, conc in enumerate(unique_concurrencies):
                    subset = df[df['concurrency'] == conc]
                    data_points = []
                    for ctx in unique_contexts:
                        val = subset[subset[x_col] == ctx][y_col].values
                        data_points.append(float(val[0]) if len(val) > 0 else 0.0)

                    datasets.append({
                        'label': f'Concurrency {int(conc)}',
                        'data': data_points,
                        'color': palette[i % len(palette)]
                    })

                fig = generator.draw_multi_line_chart(
                    unique_contexts, datasets, title,
                    'Context Length (tokens)', 'System Output Throughput (tokens/s)',
                    system_info=system_info
                )
                return generator.save_figure_to_bytes(fig, 'png')

            return None

        # ---- Other test types → unified 2×2 quad panel ----
        # Common column name resolution
        col_ttft = _resolve_col(df, 'Uncached_TTFT (s)', 'Uncached_TTFT', 'Best_TTFT (s)', 'Best_TTFT', 'TTFT_Mean (s)')
        col_prefill = _resolve_col(df, 'Max_Prefill_Speed (tokens/s)', 'Max_Prefill_Speed',
                                   'Best_Prefill_Speed')
        col_sys_input = _resolve_col(df, 'Max_System_Input_Throughput (tokens/s)',
                                     'Max_System_Input_Throughput')
        col_sys_output = _resolve_col(df, 'Max_System_Output_Throughput (tokens/s)',
                                      'Max_System_Output_Throughput', 'Max_TPS (tokens/s)', 'Max_TPS')
        col_tpot = _resolve_col(df, 'TPOT_Mean (ms)', 'TPOT_Mean')
        col_cache_rate = _resolve_col(df, 'Cache_Hit_Rate (%)', 'Cache_Hit_Rate')

        # Determine X-axis
        if test_type == 'concurrency':
            x_col = 'concurrency'
            x_label = 'Concurrency'
        elif test_type == 'prefill':
            x_col = 'input_tokens_target'
            x_label = 'Input Length (tokens)'
        elif test_type in ('long_context', 'segmented'):
            x_col = 'context_length_target'
            x_label = 'Context Length (tokens)'
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
                'y_data': _safe_col_list(df_sorted, col_ttft),
                'title': 'Time To First Token (TTFT)',
                'y_label': 'TTFT (s)',
                'color': '#4bc0c0',  # Teal - Input/Prefill related
            }

        chart_tpot = None
        if col_tpot:
            chart_tpot = {
                'y_data': _safe_col_list(df_sorted, col_tpot),
                'title': 'Average Time Per Output Token (TPOT)',
                'y_label': 'TPOT (ms)',
                'color': '#ff9f40',
            }

        bottom_charts = []

        if test_type == 'segmented' and col_cache_rate:
            bottom_charts.append({
                'y_data': _safe_col_list(df_sorted, col_cache_rate),
                'title': 'Cache Hit Rate',
                'y_label': 'Cache Hit Rate (%)',
                'color': '#36a2eb',
            })

        # Add either Prefill Speed or System Input Throughput to avoid duplication
        if test_type in ('prefill', 'segmented'):
            if col_prefill:
                bottom_charts.append({
                    'y_data': _safe_col_list(df_sorted, col_prefill),
                    'title': 'Prefill Speed',
                    'y_label': 'Prefill Speed (tokens/s)',
                    'color': '#4bc0c0',
                })
            elif col_sys_input:
                bottom_charts.append({
                    'y_data': _safe_col_list(df_sorted, col_sys_input),
                    'title': 'System Input Throughput',
                    'y_label': 'Input Throughput (tokens/s)',
                    'color': '#4bc0c0',
                })
        else:
            if col_sys_input:
                bottom_charts.append({
                    'y_data': _safe_col_list(df_sorted, col_sys_input),
                    'title': 'System Input Throughput',
                    'y_label': 'Input Throughput (tokens/s)',
                    'color': '#4bc0c0',
                })
            elif col_prefill:
                bottom_charts.append({
                    'y_data': _safe_col_list(df_sorted, col_prefill),
                    'title': 'Prefill Speed',
                    'y_label': 'Prefill Speed (tokens/s)',
                    'color': '#4bc0c0',
                })

        # Add Output Throughput
        if col_sys_output:
            bottom_charts.append({
                'y_data': _safe_col_list(df_sorted, col_sys_output),
                'title': 'System Output Throughput',
                'y_label': 'Output Throughput (tokens/s)',
                'color': '#ff9f40',  # Orange - Output/Decode related
            })

        charts = [c for c in [chart_ttft, chart_tpot] + bottom_charts if c is not None]
        
        # Max 4 panels
        charts = charts[:4]

        # Check if at least 1 valid panel exists
        if len(charts) == 0:
            return None

        fig = generator.draw_quad_chart(
            x_data, charts, title, x_label,
            system_info=system_info
        )
        return generator.save_figure_to_bytes(fig, 'png')

    except Exception as e:
        st.error(f"Summary chart export failed: {e}")
        return None

