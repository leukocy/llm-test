"""
Enhanced log viewer UI component for Streamlit.

Provides interactive log viewing with filtering, analytics, and export.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.charts import apply_theme
from utils.logger import BenchmarkLogger, LogLevel


def render_log_viewer(logger: BenchmarkLogger,
                     placeholder=None,
                     max_display=100,
                     enable_filter=True,
                     compact_mode=False):
    """
    Render enhanced log viewer

    Args:
        logger: BenchmarkLogger instance
        placeholder: Streamlit placeholder (for real-time updates)
        max_display: Maximum entries to display
        enable_filter: Whether to enable filter controls
        compact_mode: Compact mode (for real-time logging)
    """
    container = placeholder if placeholder else st.container()

    with container:
        if compact_mode:
            # Compact mode: show only recent log text
            _render_compact_log(logger, max_display)
        else:
            # Full mode: statistics, filters, multiple views
            _render_full_log_viewer(logger, max_display, enable_filter)


def _render_compact_log(logger: BenchmarkLogger, max_display=50):
    """Render compact log view (for real-time updates)"""
    entries = logger.get_recent(max_display)

    if not entries:
        st.info("ℹ️ Log window initialized... waiting for test to start")
        return

    # Showing colored text log
    log_html = _render_log_text(entries, compact=True)
    st.markdown(log_html, unsafe_allow_html=True)


def _render_full_log_viewer(logger: BenchmarkLogger, max_display=100, enable_filter=True):
    """Render full log viewer"""
    # Statistics info cards
    stats = logger.get_stats()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Logs", stats['total'])
    with col2:
        error_delta = f"-{stats['errors']}" if stats['errors'] > 0 else None
        st.metric("❌ Error", stats['errors'], delta=error_delta, delta_color="inverse")
    with col3:
        warning_delta = f"-{stats['warnings']}" if stats['warnings'] > 0 else None
        st.metric("⚠️ Warning", stats['warnings'], delta=warning_delta, delta_color="inverse")
    with col4:
        success_delta = f"+{stats['success']}" if stats['success'] > 0 else None
        st.metric("✅ succeeded", stats['success'], delta=success_delta, delta_color="normal")

    # Filter controls
    filtered_entries = logger.get_recent(max_display)

    if enable_filter:
        st.markdown("---")
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

        with filter_col1:
            level_filter = st.multiselect(
                "🔍 Log Level",
                options=[level.name for level in LogLevel],
                default=None,
                key="log_level_filter",
                help="Select log levels to display"
            )

        with filter_col2:
            search_text = st.text_input(
                "🔎 Search",
                placeholder="Enter keywords to search...",
                key="log_search",
                help="Search within log messages"
            )

        with filter_col3:
            show_metrics = st.checkbox(
                "Show Metrics",
                value=True,
                key="log_show_metrics",
                help="Show performance metrics in logs"
            )

        # Apply filters
        if level_filter:
            levels = [LogLevel[name] for name in level_filter]
            filtered_entries = logger.filter(levels=levels)
            filtered_entries = filtered_entries[-max_display:]

        if search_text:
            filtered_entries = logger.filter(search_text=search_text)
            filtered_entries = filtered_entries[-max_display:]
    else:
        show_metrics = True

    # Showing logs
    if not filtered_entries:
        st.info("📭 No matching logs")
        return

    # Tab views
    tab1, tab2, tab3 = st.tabs(["📝 Text View", "📊 Table View", "📈 Statistical Analysis"])

    with tab1:
        # Text view - with colors and formatting
        with st.expander("📜 Log Content", expanded=True):
            log_html = _render_log_text(filtered_entries, compact=False, show_metrics=show_metrics)
            st.markdown(log_html, unsafe_allow_html=True)

    with tab2:
        # Table view
        df = _logs_to_dataframe(filtered_entries)
        st.dataframe(df, width="stretch", height=400)

    with tab3:
        # Statistical analysis
        _render_log_analytics(logger, filtered_entries)


def _render_log_text(entries, compact=False, show_metrics=True):
    """Render colored text log"""
    level_colors = {
        LogLevel.DEBUG: "#6c757d",
        LogLevel.INFO: "#17a2b8",
        LogLevel.SUCCESS: "#28a745",
        LogLevel.WARNING: "#ffc107",
        LogLevel.ERROR: "#dc3545",
        LogLevel.CRITICAL: "#721c24"
    }

    level_bg_colors = {
        LogLevel.DEBUG: "#f8f9fa",
        LogLevel.INFO: "#e7f3ff",
        LogLevel.SUCCESS: "#d4edda",
        LogLevel.WARNING: "#fff3cd",
        LogLevel.ERROR: "#f8d7da",
        LogLevel.CRITICAL: "#f5c6cb"
    }

    if compact:
        # Compact mode: simple list
        html_parts = ['<div style="font-family: monospace; font-size: 12px; line-height: 1.5; max-height: 400px; overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 5px;">']
    else:
        # Full mode: with borders and background
        html_parts = ['<div style="font-family: monospace; font-size: 13px; line-height: 1.8;">']

    import html
    for entry in entries:
        color = level_colors.get(entry.level, "#333")
        bg_color = level_bg_colors.get(entry.level, "#ffffff")
        text = entry.to_text(include_metrics=show_metrics)
        safe_text = html.escape(text)

        if compact:
            html_parts.append(f'<div style="color: {color}; padding: 2px 0;">{safe_text}</div>')
        else:
            html_parts.append(
                f'<div style="color: {color}; background-color: {bg_color}; '
                f'padding: 8px 12px; margin: 4px 0; border-left: 4px solid {color}; '
                f'border-radius: 4px;">{safe_text}</div>'
            )

    html_parts.append('</div>')
    return ''.join(html_parts)


def _logs_to_dataframe(entries):
    """Convert logs to DataFrame"""
    data = []
    for entry in entries:
        row = {
            'Time': entry.timestamp.strftime("%H:%M:%S.%f")[:-3],
            'Level': entry.level.name,
            'Message': entry.message,
            'Session': entry.session_id or '-',
        }

        if entry.metrics:
            if 'ttft' in entry.metrics:
                row['TTFT'] = f"{entry.metrics['ttft']:.3f}s"
            if 'tps' in entry.metrics:
                row['TPS'] = f"{entry.metrics['tps']:.2f}"
            if 'prefill_tokens' in entry.metrics:
                row['Prefill'] = entry.metrics['prefill_tokens']
            if 'decode_tokens' in entry.metrics:
                row['Decode'] = entry.metrics['decode_tokens']

        if entry.error:
            row['Error'] = entry.error

        data.append(row)

    return pd.DataFrame(data)


def _render_log_analytics(logger: BenchmarkLogger, filtered_entries=None):
    """Render log analytics"""
    stats = logger.get_stats()

    # Log Level Distribution
    st.subheader("📊 Log Level Distribution")
    level_data = pd.DataFrame([
        {'Level': level_name, 'Count': count}
        for level_name, count in stats['by_level'].items()
        if count > 0
    ])

    if not level_data.empty:
        color_map = {
            'DEBUG': '#6c757d',
            'INFO': '#17a2b8',
            'SUCCESS': '#28a745',
            'WARNING': '#ffc107',
            'ERROR': '#dc3545',
            'CRITICAL': '#721c24'
        }

        fig = px.bar(
            level_data,
            x='Level',
            y='Count',
            color='Level',
            color_discrete_map=color_map,
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        fig = apply_theme(fig)
        st.plotly_chart(fig)
    else:
        st.info("No log data yet")

    # Timeline analysis
    if len(logger.entries) > 1:
        st.subheader("⏱️ Log Timeline")

        # Count logs per minute
        timeline_data = []
        for entry in logger.entries:
            minute = entry.timestamp.strftime("%H:%M")
            timeline_data.append({'Time': minute, 'Level': entry.level.name})

        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_grouped = timeline_df.groupby(['Time', 'Level']).size().reset_index(name='Count')

            fig = px.line(
                timeline_grouped,
                x='Time',
                y='Count',
                color='Level',
                markers=True,
                title="Log Activity Timeline"
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig)

    # Error and warning details
    if stats['errors'] > 0 or stats['warnings'] > 0:
        st.subheader("⚠️ Issue Details")

        col1, col2 = st.columns(2)

        with col1:
            if stats['errors'] > 0:
                errors = logger.filter(level=LogLevel.ERROR)
                with st.expander(f"❌ Error ({len(errors)})", expanded=True):
                    for e in errors[-5:]:  # last 5
                        # Use st.code to avoid Markdown/HTML parsing errors with raw log text
                        st.code(e.to_text(include_metrics=False), language=None, wrap_lines=True)

        with col2:
            if stats['warnings'] > 0:
                warnings = logger.filter(level=LogLevel.WARNING)
                with st.expander(f"⚠️ Warning ({len(warnings)})", expanded=False):
                    for w in warnings[-5:]:
                        # Use st.code to avoid Markdown/HTML parsing errors with raw log text
                        st.code(w.to_text(include_metrics=False), language=None, wrap_lines=True)

    # Performance Metrics Summary
    entries_with_metrics = [e for e in logger.entries if e.metrics]
    if entries_with_metrics:
        st.subheader("📈 Performance Metrics Summary")

        ttfts = [e.metrics.get('ttft') for e in entries_with_metrics if e.metrics.get('ttft')]
        tpss = [e.metrics.get('tps') for e in entries_with_metrics if e.metrics.get('tps')]

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            if ttfts:
                st.metric("Average TTFT", f"{sum(ttfts)/len(ttfts):.3f}s")
                st.caption(f"Range: {min(ttfts):.3f}s - {max(ttfts):.3f}s")

        with metric_col2:
            if tpss:
                st.metric("Average TPS", f"{sum(tpss)/len(tpss):.2f}")
                st.caption(f"Range: {min(tpss):.2f} - {max(tpss):.2f}")


def create_log_download_buttons(logger: BenchmarkLogger):
    """Create log download buttons"""
    st.markdown("### 📥 Export Logs")

    col1, col2, col3 = st.columns(3)

    with col1:
        # JSON export
        json_data = logger.export_json()
        st.download_button(
            label="📦 Download JSON",
            data=json_data,
            file_name="benchmark_log.json",
            mime="application/json",
            help="Structured JSON format with all fields"
        )

    with col2:
        # Text export
        text_data = logger.export_text(include_metrics=True)
        st.download_button(
            label="📄 Download TXT",
            data=text_data,
            file_name="benchmark_log.txt",
            mime="text/plain",
            help="Readable text format"
        )

    with col3:
        # CSV export
        csv_data = logger.export_csv()
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name="benchmark_log.csv",
            mime="text/csv",
            help="CSV format, can be opened with Excel"
        )
