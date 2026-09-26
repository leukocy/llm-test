"""
Page Layout Module

Provides page layout and navigation, including:
- CSS style definitions
- Page header
- Result display area
- Report display
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui import reports
from ui.export import safe_csv_bytes
from ui.reporting.presentation import render_scientific_panel
from ui.reporting.statistics import analysis_to_markdown, build_scientific_summary
from utils.spreadsheet import safe_download_filename


def apply_custom_css():
    """Apply custom CSS styles"""
    st.markdown(
        """
    <style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4a9eff;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #1e2129;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #30363d;
    }

    .result-table {
        background: #1e2129;
        border-radius: 0.5rem;
        padding: 1rem;
    }

    /* Scrollbar styles */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }

    /* === Semantic design system (replaces emoji) === */
    /* CSS custom properties — single source of truth for status colors */
    :root {
        --ui-success: #28a745;
        --ui-info:    #17a2b8;
        --ui-warning: #ffc107;
        --ui-danger:  #dc3545;
        --ui-muted:   #6c757d;
        --ui-accent:  #4a9eff;
    }

    /* Inline SVG icons inherit text color by default */
    .ui-icon {
        display: inline-block;
        vertical-align: middle;
        flex-shrink: 0;
    }

    /* Semantic badges for status indicators (success/info/warning/danger/muted) */
    .ui-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.85em;
        font-weight: 600;
        line-height: 1.4;
        white-space: nowrap;
        border: 1px solid transparent;
    }
    .ui-badge-success { background: rgba(40,167,69,.15);  color: var(--ui-success); border-color: rgba(40,167,69,.35); }
    .ui-badge-info    { background: rgba(23,162,184,.15); color: var(--ui-info);    border-color: rgba(23,162,184,.35); }
    .ui-badge-warning { background: rgba(255,193,7,.15);  color: var(--ui-warning); border-color: rgba(255,193,7,.35); }
    .ui-badge-danger  { background: rgba(220,53,69,.15);  color: var(--ui-danger);  border-color: rgba(220,53,69,.35); }
    .ui-badge-muted   { background: rgba(108,117,125,.15); color: var(--ui-muted);  border-color: rgba(108,117,125,.35); }

    /* Section titles with a colored accent bar and optional leading icon */
    .ui-section-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 700;
        margin: 1rem 0 0.5rem;
    }
    .ui-section-title .ui-icon { color: var(--ui-accent); }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_page_header():
    """Render page header"""
    st.title("LLM Performance Benchmark Platform")


# =====================================================================
# Unified Raw Data Display Configuration
# =====================================================================
from ui.formatters import format_results_for_display

# Internal raw test_type values (stored in results_df) → UI display names
_INTERNAL_TO_DISPLAY = {
    "concurrency": "Concurrency Test",
    "prefill": "Prefill Stress Test",
    "long_context": "Long Context Test",
    "matrix": "Concurrency-Context Matrix Test",
    "segmented": "Segmented Context Test",
    "custom_text": "Custom Text Test",
    "stability": "Stability Test",
    "all": "All Tests",
    "dataset": "Model Quality Test",
}


# Reverse mapping for quick lookup
_DISPLAY_TO_INTERNAL = {v: k for k, v in _INTERNAL_TO_DISPLAY.items()}


def _detect_test_type_from_df(df):
    """Infer the actual test type from a result DataFrame.

    Returns the UI display name (e.g. 'Concurrency Test') or None if undetectable.
    """
    if df is None or df.empty:
        return None

    cols = set(df.columns)

    # 1. Explicit test_type column takes priority
    if "test_type" in cols:
        unique_types = df["test_type"].dropna().unique()
        if len(unique_types) == 1:
            internal = str(unique_types[0]).strip().lower()
            display = _INTERNAL_TO_DISPLAY.get(internal)
            if display:
                return display

    # 2. Fallback: detect from column heuristics
    if "cumulative_mode" in cols or (
        "effective_prefill_tokens" in cols and "cache_hit_source" in cols
    ):
        return "Segmented Context Test"
    if "context_length_target" in cols and "concurrency" in cols:
        return "Concurrency-Context Matrix Test"
    # Stability must be checked before Prefill: stability CSV rows always
    # carry "input_tokens_target" (it is in the fixed csv_columns list),
    # which would otherwise be misdetected as a Prefill Stress Test.
    if "timestamp" in cols and "round" not in cols and "concurrency" in cols:
        return "Stability Test"
    if "input_tokens_target" in cols and "concurrency" not in cols:
        return "Prefill Stress Test"
    if "context_length_target" in cols:
        return "Long Context Test"
    if "concurrency" in cols:
        return "Concurrency Test"

    return None


def filter_result_rows(df: pd.DataFrame, status: str = "All", query: str = "") -> pd.DataFrame:
    """Filter the explorer only; reports and downloads retain the complete run."""
    from core.result_metrics import success_mask_from_error

    filtered = df
    if "error" in filtered.columns and status in ("Succeeded", "Failed"):
        success = success_mask_from_error(filtered["error"])
        filtered = filtered.loc[success if status == "Succeeded" else ~success]
    query = query.strip()
    if query:
        search_columns = [
            column
            for column in ("session_id", "prompt_source", "test_type", "error", "round")
            if column in filtered.columns
        ]
        if search_columns:
            haystack = filtered[search_columns].fillna("").astype(str).agg(" ".join, axis=1)
            filtered = filtered.loc[haystack.str.contains(query, case=False, regex=False, na=False)]
    return filtered


def render_results_section(test_type=None):
    """Render results display area

    Args:
        test_type: Test type, used to determine column display and formatting
    """
    results_df = st.session_state.get("results_df")
    if results_df is not None and not results_df.empty:
        st.markdown("---")
        st.header("Request explorer")

        raw_df = results_df

        # Prefer the actual test type stored in the data so column ordering
        # stays correct even when the user switches the sidebar selector.
        inferred_type = _detect_test_type_from_df(raw_df)
        display_type = inferred_type or test_type

        filter_col, search_col = st.columns([1, 2])
        with filter_col:
            status = st.radio(
                "Request status",
                ["All", "Succeeded", "Failed"],
                horizontal=True,
                disabled="error" not in raw_df.columns,
                key="request_explorer_status",
            )
        with search_col:
            query = st.text_input(
                "Search identifiers and errors",
                placeholder="Session, source, test type, error or round",
                key="request_explorer_search",
            )
        visible_df = filter_result_rows(raw_df, status=status, query=query)
        st.caption(
            f"Showing {len(visible_df):,} of {len(raw_df):,} recorded rows. Report statistics and downloads use the complete run."
        )

        # Unified formatted display
        display_df = format_results_for_display(visible_df, display_type)

        # Display formatted dataframe
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=460)

        # Expand to view raw data
        with st.expander("Raw records and provenance", expanded=False):
            st.dataframe(raw_df, use_container_width=True, hide_index=True, height=340)

        # Download button (always downloads full raw data)
        csv = safe_csv_bytes(raw_df)
        csv_name = safe_download_filename(
            Path(str(st.session_state.get("current_csv_file") or "results.csv")).name
        )
        if not csv_name.lower().endswith(".csv"):
            csv_name += ".csv"
        st.download_button(
            label="Download complete CSV",
            data=csv,
            file_name=csv_name,
            mime="text/csv",
        )

        # 数据仓库富信息：指纹卡 / 资源时序 / 等效带宽偏差 / 可对外闸门 / markdown 报告
        try:
            from ui.warehouse_report import render_warehouse_panel

            render_warehouse_panel(display_type, st.session_state.get("current_model_id", ""))
        except Exception as e:
            st.warning(f"Warehouse panel unavailable: {e}")


def render_report_section(test_type):
    """
    Render report display area

    Args:
        test_type: Current test type selected in the sidebar
    """
    results_df = st.session_state.get("results_df")
    if results_df is None or results_df.empty:
        return

    st.markdown("---")
    st.header("Test Report")

    # Extract context. After a browser refresh, results can be restored from
    # disk while sidebar widgets return their defaults, so prefer restored
    # result metadata when available.
    restored_context = (
        st.session_state.get("restored_result_context", {})
        if st.session_state.get("restored_from_csv")
        else {}
    )
    model_id = restored_context.get("model_id") or st.session_state.get(
        "current_model_id", "Unknown"
    )
    provider = restored_context.get("provider") or st.session_state.get(
        "current_provider", "Unknown"
    )
    duration = restored_context.get("duration", st.session_state.get("test_duration", 0))
    test_config = restored_context.get("test_config") or st.session_state.get("test_config", {})
    system_info = restored_context.get("system_info") or st.session_state.get("system_info", {})
    test_type = restored_context.get("test_type") or test_type

    # Infer the *actual* test type from the data itself so that switching
    # the sidebar selector does not try to render e.g. a Prefill report
    # using leftover Concurrency data (which causes missing-column errors).
    inferred_type = _detect_test_type_from_df(results_df)
    report_type = inferred_type or test_type

    analysis_type = _DISPLAY_TO_INTERNAL.get(report_type, str(report_type).strip().lower())
    try:
        analysis = build_scientific_summary(results_df, analysis_type)
    except ValueError as error:
        st.error(f"Report statistics unavailable: {error}")
        return
    render_scientific_panel(analysis)
    st.subheader("Observed extrema and request details")
    st.caption(
        "The charts below show observed best or maximum values; use the distribution and sample counts above for comparisons."
    )

    # Warn the user when the displayed report type differs from the
    # currently selected sidebar option.
    if inferred_type and inferred_type != test_type:
        st.info(
            f"Displaying results from previous **{inferred_type}**. "
            f"Switch back to that test type to run a new test."
        )

    # Report generators accept both Chinese internal names and English display names.
    internal_type = report_type

    # Auto-detect test type and generate corresponding report
    if internal_type in ("并发性能测试", "并发性能Test", "Concurrency Test"):
        st.session_state.report = reports.generate_concurrency_report(
            results_df,
            model_id=model_id,
            provider=provider,
            duration=duration,
            test_config=test_config,
            system_info=system_info,
        )
    elif internal_type in ("Prefill 压力测试", "Prefill 压力Test", "Prefill Stress Test"):
        st.session_state.report = reports.generate_prefill_report(
            results_df,
            model_id=model_id,
            provider=provider,
            duration=duration,
            test_config=test_config,
            system_info=system_info,
        )
    elif internal_type in ("长上下文测试", "长onunder文Test", "Long Context Test"):
        st.session_state.report = reports.generate_long_context_report(
            results_df,
            model_id=model_id,
            provider=provider,
            duration=duration,
            test_config=test_config,
            system_info=system_info,
        )
    elif internal_type in (
        "并发-上下文 综合测试",
        "并发-onunder文 综合Test",
        "Concurrency-Context Matrix Test",
    ):
        st.session_state.report = reports.generate_matrix_report(
            results_df,
            model_id=model_id,
            provider=provider,
            duration=duration,
            test_config=test_config,
            system_info=system_info,
        )
    elif internal_type in ("分段上下文测试", "分段onunder文Test", "Segmented Context Test"):
        st.session_state.report = reports.generate_segmented_report(
            results_df,
            model_id=model_id,
            provider=provider,
            duration=duration,
            test_config=test_config,
            system_info=system_info,
        )
    else:
        st.session_state.report = (
            f"# Test Completed ({report_type})\n\nPlease refer to the data table above for results."
        )

    # Display and export the same statistical evidence shown above.
    if st.session_state.report:
        st.session_state.report += "\n\n" + analysis_to_markdown(analysis)
    st.markdown(st.session_state.report)

    # Download report
    st.download_button(
        label="Download Report (Markdown)",
        data=st.session_state.report,
        file_name=safe_download_filename(
            f"report_{st.session_state.get('current_csv_file', 'report')}.md"
        ),
        mime="text/markdown",
    )


def render_log_section():
    """Render log viewer area"""
    if st.session_state.get("logger") and st.session_state.logger.entries:
        st.markdown("---")
        st.header("Log Viewer")

        if st.button("Open Log Viewer"):
            from ui.log_viewer import render_log_viewer

            render_log_viewer(st.session_state.logger)


class PageLayout:
    """Page layout class"""

    @staticmethod
    def render(test_type):
        """
        Main page layout render

        Args:
            test_type: Current test type
        """
        # Apply styles
        apply_custom_css()

        # Render header
        render_page_header()

        # Render report section (statistical results) - render first
        render_report_section(test_type)

        # Render results section (raw data) - render after
        render_results_section(test_type)

        # Render log section
        render_log_section()


# Backward compatibility: Create default instance
main_page_layout = PageLayout()
