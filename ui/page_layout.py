"""
Page Layout Module

Provides page layout and navigation, including:
- CSS style definitions
- Page header
- Result display area
- Report display
"""

import streamlit as st

from config.test_types import normalize_test_type, test_type_label
from ui import reports
from ui.design_system import apply_design_system, material_icon


def apply_custom_css():
    """Apply the shared application design system for standalone renders."""

    apply_design_system()


# =====================================================================
# Unified Raw Data Display Configuration
# =====================================================================
from ui.formatters import format_results_for_display


def _detect_test_type_from_df(df):
    """Infer the actual test type from a result DataFrame.

    Returns the stable test-type ID or None if undetectable.
    """
    if df is None or df.empty:
        return None

    cols = set(df.columns)

    # 1. Explicit test_type column takes priority
    if "test_type" in cols:
        unique_types = df["test_type"].dropna().unique()
        if len(unique_types) == 1:
            raw_type = str(unique_types[0]).strip().lower()
            normalized = normalize_test_type(raw_type)
            if normalized:
                return normalized

    # 2. Fallback: detect from column heuristics
    if "cumulative_mode" in cols or ("effective_prefill_tokens" in cols and "cache_hit_source" in cols):
        return "segmented"
    if "context_length_target" in cols and "concurrency" in cols:
        return "matrix"
    if "input_tokens_target" in cols:
        return "prefill"
    if "context_length_target" in cols:
        return "long_context"
    if "timestamp" in cols and "round" not in cols and "concurrency" in cols:
        return "stability"
    if "concurrency" in cols:
        return "concurrency"

    return None


def render_results_section(test_type=None):
    """Render results display area

    Args:
        test_type: Test type, used to determine column display and formatting
    """
    if not st.session_state.results_df.empty:
        st.markdown("---")
        st.header("Test results")

        raw_df = st.session_state.results_df

        # Prefer the actual test type stored in the data so column ordering
        # stays correct even when the user switches the sidebar selector.
        inferred_type = _detect_test_type_from_df(raw_df)
        display_type = inferred_type or test_type

        # Unified formatted display
        display_df = format_results_for_display(raw_df, test_type_label(display_type))

        # Display formatted dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )

        # Expand to view raw data
        with st.expander("View full raw data", expanded=False):
            st.dataframe(
                raw_df,
                use_container_width=True,
                height=300
            )

        # Download button (always downloads full raw data)
        csv = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results CSV",
            data=csv,
            file_name=f"results_{st.session_state.get('current_csv_file', 'results')}",
            mime='text/csv',
            icon=material_icon("download"),
        )

        # 数据仓库富信息：指纹卡 / 资源时序 / 等效带宽偏差 / 可对外闸门 / markdown 报告
        try:
            from ui.warehouse_report import render_warehouse_panel
            render_warehouse_panel(display_type, st.session_state.get("current_model_id", ""))
        except Exception:
            pass


def render_report_section(test_type):
    """
    Render report display area

    Args:
        test_type: Current test type selected in the sidebar
    """
    if not st.session_state.results_df.empty:
        st.markdown("---")
        st.header("Test report")

        # Extract context. After a browser refresh, results can be restored from
        # disk while sidebar widgets return their defaults, so prefer restored
        # result metadata when available.
        restored_context = (
            st.session_state.get('restored_result_context', {})
            if st.session_state.get('restored_from_csv')
            else {}
        )
        model_id = restored_context.get('model_id') or st.session_state.get('current_model_id', 'Unknown')
        provider = restored_context.get('provider') or st.session_state.get('current_provider', 'Unknown')
        duration = restored_context.get('duration', st.session_state.get('test_duration', 0))
        test_config = restored_context.get('test_config') or st.session_state.get('test_config', {})
        system_info = restored_context.get('system_info') or st.session_state.get('system_info', {})
        test_type = normalize_test_type(restored_context.get('test_type') or test_type)

        # Infer the *actual* test type from the data itself so that switching
        # the sidebar selector does not try to render e.g. a Prefill report
        # using leftover Concurrency data (which causes missing-column errors).
        inferred_type = _detect_test_type_from_df(st.session_state.results_df)
        report_type = inferred_type or test_type

        # Warn the user when the displayed report type differs from the
        # currently selected sidebar option.
        if inferred_type and inferred_type != test_type:
            st.info(
                f"Displaying results from previous **{test_type_label(inferred_type)}**. "
                f"Switch back to that test type to run a new test."
            )

        internal_type = normalize_test_type(report_type)

        # Auto-detect test type and generate corresponding report
        if internal_type == "concurrency":
            st.session_state.report = reports.generate_concurrency_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        elif internal_type == "prefill":
            st.session_state.report = reports.generate_prefill_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        elif internal_type == "long_context":
            st.session_state.report = reports.generate_long_context_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        elif internal_type == "matrix":
            st.session_state.report = reports.generate_matrix_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        elif internal_type == "segmented":
            st.session_state.report = reports.generate_segmented_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        else:
            st.session_state.report = (
                f"# Test Completed ({test_type_label(report_type)})\n\n"
                "Please refer to the data table above for results."
            )

        # Display report
        st.markdown(st.session_state.report)

        # Download report
        st.download_button(
            label="Download Markdown report",
            data=st.session_state.report,
            file_name=f"report_{st.session_state.get('current_csv_file', 'report')}.md",
            mime='text/markdown',
            icon=material_icon("download"),
        )


def render_log_section():
    """Render log viewer area"""
    if st.session_state.get('logger') and st.session_state.logger.entries:
        st.markdown("---")
        st.header("Log viewer")

        if st.button("Open log viewer", icon=material_icon("open_in_new")):
            from ui.log_viewer import render_log_viewer
            render_log_viewer(st.session_state.logger)


class PageLayout:
    """Page layout class"""

    @staticmethod
    def show_empty_state():
        """Render empty state prompt"""
        st.info(
            "Choose a test type, configure its parameters, and run the benchmark. "
            "Results, reports, and request logs will appear here when the run completes."
        )

    @staticmethod
    def render(test_type):
        """
        Main page layout render

        Args:
            test_type: Current test type
        """
        # Apply styles
        apply_custom_css()

        # Render report section (statistical results) - render first
        render_report_section(test_type)

        # Render results section (raw data) - render after
        render_results_section(test_type)

        # Render log section
        render_log_section()


# Backward compatibility: Create default instance
main_page_layout = PageLayout()
