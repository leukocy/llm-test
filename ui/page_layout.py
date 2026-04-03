"""
Page Layout Module

Provides page layout and navigation, including:
- CSS style definitions
- Page header
- Result display area
- Report display
"""

import pandas as pd
import streamlit as st

from ui import reports


def apply_custom_css():
    """Apply custom CSS styles"""
    st.markdown("""
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
    </style>
    """, unsafe_allow_html=True)


def render_page_header():
    """Render page header"""
    st.title("LLM Performance Benchmark Platform")


# =====================================================================
# Unified Raw Data Display Configuration
# =====================================================================
from ui.formatters import format_results_for_display

# Test type name mapping: English → Chinese (for backward compatibility with report generators)
_TEST_TYPE_MAP = {
    "Concurrency Test": "并发性能Test",
    "Prefill Stress Test": "Prefill 压力Test",
    "Long Context Test": "长onunder文Test",
    "Concurrency-Context Matrix Test": "并发-onunder文 综合Test",
    "Segmented Context Test": "分段onunder文Test",
    "Custom Text Test": "Custom文本Test",
    "All Tests": "全部Test",
    "Stability Test": "稳定性Test",
}


def render_results_section(test_type=None):
    """Render results display area

    Args:
        test_type: Test type, used to determine column display and formatting
    """
    if not st.session_state.results_df.empty:
        st.markdown("---")
        st.header("📊 Test Results")

        raw_df = st.session_state.results_df

        # Unified formatted display
        display_df = format_results_for_display(raw_df, test_type)

        # Display formatted dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )

        # Expand to view raw data
        with st.expander("🔍 View Full Raw Data", expanded=False):
            st.dataframe(
                raw_df,
                use_container_width=True,
                height=300
            )

        # Download button (always downloads full raw data)
        csv = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results CSV (Full Data)",
            data=csv,
            file_name=f"results_{st.session_state.get('current_csv_file', 'results')}",
            mime='text/csv'
        )


def render_report_section(test_type):
    """
    Render report display area

    Args:
        test_type: Test type
    """
    if not st.session_state.results_df.empty:
        st.markdown("---")
        st.header("📄 Test Report")

        # Extract context
        model_id = st.session_state.get('current_model_id', 'Unknown')
        provider = st.session_state.get('current_provider', 'Unknown')
        duration = st.session_state.get('test_duration', 0)
        test_config = st.session_state.get('test_config', {})
        system_info = st.session_state.get('system_info', {})

        # Map test type for backward compatibility
        internal_type = _TEST_TYPE_MAP.get(test_type, test_type)

        # Auto-detect test type and generate corresponding report
        if internal_type in ("并发性能Test", "Concurrency Test"):
            st.session_state.report = reports.generate_concurrency_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        elif internal_type in ("Prefill 压力Test", "Prefill Stress Test"):
            st.session_state.report = reports.generate_prefill_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        elif internal_type in ("长onunder文Test", "Long Context Test"):
            st.session_state.report = reports.generate_long_context_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        elif internal_type in ("并发-onunder文 综合Test", "Concurrency-Context Matrix Test"):
            st.session_state.report = reports.generate_matrix_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        elif internal_type in ("分段onunder文Test", "Segmented Context Test"):
            st.session_state.report = reports.generate_segmented_report(
                st.session_state.results_df,
                model_id=model_id,
                provider=provider,
                duration=duration,
                test_config=test_config,
                system_info=system_info
            )
        else:
            st.session_state.report = f"# Test Completed ({test_type})\n\nPlease refer to the data table above for results."

        # Display report
        st.markdown(st.session_state.report)

        # Download report
        st.download_button(
            label="📥 Download Report (Markdown)",
            data=st.session_state.report,
            file_name=f"report_{st.session_state.get('current_csv_file', 'report')}.md",
            mime='text/markdown'
        )


def render_log_section():
    """Render log viewer area"""
    if st.session_state.get('logger') and st.session_state.logger.entries:
        st.markdown("---")
        st.header("📋 Log Viewer")

        if st.button("🔍 Open Log Viewer"):
            from ui.log_viewer import render_log_viewer
            render_log_viewer(st.session_state.logger)


class PageLayout:
    """Page layout class"""

    @staticmethod
    def show_empty_state():
        """Render empty state prompt"""
        st.info("""
        ### 👋 Welcome to LLM Performance Benchmark Platform V2

        Please configure test parameters in the left sidebar, then select a test type to begin.

        **Features:**
        - ⚡ Concurrency Test
        - 🔥 Prefill Stress Test
        - 📏 Long Context Test
        - 🔬 Matrix Test
        - 📄 Custom Text Test
        - 🎯 All Tests
        - ⏱️ Stability Test

        **V2 New Features:**
        - 📦 Modular Architecture
        - 🔧 Cleaner Code Organization
        - 🚀 Better Maintainability
        """)

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
