"""
Batch Test UI Components

Provides batch test user interface:
- Batch test configuration editor
- Batch test execution interface
- Batch test results display
- Batch test history
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.batch_test import BatchTestConfig, BatchTestItem, batch_test_manager
from ui.design_system import material_icon
from ui.reporting.statistics import wilson_interval
from ui.reporting.theme import INDIGO, INK, PAPER, SUBTLE, TEAL
from utils.spreadsheet import safe_download_filename

# ============================================================================
# Batch test configuration editor
# ============================================================================


def render_batch_config_editor():
    """Render batch test configuration editor"""
    st.subheader("Configure Batch Test")

    # Basic Information
    config_name = st.text_input(
        "Batch Test Name",
        key="batch_config_name",
        placeholder="For example: Multi-model performance comparison test",
    )

    config_desc = st.text_area(
        "Description (optional)",
        key="batch_config_desc",
        placeholder="Describe the purpose of this batch test...",
    )

    # Execution Options
    col1, col2, col3 = st.columns(3)

    with col1:
        parallel = st.checkbox(
            "Parallel Execution",
            value=False,
            help="Run multiple tests simultaneously (experimental)",
        )

    with col2:
        max_parallel = st.number_input("Max Parallel", min_value=1, max_value=10, value=2)

    with col3:
        stop_on_error = st.checkbox(
            "Stop on Error", value=False, help="Stop batch test when a test fails"
        )

    # Test Item Configuration
    st.markdown("---")
    st.markdown("### Test Item Configuration")

    # Initialize test items list
    if "batch_test_items" not in st.session_state:
        st.session_state.batch_test_items = []

    # Add test item button
    if st.button("Add Test Item", type="secondary", icon=material_icon("add")):
        st.session_state.batch_test_items.append(
            {
                "name": f"Test {len(st.session_state.batch_test_items) + 1}",
                "api_base_url": st.session_state.get("current_api_base", ""),
                "model_id": st.session_state.get("current_model_id", ""),
                "api_key": st.session_state.get("current_api_key", ""),
                "test_type": "concurrency",
                "concurrency": 1,
                "max_tokens": 512,
                "temperature": 0.0,
                "thinking_enabled": False,
                "enabled": True,
            }
        )
        st.rerun()

    # Display and edit test items
    if st.session_state.batch_test_items:
        for i, item in enumerate(st.session_state.batch_test_items):
            with st.expander(item.get("name", f"Test Item {i + 1}"), expanded=False):
                col_edit, col_del, col_move = st.columns([3, 1, 1])

                with col_edit:
                    # Edit test item
                    new_name = st.text_input(
                        "Test Name", value=item.get("name", ""), key=f"item_{i}_name"
                    )
                    new_api = st.text_input(
                        "API URL",
                        value=item.get("api_base_url", ""),
                        key=f"item_{i}_api",
                    )
                    new_model = st.text_input(
                        "Model ID",
                        value=item.get("model_id", ""),
                        key=f"item_{i}_model",
                    )
                    new_concurrency = st.number_input(
                        "Concurrency",
                        value=item.get("concurrency", 1),
                        key=f"item_{i}_concurrency",
                    )
                    new_enabled = st.checkbox(
                        "Enabled",
                        value=item.get("enabled", True),
                        key=f"item_{i}_enabled",
                    )

                    if st.button(
                        "Save Changes",
                        key=f"save_item_{i}",
                        icon=material_icon("save"),
                    ):
                        st.session_state.batch_test_items[i].update(
                            {
                                "name": new_name,
                                "api_base_url": new_api,
                                "model_id": new_model,
                                "concurrency": new_concurrency,
                                "enabled": new_enabled,
                            }
                        )
                        st.rerun()

                with col_del:
                    if st.button(
                        "Delete",
                        key=f"del_item_{i}",
                        help="Delete this test item",
                        icon=material_icon("delete"),
                    ):
                        st.session_state.batch_test_items.pop(i)
                        st.rerun()

    # Quick Add Feature
    st.markdown("---")
    st.markdown("### Quick Add")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Add from Presets**")
        # LYest saved presets for quick add here

    with col_b:
        st.markdown("**Batch Import**")
        uploaded_file = st.file_uploader(
            "Upload configuration file (JSON)",
            type=["json"],
            key="batch_import_upload",
            help="Upload batch test configuration file",
        )

        if uploaded_file:
            if st.button(
                "Import Configuration",
                key="batch_import_btn",
                icon=material_icon("upload_file"),
            ):
                try:
                    import json

                    data = json.load(uploaded_file)

                    if "items" in data:
                        # Import test items
                        for item_data in data["items"]:
                            st.session_state.batch_test_items.append(item_data)

                        st.success(f"Imported {len(data['items'])}  test items")
                        st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    # Save config button
    st.markdown("---")
    col_save, col_clear = st.columns(2)

    with col_save:
        if st.button(
            "Save Batch Test Configuration",
            type="primary",
            use_container_width=True,
            icon=material_icon("save"),
        ):
            if not config_name:
                st.error("Please enter a batch test name")
            elif not st.session_state.batch_test_items:
                st.error("Please add at least one test item")
            else:
                # Create configuration
                items = [BatchTestItem(**item) for item in st.session_state.batch_test_items]
                config = BatchTestConfig(
                    name=config_name,
                    description=config_desc,
                    items=items,
                    parallel=parallel,
                    max_parallel=max_parallel,
                    stop_on_error=stop_on_error,
                )

                if batch_test_manager.save_config(config):
                    st.success(f"Batch test configuration '{config_name}' saved")
                    st.session_state.last_saved_batch_config = config_name

    with col_clear:
        if st.button(
            "Clear All",
            use_container_width=True,
            icon=material_icon("delete_sweep"),
        ):
            st.session_state.batch_test_items = []
            st.rerun()


# ============================================================================
# Batch test execution interface
# ============================================================================


def render_batch_test_executor():
    """Render batch test execution interface"""
    st.subheader("Execute Batch Test")

    # Select saved configuration
    saved_configs = batch_test_manager.list_configs()

    if not saved_configs:
        st.info("No saved batch test configurations yet. Please create one above.")
        return

    config_names = [c["name"] for c in saved_configs]
    selected_config_name = st.selectbox(
        "Select Batch Test Configuration", config_names, key="batch_config_selector"
    )

    if not selected_config_name:
        return

    # Display configuration info
    config = batch_test_manager.load_config(selected_config_name)
    if not config:
        st.error(f"Cannot load configuration: {selected_config_name}")
        return

    # Display configuration summary
    with st.expander("Configuration Details", expanded=False):
        st.markdown(f"**Description:** {config.description or 'None'}")
        st.markdown(f"**Number of test items:** {len(config.items)}")
        st.markdown(f"**Parallel execution:** {'Yes' if config.parallel else 'No'}")
        if config.parallel:
            st.markdown(f"**Max parallel:** {config.max_parallel}")

        st.markdown("**Test Item list:**")
        for item in config.items:
            status = "Enabled" if item.enabled else "Paused"
            st.markdown(f"- **{item.name}** ({item.model_id}) · {status}")

    # Execute button
    st.markdown("---")

    col_start, col_stop = st.columns(2)

    with col_start:
        if st.button(
            "Start Batch Test",
            type="primary",
            use_container_width=True,
            icon=material_icon("play_arrow"),
        ):
            st.session_state.batch_test_running = True
            st.session_state.batch_test_config = config
            st.rerun()

    with col_stop:
        if st.button(
            "Stop Test",
            disabled=not st.session_state.get("batch_test_running", False),
            use_container_width=True,
        ):
            st.session_state.batch_test_stop_requested = True
            # 进程级取消信号：批量停止 + 中断进行中的 LLM 调用
            try:
                from core import cancel_state

                cancel_state.request_batch_stop()
                cancel_state.request_stop()
            except ImportError:
                pass
            st.toast("Stopping batch test...", icon="Stop")
            st.rerun()

    # Display progress
    if st.session_state.get("batch_test_running"):
        _run_batch_test(config)


def _run_batch_test(config: BatchTestConfig):
    """Execute the batch test synchronously (Streamlit reruns keep the page alive)."""
    import asyncio

    from core.batch_test import BatchTestScheduler

    st.markdown("---")
    st.subheader("Test Progress")

    progress_bar = st.progress(0.0)
    metrics_cols = st.columns(4)
    current_item_box = st.empty()
    log_expander = st.expander("Execution Log", expanded=False)
    log_box = log_expander.empty()
    logs: list[str] = []

    def _render_progress(progress):
        progress_bar.progress(min(progress.progress_percentage / 100, 1.0))
        metrics_cols[0].metric("Completed", f"{progress.completed_items}/{progress.total_items}")
        metrics_cols[1].metric("Failed", f"{progress.failed_items}")
        metrics_cols[2].metric("Skipped", f"{progress.skipped_items}")
        metrics_cols[3].metric("Elapsed Time", f"{progress.elapsed_time:.1f}s")
        if progress.current_item:
            current_item_box.info(f"Currently executing: {progress.current_item}")

    def _on_progress(progress):
        st.session_state.batch_test_progress = progress
        _render_progress(progress)

    def _on_log(message: str):
        logs.append(message)
        st.session_state.batch_test_logs = logs
        log_box.code("\n".join(logs[-10:]))

    scheduler = BatchTestScheduler(
        config,
        test_function=None,  # unused: scheduler runs items via BenchmarkRunner
        progress_callback=_on_progress,
        log_callback=_on_log,
    )
    st.session_state.batch_test_scheduler = scheduler

    try:
        result = asyncio.run(scheduler.run())
    except Exception as e:
        st.error(f"Batch test failed: {e}")
        return
    finally:
        st.session_state.batch_test_running = False
        st.session_state.batch_test_scheduler = None

    if result is None:
        st.warning("Batch test was stopped before producing a result.")
        return

    if batch_test_manager.save_result(result):
        st.success(
            f"Batch test finished: {result.completed_items}/{result.total_items} "
            f"completed, {result.failed_items} failed. See the Results tab."
        )
    else:
        st.warning("Batch test finished but the result could not be saved.")


# ============================================================================
# Batch test results display
# ============================================================================


def add_batch_success_intervals(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add binomial intervals only where recorded outcome counts are available."""
    display = comparison.copy()
    if "Requests" not in display or "Succeeded" not in display:
        return display
    bounds: list[tuple[float | None, float | None]] = []
    for total, succeeded in zip(display["Requests"], display["Succeeded"], strict=True):
        if pd.isna(total) or pd.isna(succeeded):
            bounds.append((None, None))
            continue
        interval = wilson_interval(int(succeeded), int(total))
        bounds.append((interval[0] * 100, interval[1] * 100) if interval else (None, None))
    display["Success CI low (%)"] = [bound[0] for bound in bounds]
    display["Success CI high (%)"] = [bound[1] for bound in bounds]
    return display


def build_batch_success_figure(comparison: pd.DataFrame) -> go.Figure | None:
    """Compare observed batch success rates with count-based uncertainty."""
    required = {
        "Test名称",
        "Model",
        "Requests",
        "Request Success (%)",
        "Success CI low (%)",
        "Success CI high (%)",
    }
    if not required.issubset(comparison.columns):
        return None
    valid = comparison.dropna(subset=list(required - {"Test名称", "Model"})).copy()
    if valid.empty:
        return None
    valid["Label"] = valid["Test名称"].astype(str) + " · " + valid["Model"].astype(str)
    values = valid["Request Success (%)"].astype(float)
    lower = valid["Success CI low (%)"].astype(float)
    upper = valid["Success CI high (%)"].astype(float)
    fig = go.Figure(
        go.Scatter(
            x=values,
            y=valid["Label"],
            mode="markers",
            marker={"color": INDIGO, "size": 11},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": upper - values,
                "arrayminus": values - lower,
                "color": TEAL,
                "thickness": 2,
                "width": 5,
            },
            customdata=valid[["Requests"]].to_numpy(),
            hovertemplate="%{y}<br>Success: %{x:.1f}%<br>Recorded requests: %{customdata[0]}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "Request success by test item", "x": 0, "font": {"color": INK, "size": 17}},
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font={"color": INK},
        xaxis={"title": "Success (%)", "range": [0, 105], "gridcolor": SUBTLE},
        yaxis={"title": None, "showgrid": False},
        margin={"l": 100, "r": 45, "t": 65, "b": 55},
        height=max(310, 55 * len(valid) + 145),
    )
    return fig


def render_batch_test_results():
    """Render batch test results"""
    st.subheader("Batch Test Results")

    # Get all results
    results = batch_test_manager.list_results()

    if not results:
        st.info("No batch test results yet")
        return

    # Select results to view
    result_options = [f"{r['batch_name']} ({r['start_time']})" for r in results]
    selected_result = st.selectbox(
        "Select Test Results", result_options, key="batch_result_selector"
    )

    if not selected_result:
        return

    # Parse result
    result_name = selected_result.split("(")[0].strip()
    result = load_batch_result(result_name)

    if not result:
        st.error(f"Cannot load result: {result_name}")
        return

    # Display result summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Items", f"{result.total_items}")

    with col2:
        st.metric("Completed", f"{result.completed_items}")

    with col3:
        st.metric("Failed", f"{result.failed_items}")

    # Display comparison table
    st.markdown("---")
    st.markdown("### Test Comparison")

    comparison_df = result.get_comparison_df()
    if not comparison_df.empty:
        from ui.export import safe_csv_bytes
        from ui.reporting.html_export import build_html_report

        comparison_df = add_batch_success_intervals(comparison_df)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.caption(
            "Request success includes partial failures. TTFT and TPS use positive values from successful requests. "
            "Historical batch results without outcome counts show unavailable intervals."
        )
        success_figure = build_batch_success_figure(comparison_df)
        if success_figure is not None:
            st.plotly_chart(success_figure, use_container_width=True)

        # Download button
        export_col1, export_col2 = st.columns(2)
        export_col1.download_button(
            "Download comparison CSV",
            data=safe_csv_bytes(comparison_df),
            file_name=safe_download_filename(f"{result.batch_name}_comparison.csv"),
            mime="text/csv",
            icon=material_icon("download"),
        )
        html_report = build_html_report(
            [success_figure] if success_figure is not None else [],
            [("Observed test items", comparison_df)],
            [
                "Method: Request success includes every recorded outcome; timing quantiles use valid successful measurements.",
                "Comparability: Check workload, configuration and sample counts before ranking items.",
            ],
            title=f"Batch comparison - {result.batch_name}",
            deck="Batch outcomes, request-level statistics, and comparability notes in one portable report.",
            table_heading="Batch comparison",
            table_subtitle="Observed outcomes and timing distribution",
            overview_cards=(
                ("Test items", str(result.total_items), "Configured items"),
                ("Completed", str(result.completed_items), "Scheduler count"),
                ("Failed", str(result.failed_items), "Scheduler count"),
                ("Duration", f"{result.duration_seconds:.1f} s", "Batch wall time"),
            ),
        )
        export_col2.download_button(
            "Download batch HTML report",
            data=html_report,
            file_name=safe_download_filename(f"{result.batch_name}_comparison.html"),
            mime="text/html",
        )
    else:
        st.info("No data available")

    # Display detailed results
    st.markdown("---")
    st.markdown("### Detailed Results")

    with st.expander("View Detailed Results", expanded=False):
        for item_result in result.item_results:
            st.markdown(f"**{item_result.get('name', 'Unknown')}**")
            st.json(item_result)
            st.markdown("---")


def load_batch_result(name: str):
    """Load batch test results"""
    try:
        results_dir = Path("batch_tests/results")
        matching_files = list(results_dir.glob(f"{name}*.json"))

        if not matching_files:
            return None

        # Use latest result file
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)

        with open(latest_file, encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct BatchTestResult object
        from core.batch_test import BatchTestResult

        return BatchTestResult(
            batch_name=data["batch_name"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            duration_seconds=data["duration_seconds"],
            item_results=data["item_results"],
            total_items=data["total_items"],
            completed_items=data["completed_items"],
            failed_items=data["failed_items"],
        )
    except Exception as e:
        print(f"Failed to load result: {e}")
        return None


# ============================================================================
# Batch Test Main Interface
# ============================================================================


def render_batch_test_main():
    """Render batch test main interface"""
    # Tabs
    tab_config, tab_execute, tab_results, tab_history = st.tabs(
        [
            "Configure",
            "Execute",
            "Results",
            "History",
        ]
    )

    with tab_config:
        render_batch_config_editor()

    with tab_execute:
        render_batch_test_executor()

    with tab_results:
        render_batch_test_results()

    with tab_history:
        render_batch_test_history()


def render_batch_test_history():
    """Render batch test history."""
    st.subheader("Batch Test History")

    results = batch_test_manager.list_results()

    if not results:
        st.info("No batch test history")
        return

    # Display history
    for result in results:
        with st.expander(f"{result['batch_name']} - {result['start_time']}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.write(f"**Total Tests:** {result['total_items']}")

            with col2:
                st.write(f"**Completed:** {result['completed_items']}")

            with col3:
                st.write(f"**Failed:** {result['failed_items']}")

            with col4:
                duration = result.get("duration_seconds", 0)
                st.write(f"**Duration:** {duration:.1f}s")


# ============================================================================
# Helper Functions
# ============================================================================
