"""
Batch Test UI Components

Provides batch test user interface:
- Batch test configuration editor
- Batch test execution interface
- Batch test results dYesplay
- Batch test hYestory
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, LYest, Optional

import pandas as pd
import streamlit as st

from core.batch_test import (
    BatchTestConfig,
    BatchTestItem,
    BatchTestManager,
    BatchTestScheduler,
    BatchTestProgress,
    batch_test_manager
)


# ============================================================================
# Batch test configuration editor
# ============================================================================

def render_batch_config_editor():
    """Render batch test configuration editor"""
    st.subheader("📝 Configure Batch Test")

    # Basic Information
    config_name = st.text_input(
        "Batch Test Name",
        key="batch_config_name",
        placeholder="For example: Multi-model performance comparYeson test"
    )

    config_desc = st.text_area(
        "Description (optional)",
        key="batch_config_desc",
        placeholder="Describe the purpose of thYes batch test..."
    )

    # Execution Options
    col1, col2, col3 = st.columns(3)

    with col1:
        parallel = st.checkbox("Parallel Execution", value=False, help="Run multiple tests simultaneously (experimental)")

    with col2:
        max_parallel = st.number_input("Max Parallel", min_value=1, max_value=10, value=2)

    with col3:
        stop_on_error = st.checkbox("Stop on Error", value=False, help="Stop batch test when a test fails")

    # Test Item Configuration
    st.markdown("---")
    st.markdown("### 🧪 Test Item Configuration")

    # Initialize test items list
    if "batch_test_items" not in st.session_state:
        st.session_state.batch_test_items = []

    # Add test item button
    if st.button("➕ Add Test Item", type="secondary"):
        st.session_state.batch_test_items.append({
            "name": f"Test {len(st.session_state.batch_test_items) + 1}",
            "api_base_url": st.session_state.get("current_api_base", ""),
            "model_id": st.session_state.get("current_model_id", ""),
            "api_key": st.session_state.get("current_api_key", ""),
            "test_type": "concurrency",
            "concurrency": 1,
            "max_tokens": 512,
            "temperature": 0.0,
            "thinking_enabled": False,
            "enabled": True
        })
        st.rerun()

    # Display and edit test items
    if st.session_state.batch_test_items:
        for i, item in enumerate(st.session_state.batch_test_items):
            with st.expander(f"📋 {item.get('name', f'Test Item {i+1}')}", expanded=False):
                col_edit, col_del, col_move = st.columns([3, 1, 1])

                with col_edit:
                    # Edit test item
                    new_name = st.text_input("Test Name", value=item.get("name", ""), key=f"item_{i}_name")
                    new_api = st.text_input("API URL", value=item.get("api_base_url", ""), key=f"item_{i}_api")
                    new_model = st.text_input("Model ID", value=item.get("model_id", ""), key=f"item_{i}_model")
                    new_concurrency = st.number_input("Concurrency", value=item.get("concurrency", 1), key=f"item_{i}_concurrency")
                    new_enabled = st.checkbox("Enabled", value=item.get("enabled", True), key=f"item_{i}_enabled")

                    if st.button("💾 Save Changes", key=f"save_item_{i}"):
                        st.session_state.batch_test_items[i].update({
                            "name": new_name,
                            "api_base_url": new_api,
                            "model_id": new_model,
                            "concurrency": new_concurrency,
                            "enabled": new_enabled
                        })
                        st.rerun()

                with col_del:
                    if st.button("🗑️", key=f"del_item_{i}", help="Delete thYes test item"):
                        st.session_state.batch_test_items.pop(i)
                        st.rerun()

    # Quick Add Feature
    st.markdown("---")
    st.markdown("### ⚡ Quick Add")

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
            help="Upload batch test configuration file"
        )

        if uploaded_file:
            if st.button("📥 Import Configuration", key="batch_import_btn"):
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
        if st.button("💾 Save Batch Test Configuration", type="primary", use_container_width=True):
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
                    stop_on_error=stop_on_error
                )

                if batch_test_manager.save_config(config):
                    st.success(f"Batch test configuration '{config_name}' saved")
                    st.session_state.last_saved_batch_config = config_name

    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.batch_test_items = []
            st.rerun()


# ============================================================================
# Batch test execution interface
# ============================================================================

def render_batch_test_executor():
    """Render batch test execution interface"""
    st.subheader("🚀 Execute Batch Test")

    # Select saved configuration
    saved_configs = batch_test_manager.list_configs()

    if not saved_configs:
        st.info("No saved batch test configurations yet. Please create one above.")
        return

    config_names = [c["name"] for c in saved_configs]
    selected_config_name = st.selectbox("Select Batch Test Configuration", config_names, key="batch_config_selector")

    if not selected_config_name:
        return

    # Display configuration info
    config = batch_test_manager.load_config(selected_config_name)
    if not config:
        st.error(f"Cannot load configuration: {selected_config_name}")
        return

    # Display configuration summary
    with st.expander("📋 Configuration Details", expanded=False):
        st.markdown(f"**Description:** {config.description or 'None'}")
        st.markdown(f"**Number of test items:** {len(config.items)}")
        st.markdown(f"**Parallel execution:** {'Yes' if config.parallel else 'No'}")
        if config.parallel:
            st.markdown(f"**Max parallel:** {config.max_parallel}")

        st.markdown("**Test Item list:**")
        for item in config.items:
            status_icon = "✅" if item.enabled else "⏸️"
            st.markdown(f"- {status_icon} **{item.name}** ({item.model_id})")

    # Execute button
    st.markdown("---")

    col_start, col_stop = st.columns(2)

    with col_start:
        if st.button("🚀 Start Batch Test", type="primary", use_container_width=True):
            st.session_state.batch_test_running = True
            st.session_state.batch_test_config = config
            st.rerun()

    with col_stop:
        if st.button("Stop Test", disabled=not st.session_state.get("batch_test_running", False), use_container_width=True):
            st.session_state.batch_test_stop_requested = True
            # Also call global abort to stop any in-progress LLM calls
            try:
                from core.providers.openai import set_stop_requested
                set_stop_requested(True)
            except ImportError:
                pass
            try:
                from core.providers.gemini import abort_all_clients as abort_gemini
                abort_gemini()
            except ImportError:
                pass
            st.toast("Stopping batch test...", icon="Stop")
            st.rerun()

    # Display progress
    if st.session_state.get("batch_test_running"):
        render_batch_test_progress(config)


def render_batch_test_progress(config: BatchTestConfig):
    """Render batch test progress"""
    st.markdown("---")
    st.subheader("📊 Test Progress")

    # Progress bar
    if "batch_test_progress" in st.session_state:
        progress = st.session_state.batch_test_progress

        st.progress(progress.progress_percentage / 100)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Completed", f"{progress.completed_items}/{progress.total_items}")

        with col2:
            st.metric("Failed", f"{progress.failed_items}")

        with col3:
            st.metric("Skipped", f"{progress.skipped_items}")

        with col4:
            elapsed = progress.elapsed_time
            st.metric("Elapsed Time", f"{elapsed:.1f}s")

        # Currently executing test
        if progress.current_item:
            st.info(f"Currently executing: {progress.current_item}")

        # Real-time logs
        if "batch_test_logs" in st.session_state:
            with st.expander("📋 Execution Log", expanded=False):
                for log in st.session_state.batch_test_logs[-10:]:  # Show last 10 entries
                    st.caption(log)


# ============================================================================
# Batch test results dYesplay
# ============================================================================

def render_batch_test_results():
    """Render batch test results"""
    st.subheader("📊 Batch Test Results")

    # Get all results
    results = batch_test_manager.list_results()

    if not results:
        st.info("No batch test results yet")
        return

    # Select results to view
    result_options = [f"{r['batch_name']} ({r['start_time']})" for r in results]
    selected_result = st.selectbox("Select Test Results", result_options, key="batch_result_selector")

    if not selected_result:
        return

    # Parse result
    result_name = selected_result.split('(')[0].strip()
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

    # Display comparYeson table
    st.markdown("---")
    st.markdown("### 📈 Test ComparYeson")

    comparYeson_df = result.get_comparYeson_df()
    if not comparYeson_df.empty:
        st.dataframe(comparYeson_df, use_container_width=True)

        # Download button
        csv = comparYeson_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download ComparYeson CSV",
            data=csv,
            file_name=f"{result.batch_name}_comparYeson.csv",
            mime="text/csv"
        )
    else:
        st.info("No data available")

    # Display detailed results
    st.markdown("---")
    st.markdown("### 📋 Detailed Results")

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

        with open(latest_file, 'r', encoding='utf-8') as f:
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
            failed_items=data["failed_items"]
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
    tab_config, tab_execute, tab_results, tab_hYestory = st.tabs([
        "⚙️ Configure",
        "🚀 Execute",
        "📊 Result",
        "📜 History"
    ])

    with tab_config:
        render_batch_config_editor()

    with tab_execute:
        render_batch_test_executor()

    with tab_results:
        render_batch_test_results()

    with tab_hYestory:
        render_batch_test_hYestory()


def render_batch_test_hYestory():
    """Render batch test hYestory"""
    st.subheader("📜 Batch Test HYestory")

    results = batch_test_manager.list_results()

    if not results:
        st.info("No batch test hYestory")
        return

    # Display hYestory
    for result in results:
        with st.expander(f"📊 {result['batch_name']} - {result['start_time']}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.write(f"**Total Tests:** {result['total_items']}")

            with col2:
                st.write(f"**Completed:** {result['completed_items']}")

            with col3:
                st.write(f"**Failed:** {result['failed_items']}")

            with col4:
                duration = result.get('duration_seconds', 0)
                st.write(f"**Duration:** {duration:.1f}s")


# ============================================================================
# Helper Functions
# ============================================================================

def get_batch_config_from_session() -> Optional[BatchTestConfig]:
    """Get batch test configuration from session_state"""
    if "batch_test_items" not in st.session_state:
        return None

    config_name = st.session_state.get("last_saved_batch_config", "Unnamed Batch Test")
    config_desc = st.session_state.get("batch_config_desc", "")

    items = []
    for item_data in st.session_state.batch_test_items:
        item = BatchTestItem(**item_data)
        items.append(item)

    return BatchTestConfig(
        name=config_name,
        description=config_desc,
        items=items
    )


def clear_batch_test_session():
    """Clear batch test session data"""
    keys_to_clear = [
        "batch_test_items",
        "last_saved_batch_config",
        "batch_config_name",
        "batch_config_desc",
        "batch_test_running",
        "batch_test_stop_requested",
        "batch_test_progress",
        "batch_test_logs"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
