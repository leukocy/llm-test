"""
History Browser UI

Provides test history browsing, search, and export functionality.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from core.database import db_manager
from core.models import TestRun


def render_history_browser():
    """Render history browser"""

    st.subheader("📚 Test History")

    # Initialize
    db = db_manager

    # Search and filter
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input("Search", placeholder="Enter model ID, label, or notes...")

    with col2:
        filter_type = st.selectbox(
            "Test Type",
            ["All", "concurrency", "prefill", "segmented_prefill", "long_context", "matrix", "custom", "stability"]
        )

    with col3:
        filter_days = st.selectbox(
            "Time Range",
            ["All", "Today", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
        )

    # Build query
    runs = _query_runs(db, search_query, filter_type, filter_days)

    # Showing statistics
    _render_stats(db, runs)

    st.markdown("---")

    # Showing list
    if not runs:
        st.info("No test records yet")
        return None

    # Pagination
    page_size = 10
    total_pages = (len(runs) + page_size - 1) // page_size
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_runs = runs[start_idx:end_idx]

    # Showing records
    for run in page_runs:
        _render_run_card(run, db)

    # Pagination info
    st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(runs))}  of {len(runs)} records")


def _query_runs(db, search_query: str, filter_type: str, filter_days: str) -> List[TestRun]:
    """Query test runs"""
    # Base query
    if search_query:
        runs = db.search_runs(search_query, limit=500)
    else:
        runs = db.get_recent_runs(limit=500)

    # Type filter
    if filter_type != "All":
        runs = [r for r in runs if r.test_type == filter_type]

    # Time filter
    if filter_days != "All":
        now = datetime.now()
        if filter_days == "Today":
            cutoff = now - timedelta(days=1)
        elif filter_days == "Last 7 Days":
            cutoff = now - timedelta(days=7)
        elif filter_days == "Last 30 Days":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(days=90)

        runs = [r for r in runs if r.created_at and r.created_at >= cutoff]

    return runs


def _render_stats(db, runs: List[TestRun]):
    """Showing statistics info"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tests", len(runs))

    with col2:
        completed = len([r for r in runs if r.status == "completed"])
        st.metric("Completed", completed)

    with col3:
        models = set(r.model_id for r in runs if r.model_id)
        st.metric("Models", len(models))

    with col4:
        if runs:
            avg_ttft = sum(r.avg_ttft for r in runs if r.avg_ttft) / max(1, len([r for r in runs if r.avg_ttft]))
            st.metric("Average TTFT", f"{avg_ttft:.3f}s")
        else:
            st.metric("Average TTFT", "-")


def _render_run_card(run: TestRun, db):
    """Render single test run card"""

    # Status colors
    status_colors = {
        "completed": "🟢",
        "running": "🔵",
        "paused": "🟡",
        "cancelled": "⚪",
        "failed": "🔴",
    }
    status_icon = status_colors.get(run.status, "⚪")

    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            st.markdown(f"**{status_icon} {run.test_type}** - {run.model_id}")
            if run.tags:
                tags_str = " ".join([f"`{t}`" for t in run.tags[:3]])
                st.markdown(f"<small>{tags_str}</small>", unsafe_allow_html=True)

        with col2:
            if run.created_at:
                st.caption(f"📅 {run.created_at.strftime('%Y-%m-%d %H:%M')}")

        with col3:
            progress = f"{run.completed_requests}/{run.total_requests}"
            st.caption(f"📊 {progress}")

        with col4:
            if run.avg_ttft:
                st.caption(f"⏱️ TTFT: {run.avg_ttft:.3f}s")
            if run.avg_tps:
                st.caption(f"🚀 TPS: {run.avg_tps:.1f}")

        # Action buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        with btn_col1:
            if st.button("📋 Details", key=f"detail_{run.id}"):
                st.session_state.selected_run_id = run.id
                st.rerun()

        with btn_col2:
            if st.button("📊 Metrics", key=f"metrics_{run.id}"):
                _show_run_metrics(run, db)

        with btn_col3:
            if st.button("📥 Export", key=f"export_{run.id}"):
                _export_run(run, db)

        with btn_col4:
            if st.button("🗑️ Delete", key=f"delete_{run.id}"):
                if db.runs.delete_by_id(run.id):
                    st.success("Deleted")
                    st.rerun()

        st.markdown("---")


def _show_run_metrics(run: TestRun, db):
    """Showing detailed test run metrics"""
    with st.expander(f"📊 Detailed Metrics - {run.test_id}", expanded=True):
        # Get result statistics
        stats = db.results.get_aggregate_metrics(run.id)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Requests", stats.get("total_requests", 0))
            st.metric("Average TTFT", f"{stats.get('avg_ttft', 0):.3f}s" if stats.get('avg_ttft') else "-")
            st.metric("Min TTFT", f"{stats.get('min_ttft', 0):.3f}s" if stats.get('min_ttft') else "-")

        with col2:
            st.metric("Average TPS", f"{stats.get('avg_tps', 0):.1f}" if stats.get('avg_tps') else "-")
            st.metric("Max TPS", f"{stats.get('max_tps', 0):.1f}" if stats.get('max_tps') else "-")
            st.metric("Min TPS", f"{stats.get('min_tps', 0):.1f}" if stats.get('min_tps') else "-")

        with col3:
            st.metric("Total Tokens", stats.get("total_prefill_tokens", 0) + stats.get("total_decode_tokens", 0))
            st.metric("Cache Hits", stats.get("total_cache_hit_tokens", 0))
            st.metric("Average Prefill Speed", f"{stats.get('avg_prefill_speed', 0):.1f}" if stats.get('avg_prefill_speed') else "-")


def _export_run(run: TestRun, db):
    """Export test run"""
    with st.expander(f"📥 Export - {run.test_id}", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Export JSON", key=f"json_{run.id}"):
                path = db.export_run_json(run.id)
                if path:
                    st.success(f"Exported: {path}")

        with col2:
            if st.button("Export CSV", key=f"csv_{run.id}"):
                path = db.export_run_csv(run.id)
                if path:
                    st.success(f"Exported: {path}")

        with col3:
            if st.button("Export Excel", key=f"excel_{run.id}"):
                path = db.export_run_excel(run.id)
                if path:
                    st.success(f"Exported: {path}")


def render_import_panel():
    """Render data import panel"""

    st.subheader("📥 Import Historical Data")

    st.info("Import existing CSV test results into the database")

    # Select import source
    source_type = st.radio("Import Source", ["Single File", "Entire Directory"])

    if source_type == "Single File":
        csv_file = st.text_input("CSV file path", placeholder="e.g.: raw_data/model_id/benchmark_results.csv")

        col1, col2 = st.columns(2)
        with col1:
            model_id = st.text_input("Model ID (optional, leave blank for auto-detection)")
        with col2:
            test_type = st.text_input("Test Type (optional)")

        if st.button("Start Import") and csv_file:
            with st.spinner("Importing..."):
                imported, errors = db_manager.import_csv(csv_file, model_id or None, test_type or None)
                if imported > 0:
                    st.success(f"Successfully imported {imported} records")
                if errors:
                    for err in errors[:5]:
                        st.warning(err)

    else:
        directory = st.text_input("Directory Path", placeholder="e.g.: raw_data/")

        if st.button("Scan and Import") and directory:
            with st.spinner("Scanning..."):
                success, total, errors = db_manager.importer.import_directory(directory)

            st.success(f"Complete! Successfully imported {success} files, total {total} records")

            if errors:
                with st.expander(f"View Errors ({len(errors)} items)"):
                    for err in errors[:20]:
                        st.warning(err)


def render_dashboard():
    """Render dashboard"""

    st.subheader("📊 Dashboard")

    db = db_manager
    stats = db.get_dashboard_stats()

    # Overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Tests", stats["total_runs"])

    with col2:
        db_size_mb = stats["db_size_bytes"] / 1024 / 1024
        st.metric("Database Size", f"{db_size_mb:.2f} MB")

    with col3:
        backups = db.list_backups()
        st.metric("Backups", len(backups))

    # Statistics by model
    st.markdown("#### Statistics by Model")
    model_stats = stats.get("by_model", [])
    if model_stats:
        import pandas as pd
        df = pd.DataFrame(model_stats)
        st.bar_chart(df.set_index("model_id")["count"])

    # Statistics by type
    st.markdown("#### Statistics by Test Type")
    type_stats = stats.get("by_type", [])
    if type_stats:
        import pandas as pd
        df = pd.DataFrame(type_stats)
        st.bar_chart(df.set_index("test_type")["count"])

    # Recent tests
    st.markdown("#### Recent Tests")
    recent = stats.get("recent_runs", [])
    if recent:
        for run in recent[:5]:
            status_icons = {"completed": "🟢", "running": "🔵", "failed": "🔴"}
            icon = status_icons.get(run.status, "⚪")
            st.markdown(f"- {icon} **{run.test_type}** - {run.model_id} ({run.created_at.strftime('%Y-%m-%d %H:%M') if run.created_at else ''})")
