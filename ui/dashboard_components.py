"""
Streamlit UI components for realtime dashboard.

Provides ready-to-use UI elements that can be embedded in the main app.
"""
import streamlit as st

from ui.realtime_dashboard import RealtimeDashboard


def render_dashboard_ui(dashboard: RealtimeDashboard):
    """
    Render the full realtime dashboard UI.

    Args:
        dashboard: RealtimeDashboard instance
    """
    # Metrics cards section
    st.subheader("📊 Real-time Test Metrics")
    metrics = dashboard.get_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🟢 Active Requests",
            value=metrics['active'],
            help="Number of requests currently being processed"
        )

    with col2:
        st.metric(
            label="✅ Completed",
            value=metrics['completed'],
            delta=f"+{metrics['completed']}" if metrics['completed'] > 0 else None,
            help="Total successfully completed requests"
        )

    with col3:
        st.metric(
            label="❌ Failed",
            value=metrics['failed'],
            delta=f"+{metrics['failed']}" if metrics['failed'] > 0 else None,
            delta_color="inverse",
            help="Total failed requests"
        )

    with col4:
        st.metric(
            label="⚡ Success Rate",
            value=f"{metrics['success_rate']:.1f}%",
            help="Request success rate percentage"
        )

    # Second row of metrics
    col5, col6 = st.columns(2)

    with col5:
        st.metric(
            label="⏱️ Average TTFT",
            value=f"{metrics['avg_ttft']:.3f}seconds",
            help="Average time to first token"
        )

    with col6:
        st.metric(
            label="🚀 Average TPS",
            value=f"{metrics['avg_tps']:.2f} tokens/s",
            help="Average tokens generated per second"
        )

    st.markdown("---")

    # Performance chart
    chart = dashboard.create_realtime_chart()
    if chart:
        st.plotly_chart(chart)
    else:
        st.info("Waiting for test data...")


def render_compact_metrics(dashboard: RealtimeDashboard):
    """
    Render a compact version of metrics (for sidebar or limited space).

    Args:
        dashboard: RealtimeDashboard instance
    """
    metrics = dashboard.get_metrics()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Active", metrics['active'])
        st.metric("Completed", metrics['completed'])

    with col2:
        st.metric("Failed", metrics['failed'])
        st.metric(f"{metrics['success_rate']:.0f}%", "Success Rate")


def render_request_grid(dashboard: RealtimeDashboard, max_display=64):
    """
    Render request status grid.

    Args:
        dashboard: RealtimeDashboard instance
        max_display: Maximum number of requests to display
    """
    st.subheader("🔲 Request Status Grid")

    grid_data = dashboard.get_request_grid_data(max_display=max_display)

    if not grid_data['ids']:
        st.info("No requests yet")
        return

    # Status icons mapping
    status_icons = {
        'waiting': '⏸️',
        'running': '🟢',
        'completed': '✅',
        'failed': '❌'
    }

    # Create grid layout (8 columns)
    cols_per_row = 8
    ids = grid_data['ids']
    states = grid_data['states']

    for start_idx in range(0, len(ids), cols_per_row):
        cols = st.columns(cols_per_row)
        end_idx = min(start_idx + cols_per_row, len(ids))

        for i, col in enumerate(cols[:end_idx - start_idx]):
            idx = start_idx + i
            with col:
                icon = status_icons.get(states[idx], '⚪')
                st.markdown(f"{icon} {ids[idx]}", unsafe_allow_html=False)


def create_dashboard_placeholders():
    """
    Create Streamlit placeholders for realtime dashboard updates.

    Returns:
        dict: Dictionary containing placeholder objects
    """
    return {
        'metrics': st.empty(),
        'chart': st.empty(),
        'grid': st.empty()
    }


def update_dashboard_placeholders(placeholders: dict, dashboard: RealtimeDashboard):
    """
    Update all dashboard placeholders.

    Args:
        placeholders: Dictionary of Streamlit placeholders
        dashboard: RealtimeDashboard instance
    """
    with placeholders['metrics'].container():
        metrics = dashboard.get_metrics()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active", metrics['active'])
        with col2:
            st.metric("Completed", metrics['completed'])
        with col3:
            st.metric("Failed", metrics['failed'])
        with col4:
            st.metric(f"{metrics['avg_ttft']:.3f}s", "AverageTTFT")

    with placeholders['chart'].container():
        chart = dashboard.create_realtime_chart()
        if chart:
            st.plotly_chart(chart, key=f"chart_{dashboard.completed_requests}")
