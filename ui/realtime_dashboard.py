"""
Real-time Dashboard for Test Visualization

Provides live metrics and performance charts during benchmark tests.
"""
from collections import deque
from typing import Any, Dict, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class RealtimeDashboard:
    """Real-time dashboard for monitoring test progress."""

    def __init__(self, max_points=100):
        """
        Initialize dashboard.

        Args:
            max_points: Maximum number of data points to keep in charts
        """
        self.max_points = max_points

        # Time series data
        self.timestamps = deque(maxlen=max_points)
        self.ttft_values = deque(maxlen=max_points)
        self.tps_values = deque(maxlen=max_points)

        # Counters
        self.active_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.total_requests = 0

        # Request states tracking
        self.request_states = {}  # {request_id: status}

        # Performance aggregates
        self.ttft_sum = 0
        self.tps_sum = 0
        self.valid_count = 0

    def update(self, timestamp: float, ttft: float, tps: float, status: str, session_id: int | None = None):
        """
        Update dashboard with new data point.

        Args:
            timestamp: Unix timestamp
            ttft: Time to first token
            tps: Tokens per second
            status: 'success' or 'failed'
            session_id: Optional session identifier
        """
        # Update time series if valid data
        if status == 'success' and ttft is not None and tps is not None:
            self.timestamps.append(timestamp)
            self.ttft_values.append(ttft)
            self.tps_values.append(tps)

            # Update aggregates
            self.ttft_sum += ttft
            self.tps_sum += tps
            self.valid_count += 1

        # Update counters
        if status == 'success':
            self.completed_requests += 1
        elif status == 'failed':
            self.failed_requests += 1

        # Update request state if session_id provided
        if session_id is not None:
            self.request_states[session_id] = 'completed' if status == 'success' else 'failed'

    def update_request_state(self, session_id: int, state: str):
        """
        Update the state of a specific request.

        Args:
            session_id: Request identifier
            state: 'waiting', 'running', 'completed', or 'failed'
        """
        self.request_states[session_id] = state

        if state == 'running':
            self.active_requests += 1
        elif state in ('completed', 'failed'):
            self.active_requests = max(0, self.active_requests - 1)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current metric values.

        Returns:
            Dictionary of current metrics
        """
        avg_ttft = self.ttft_sum / self.valid_count if self.valid_count > 0 else 0
        avg_tps = self.tps_sum / self.valid_count if self.valid_count > 0 else 0

        return {
            'active': self.active_requests,
            'completed': self.completed_requests,
            'failed': self.failed_requests,
            'total': self.total_requests,
            'avg_ttft': avg_ttft,
            'avg_tps': avg_tps,
            'success_rate': (self.completed_requests / max(1, self.completed_requests + self.failed_requests)) * 100
        }

    def create_realtime_chart(self) -> go.Figure | None:
        """
        Create real-time performance chart.

        Returns:
            Plotly figure or None if insufficient data
        """
        if len(self.timestamps) == 0:
            return None

        # Convert timestamps to relative seconds for better readability
        start_time = self.timestamps[0]
        relative_times = [(t - start_time) for t in self.timestamps]

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add TTFT trace
        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=list(self.ttft_values),
                mode='lines+markers',
                name='TTFT (Time to First Token)',
                line={'color': '#3b82f6', 'width': 2},
                marker={'size': 4},
                hovertemplate='<b>TTFT</b>: %{y:.3f}s<extra></extra>'
            ),
            secondary_y=False
        )

        # Add TPS trace
        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=list(self.tps_values),
                mode='lines+markers',
                name='TPS (Tokens per Second)',
                line={'color': '#10b981', 'width': 2},
                marker={'size': 4},
                hovertemplate='<b>TPS</b>: %{y:.2f} tokens/s<extra></extra>'
            ),
            secondary_y=True
        )

        # Update layout
        fig.update_layout(
            title={
                'text': "Real-time Performance Metrics",
                'font': {'size': 18, 'color': '#1f2937'}
            },
            xaxis={
                'title': "Test Time (seconds)",
                'gridcolor': '#e5e7eb'
            },
            yaxis={
                'title': "TTFT (seconds)",
                'title_font': {'color': '#3b82f6'},
                'tickfont': {'color': '#3b82f6'},
                'gridcolor': '#e5e7eb'
            },
            yaxis2={
                'title': "TPS (tokens/seconds)",
                'title_font': {'color': '#10b981'},
                'tickfont': {'color': '#10b981'}
            },
            hovermode='x unified',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin={'l': 60, 'r': 60, 't': 60, 'b': 60},
            legend={
                'orientation': "h",
                'yanchor': "bottom",
                'y': 1.02,
                'xanchor': "right",
                'x': 1
            }
        )

        return fig

    def get_request_grid_data(self, max_display=64) -> dict[str, list]:
        """
        Get formatted request state data for grid display.

        Args:
            max_display: Maximum number of requests to display

        Returns:
            Dictionary with request IDs and their states
        """
        # Get most recent requests
        states = list(self.request_states.items())[-max_display:]

        return {
            'ids': [str(sid) for sid, _ in states],
            'states': [state for _, state in states]
        }

    def reset(self):
        """Reset all dashboard data."""
        self.timestamps.clear()
        self.ttft_values.clear()
        self.tps_values.clear()
        self.active_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.total_requests = 0
        self.request_states.clear()
        self.ttft_sum = 0
        self.tps_sum = 0
        self.valid_count = 0
