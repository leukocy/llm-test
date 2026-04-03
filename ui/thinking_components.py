"""
Phase 6: Enhanced UI Components

Provides specialized UI components for reasoning models:
- Collapsible reasoning content display
- Thinking metrics visualization
- Report export buttons
- Real-time streaming preview
"""

import base64
import json
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import streamlit as st

# Try importing core modules
try:
    from core.evaluation_report import EvaluationReport, ReportBuilder
    from core.metrics import ThinkingMetricsResult, format_metrics_report
    from core.response_parser import ParsedResponse
    from core.thinking_params import PLATFORM_FEATURES, get_platform_features
except ImportError:
    # If import fails, provide placeholder
    ThinkingMetricsResult = None
    EvaluationReport = None


def render_reasoning_expander(
    reasoning_content: str,
    content: str,
    title: str = "🧠 Reasoning process",
    expanded: bool = False,
    max_height: int = 400
):
    """
    Render collapsible reasoning content display

    Args:
        reasoning_content: Reasoning content
        content: Main body content
        title: Collapsible section title
        expanded: Whether to expand by Default
        max_height: Maximum height (px)
    """
    if reasoning_content:
        with st.expander(f"{title} ({len(reasoning_content)} chars)", expanded=expanded):
            st.markdown(
                f"""
                <div style="
                    max-height: {max_height}px;
                    overflow-y: auto;
                    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
                    padding: 16px;
                    border-radius: 8px;
                    border-left: 4px solid #60a5fa;
                    font-family: 'Consolas', monospace;
                    font-size: 14px;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    color: #e2e8f0;
                ">
                    {reasoning_content}
                </div>
                """,
                unsafe_allow_html=True
            )

    if content:
        st.markdown("**📝 Final Output:**")
        st.markdown(
            f"""
            <div style="
                background: #1e293b;
                padding: 16px;
                border-radius: 8px;
                border-left: 4px solid #22c55e;
                max-height: {max_height}px;
                overflow-y: auto;
            ">
                {content}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_thinking_metrics_cards(metrics: dict[str, Any], cols: int = 4):
    """
    Render thinking metrics cards

    Args:
        metrics: Metrics dict or ThinkingMetricsResult
        cols: Cards per row
    """
    # Convert to dict
    if hasattr(metrics, '__dataclass_fields__'):
        metrics = {k: getattr(metrics, k) for k in metrics.__dataclass_fields__}

    # Define metrics to display
    display_metrics = [
        ("TTFT", metrics.get("ttft_ms"), "ms", "⚡", "#eab308"),
        ("TTUT", metrics.get("ttut_ms"), "ms", "⏱️", "#22c55e"),
        ("Reasoning Tokens", metrics.get("reasoning_tokens"), "", "🧠", "#60a5fa"),
        ("Reasoning Ratio", metrics.get("reasoning_ratio"), "%", "📊", "#a855f7"),
        ("Total Tokens", metrics.get("total_tokens"), "", "📝", "#64748b"),
        ("Est. Cost", metrics.get("estimated_cost_usd"), "$", "💰", "#f97316"),
    ]

    # Create columns
    columns = st.columns(cols)

    for i, (label, value, unit, icon, color) in enumerate(display_metrics):
        with columns[i % cols]:
            if value is not None:
                if unit == "%":
                    display_value = f"{value * 100:.1f}%"
                elif unit == "$":
                    display_value = f"${value:.6f}"
                elif unit == "ms":
                    display_value = f"{value:.0f}ms"
                else:
                    display_value = f"{value:,}" if isinstance(value, int) else f"{value}"
            else:
                display_value = "N/A"

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                    padding: 16px;
                    border-radius: 12px;
                    border: 1px solid #334155;
                    text-align: center;
                    margin-bottom: 12px;
                ">
                    <div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>
                    <div style="font-size: 24px; font-weight: 700; color: {color};">{display_value}</div>
                    <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_latency_gauge(ttft: float, ttut: float, total: float, max_value: float = 10000):
    """
    Render latency gauge dashboard

    Args:
        ttft: Time to first token (ms)
        ttut: Time to first useful token (ms)
        total: Total time (ms)
        max_value: Maximum value (ms)
    """
    fig = go.Figure()

    # TTFT gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=ttft or 0,
        title={"text": "TTFT (ms)", "font": {"size": 14}},
        domain={"x": [0, 0.3], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, max_value / 5]},
            "bar": {"color": "#eab308"},
            "steps": [
                {"range": [0, 500], "color": "rgba(34, 197, 94, 0.15)"},
                {"range": [500, 1500], "color": "rgba(234, 179, 8, 0.15)"},
                {"range": [1500, max_value / 5], "color": "rgba(239, 68, 68, 0.15)"}
            ]
        }
    ))

    # TTUT gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=ttut or 0,
        title={"text": "TTUT (ms)", "font": {"size": 14}},
        domain={"x": [0.35, 0.65], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, max_value / 2]},
            "bar": {"color": "#22c55e"},
            "steps": [
                {"range": [0, 2000], "color": "rgba(34, 197, 94, 0.15)"},
                {"range": [2000, 5000], "color": "rgba(234, 179, 8, 0.15)"},
                {"range": [5000, max_value / 2], "color": "rgba(239, 68, 68, 0.15)"}
            ]
        }
    ))

    # Total time gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=total or 0,
        title={"text": "Total (ms)", "font": {"size": 14}},
        domain={"x": [0.7, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": "#60a5fa"},
            "steps": [
                {"range": [0, 5000], "color": "rgba(34, 197, 94, 0.15)"},
                {"range": [5000, 10000], "color": "rgba(234, 179, 8, 0.15)"},
                {"range": [10000, max_value], "color": "rgba(239, 68, 68, 0.15)"}
            ]
        }
    ))


    fig.update_layout(
        height=200,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}
    )

    st.plotly_chart(fig, use_container_width=True)


def render_token_pie_chart(reasoning_tokens: int, content_tokens: int, prompt_tokens: int = 0):
    """
    Render token distribution pie chart

    Args:
        reasoning_tokens: Reasoning Tokens
        content_tokens: Content Tokens
        prompt_tokens: Input Tokens
    """
    labels = []
    values = []
    colors = []

    if prompt_tokens > 0:
        labels.append("Input")
        values.append(prompt_tokens)
        colors.append("#64748b")

    if reasoning_tokens > 0:
        labels.append("Reasoning")
        values.append(reasoning_tokens)
        colors.append("#60a5fa")

    if content_tokens > 0:
        labels.append("Content")
        values.append(content_tokens)
        colors.append("#22c55e")

    if not values:
        st.info("No token data yet")
        return

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo="percent+label",
        textposition="outside"
    )])

    fig.update_layout(
        height=250,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        showlegend=False,
        annotations=[{"text": "Token<br>Distribution", "x": 0.5, "y": 0.5, "font_size": 14, "showarrow": False}]
    )

    st.plotly_chart(fig, use_container_width=True)


def render_platform_info(platform: str):
    """
    Render platform info card

    Args:
        platform: Platform identifier
    """
    features = PLATFORM_FEATURES.get(platform, {})

    if not features:
        st.info(f"Unknown platform: {platform}")
        return

    name = features.get("name", platform)
    location = features.get("thinking_param_location", "unknown")
    supports_budget = "✅" if features.get("supports_budget") else "❌"
    supports_effort = "✅" if features.get("supports_effort") else "❌"
    reasoning_field = features.get("reasoning_output_field", "N/A")
    notes = features.get("notes", "")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #334155;
        ">
            <h3 style="margin: 0 0 16px 0; color: #e2e8f0;">🏢 {name}</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                <div>
                    <span style="color: #94a3b8;">Param Location:</span>
                    <span style="color: #60a5fa; margin-left: 8px;">{location}</span>
                </div>
                <div>
                    <span style="color: #94a3b8;">Reasoning Field:</span>
                    <span style="color: #22c55e; margin-left: 8px;">{reasoning_field}</span>
                </div>
                <div>
                    <span style="color: #94a3b8;">Budget Support:</span>
                    <span style="margin-left: 8px;">{supports_budget}</span>
                </div>
                <div>
                    <span style="color: #94a3b8;">Effort Support:</span>
                    <span style="margin-left: 8px;">{supports_effort}</span>
                </div>
            </div>
            {f'<p style="margin: 16px 0 0 0; color: #94a3b8; font-size: 13px;">💡 {notes}</p>' if notes else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_export_buttons(
    report_builder: "ReportBuilder",
    filename_prefix: str = "evaluation_report"
):
    """
    Render report export buttons

    Args:
        report_builder: Report builder instance
        filename_prefix: Filename prefix
    """
    st.markdown("### 📥 Export Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 JSON", use_container_width=True):
            json_str = json.dumps(report_builder.to_dict(), ensure_ascii=False, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="{filename_prefix}.json">Click to download JSON</a>'
            st.markdown(href, unsafe_allow_html=True)

    with col2:
        if st.button("📑 Markdown", use_container_width=True):
            # Temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                report_builder.export_markdown(f.name)
                f.seek(0)
            with open(f.name, encoding='utf-8') as f:
                md_content = f.read()
            b64 = base64.b64encode(md_content.encode()).decode()
            href = f'<a href="data:text/markdown;base64,{b64}" download="{filename_prefix}.md">Click to download Markdown</a>'
            st.markdown(href, unsafe_allow_html=True)

    with col3:
        if st.button("🌐 HTML", use_container_width=True):
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                report_builder.export_html(f.name)
            with open(f.name, encoding='utf-8') as f:
                html_content = f.read()
            b64 = base64.b64encode(html_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="{filename_prefix}.html">Click to download HTML</a>'
            st.markdown(href, unsafe_allow_html=True)


def render_stream_preview(
    full_content: str = "",
    reasoning_content: str = "",
    is_streaming: bool = False,
    show_reasoning: bool = True
):
    """
    Render real-time streaming preview

    Args:
        full_content: Current accumulated body content
        reasoning_content: Current accumulated reasoning content
        is_streaming: Whether currently streaming
        show_reasoning: Whether to show reasoning content
    """
    # Status indicator
    status = "🔄 Receiving..." if is_streaming else "✅ Complete"
    st.markdown(f"**Status:** {status}")

    # Reasoning preview
    if show_reasoning and reasoning_content:
        st.markdown("**🧠 Reasoning process:**")
        reasoning_container = st.container()
        with reasoning_container:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid #60a5fa;
                    max-height: 200px;
                    overflow-y: auto;
                    font-family: monospace;
                    font-size: 13px;
                    color: #e2e8f0;
                    white-space: pre-wrap;
                ">
                    {reasoning_content[-2000:] if len(reasoning_content) > 2000 else reasoning_content}
                    {'<span style="animation: blink 1s infinite;">▌</span>' if is_streaming else ''}
                </div>
                <style>
                    @keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
                </style>
                """,
                unsafe_allow_html=True
            )

    # Content preview
    if full_content:
        st.markdown("**📝 Output Content:**")
        st.markdown(
            f"""
            <div style="
                background: #1e293b;
                padding: 12px;
                border-radius: 8px;
                border-left: 4px solid #22c55e;
                max-height: 300px;
                overflow-y: auto;
            ">
                {full_content[-3000:] if len(full_content) > 3000 else full_content}
                {'<span style="animation: blink 1s infinite;">▌</span>' if is_streaming else ''}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_quality_score_card(
    overall: float | None = None,
    accuracy: float | None = None,
    coherence: float | None = None,
    completeness: float | None = None
):
    """
    Render quality score card

    Args:
        overall: Overall score
        accuracy: Accuracy
        coherence: Coherence
        completeness: Completeness
    """
    def get_color(score):
        if score is None:
            return "#64748b"
        if score >= 8:
            return "#22c55e"
        if score >= 6:
            return "#eab308"
        return "#ef4444"

    def format_score(score):
        return f"{score:.1f}" if score is not None else "N/A"

    html_content = f"""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="font-size: 48px; font-weight: 800; color: {get_color(overall)};">
            {format_score(overall)}
        </div>
        <div style="font-size: 14px; color: #94a3b8; margin-bottom: 20px;">Overall Score / 10</div>

        <div style="display: flex; justify-content: space-around;">
            <div style="flex: 1;">
                <div style="font-size: 24px; font-weight: 600; color: {get_color(accuracy)};">{format_score(accuracy)}</div>
                <div style="font-size: 12px; color: #94a3b8;">Accuracy</div>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 24px; font-weight: 600; color: {get_color(coherence)};">{format_score(coherence)}</div>
                <div style="font-size: 12px; color: #94a3b8;">Coherence</div>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 24px; font-weight: 600; color: {get_color(completeness)};">{format_score(completeness)}</div>
                <div style="font-size: 12px; color: #94a3b8;">Completeness</div>
            </div>
        </div>
    </div>
    """

    st.markdown(html_content, unsafe_allow_html=True)

