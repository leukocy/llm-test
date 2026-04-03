"""
EvaluationResultVisualizationDashboard (Evaluation Results Dashboard)

Streamlit component for visualizing evaluation results.
"""

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_evaluation_dashboard(
    results: dict[str, Any],
    title: str = "📊 Evaluation Results Dashboard"
):
    """
    Render full evaluation results dashboard

    Args:
        results: Evaluation results dict containing models, summary, etc.
        title: Dashboard title
    """
    st.title(title)

    # Meta info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Name", results.get('test_name', 'N/A'))
    with col2:
        st.metric("Test Time", results.get('start_time', 'N/A')[:19])
    with col3:
        st.metric("Duration", f"{results.get('duration_seconds', 0):.1f}s")

    st.markdown("---")

    # Model comparison table
    if 'models' in results:
        render_model_comparison_table(results['models'])

    st.markdown("---")

    # Chart area
    col1, col2 = st.columns(2)

    with col1:
        if 'models' in results:
            render_accuracy_comparison(results['models'])

    with col2:
        if 'models' in results:
            render_latency_comparison(results['models'])


def render_model_comparison_table(models: dict[str, Any]):
    """Render model comparison table"""
    st.subheader("📋 Model Performance Comparison")

    rows = []
    for model_key, data in models.items():
        rows.append({
            "Model": data.get('model_id', model_key),
            "Platform": data.get('platform', '-'),
            "Accuracy": f"{data.get('accuracy', 0) * 100:.1f}%",
            "Sample count": data.get('total_samples', 0),
            "Correct": data.get('correct_samples', 0),
            "Avg Latency": f"{data.get('avg_latency_ms', 0):.0f}ms",
            "Reasoning Quality": f"{data.get('avg_reasoning_quality', 0):.1f}/10"
        })

    df = pd.DataFrame(rows)

    # Highlight best values
    st.dataframe(df, use_container_width=True)


def render_accuracy_comparison(models: dict[str, Any]):
    """Render accuracy comparison chart"""
    st.subheader("🎯 Accuracy Comparison")

    model_names = []
    accuracies = []
    colors = []

    # Color scheme
    color_palette = ['#60a5fa', '#22c55e', '#eab308', '#ef4444', '#a855f7', '#f97316']

    max_acc = 0

    for i, (key, data) in enumerate(models.items()):
        model_names.append(data.get('model_id', key))
        acc = data.get('accuracy', 0) * 100
        accuracies.append(acc)

        if acc > max_acc:
            max_acc = acc

        colors.append(color_palette[i % len(color_palette)])

    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=accuracies,
            marker_color=colors,
            text=[f"{a:.1f}%" for a in accuracies],
            textposition='outside'
        )
    ])

    fig.update_layout(
        height=300,
        margin={"l": 20, "r": 20, "t": 30, "b": 50},
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, max(100, max(accuracies) * 1.1)],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}
    )

    st.plotly_chart(fig, use_container_width=True)


def render_latency_comparison(models: dict[str, Any]):
    """Render latency comparison chart"""
    st.subheader("⏱️ Latency Comparison")

    model_names = []
    latencies = []

    for key, data in models.items():
        model_names.append(data.get('model_id', key))
        latencies.append(data.get('avg_latency_ms', 0))

    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=latencies,
            marker_color='#22c55e',
            text=[f"{l:.0f}ms" for l in latencies],
            textposition='outside'
        )
    ])

    fig.update_layout(
        height=300,
        margin={"l": 20, "r": 20, "t": 30, "b": 50},
        yaxis_title="Average Latency (ms)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}
    )

    st.plotly_chart(fig, use_container_width=True)


def render_failure_analysis_panel(failure_report: dict[str, Any]):
    """Render failure analysis panel"""
    st.subheader("❌ Failure Analysis")

    # Failure rate
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Failure Rate",
            f"{failure_report.get('failure_rate', 0):.1f}%",
            delta=None
        )
    with col2:
        st.metric(
            "Failed Samples",
            failure_report.get('failed_samples', 0)
        )

    # Category distribution pie chart
    category_dist = failure_report.get('category_distribution', {})
    if category_dist:
        fig = go.Figure(data=[go.Pie(
            labels=list(category_dist.keys()),
            values=list(category_dist.values()),
            hole=0.4,
            textinfo='percent+label',
            marker_colors=['#ef4444', '#eab308', '#f97316', '#a855f7', '#64748b', '#3b82f6']
        )])

        fig.update_layout(
            height=300,
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Improvement suggestions
    suggestions = failure_report.get('improvement_suggestions', [])
    if suggestions:
        st.markdown("**💡 Improvement Suggestions:**")
        for i, suggestion in enumerate(suggestions[:5]):
            st.markdown(f"  {i+1}. {suggestion}")


def render_consistency_panel(consistency_report: dict[str, Any]):
    """Render consistency test panel"""
    st.subheader("🔄 Consistency Test")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Overall Consistency",
            f"{consistency_report.get('overall_consistency', 0) * 100:.1f}%"
        )
    with col2:
        st.metric(
            "Overall Accuracy",
            f"{consistency_report.get('overall_accuracy', 0) * 100:.1f}%"
        )
    with col3:
        st.metric(
            "Stable Sample Ratio",
            f"{consistency_report.get('stable_sample_count', 0)}/{consistency_report.get('total_samples', 0)}"
        )

    # Suggestion
    recommendations = consistency_report.get('recommendations', [])
    if recommendations:
        st.info(" | ".join(recommendations))


def render_robustness_panel(robustness_report: dict[str, Any]):
    """Render robustness test panel"""
    st.subheader("🛡️ Robustness Test")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Accuracy",
            f"{robustness_report.get('original_accuracy', 0) * 100:.1f}%"
        )
    with col2:
        acc_drop = robustness_report.get('accuracy_drop', 0) * 100
        st.metric(
            "Accuracy Drop",
            f"{acc_drop:.1f}%",
            delta=f"-{acc_drop:.1f}%" if acc_drop > 0 else "0%",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            "Overall Robustness",
            f"{robustness_report.get('overall_robustness', 0) * 100:.1f}%"
        )

    # Sensitivity by perturbation type
    sensitivity = robustness_report.get('sensitivity_by_type', {})
    if sensitivity:
        st.markdown("**Perturbation Sensitivity Analysis:**")

        fig = go.Figure(data=[go.Bar(
            x=list(sensitivity.keys()),
            y=[v * 100 for v in sensitivity.values()],
            marker_color=['#ef4444' if v > 0.3 else '#eab308' if v > 0.1 else '#22c55e'
                         for v in sensitivity.values()],
            text=[f"{v*100:.0f}%" for v in sensitivity.values()],
            textposition='outside'
        )])

        fig.update_layout(
            height=250,
            margin={"l": 20, "r": 20, "t": 30, "b": 50},
            yaxis_title="Sensitivity (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"}
        )

        st.plotly_chart(fig, use_container_width=True)


def render_reasoning_quality_breakdown(results: list[dict[str, Any]]):
    """Render reasoning quality breakdown"""
    st.subheader("🧠 Reasoning Quality Analysis")

    if not results:
        st.info("No reasoning quality data yet")
        return

    # Extract average scores per dimension
    dimensions = ['coherence', 'completeness', 'relevance', 'correctness', 'efficiency']
    dim_labels = ['Coherence', 'Completeness', 'Relevance', 'Correctness', 'Efficiency']

    scores = []
    for dim in dimensions:
        values = [r.get(f'reasoning_{dim}', 0) for r in results if r.get(f'reasoning_{dim}', 0) > 0]
        scores.append(sum(values) / len(values) if values else 0)

    # Radar Chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  # Close the loop
        theta=dim_labels + [dim_labels[0]],
        fill='toself',
        fillcolor='rgba(96, 165, 250, 0.3)',
        line_color='#60a5fa',
        name='Reasoning Quality'
    ))

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 10]
            }
        },
        height=300,
        margin={"l": 50, "r": 50, "t": 30, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_sample_details_table(samples: list[dict[str, Any]], max_display: int = 20):
    """Render sample details table"""
    st.subheader("📝 Sample Details")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_option = st.selectbox(
            "Filter",
            ["All", "Correct Only", "Errors Only"],
            key="sample_filter"
        )
    with col2:
        sort_option = st.selectbox(
            "Sort",
            ["Default", "By Latency", "By Reasoning Quality"],
            key="sample_sort"
        )

    # Filter
    filtered = samples
    if filter_option == "Correct Only":
        filtered = [s for s in samples if s.get('is_correct')]
    elif filter_option == "Errors Only":
        filtered = [s for s in samples if not s.get('is_correct')]

    # Sort
    if sort_option == "By Latency":
        filtered = sorted(filtered, key=lambda x: x.get('latency_ms', 0), reverse=True)
    elif sort_option == "By Reasoning Quality":
        filtered = sorted(filtered, key=lambda x: x.get('reasoning_quality', 0), reverse=True)

    # Display
    for sample in filtered[:max_display]:
        with st.expander(
            f"{'✅' if sample.get('is_correct') else '❌'} Sample {sample.get('sample_id', 'N/A')}"
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Question:** {sample.get('question', '')[:200]}...")
                st.markdown(f"**Correct Answer:** {sample.get('correct_answer', '')}")
                st.markdown(f"**Predicted Answer:** {sample.get('predicted_answer', '')}")

            with col2:
                st.metric("Latency", f"{sample.get('latency_ms', 0):.0f}ms")
                st.metric("Reasoning Quality", f"{sample.get('reasoning_quality', 0):.1f}/10")

            if sample.get('reasoning_content'):
                st.markdown("**Reasoning process:**")
                st.code(sample['reasoning_content'][:500], language=None)


def render_export_section(results: dict[str, Any]):
    """Render export section"""
    st.subheader("📥 Export Report")

    import base64
    import json

    col1, col2, col3 = st.columns(3)

    with col1:
        json_str = json.dumps(results, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="evaluation_report.json">📄 Download JSON</a>'
        st.markdown(href, unsafe_allow_html=True)

    with col2:
        # Generate CSV
        if 'models' in results:
            rows = []
            for key, data in results['models'].items():
                rows.append({
                    "model": data.get('model_id', key),
                    "accuracy": data.get('accuracy', 0),
                    "latency_ms": data.get('avg_latency_ms', 0),
                    "reasoning_quality": data.get('avg_reasoning_quality', 0)
                })
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:text/csv;base64,{b64}" download="evaluation_summary.csv">📊 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    with col3:
        st.markdown("🔗 [View Full Report]()")


def render_mini_dashboard(
    accuracy: float,
    latency_ms: float,
    reasoning_quality: float,
    token_usage: dict[str, int] = None
):
    """
    Render mini dashboard (for sidebar or small space)
    """
    # Accuracy progress bar
    st.markdown("**Accuracy**")
    st.progress(accuracy)
    st.caption(f"{accuracy * 100:.1f}%")

    # Latency
    st.markdown("**Average Latency**")
    color = "#22c55e" if latency_ms < 1000 else "#eab308" if latency_ms < 3000 else "#ef4444"
    st.markdown(
        f'<div style="font-size: 24px; color: {color};">{latency_ms:.0f}ms</div>',
        unsafe_allow_html=True
    )

    # Reasoning Quality
    st.markdown("**Reasoning Quality**")
    st.markdown(
        f'<div style="font-size: 24px; color: #60a5fa;">{reasoning_quality:.1f}/10</div>',
        unsafe_allow_html=True
    )

    # Token use
    if token_usage:
        st.markdown("**Token use**")
        st.caption(f"Input: {token_usage.get('input', 0):,}")
        st.caption(f"Reasoning: {token_usage.get('reasoning', 0):,}")
        st.caption(f"Output: {token_usage.get('output', 0):,}")
