"""
Quality Test Reports Module
Quality test report module - generates quality assessment visualization reports
"""

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evaluators.base_evaluator import EvaluationResult


def generate_quality_summary(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    """
    Generate quality assessment summary table

    Args:
        results: Evaluation results dict {dataset_name: EvaluationResult}

    Returns:
        Summary DataFrame
    """
    if not results:
        return pd.DataFrame()

    data = []
    for name, result in results.items():
        stats = result.performance_stats or {}

        # Calculate AI judge correction count
        judge_corrected = sum(1 for s in result.details if getattr(s, 'is_judge_corrected', False))

        row = {
            "Dataset": name,
            "Model": result.model_id,
            "Accuracy": result.accuracy,
            "Correct": result.correct_samples,
            "Total Samples": result.total_samples,
            "Judge Corrections": judge_corrected, # New column
            "Duration (s)": round(result.duration_seconds, 1),
        }

        # Add performance metrics (if available)
        if stats:
            row["Avg TTFT(ms)"] = round(stats.get('avg_ttft_ms', 0), 0)
            row["Avg TPS"] = round(stats.get('avg_tps', 0), 1)
            row["Input Tokens"] = stats.get('total_input_tokens', 0)
            row["Output Tokens"] = stats.get('total_output_tokens', 0)

        row["Eval Time"] = result.timestamp
        data.append(row)

    df = pd.DataFrame(data)
    return df



def render_accuracy_chart(results: dict[str, EvaluationResult]) -> go.Figure:
    """
    Render accuracy bar chart
    """
    if not results:
        return go.Figure()

    datasets = list(results.keys())
    accuracies = [results[d].accuracy * 100 for d in datasets]

    fig = go.Figure(data=[
        go.Bar(
            x=datasets,
            y=accuracies,
            text=[f"{acc:.1f}%" for acc in accuracies],
            textposition='outside',
            marker_color='#4CAF50',
            marker_line_color='#2E7D32',
            marker_line_width=1
        )
    ])

    fig.update_layout(
        title="📊 Accuracy by Dataset",
        xaxis_title="Dataset",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        showlegend=False,
        height=400
    )

    return fig


def render_radar_chart(results: dict[str, EvaluationResult]) -> go.Figure:
    """
    Render capability radar chart (for multi-dataset evaluation)
    """
    if not results or len(results) < 3:
        return go.Figure()

    categories = list(results.keys())
    values = [results[d].accuracy * 100 for d in categories]

    # Close the radar chart
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Model Capability',
        line_color='#2196F3',
        fillcolor='rgba(33, 150, 243, 0.3)'
    ))

    fig.update_layout(
        polar={
            'radialaxis': {
                'visible': True,
                'range': [0, 100]
            }
        },
        title="🎯 Model Capability Radar",
        showlegend=False,
        height=450
    )

    return fig


def render_category_heatmap(result: EvaluationResult) -> go.Figure:
    """
    Render per-category accuracy heatmap
    """
    if not result.by_category:
        return go.Figure()

    # Extract data
    categories = list(result.by_category.keys())
    accuracies = [result.by_category[c].get('accuracy', 0) * 100 for c in categories]
    counts = [result.by_category[c].get('count', 0) for c in categories]

    # Sort by accuracy
    sorted_data = sorted(zip(categories, accuracies, counts, strict=False), key=lambda x: x[1], reverse=True)
    categories, accuracies, counts = zip(*sorted_data, strict=False) if sorted_data else ([], [], [])

    # Limit display count
    max_display = 20
    if len(categories) > max_display:
        categories = categories[:max_display]
        accuracies = accuracies[:max_display]
        counts = counts[:max_display]

    fig = go.Figure(data=[
        go.Bar(
            y=list(categories),
            x=list(accuracies),
            orientation='h',
            text=[f"{acc:.1f}% (n={cnt})" for acc, cnt in zip(accuracies, counts, strict=False)],
            textposition='outside',
            marker_color=list(accuracies),
            marker_colorscale='RdYlGn',
            marker_cmin=0,
            marker_cmax=100
        )
    ])

    fig.update_layout(
        title=f"📈 {result.dataset_name} Per-Category Accuracy (Top {min(len(categories), max_display)})",
        xaxis_title="Accuracy (%)",
        yaxis_title="Category",
        xaxis_range=[0, 110],
        template="plotly_white",
        height=max(400, len(categories) * 25),
        showlegend=False
    )

    return fig


def render_error_analysis(result: EvaluationResult, max_errors: int = 20) -> None:
    """
    Render enhanced error analysis panel

    Args:
        result: Evaluation result
        max_errors: No longer returns DataFrame directly, handles rendering and download here
    """
    # Extract error samples
    errors = [d for d in result.details if not d.is_correct]

    if not errors:
        st.success("🎉 Great! No error samples found in this dataset.")
        return

    # --- New: Showing automated failure analysis report ---
    if hasattr(result, 'extended_metrics') and result.extended_metrics and 'failure_analysis' in result.extended_metrics:
        fa = result.extended_metrics['failure_analysis']
        with st.expander("🤖 Automated Failure Analysis Report", expanded=True):
            st.markdown(f"**Overall Failure Rate**: `{fa.get('failure_rate', 0):.1f}%`")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🚨 Top Failure Causes")
                for issue in fa.get('top_issues', []):
                    st.write(f"- {issue}")
            
            with c2:
                st.markdown("##### 💡 Improvement Suggestions")
                for suggestion in fa.get('suggestions', []):
                    st.info(suggestion)
            
            # Distribution chart
            if fa.get('category_distribution'):
                dist = fa['category_distribution']
                fig_dist = px.pie(
                    values=list(dist.values()),
                    names=list(dist.keys()),
                    title="Failure Cause Distribution"
                )
                fig_dist.update_layout(height=300)
                st.plotly_chart(fig_dist, use_container_width=True)
    # ----------------------------------

    # 1. Control bar: filter and export
    col_filter, col_export = st.columns([3, 1])

    with col_filter:
        # Extract categories for filtering
        categories = sorted({err.category for err in errors})
        selected_category = st.selectbox(
            f"Filter Error Category (total {len(errors)} errors)",
            ["All"] + categories,
            key=f"err_filter_{result.dataset_name}_{result.model_id}"
        )

    # Filter data
    filtered_errors = errors if selected_category == "All" else [e for e in errors if e.category == selected_category]

    with col_export:
        # Build complete CSV data for download
        export_data = []
        for err in errors: # Export all errors, not just filtered ones
            export_data.append({
                "Sample ID": err.sample_id,
                "Category": err.category,
                "Question": err.question,
                "Correct Answer": err.correct_answer,
                "Predicted Answer": err.predicted_answer,
                "Full Prompt": err.prompt,
                "Full Response": err.model_response
            })

        if export_data:
            csv_df = pd.DataFrame(export_data)
            st.download_button(
                label="📥 Download Error Report",
                data=csv_df.to_csv(index=False),
                file_name=f"errors_{result.dataset_name}_{result.model_id}.csv",
                mime="text/csv",
                help="Download all error samples with prompts and full responses"
            )

    # 2. Error list overview (display section)
    st.markdown("#### ❌ Error List")

    display_data = []
    for err in filtered_errors[:max_errors]:
        display_data.append({
            "ID": err.sample_id,
            "Category": err.category,
            "Question Summary": (err.question[:60] + "...") if len(err.question) > 60 else err.question,
            "Correct Answer": err.correct_answer,
            "Model Prediction": err.predicted_answer
        })

    if display_data:
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
        if len(filtered_errors) > max_errors:
            st.caption(f"*Showing only the first {max_errors} items. Use the button above to download the full report or the tool below for details.*")

    # 3. Deep diagnosis tool
    st.markdown("#### 🕵️ Deep Diagnosis")
    st.caption("Select a sample ID to view the full prompt and model raw response for error analysis.")

    selected_error_id = st.selectbox(
        "Select Sample ID",
        options=[err.sample_id for err in filtered_errors],
        format_func=lambda x: f"ID: {x}",
        key=f"err_select_{result.dataset_name}_{result.model_id}"
    )

    # Find selected error details
    target_error = next((e for e in filtered_errors if e.sample_id == selected_error_id), None)

    if target_error:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Question (Prompt)**")
                st.text_area("Input", value=target_error.prompt, height=300, disabled=True, key=f"p_{selected_error_id}")
                st.markdown(f"**Correct Answer**: `{target_error.correct_answer}`")

            with c2:
                st.markdown("**Model Full Response**")
                st.text_area("Output", value=target_error.model_response, height=300, disabled=True, key=f"r_{selected_error_id}")
                st.markdown(f"**Extracted Prediction**: `{target_error.predicted_answer}`")


def render_performance_stats(results: dict[str, EvaluationResult]) -> None:
    """
    Render performance metrics panel
    """
    st.markdown("### ⚡ Performance Metrics")

    # Collect results with performance data
    perf_data = []
    for name, result in results.items():
        stats = result.performance_stats or {}
        if stats:
            perf_data.append({
                "Dataset": name,
                "Avg TTFT (ms)": round(stats.get('avg_ttft_ms', 0), 1),
                "Avg TPS": round(stats.get('avg_tps', 0), 1),
                "Max TPS": round(stats.get('max_tps', 0), 1),
                "Avg Latency (ms)": round(stats.get('avg_latency_ms', 0), 0),
                "Input Tokens": stats.get('total_input_tokens', 0),
                "Output Tokens": stats.get('total_output_tokens', 0),
                "Success Rate": f"{stats.get('success_rate', 0):.1%}"
            })

    if not perf_data:
        st.info("No performance metrics data yet")
        return

    # Performance summary table
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df)

    # Chart display
    col1, col2 = st.columns(2)

    with col1:
        # TTFT chart
        datasets = [d["Dataset"] for d in perf_data]
        ttfts = [d["Avg TTFT (ms)"] for d in perf_data]

        fig_ttft = go.Figure(data=[
            go.Bar(
                x=datasets,
                y=ttfts,
                text=[f"{t:.0f}ms" for t in ttfts],
                textposition='outside',
                marker_color='#FF9800'
            )
        ])
        fig_ttft.update_layout(
            title="⏱️ Average TTFT",
            xaxis_title="Dataset",
            yaxis_title="TTFT (ms)",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_ttft)

    with col2:
        # TPS chart
        tps_values = [d["Avg TPS"] for d in perf_data]

        fig_tps = go.Figure(data=[
            go.Bar(
                x=datasets,
                y=tps_values,
                text=[f"{t:.1f}" for t in tps_values],
                textposition='outside',
                marker_color='#2196F3'
            )
        ])
        fig_tps.update_layout(
            title="🚀 Average TPS",
            xaxis_title="Dataset",
            yaxis_title="Tokens/Second",
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_tps)


def render_quality_report(
    results: dict[str, EvaluationResult],
    model_id: str,
    show_details: bool = True
):
    """
    Render full quality assessment report within Streamlit

    Args:
        results: Evaluation result
        model_id: Model ID
        show_details: Whether to show detailed information
    """
    if not results:
        st.warning("No evaluation results yet")
        return

    # Title
    st.header("📋 Model Quality Assessment Report")
    st.subheader(f"Model: `{model_id}`")

    # Summary table
    st.markdown("### 📊 Evaluation Summary")
    summary_df = generate_quality_summary(results)

    # Format accuracy column
    if "Accuracy" in summary_df.columns:
        summary_df["Accuracy"] = summary_df["Accuracy"].apply(lambda x: f"{x:.2%}")

    st.dataframe(summary_df)

    # Accuracy bar chart
    col1, col2 = st.columns(2)

    with col1:
        accuracy_chart = render_accuracy_chart(results)
        st.plotly_chart(accuracy_chart)

    with col2:
        if len(results) >= 3:
            radar_chart = render_radar_chart(results)
            st.plotly_chart(radar_chart)
        else:
            st.info("Need at least 3 datasets to display radar chart")

    # Performance metrics panel
    render_performance_stats(results)

    # Per-category details
    if show_details:
        st.markdown("### 📈 Per-Category Analysis")

        for dataset_name, result in results.items():
            with st.expander(f"🔍 {dataset_name} Detailed Analysis", expanded=True):
                # Per-category accuracy
                if result.by_category:
                    heatmap = render_category_heatmap(result)
                    st.plotly_chart(heatmap)

                # AI Judge correction records
                corrected_samples = [s for s in result.details if getattr(s, "is_judge_corrected", False)]
                if corrected_samples:
                    st.info(f"⚖️ AI Judge successfully corrected {len(corrected_samples)} misjudged samples! (these are now included in the accuracy calculation)")
                    with st.expander("View Judge Correction Details", expanded=False):
                        for s in corrected_samples:
                            st.markdown(f"**Sample ID: {s.sample_id}**")
                            st.text(f"Question: {s.question[:80]}..." if s.question else "Question: (Prompt only)")
                            st.code(f"Model output: {s.model_response[:200]}...", language=None)
                            st.caption(f"Correct Answer: {s.correct_answer} | Extracted prediction (wrong): {s.predicted_answer}")
                            st.divider()

                # Error analysis
                render_error_analysis(result)

    # Export options
    st.markdown("### 💾 Export Results")

    col_export1, col_export2 = st.columns(2)

    with col_export1:
        # CSV Export
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary CSV",
            data=csv_data,
            file_name=f"quality_summary_{model_id}.csv",
            mime="text/csv"
        )

    with col_export2:
        # JSON Export
        import json
        json_data = {
            name: result.to_dict()
            for name, result in results.items()
        }
        st.download_button(
            label="📥 Download Detailed JSON",
            data=json.dumps(json_data, ensure_ascii=False, indent=2),
            file_name=f"quality_details_{model_id}.json",
            mime="application/json"
        )


def render_model_comparison(
    all_results: dict[str, dict[str, EvaluationResult]]
):
    """
    Render multi-model comparison report

    Args:
        all_results: {model_id: {dataset_name: EvaluationResult}}
    """
    if not all_results:
        st.warning("No comparison data yet")
        return

    st.header("📊 Model Capability Comparison")

    # Collect all datasets
    all_datasets = set()
    for model_results in all_results.values():
        all_datasets.update(model_results.keys())
    all_datasets = sorted(all_datasets)

    # Build comparison data
    comparison_data = []
    for model_id, model_results in all_results.items():
        row = {"Model": model_id}
        for dataset in all_datasets:
            if dataset in model_results:
                row[dataset] = model_results[dataset].accuracy
            else:
                row[dataset] = None
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Showing comparison table
    st.markdown("### Accuracy Comparison Table")

    # Format
    styled_df = df.copy()
    for col in all_datasets:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(
                lambda x: f"{x:.2%}" if x is not None else "N/A"
            )

    st.dataframe(styled_df)

    # Comparison bar chart
    st.markdown("### Visual Comparison")

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, (model_id, model_results) in enumerate(all_results.items()):
        datasets = list(model_results.keys())
        accuracies = [model_results[d].accuracy * 100 for d in datasets]

        fig.add_trace(go.Bar(
            name=model_id,
            x=datasets,
            y=accuracies,
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        barmode='group',
        title="Accuracy Comparison Across Models and Datasets",
        xaxis_title="Dataset",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig)
