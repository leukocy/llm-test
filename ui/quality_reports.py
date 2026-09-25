"""
Quality Test Reports Module
Quality test report module - generates quality assessment visualization reports
"""

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evaluators.base_evaluator import EvaluationResult
from ui.reporting.statistics import wilson_interval
from ui.reporting.theme import INDIGO, INK, PAPER, SUBTLE, TEAL
from utils.spreadsheet import safe_download_filename


def _positive_measurement(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _quality_accuracy(result: EvaluationResult) -> tuple[float | None, tuple[float, float] | None]:
    """Use observed counts for an unweighted accuracy and its Wilson interval."""
    total = result.total_samples
    correct = result.correct_samples
    interval = wilson_interval(correct, total)
    if interval is None:
        return None, None
    return correct / total, interval


def _quality_chart_style(fig: go.Figure, title: str, height: int = 370) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 17, "color": INK}},
        font={"family": "Inter, Segoe UI, PingFang SC, sans-serif", "color": INK},
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        margin={"l": 90, "r": 50, "t": 58, "b": 48},
        height=height,
        showlegend=False,
        xaxis={"title": "Accuracy (%)", "range": [0, 105], "gridcolor": SUBTLE, "zeroline": False},
        yaxis={"title": None, "showgrid": False},
    )
    return fig


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
        provenance = (result.config or {}).get("dataset_provenance", {})
        accuracy, interval = _quality_accuracy(result)

        # Calculate AI judge correction count
        judge_corrected = sum(1 for s in result.details if getattr(s, "is_judge_corrected", False))

        # Calculate evaluation method breakdown
        eval_methods = _compute_eval_method_breakdown(result.details)

        row = {
            "Dataset": name,
            "Dataset Source": provenance.get("source", "legacy_unverified"),
            "Sample SHA-256": provenance.get("sample_sha256", ""),
            "Few-shot SHA-256": provenance.get("few_shot_sha256", ""),
            "Model": result.model_id,
            "Accuracy": accuracy,
            "Correct": result.correct_samples,
            "Total Samples": result.total_samples,
            "Accuracy CI low (%)": interval[0] * 100 if interval else None,
            "Accuracy CI high (%)": interval[1] * 100 if interval else None,
            "Judge Corrections": judge_corrected,
            "Parse Methods": eval_methods,
            "Duration (s)": round(result.duration_seconds, 1),
        }

        # Add performance metrics (if available)
        if stats:
            ttft = _positive_measurement(stats.get("avg_ttft_ms"))
            tps = _positive_measurement(stats.get("avg_tps"))
            row["Avg TTFT(ms)"] = round(ttft, 1) if ttft is not None else None
            row["Avg TPS"] = round(tps, 1) if tps is not None else None
            row["Input Tokens"] = stats.get("total_input_tokens")
            row["Output Tokens"] = stats.get("total_output_tokens")

        row["Eval Time"] = result.timestamp
        data.append(row)

    df = pd.DataFrame(data)
    return df


def _compute_eval_method_breakdown(details: list) -> str:
    """Compute a compact summary of evaluation methods used across samples."""
    if not details:
        return ""
    from collections import Counter

    methods: Counter[str] = Counter()
    parse_methods: Counter[str] = Counter()
    for s in details:
        if getattr(s, "evaluation_method", ""):
            methods[s.evaluation_method] += 1
        if getattr(s, "answer_parse_method", ""):
            parse_methods[s.answer_parse_method] += 1

    # Combine into a single display string
    all_labels = []
    for m, c in methods.most_common(3):
        all_labels.append(f"{m}:{c}")
    for m, c in parse_methods.most_common(3):
        if m and m not in methods:
            all_labels.append(f"{m}:{c}")
    return ", ".join(all_labels) if all_labels else ""


def render_accuracy_chart(results: dict[str, EvaluationResult]) -> go.Figure:
    """Show dataset accuracy with sample counts and Wilson intervals."""
    if not results:
        return go.Figure()

    rows = []
    for name, result in results.items():
        accuracy, interval = _quality_accuracy(result)
        if accuracy is not None and interval is not None:
            rows.append(
                (name, accuracy * 100, interval[0] * 100, interval[1] * 100, result.total_samples)
            )
    if not rows:
        return go.Figure()
    names, values, lows, highs, counts = zip(*rows, strict=True)
    fig = go.Figure(
        go.Scatter(
            x=values,
            y=names,
            mode="markers",
            marker={"size": 12, "color": INDIGO},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [high - value for value, high in zip(values, highs, strict=True)],
                "arrayminus": [value - low for value, low in zip(values, lows, strict=True)],
                "color": TEAL,
                "thickness": 2,
                "width": 5,
            },
            customdata=list(zip(counts, lows, highs, strict=True)),
            hovertemplate="%{y}<br>Accuracy: %{x:.1f}%<br>n=%{customdata[0]}<br>95% Wilson: %{customdata[1]:.1f}–%{customdata[2]:.1f}%<extra></extra>",
        )
    )
    return _quality_chart_style(
        fig, "Accuracy and sampling uncertainty", max(300, 58 * len(rows) + 130)
    )


def render_radar_chart(results: dict[str, EvaluationResult]) -> go.Figure:
    """Show workload size without treating different datasets as a common capability scale."""
    if not results:
        return go.Figure()
    fig = go.Figure(
        go.Bar(
            x=[result.total_samples for result in results.values()],
            y=list(results),
            orientation="h",
            marker_color=TEAL,
            hovertemplate="%{y}<br>Evaluated samples: %{x:,}<extra></extra>",
        )
    )
    _quality_chart_style(fig, "Evaluated sample coverage", max(300, 58 * len(results) + 130))
    fig.update_xaxes(title="Samples", range=None)
    return fig


def render_category_heatmap(result: EvaluationResult) -> go.Figure:
    """
    Render per-category accuracy heatmap
    """
    if not result.by_category:
        return go.Figure()

    # Extract data (accuracy may be explicitly None in deserialized results)
    categories = list(result.by_category.keys())
    accuracies = []
    for category in categories:
        accuracy = result.by_category[category].get("accuracy")
        accuracies.append(accuracy * 100 if accuracy is not None else float("nan"))
    counts = [result.by_category[c].get("count", 0) for c in categories]

    # Sort by accuracy
    sorted_data = sorted(
        zip(categories, accuracies, counts, strict=False),
        key=lambda x: x[1] if pd.notna(x[1]) else -1,
        reverse=True,
    )
    if sorted_data:
        cat_seq, acc_seq, cnt_seq = zip(*sorted_data, strict=False)
        categories = list(cat_seq)
        accuracies = list(acc_seq)
        counts = list(cnt_seq)

    # Limit display count
    max_display = 20
    if len(categories) > max_display:
        categories = categories[:max_display]
        accuracies = accuracies[:max_display]
        counts = counts[:max_display]

    fig = go.Figure(
        data=[
            go.Bar(
                y=list(categories),
                x=list(accuracies),
                orientation="h",
                text=[
                    f"{acc:.1f}% (n={cnt})" if pd.notna(acc) else f"No score (n={cnt})"
                    for acc, cnt in zip(accuracies, counts, strict=False)
                ],
                textposition="outside",
                marker_color=INDIGO,
            )
        ]
    )

    fig.update_layout(
        title=f"{result.dataset_name} Per-Category Accuracy (Top {min(len(categories), max_display)})",
        xaxis_title="Accuracy (%)",
        yaxis_title="Category",
        xaxis_range=[0, 110],
        height=max(400, len(categories) * 25),
        showlegend=False,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font={"color": INK},
        xaxis={"gridcolor": SUBTLE},
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
        st.success("Great! No error samples found in this dataset.")
        return

    # --- New: Showing automated failure analysis report ---
    if (
        hasattr(result, "extended_metrics")
        and result.extended_metrics
        and "failure_analysis" in result.extended_metrics
    ):
        fa = result.extended_metrics["failure_analysis"]
        with st.expander("Automated Failure Analysis Report", expanded=True):
            st.markdown(f"**Overall Failure Rate**: `{fa.get('failure_rate', 0):.1f}%`")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Top Failure Causes")
                for issue in fa.get("top_issues", []):
                    st.write(f"- {issue}")

            with c2:
                st.markdown("##### Improvement Suggestions")
                for suggestion in fa.get("suggestions", []):
                    st.info(suggestion)

            # Distribution chart
            if fa.get("category_distribution"):
                dist = fa["category_distribution"]
                fig_dist = px.pie(
                    values=list(dist.values()),
                    names=list(dist.keys()),
                    title="Failure Cause Distribution",
                )
                fig_dist.update_layout(height=300)
                st.plotly_chart(fig_dist, use_container_width=True)
    # ----------------------------------

    # 1. Control bar: filter and export
    col_filter, col_export = st.columns([3, 1])

    with col_filter:
        # Extract categories for filtering (tolerate None categories)
        categories = sorted({str(err.category or "Uncategorized") for err in errors})
        selected_category = st.selectbox(
            f"Filter Error Category (total {len(errors)} errors)",
            ["All"] + categories,
            key=f"err_filter_{result.dataset_name}_{result.model_id}",
        )

    # Filter data
    filtered_errors = (
        errors
        if selected_category == "All"
        else [e for e in errors if str(e.category or "Uncategorized") == selected_category]
    )

    with col_export:
        # Build complete CSV data for download
        export_data = []
        for err in errors:  # Export all errors, not just filtered ones
            row = {
                "Sample ID": err.sample_id,
                "Category": err.category,
                "Question": err.question,
                "Correct Answer": err.correct_answer,
                "Predicted Answer": err.predicted_answer,
                "Full Prompt": err.prompt,
                "Full Response": err.model_response,
            }
            # Include evaluation metadata if available
            eval_method = getattr(err, "evaluation_method", "")
            parse_method = getattr(err, "answer_parse_method", "")
            confidence = getattr(err, "answer_parse_confidence", 0) or 0
            if eval_method:
                row["Evaluation Method"] = eval_method
            if parse_method:
                row["Parse Method"] = parse_method
            if confidence > 0:
                row["Parse Confidence"] = f"{confidence:.2f}"
            export_data.append(row)

        if export_data:
            from ui.export import safe_csv_bytes

            csv_df = pd.DataFrame(export_data)
            st.download_button(
                label="Download Error Report",
                data=safe_csv_bytes(csv_df),
                file_name=safe_download_filename(
                    f"errors_{result.dataset_name}_{result.model_id}.csv"
                ),
                mime="text/csv",
                help="Download all error samples with prompts and full responses",
            )

    # 2. Error list overview (display section)
    st.markdown("#### Error List")

    display_data = []
    for err in filtered_errors[:max_errors]:
        row = {
            "ID": err.sample_id,
            "Category": err.category,
            "Question Summary": (
                (err.question[:60] + "...") if len(err.question) > 60 else err.question
            ),
            "Correct Answer": err.correct_answer,
            "Model Prediction": err.predicted_answer,
        }
        # Add parse method and confidence if available
        parse_method = getattr(err, "answer_parse_method", "")
        if parse_method:
            row["Parse Method"] = parse_method
        confidence = getattr(err, "answer_parse_confidence", 0) or 0
        if confidence > 0:
            row["Confidence"] = f"{confidence:.0%}"
        display_data.append(row)

    if display_data:
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
        if len(filtered_errors) > max_errors:
            st.caption(
                f"*Showing only the first {max_errors} items. Use the button above to download the full report or the tool below for details.*"
            )

    # 3. Deep diagnosis tool
    st.markdown("#### Deep Diagnosis")
    st.caption(
        "Select a sample ID to view the full prompt and model raw response for error analysis."
    )

    selected_error_id = st.selectbox(
        "Select Sample ID",
        options=[err.sample_id for err in filtered_errors],
        format_func=lambda x: f"ID: {x}",
        key=f"err_select_{result.dataset_name}_{result.model_id}",
    )

    # Find selected error details
    target_error = next((e for e in filtered_errors if e.sample_id == selected_error_id), None)

    if target_error:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Question (Prompt)**")
                st.text_area(
                    "Input",
                    value=target_error.prompt,
                    height=300,
                    disabled=True,
                    key=f"p_{selected_error_id}",
                )
                st.markdown(f"**Correct Answer**: `{target_error.correct_answer}`")

            with c2:
                st.markdown("**Model Full Response**")
                st.text_area(
                    "Output",
                    value=target_error.model_response,
                    height=300,
                    disabled=True,
                    key=f"r_{selected_error_id}",
                )

                # Parse details
                parse_method = getattr(target_error, "answer_parse_method", "")
                confidence = getattr(target_error, "answer_parse_confidence", 0) or 0
                eval_method = getattr(target_error, "evaluation_method", "")

                st.markdown(f"**Extracted Prediction**: `{target_error.predicted_answer}`")

                if parse_method or eval_method:
                    detail_parts = []
                    if eval_method:
                        detail_parts.append(f"Eval: {eval_method}")
                    if parse_method:
                        detail_parts.append(f"Parse: {parse_method}")
                    if confidence > 0:
                        detail_parts.append(f"Confidence: {confidence:.0%}")
                    st.caption(" | ".join(detail_parts))


def render_eval_method_breakdown(result: EvaluationResult) -> None:
    """Render evaluation method and parse method breakdown for a dataset."""
    if not result.details:
        return

    from collections import Counter

    eval_methods: Counter[str] = Counter()
    parse_methods: Counter[str] = Counter()
    confidences: list = []

    for s in result.details:
        em = getattr(s, "evaluation_method", "")
        pm = getattr(s, "answer_parse_method", "")
        conf = getattr(s, "answer_parse_confidence", 0) or 0
        if em:
            eval_methods[em] += 1
        if pm:
            parse_methods[pm] += 1
        if conf > 0:
            confidences.append(conf)

    if not eval_methods and not parse_methods:
        return

    st.markdown("##### Evaluation Method Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        if eval_methods:
            method_data = [{"Method": m, "Count": c} for m, c in eval_methods.most_common()]
            st.dataframe(pd.DataFrame(method_data), hide_index=True, use_container_width=True)
        else:
            st.caption("No evaluation method data")

    with col2:
        if parse_methods:
            parse_data = [{"Parse Method": m, "Count": c} for m, c in parse_methods.most_common()]
            st.dataframe(pd.DataFrame(parse_data), hide_index=True, use_container_width=True)
        else:
            st.caption("No parse method data")

    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        high_conf = sum(1 for c in confidences if c >= 0.8)
        low_conf = sum(1 for c in confidences if c < 0.5)
        st.caption(
            f"Parse confidence: avg {avg_conf:.0%} | "
            f"high (>=80%): {high_conf} | low (<50%): {low_conf} | "
            f"total: {len(confidences)}"
        )


def render_performance_stats(results: dict[str, EvaluationResult]) -> None:
    """Render sample-level timing distributions when measurements are present."""
    st.markdown("### Performance Metrics")
    perf_df = build_quality_performance_summary(results)
    if perf_df.empty:
        st.info("No performance metrics data yet")
        return
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.caption(
        "P50/P95 TTFT and P50/P10 TPS use successful samples with positive measurements. "
        "Aggregate averages are shown only when supplied by the evaluator. "
        "Dataset workloads may differ."
    )

    for fig in build_quality_performance_figures(perf_df):
        st.plotly_chart(fig, use_container_width=True)


def build_quality_performance_figures(perf_df: pd.DataFrame) -> list[go.Figure]:
    """Build the same sample distribution charts for screen and HTML export."""
    if perf_df.empty:
        return []
    figures = []
    chart_specs = (
        ("TTFT (ms)", "TTFT P50 (ms)", "TTFT P95 (ms)", "TTFT valid n"),
        ("TPS (tokens/s)", "TPS P50", "TPS P10", "TPS valid n"),
    )
    for title, median_col, tail_col, n_col in chart_specs:
        if median_col not in perf_df or perf_df[median_col].notna().sum() == 0:
            continue
        fig = go.Figure()
        for column, label, color, symbol in (
            (median_col, "P50", INDIGO, "circle"),
            (tail_col, "P95" if "P95" in tail_col else "P10", TEAL, "diamond"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=perf_df[column],
                    y=perf_df["Dataset"],
                    mode="markers",
                    name=label,
                    marker={"color": color, "size": 10, "symbol": symbol},
                    customdata=perf_df[[n_col]].to_numpy(),
                    hovertemplate="%{y}<br>"
                    + label
                    + ": %{x:,.2f}<br>Valid n: %{customdata[0]}<extra></extra>",
                )
            )
        _quality_chart_style(fig, title, max(300, 52 * len(perf_df) + 130))
        fig.update_xaxes(title=title, range=None)
        fig.update_layout(
            showlegend=True, legend={"orientation": "h", "y": 1.13, "x": 1, "xanchor": "right"}
        )
        figures.append(fig)
    return figures


def build_quality_performance_summary(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    """Keep aggregate averages distinct from measured sample quantiles."""
    records = []
    for name, result in results.items():
        valid = [sample for sample in result.details if not sample.error]
        ttft = pd.Series(
            [
                value
                for sample in valid
                if (value := _positive_measurement(sample.ttft_ms)) is not None
            ],
            dtype=float,
        )
        tps = pd.Series(
            [value for sample in valid if (value := _positive_measurement(sample.tps)) is not None],
            dtype=float,
        )
        stats = result.performance_stats or {}
        average_ttft = _positive_measurement(stats.get("avg_ttft_ms"))
        average_tps = _positive_measurement(stats.get("avg_tps"))
        if not len(ttft) and not len(tps) and average_ttft is None and average_tps is None:
            continue
        records.append(
            {
                "Dataset": name,
                "TTFT valid n": len(ttft),
                "TTFT P50 (ms)": float(ttft.median()) if len(ttft) else None,
                "TTFT P95 (ms)": float(ttft.quantile(0.95)) if len(ttft) else None,
                "TPS valid n": len(tps),
                "TPS P50": float(tps.median()) if len(tps) else None,
                "TPS P10": float(tps.quantile(0.10)) if len(tps) else None,
                "Avg TTFT (ms)": average_ttft,
                "Avg TPS": average_tps,
            }
        )
    return pd.DataFrame.from_records(records)


def render_quality_report(
    results: dict[str, EvaluationResult], model_id: str, show_details: bool = True
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
    st.header("Model Quality Assessment Report")
    st.subheader(f"Model: `{model_id}`")

    demo_datasets = [
        name
        for name, result in results.items()
        if (result.config or {}).get("dataset_provenance", {}).get("source") == "embedded_demo"
    ]
    if demo_datasets:
        st.warning(
            "Demo samples were used for: "
            + ", ".join(demo_datasets)
            + ". These scores are for demonstration only."
        )

    # Summary table
    st.markdown("### Evaluation Summary")
    summary_df = generate_quality_summary(results)

    display_df = summary_df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].map(
        lambda value: f"{value:.2%}" if pd.notna(value) else "Unavailable"
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(
        "Accuracy is correct / evaluated samples. Intervals use the 95% Wilson method; "
        "they describe sampling uncertainty only when items can be treated as approximately independent. "
        "Dataset scores represent different tasks and should not be averaged into one capability score."
    )
    for name, result in results.items():
        accuracy, _ = _quality_accuracy(result)
        if accuracy is not None and abs(accuracy - result.accuracy) > 1e-6:
            st.warning(
                f"{name}: stored accuracy differs from correct / total; charts and summary use the observed counts."
            )
        if result.total_samples < 20:
            st.warning(
                f"{name}: only {result.total_samples} evaluated samples; interpret the score cautiously."
            )

    # Accuracy bar chart
    col1, col2 = st.columns(2)

    with col1:
        accuracy_chart = render_accuracy_chart(results)
        st.plotly_chart(accuracy_chart)

    with col2:
        coverage_chart = render_radar_chart(results)
        st.plotly_chart(coverage_chart, use_container_width=True)

    # Performance metrics panel
    render_performance_stats(results)

    # Per-category details
    if show_details:
        st.markdown("### Per-Category Analysis")

        for dataset_name, result in results.items():
            with st.expander(f"{dataset_name} Detailed Analysis", expanded=True):
                # Per-category accuracy
                if result.by_category:
                    heatmap = render_category_heatmap(result)
                    st.plotly_chart(heatmap)

                # Evaluation method breakdown
                render_eval_method_breakdown(result)

                # AI Judge correction records
                corrected_samples = [
                    s for s in result.details if getattr(s, "is_judge_corrected", False)
                ]
                if corrected_samples:
                    st.info(
                        f"AI Judge successfully corrected {len(corrected_samples)} misjudged samples! (these are now included in the accuracy calculation)"
                    )
                    with st.expander("View Judge Correction Details", expanded=False):
                        for s in corrected_samples:
                            st.markdown(f"**Sample ID: {s.sample_id}**")
                            st.text(
                                f"Question: {s.question[:80]}..."
                                if s.question
                                else "Question: (Prompt only)"
                            )
                            st.code(
                                f"Model output: {s.model_response[:200]}...",
                                language=None,
                            )
                            st.caption(
                                f"Correct Answer: {s.correct_answer} | Extracted prediction (wrong): {s.predicted_answer}"
                            )
                            st.divider()

                # Error analysis
                render_error_analysis(result)

    # Export options
    st.markdown("### Export Results")

    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        # CSV Export
        from ui.export import safe_csv_bytes

        csv_data = safe_csv_bytes(summary_df)
        st.download_button(
            label="Download Summary CSV",
            data=csv_data,
            file_name=safe_download_filename(f"quality_summary_{model_id}.csv"),
            mime="text/csv",
        )

    with col_export2:
        # JSON Export
        import json

        json_data = {name: result.to_dict() for name, result in results.items()}
        st.download_button(
            label="Download Detailed JSON",
            data=json.dumps(json_data, ensure_ascii=False, indent=2),
            file_name=safe_download_filename(f"quality_details_{model_id}.json"),
            mime="application/json",
        )

    with col_export3:
        from ui.reporting.html_export import build_html_report

        perf_df = build_quality_performance_summary(results)
        tables = [("Quality outcomes", summary_df)]
        if not perf_df.empty:
            tables.append(("Sample-level performance", perf_df))
        html_report = build_html_report(
            [
                render_accuracy_chart(results),
                render_radar_chart(results),
                *build_quality_performance_figures(perf_df),
            ],
            tables,
            [
                "Method: Accuracy uses correct / evaluated samples with 95% Wilson intervals.",
                "Comparability: Datasets cover different tasks; compare like workloads and inspect sample counts.",
                "Timing: P50/P95 TTFT and P50/P10 TPS use positive measurements from successful samples only.",
            ],
            title=f"Model quality assessment - {model_id}",
            overview_cards=(
                ("Model", model_id, "Evaluated model identifier"),
                ("Datasets", str(len(results)), "Distinct evaluation workloads"),
                (
                    "Evaluated samples",
                    f"{sum(r.total_samples for r in results.values()):,}",
                    "Across all datasets",
                ),
                (
                    "Correct answers",
                    f"{sum(r.correct_samples for r in results.values()):,}",
                    "Count only; dataset scores remain separate",
                ),
            ),
        )
        st.download_button(
            label="Download quality HTML report",
            data=html_report,
            file_name=safe_download_filename(f"quality_report_{model_id}.html"),
            mime="text/html",
        )
