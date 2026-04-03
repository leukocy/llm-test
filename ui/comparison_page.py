import os

import pandas as pd
import plotly.express as px
import streamlit as st

from core.history_manager import HistoryManager
from ui.charts import plot_performance_summary
from ui.reports import COLUMN_TOOLTIPS
from utils.helpers import reorder_dataframe_columns


def render_comparison_page():
    """Render the Historical Comparison UI."""
    st.header("📈 Historical Comparison")

    manager = HistoryManager()
    history = manager.list_history()

    if not history:
        st.info("No history yet. Please run some benchmark tests first.")
        return

    # --- Selection ---
    st.subheader("Select Records to Compare")

    # Create options for multiselect
    options = {item['display_name']: item['path'] for item in history}

    selected_names = st.multiselect(
        "Select Records (multi-select)",
        options=list(options.keys()),
        format_func=lambda x: f"{x} ({options[x].split(os.sep)[-2] if os.sep in options[x] else ''})"
    )

    if not selected_names:
        st.info("Please select at least two records to compare.")
        return

    selected_paths = [options[name] for name in selected_names]

    # --- Loading Data ---
    results_map = manager.load_results(selected_paths)

    if not results_map:
        st.error("Unable to load selected records.")
        return

    # Combine data for visualization
    combined_df = pd.DataFrame()
    for _name, df in results_map.items():
        # Ensure critical columns exist
        if 'ttft' in df.columns: # Basic check
             combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df.empty:
        st.warning("Selected records do not contain valid test data.")
        return

    # Backward compatibility: Rename 'system_throughput' to 'system_output_throughput' if needed
    if 'system_output_throughput' not in combined_df.columns and 'system_throughput' in combined_df.columns:
        combined_df['system_output_throughput'] = combined_df['system_throughput']
    # Ensure column exists
    if 'system_output_throughput' not in combined_df.columns:
        combined_df['system_output_throughput'] = 0


    st.markdown("---")

    # --- Visualizations ---
    st.subheader("📊 Comparison Charts")

    # 1. Radar Chart Comparison
    st.markdown("### Overall Performance Comparison (Radar Chart)")

    # Calculate aggregate metrics for Radar Chart
    # We want per-model averages for: TTFT, TPS, TPOT, Success Rate, System Throughput

    # Prepare data for Radar Chart
    radar_data = combined_df.groupby('source_file').agg({
        'ttft': 'min', # Best TTFT is better
        'tps': 'max',  # Max TPS is better
        'system_output_throughput': 'max',
        'error': lambda x: x.isnull().mean(), # Success Rate
        'prefill_speed': 'max'
    }).reset_index()

    radar_data = radar_data.rename(columns={
        'ttft': 'Best_TTFT',
        'tps': 'Max_Single_TPS',
        'system_output_throughput': 'Max_Sys_Output_Tps',
        'error': 'Success_Rate',
        'prefill_speed': 'Pre_Speed'
    })

    # Use 'source_file' (which contains filename) as the label
    radar_chart = plot_performance_summary(
        radar_data,
        metrics=['Best_TTFT', 'Max_Single_TPS', 'Max_Sys_Output_Tps', 'Success_Rate', 'Pre_Speed'],
        title="Multi-Model Performance Comparison"
    )
    if radar_chart:
        st.plotly_chart(radar_chart)

    st.markdown("---")
    st.markdown("### Individual Metric Details")
    metric = st.selectbox("Select Comparison Metric", ["ttft", "tps", "tpot", "prefill_speed", "system_output_throughput", "system_input_throughput"])

    # Check if metric exists in data
    if metric not in combined_df.columns:
        st.error(f"Metric '{metric}' does not exist in selected data.")
    else:
        # Group by source_file and calculate mean/median
        agg_df = combined_df.groupby('source_file')[metric].agg(['mean', 'median', 'std']).reset_index()

        # Bar Chart
        fig = px.bar(
            agg_df,
            x='source_file',
            y='mean',
            error_y='std',
            title=f"Average {metric.upper()} Comparison",
            labels={'mean': f'Average {metric.upper()}', 'source_file': 'Run'},
            color='source_file'
        )
        st.plotly_chart(fig)

        # Box Plot for distribution
        fig_box = px.box(
            combined_df,
            x='source_file',
            y=metric,
            title=f"{metric.upper()} Distribution",
            color='source_file'
        )
        st.plotly_chart(fig_box)

    # --- Data Table ---
    with st.expander("📋 View Detailed Data", expanded=False):
        if not combined_df.empty:
            combined_df = reorder_dataframe_columns(combined_df)

            # Use columns present in the dataframe
            list(combined_df.columns)

            st.dataframe(
                combined_df,
                width="stretch",
                column_config={
                     **{k: st.column_config.Column(help=v) for k, v in COLUMN_TOOLTIPS.items()},
                    "ttft": st.column_config.ProgressColumn(
                        "TTFT (s)",
                        format="%.4f s",
                        min_value=0,
                        max_value=combined_df['ttft'].max() * 1.2 if 'ttft' in combined_df else 1,
                    ),
                    "tps": st.column_config.ProgressColumn(
                        "TPS (t/s)",
                        format="%d t/s",
                        min_value=0,
                        max_value=combined_df['tps'].max() * 1.1 if 'tps' in combined_df else 100,
                    ),
                     "system_output_throughput": st.column_config.ProgressColumn(
                        "Sys Output Thrpt",
                        format="%d t/s",
                        min_value=0,
                        max_value=combined_df['system_output_throughput'].max() * 1.1 if 'system_output_throughput' in combined_df else 100,
                    ),
                    "error": "Error"
                }
            )
