import colorsys
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.metric_sanitizer import sanitize_performance_metrics
from ui.reporting.theme import AMBER, BLUE, INDIGO, INK, MUTED, PAPER, SUBTLE, TEAL


# ===== SMART VALUE FORMATTING =====
def get_smart_format_string(max_value):
    """
    Returns the appropriate format string based on max value.
    - max_value > 100: integer (.0f)
    - max_value > 10: 1 decimal place (.1f)
    - other: 2 decimal places (.2f)
    """
    if max_value > 100:
        return ".0f"
    elif max_value > 10:
        return ".1f"
    else:
        return ".2f"


def smart_format_value(value, max_value=None):
    """Format a single value"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if max_value is None:
        max_value = abs(value) if value else 0

    if max_value > 100:
        return f"{value:.0f}"
    elif max_value > 10:
        return f"{value:.1f}"
    else:
        return f"{value:.2f}"


# ===== COLOR HELPERS =====
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def generate_color_gradient(base_hex, n_steps):
    """
    Generates a list of n_steps colors forming a gradient based on the base_hex.
    The gradient goes from light/faint to the full base color (or darker).
    """
    if n_steps < 1:
        return []
    if n_steps == 1:
        return [base_hex]

    base_rgb = hex_to_rgb(base_hex)
    # Convert to HSV to manipulate saturation/value
    h, s, v = colorsys.rgb_to_hsv(base_rgb[0] / 255.0, base_rgb[1] / 255.0, base_rgb[2] / 255.0)

    colors = []
    # Strategy: Vary Saturation and Value to create distinct shades.
    # We want to represent intensity.
    # For Concurrency: Low (1) -> High (N).
    # Typically, light colors = low intensity, dark/saturated = high intensity.

    for i in range(n_steps):
        # Linear interpolation factor
        t = i / (n_steps - 1)

        # New Strategy:
        # Start (Low Concurrency): Less Saturated (paler), Higher Value (lighter)
        # End (High Concurrency): Fully Saturated, Slightly Darker Value

        new_s = s * (0.4 + 0.6 * t)  # Saturation: 40% -> 100%
        new_v = v * (1.0 - 0.3 * t)  # Value: 100% -> 70% (Darker)

        r, g, b = colorsys.hsv_to_rgb(h, new_s, new_v)
        colors.append(rgb_to_hex((r * 255, g * 255, b * 255)))

    return colors


# ===== UNIFIED CHART THEME =====
CHART_THEME = {
    "plot_bgcolor": PAPER,
    "paper_bgcolor": PAPER,
    "font": {
        "family": 'Inter, "Segoe UI", "PingFang SC", sans-serif',
        "size": 12,
        "color": INK,
    },
    "title": {
        "font": {"size": 17, "color": INK},
        "x": 0,
        "xanchor": "left",
        "pad": {"b": 10},
    },
    "xaxis": {
        "showgrid": False,
        "showline": True,
        "linewidth": 1,
        "linecolor": SUBTLE,
        "title_font": {"size": 13, "color": MUTED},
        "tickfont": {"size": 11},
    },
    "yaxis": {
        "showgrid": True,
        "gridcolor": SUBTLE,
        "gridwidth": 1,
        "showline": True,
        "linewidth": 1,
        "linecolor": SUBTLE,
        "title_font": {"size": 13, "color": MUTED},
        "tickfont": {"size": 11},
    },
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": "white",
        "font_size": 11,
        "font_family": "monospace",
        "bordercolor": INDIGO,
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "bgcolor": "rgba(255, 255, 255, 0.8)",
        "bordercolor": SUBTLE,
        "borderwidth": 1,
    },
    "margin": {"l": 60, "r": 30, "t": 70, "b": 60},
}

# Color palette
COLORS = {
    "primary": INDIGO,
    "success": TEAL,
    "warning": AMBER,
    "danger": "#b42318",
    "info": BLUE,
    "gradient": [INDIGO, BLUE, TEAL, AMBER],
}


def apply_theme(fig):
    """Apply unified theme to plotly figure."""
    fig.update_layout(**CHART_THEME)
    return fig


def plot_plotly_line(
    df,
    x,
    y,
    title,
    xlabel,
    ylabel,
    model_id,
    total_runs,
    color=None,
    hover_data=None,
    error_y_col=None,
    show_relative=False,
    force_linear_scale=False,
    provider=None,
    line_color=None,
):
    try:
        df = sanitize_performance_metrics(df, [y])

        # Legacy report builders carry a short string label for display. Use
        # the underlying numeric workload size so spacing has real meaning.
        if x == "concurrency_str" and "concurrency" in df:
            plot_x = "concurrency"
        elif x == "x_label" and "context_length_target" in df:
            plot_x = "context_length_target"
        elif x == "x_label" and "input_tokens_target" in df:
            plot_x = "input_tokens_target"
        else:
            plot_x = x
        numeric_x = pd.api.types.is_numeric_dtype(df[plot_x])
        if numeric_x:
            df = df.sort_values(plot_x)

        # Simplified title,removed hardware/model info subtitle for clean UI
        fig_title = title

        # Determine target color
        legacy_colors = {"#4bc0c0": TEAL, "#ff9f40": AMBER, "#36a2eb": BLUE}
        target_color = (
            legacy_colors.get(str(line_color).lower(), line_color) if line_color else INDIGO
        )

        # Color sequence logic
        color_seq = None
        if color is not None and line_color is not None:
            # If group column specified(color) and base color(line_color)
            # generate gradient based on base color
            n_groups = df[color].nunique()
            color_seq = generate_color_gradient(target_color, n_groups)
        elif color is None:
            color_seq = [target_color]  # Single line single color

        # Create chart
        fig = px.line(
            df,
            x=plot_x,
            y=y,
            title=fig_title,
            labels={plot_x: xlabel, y: ylabel},
            markers=True,
            color=color,
            color_discrete_sequence=color_seq,
            hover_data=hover_data,
            error_y=error_y_col,
        )

        # Add relative performance view
        if (
            show_relative and y in df.columns and color is None
        ):  # Relative perf view currently does not support multiple lines
            best_value = df[y].max()
            if pd.notna(best_value) and best_value > 0:
                df_rel = df.copy()
                df_rel[f"{y}_relative"] = (df[y] / best_value * 100).round(1)

                # Add secondary relative performance line
                fig.add_trace(
                    go.Scatter(
                        x=df_rel[plot_x],
                        y=df_rel[f"{y}_relative"],
                        mode="lines+markers",
                        name="Relative Perf (%)",
                        line={
                            "color": AMBER,
                            "dash": "dash",
                        },  # Yellow for relative
                        marker={"size": 7, "color": AMBER},
                        yaxis="y2",
                    )
                )

                # Set dual y-axis
                fig.update_layout(
                    yaxis2={
                        "title": "Relative Perf (%)",
                        "overlaying": "y",
                        "side": "right",
                        "range": [0, 105],
                        "ticksuffix": "%",
                    }
                )

                # Update main y-axis label
                if "System Throughput" in ylabel or "Throughput" in ylabel:
                    ylabel_with_rel = f"{ylabel} (max=100%)"
                else:
                    ylabel_with_rel = f"{ylabel} (Relative to Best)"

                fig.update_layout(yaxis_title=ylabel_with_rel)

        # Improved axis normalization
        y_values = pd.to_numeric(df[y], errors="coerce").to_numpy()
        if error_y_col:
            err_values = pd.to_numeric(df[error_y_col], errors="coerce").fillna(0).to_numpy()
            y_values = np.concatenate([y_values, y_values + err_values])
        finite_y = y_values[np.isfinite(y_values)]
        y_max = np.max(finite_y) if len(finite_y) > 0 else 1
        y_min = np.min(finite_y) if len(finite_y) > 0 else 0

        # Dynamically set Y-axis range. All-zero/all-negative data must still
        # produce a valid (lower < upper) axis range for plotly.
        y_padding = (y_max - y_min) * 0.1 if y_max != y_min else abs(y_max) * 0.1
        y_lower = max(0, y_min - y_padding)
        y_upper = y_max + y_padding
        if y_upper <= y_lower:
            y_upper = y_lower + max(abs(y_lower), 1.0) * 0.1 or 1.0
        fig.update_yaxes(range=[y_lower, y_upper])

        # Improved log scale detection logic
        # 1. Filter out <= 0 values to avoid 0 values causing ratio calculation errors
        positive_y = y_values[y_values > 0]
        calc_min = np.min(positive_y) if len(positive_y) > 0 else y_max

        # 2. Calculate ratio
        ratio = y_max / calc_min if calc_min > 0 else 0

        # 3. Only when ratio is extremely large (> 100,000) enable log scale
        if not force_linear_scale and (ratio > 100000):
            fig.update_yaxes(
                type="log",
                range=[math.log10(calc_min) - 0.08, math.log10(y_max) + 0.08],
            )
            fig_title = title + " [Log Scale]"

        # Improved layout and styles
        fig.update_layout(
            title_x=0,  # Left-align title
            xaxis_title=xlabel,
            plot_bgcolor=PAPER,
            paper_bgcolor=PAPER,
            font_color=INK,
            xaxis={
                "title_font": {"size": 14, "weight": "bold"},
                "tickfont": {"size": 12},
                "type": "linear" if numeric_x else "category",
                "showgrid": False,
            },
            yaxis={
                "title_font": {"size": 14, "weight": "bold"},
                "tickfont": {"size": 12},
                "gridcolor": SUBTLE,
                "showgrid": True,
            },
        )

        # Distinguish confidence interval and error bars
        # Common style update - Use smart formatting
        fig.update_traces(
            error_y_color=AMBER,
            error_y_thickness=2,
            error_y_width=3,
        )

        # Special style for single line (custom color)
        if color is None:
            fig.update_traces(
                marker={
                    "size": 8,
                    "color": target_color,
                    "line": {"width": 1, "color": PAPER},
                },
                line={"width": 2.5, "color": target_color},
            )
        else:
            fig.update_traces(
                marker={"size": 7, "line": {"width": 1, "color": PAPER}},
                line={"width": 2},
            )

        # Title and layout enhancement
        fig.update_layout(
            title={
                "text": fig_title,
                "font": {
                    "size": 17,
                    "color": INK,
                    "family": "Inter, Segoe UI, sans-serif",
                },
                "x": 0,
                "y": 0.95,
            },
            xaxis={
                "title_font": {"size": 14, "weight": "bold"},
                "tickfont": {"size": 12},
                "type": "linear" if numeric_x else "category",
                "showgrid": False,
            },
            yaxis={
                "title_font": {"size": 14, "weight": "bold"},
                "tickfont": {"size": 12},
            },
        )

        return fig
    except Exception as e:
        st.error(f"Plotly line chart failed: {e}")
        return None


def plot_performance_summary(df, metrics, title="Performance Summary"):
    """
    Multi-metric performance summary chart (Radar Chart).

    Shows multiple metrics in a single radar/spider chart.
    Metrics are normalized to 0-100 relative to the dataset max/min.
    """
    try:
        if not metrics:
            st.warning("No metrics provided for performance summary chart.")
            return None

        # Normalize metrics to 0-100 scale
        df_norm = df.copy()

        # Define 'higher is better' metrics vs 'lower is better'
        higher_is_better = [
            "tps",
            "system_output_throughput",
            "system_input_throughput",
            "system_throughput",
            "prefill_speed",
            "rps",
            "success_rate",
        ]

        for metric in metrics:
            if metric in df.columns:
                min_val = df[metric].min()
                max_val = df[metric].max()

                if pd.isna(min_val) or pd.isna(max_val):
                    # All-NaN column: nothing to normalize
                    df_norm[f"{metric}_norm"] = 0
                elif max_val == min_val:
                    df_norm[f"{metric}_norm"] = 100  # If all same, give full score
                else:
                    if any(h in metric.lower() for h in higher_is_better):
                        # Higher is better: (val - min) / (max - min) * 100
                        df_norm[f"{metric}_norm"] = (
                            (df[metric] - min_val) / (max_val - min_val)
                        ) * 100
                    else:
                        # Lower is better (Latency): (max - val) / (max - min) * 100
                        # So lowest value gets 100, highest gets 0
                        df_norm[f"{metric}_norm"] = (
                            (max_val - df[metric]) / (max_val - min_val)
                        ) * 100

        # Create radar chart
        fig = go.Figure()

        for idx, row in df_norm.iterrows():
            # Get original values for hover text
            values_norm = []
            values_text = []

            for m in metrics:
                if m in df.columns:
                    values_norm.append(row.get(f"{m}_norm", 0))
                    original_val = row[m]
                    values_text.append(f"{original_val:.2f}")
                else:
                    values_norm.append(0)
                    values_text.append("N/A")

            # Close the loop
            values_norm.append(values_norm[0])
            metrics_labels = metrics + [metrics[0]]

            name_label = str(row.get(df.columns[0], idx))
            if "session_id" in df.columns and len(str(row.get("session_id", ""))) > 8:
                # If session ID is used as name, shorten it
                name_label = str(row["session_id"])[:8] + "..."
            elif "concurrency" in df.columns:
                name_label = f"Concurrency {row['concurrency']}"

            fig.add_trace(
                go.Scatterpolar(
                    r=values_norm,
                    theta=metrics_labels,
                    fill="toself",
                    name=name_label,
                    text=values_text,
                    hoverinfo="text+name",
                )
            )

        fig.update_layout(
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": [0, 100],
                    "showticklabels": False,  # Hide normalized numbers to avoid confusion
                }
            },
            title={"text": title, "x": 0.5},
            showlegend=True,
            margin={"l": 80, "r": 80, "t": 50, "b": 50},  # Increase margins for labels
        )

        fig = apply_theme(fig)

        return fig
    except Exception as e:
        st.error(f"Performance summary chart failed: {e}")
        return None
