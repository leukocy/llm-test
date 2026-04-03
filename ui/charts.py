import colorsys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ===== SMART VALUE FORMATTING =====
def get_smart_format_string(max_value):
    """
    Returns the appropriate format string based on max value.
    - max_value > 100: integer (.0f)
    - max_value > 10: 1 decimal place (.1f)
    - other: 2 decimal places (.2f)
    """
    if max_value > 100:
        return '.0f'
    elif max_value > 10:
        return '.1f'
    else:
        return '.2f'


def smart_format_value(value, max_value=None):
    """Format a single value"""
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
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'

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
    h, s, v = colorsys.rgb_to_hsv(base_rgb[0]/255.0, base_rgb[1]/255.0, base_rgb[2]/255.0)

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

        new_s = s * (0.4 + 0.6 * t)       # Saturation: 40% -> 100%
        new_v = v * (1.0 - 0.3 * t)       # Value: 100% -> 70% (Darker)

        r, g, b = colorsys.hsv_to_rgb(h, new_s, new_v)
        colors.append(rgb_to_hex((r*255, g*255, b*255)))

    return colors

# ===== UNIFIED CHART THEME =====
CHART_THEME = {
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'font': {
        'family': 'Arial, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        'size': 12,
        'color': '#333'
    },
    'title': {
        'font': {'size': 16, 'color': '#1a1a1a', 'family': 'Arial'},
        'x': 0.5,
        'xanchor': 'center',
        'pad': {'b': 10}
    },
    'xaxis': {
        'showgrid': False,
        'showline': True,
        'linewidth': 1,
        'linecolor': '#e6e9ef',
        'title_font': {'size': 13, 'color': '#555'},
        'tickfont': {'size': 11}
    },
    'yaxis': {
        'showgrid': True,
        'gridcolor': '#f0f2f6',
        'gridwidth': 1,
        'showline': True,
        'linewidth': 1,
        'linecolor': '#e6e9ef',
        'title_font': {'size': 13, 'color': '#555'},
        'tickfont': {'size': 11}
    },
    'hovermode': 'x unified',
    'hoverlabel': {
        'bgcolor': 'white',
        'font_size': 11,
        'font_family': 'monospace',
        'bordercolor': '#007bff'
    },
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'right',
        'x': 1,
        'bgcolor': 'rgba(255, 255, 255, 0.8)',
        'bordercolor': '#e6e9ef',
        'borderwidth': 1
    },
    'margin': {'l': 60, 'r': 40, 't': 80, 'b': 60}
}

# Color palette
COLORS = {
    'primary': '#007bff',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'gradient': ['#007bff', '#0056b3', '#003d82', '#002455']
}


def apply_theme(fig):
    """Apply unified theme to plotly figure."""
    fig.update_layout(**CHART_THEME)
    return fig

def plot_plotly_bar(df, x, y, title, xlabel, ylabel, model_id, total_runs, error_y_col=None, show_relative=False, force_linear_scale=False, hover_data=None, provider=None):
    try:
        # Simplified title
        fig_title = title

        # Default color (Teal)
        color_seq = ['#4bc0c0']

        # Create chart
        fig = px.bar(df, x=x, y=y, text_auto='.2f',
                     title=fig_title,
                     labels={x: xlabel, y: ylabel},
                     color_discrete_sequence=color_seq,
                     error_y=error_y_col,
                     hover_data=hover_data)

        # === Enhanced: Annotate Decode Tokens for TPS charts ===
        if "TPS" in ylabel or "Throughput" in ylabel:
            # Check if we have decode token info in hover_data
            decode_col = None
            if hover_data:
                for col in hover_data:
                    if "Decode_Tokens" in col:
                        decode_col = col
                        break

            if decode_col and decode_col in df.columns:
                # Create custom text with value and token count
                custom_text = df.apply(lambda row: f"{row[y]:.2f}<br>({int(row[decode_col])} tok)", axis=1)
                fig.update_traces(text=custom_text, textposition='outside')
        # =======================================================

        # Add relative performance view (if enabled and y is numeric)
        if show_relative and y in df.columns:
            best_value = df[y].max()
            df_rel = df.copy()
            df_rel[f'{y}_relative'] = (df_rel[y] / best_value * 100).round(1)
            df_rel[f'{y}_text'] = df_rel[f'{y}_relative'].apply(lambda x: f"{x:.1f}%")

            # Use relative performance as text display
            fig.update_traces(text=df_rel[f'{y}_text'], textposition='outside')

            # Update y-axis label, add relative performance annotation
            if 'System Throughput' in ylabel or 'Throughput' in ylabel:
                ylabel_with_rel = f"{ylabel} (max=100%)"
            else:
                ylabel_with_rel = f"{ylabel} (Relative Perf)"

            fig.update_layout(yaxis_title=ylabel_with_rel)

        # Improved axis normalization
        y_values = df[y].values if not error_y_col else np.concatenate([df[y].values, df[y].values + df[error_y_col].fillna(0).values])
        y_max = np.max(y_values) if len(y_values) > 0 else 1
        y_min = np.min(y_values) if len(y_values) > 0 else 0

        # Dynamically set y-axis range with sufficient margin
        y_padding = (y_max - y_min) * 0.1 if y_max != y_min else y_max * 0.1
        fig.update_yaxes(range=[max(0, y_min - y_padding), y_max + y_padding])

        # Improved log scale detection logic
        # 1. Filter out <= 0 values to avoid 0 values causing ratio calculation errors
        positive_y = y_values[y_values > 0]
        if len(positive_y) > 0:
            calc_min = np.min(positive_y)
        else:
            calc_min = y_max # If all zeros, do not trigger

        # 2. Calculate ratio
        ratio = y_max / calc_min if calc_min > 0 else 0

        # 3. Only when ratio is extremely large (> 100,000) enable log scale
        if not force_linear_scale and (ratio > 100000):
            fig.update_yaxes(type="log")
            fig.update_layout(title=title + " [Log Scale]")

        # Improved layout and styles
        fig.update_layout(
            title_x=0,  # Left-align title
            xaxis_title=xlabel,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#333',
            xaxis={'showgrid': False},
            yaxis={'gridcolor': '#e6e9ef', 'showgrid': True},
        )

        # Distinguish confidence interval and error bar colors
        fig.update_traces(
            marker_color='#007bff',
            textposition='outside',
            error_y_color='rgba(255, 0, 0, 0.7)',  # Confidence interval in red
            error_y_thickness=2,
            error_y_width=3
        )

        return fig
    except Exception as e:
        st.error(f"Plotly bar chart failed: {e}")
        return None

def plot_plotly_line(df, x, y, title, xlabel, ylabel, model_id, total_runs, color=None, hover_data=None, error_y_col=None, show_relative=False, force_linear_scale=False, provider=None, line_color=None):
    try:
        # Simplified title,removed hardware/model info subtitle for clean UI
        fig_title = title

        # Determine target color
        target_color = line_color if line_color else '#4bc0c0'

        # Color sequence logic
        color_seq = None
        if color is not None and line_color is not None:
             # If group column specified(color) and base color(line_color)
             # generate gradient based on base color
             n_groups = df[color].nunique()
             color_seq = generate_color_gradient(target_color, n_groups)
        elif color is None:
             color_seq = [target_color] # Single line single color

        # Create chart
        fig = px.line(df, x=x, y=y, title=fig_title,
                      labels={x: xlabel, y: ylabel},
                      markers=True,
                      color=color,
                      color_discrete_sequence=color_seq,
                      hover_data=hover_data,
                      error_y=error_y_col,
                      text=y) # Display value labels

        # Add relative performance view
        if show_relative and y in df.columns and color is None: # Relative perf view currently does not support multiple lines
            best_value = df[y].max()
            df_rel = df.copy()
            df_rel[f'{y}_relative'] = (df[y] / best_value * 100).round(1)

            # Add secondary relative performance line
            fig.add_trace(
                go.Scatter(
                    x=df_rel[x],
                    y=df_rel[f'{y}_relative'],
                    mode='lines+markers',
                    name='Relative Perf (%)',
                    line={'color': '#ffcd56', 'dash': 'dash'}, # Yellow for relative
                    marker={'size': 8, 'color': '#ffcd56'},
                    yaxis='y2'
                )
            )

            # Set dual y-axis
            fig.update_layout(
                yaxis2={
                    'title': "Relative Perf (%)",
                    'overlaying': 'y',
                    'side': 'right',
                    'range': [0, 105],
                    'ticksuffix': "%"
                }
            )

            # Update main y-axis label
            if 'System Throughput' in ylabel or 'Throughput' in ylabel:
                ylabel_with_rel = f"{ylabel} (max=100%)"
            else:
                ylabel_with_rel = f"{ylabel} (Relative to Best)"

            fig.update_layout(yaxis_title=ylabel_with_rel)

        # Improved axis normalization
        y_values = df[y].values if not error_y_col else np.concatenate([df[y].values, df[y].values + df[error_y_col].fillna(0).values])
        y_max = np.max(y_values) if len(y_values) > 0 else 1
        y_min = np.min(y_values) if len(y_values) > 0 else 0

        # Dynamically set Y-axis range
        y_padding = (y_max - y_min) * 0.1 if y_max != y_min else y_max * 0.1
        fig.update_yaxes(range=[max(0, y_min - y_padding), y_max + y_padding])

        # Improved log scale detection logic
        # 1. Filter out <= 0 values to avoid 0 values causing ratio calculation errors
        positive_y = y_values[y_values > 0]
        calc_min = np.min(positive_y) if len(positive_y) > 0 else y_max

        # 2. Calculate ratio
        ratio = y_max / calc_min if calc_min > 0 else 0

        # 3. Only when ratio is extremely large (> 100,000) enable log scale
        if not force_linear_scale and (ratio > 100000):
            fig.update_yaxes(type="log")
            fig.update_layout(title=title + " [Log Scale]")

        # Improved layout and styles
        fig.update_layout(
            title_x=0,  # Left-align title
            xaxis_title=xlabel,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#333',
             xaxis={
                 'title_font': {'size': 14, 'weight': 'bold'},
                 'tickfont': {'size': 12},
                 'type': 'category', # Force X-axis to categorical for equal spacing
                 'showgrid': False
             },
             yaxis={
                 'title_font': {'size': 14, 'weight': 'bold'},
                 'tickfont': {'size': 12},
                 'gridcolor': '#e6e9ef',
                 'showgrid': True
             }
        )

        # Distinguish confidence interval and error bars
        # Common style update - Use smart formatting
        smart_fmt = get_smart_format_string(y_max)
        fig.update_traces(
            textposition="top center",
            texttemplate=f'<b>%{{y:{smart_fmt}}}</b>',    # Bold values, smart formatting
            textfont={'size': 14},            # Larger value font
            error_y_color='rgba(255, 0, 0, 0.7)',
            error_y_thickness=2,
            error_y_width=3
        )

        # Special style for single line (custom color)
        if color is None:
            fig.update_traces(
                # Simulate static chart hollow/bullseye effect: white fill + thick colored border
                marker={
                    'size': 14,
                    'color': 'white',
                    'line': {'width': 3, 'color': target_color}
                },
                line={'width': 4, 'color': target_color} # Thicker line
            )
        else:
            # Multi-line: keep auto-color (or gradient), enlarge markers
            fig.update_traces(
                 marker={'size': 10, 'line': {'width': 2, 'color': 'white'}},
                 line={'width': 3}
            )

        # Title and layout enhancement
        fig.update_layout(
             title={
                 'text': fig_title,
                 'font': {'size': 22, 'color': 'black', 'family': 'sans-serif', 'weight': 'bold'}, # Larger and darker title
                 'x': 0.5, # Center
                 'y': 0.95
             },
             xaxis={
                 'title_font': {'size': 14, 'weight': 'bold'},
                 'tickfont': {'size': 12},
                 'type': 'category', # Force X-axis to categorical for equal spacing
                 'showgrid': False
             },
             yaxis={
                 'title_font': {'size': 14, 'weight': 'bold'},
                 'tickfont': {'size': 12}
             }
        )

        return fig
    except Exception as e:
        st.error(f"Plotly line chart failed: {e}")
        return None

def plot_relative_performance_bar(df, x, y, title, xlabel, ylabel, model_id, total_runs):
    """Dedicated bar chart for displaying relative performance comparison"""
    try:
        df_rel = df.copy()
        best_value = df_rel[y].max()
        df_rel[f'{y}_relative'] = (df_rel[y] / best_value * 100).round(1)
        df_rel[f'{y}_absolute'] = df_rel[y]

        fig = go.Figure()

        # Add absolute value bars (light color)
        fig.add_trace(
            go.Bar(
                x=df_rel[x],
                y=df_rel[f'{y}_absolute'],
                name='Absolute',
                marker_color='rgba(0, 123, 255, 0.3)',
                text=[f"{val:.2f}" for val in df_rel[f'{y}_absolute']],
                textposition='outside',
                yaxis='y'
            )
        )

        # Add relative performance bar chart (dark color)
        fig.add_trace(
            go.Bar(
                x=df_rel[x],
                y=df_rel[f'{y}_relative'],
                name='Relative Perf (%)',
                marker_color='#007bff',
                text=[f"{val:.1f}%" for val in df_rel[f'{y}_relative']],
                textposition='outside',
                yaxis='y2'
            )
        )

        # Set dual y-axis
        fig.update_layout(
            title=f"{title} - Relative Performance Comparison (Model: {model_id})",
            xaxis_title=xlabel,
            yaxis={
                'title': ylabel,
                'side': 'left',
                'showgrid': False
            },
            yaxis2={
                'title': "Relative Perf (%)",
                'overlaying': 'y',
                'side': 'right',
                'range': [0, 105],
                'ticksuffix': "%"
            },
            title_x=0,  # Left-align title
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#333',
            barmode='group',
            showlegend=True
        )

        return fig
    except Exception as e:
        st.error(f"Relative performance chart failed: {e}")
        return None


def plot_box_plot(df, x, y, title, xlabel, ylabel, color=None):
    """
    Box plot for performance distribution analysis.

    Shows quartiles, outliers, and data distribution.
    """
    try:
        fig = px.box(
            df, x=x, y=y,
            title=title,
            labels={x: xlabel, y: ylabel},
            color=color,
            points='all',  # Show all data points
            notched=True   # Show confidence interval
        )

        # Apply theme
        fig = apply_theme(fig)

        # Customize box colors
        fig.update_traces(
            marker={'size': 4, 'opacity': 0.6, 'color': COLORS['primary']},
            line={'color': COLORS['primary'], 'width': 2}
        )

        return fig
    except Exception as e:
        st.error(f"Box plot generation failed: {e}")
        return None


def plot_violin(df, x, y, title, xlabel, ylabel, color=None):
    """
    Violin plot combining box plot and density distribution.

    Better for showing distribution shape.
    """
    try:
        fig = px.violin(
            df, x=x, y=y,
            title=title,
            labels={x: xlabel, y: ylabel},
            color=color,
            box=True,      # Overlay box plot
            points='all'   # Show all points
        )

        # Apply theme
        fig = apply_theme(fig)

        # Customize violin
        fig.update_traces(
            meanline_visible=True,
            marker={'size': 3, 'opacity': 0.5},
            line={'color': COLORS['primary'], 'width': 2}
        )

        return fig
    except Exception as e:
        st.error(f"Violin plot generation failed: {e}")
        return None


def plot_scatter_with_trend(df, x, y, title, xlabel, ylabel, size=None, color=None, trendline='ols'):
    """
    Scatter plot with regression trendline.

    Args:
        trendline: 'ols' (linear), 'lowess' (local weighted), or None
    """
    try:
        fig = px.scatter(
            df, x=x, y=y,
            title=title,
            labels={x: xlabel, y: ylabel},
            size=size,
            color=color,
            trendline=trendline if trendline else None,
            trendline_color_override=COLORS['danger']
        )

        # Apply theme
        fig = apply_theme(fig)

        # Customize markers
        fig.update_traces(
            marker={
                'size': 10,
                'opacity': 0.7,
                'line': {'width': 1, 'color': 'white'}
            }
        )

        return fig
    except Exception as e:
        st.error(f"Scatter plot generation failed: {e}")
        return None


def plot_performance_summary(df, metrics, title="Performance Summary"):
    """
    Multi-metric performance summary chart (Radar Chart).

    Shows multiple metrics in a single radar/spider chart.
    Metrics are normalized to 0-100 relative to the dataset max/min.
    """
    try:
        # Normalize metrics to 0-100 scale
        df_norm = df.copy()

        # Define 'higher is better' metrics vs 'lower is better'
        higher_is_better = ['tps', 'system_output_throughput', 'system_input_throughput', 'system_throughput', 'prefill_speed', 'rps', 'success_rate']

        for metric in metrics:
            if metric in df.columns:
                min_val = df[metric].min()
                max_val = df[metric].max()

                if max_val == min_val:
                    df_norm[f'{metric}_norm'] = 100 # If all same, give full score
                else:
                    if any(h in metric.lower() for h in higher_is_better):
                        # Higher is better: (val - min) / (max - min) * 100
                        df_norm[f'{metric}_norm'] = ((df[metric] - min_val) / (max_val - min_val)) * 100
                    else:
                        # Lower is better (Latency): (max - val) / (max - min) * 100
                        # So lowest value gets 100, highest gets 0
                        df_norm[f'{metric}_norm'] = ((max_val - df[metric]) / (max_val - min_val)) * 100

        # Create radar chart
        fig = go.Figure()

        for idx, row in df_norm.iterrows():
            # Get original values for hover text
            values_norm = []
            values_text = []

            for m in metrics:
                if m in df.columns:
                    values_norm.append(row.get(f'{m}_norm', 0))
                    original_val = row[m]
                    values_text.append(f"{original_val:.2f}")
                else:
                    values_norm.append(0)
                    values_text.append("N/A")

            # Close the loop
            values_norm.append(values_norm[0])
            metrics_labels = metrics + [metrics[0]]

            name_label = str(row.get(df.columns[0], idx))
            if 'session_id' in df.columns and len(str(row.get('session_id', ''))) > 8:
                 # If session ID is used as name, shorten it
                 name_label = str(row['session_id'])[:8] + "..."
            elif 'concurrency' in df.columns:
                 name_label = f"Concurrency {row['concurrency']}"

            fig.add_trace(go.Scatterpolar(
                r=values_norm,
                theta=metrics_labels,
                fill='toself',
                name=name_label,
                text=values_text,
                hoverinfo='text+name'
            ))

        fig.update_layout(
            polar={
                'radialaxis': {
                    'visible': True,
                    'range': [0, 100],
                    'showticklabels': False # Hide normalized numbers to avoid confusion
                }
            },
            title={
                'text': title,
                'x': 0.5
            },
            showlegend=True,
            margin={'l': 80, 'r': 80, 't': 50, 'b': 50} # Increase margins for labels
        )

        fig = apply_theme(fig)

        return fig
    except Exception as e:
        st.error(f"Performance summary chart failed: {e}")
        return None
