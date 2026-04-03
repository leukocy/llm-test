"""
Static Chart Generator

Inspired by llm-performance-test.html Canvas drawing style,
uses Matplotlib to generate beautiful static performance test result images.

Features:
- Clean white background + grid lines
- Circular data points with shadow
- Data value labels displayed above data points
- Smooth line chart
- Can be exported as static PNG images
"""

import base64
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===== Color Scheme =====
COLORS = {
    'prefill': '#4bc0c0',       # Teal - Prefill speed
    'output': '#ff6384',        # Pink - Output speed
    'ttft': '#36a2eb',          # Blue - TTFT
    'throughput': '#9966ff',    # Purple - Throughput
    'tps': '#ff9f40',           # Orange - TPS
    'success': '#4bc07a',       # Green - Success rate
    'primary': '#007bff',       # Primary - Blue
    'secondary': '#6c757d',     # Secondary - Gray
    'grid': '#e0e0e0',          # Grid line color
    'axis': '#000000',          # Axis color
    'text': '#000000',          # Text color
    'background': '#ffffff',    # Background color
}

# ===== Font Configuration =====
FONT_CONFIG = {
    'family': 'Microsoft YaHei, SimHei, Arial, sans-serif',
    'title_size': 20,
    'label_size': 16,
    'tick_size': 15,
    'data_point_size': 15,
    'info_size': 15,
}


# ===== Smart Value Formatting =====
def smart_format_value(value: float, max_value: float | None = None) -> str:
    """
    Smart format value, automatically selects appropriate decimal places based on value magnitude.

    Rules:
    - max_value > 100: Integer (0 decimal places)
    - max_value > 10: 1 decimal place
    - other: 2 decimal places

    Args:
        value: Value to format
        max_value: Max value of column/series for format decision. If None, uses value itself.

    Returns:
        Formatted string
    """
    if max_value is None:
        max_value = abs(value)

    if max_value > 100:
        return f"{value:.0f}"
    elif max_value > 10:
        return f"{value:.1f}"
    else:
        return f"{value:.2f}"


def get_smart_formatter(max_value: float):
    """
    Returns a matplotlib-compatible format function for Y-axis ticks.

    Args:
        max_value: Max value of this axis

    Returns:
        Format function (value, position) -> str
    """
    if max_value > 100:
        return lambda x, p: f'{x:.0f}'
    elif max_value > 10:
        return lambda x, p: f'{x:.1f}'
    else:
        return lambda x, p: f'{x:.2f}'


class StaticChartGenerator:
    """Static chart generator for creating beautiful performance test result charts"""

    def __init__(self, dpi: int = 100):
        """
        Initialize chart generator

        Args:
            dpi: Image resolution
        """
        self.dpi = dpi
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Configure matplotlib global settings"""
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.dpi'] = self.dpi

    def _get_nice_max_value(self, max_val: float) -> float:
        """
        Get the nearest nice round number >= max
        (e.g., 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 etc.)

        Args:
            max_val: Maximum value

        Returns:
            Normalized maximum value
        """
        if max_val <= 0:
            return 1

        exp = np.floor(np.log10(max_val))
        base = 10 ** exp

        nice_max = base
        if max_val > base:
            if max_val <= 2 * base:
                nice_max = 2 * base
            elif max_val <= 5 * base:
                nice_max = 5 * base
            else:
                nice_max = 10 * base

        return nice_max

    def draw_line_chart(
        self,
        x_data: list[float],
        y_data: list[float],
        title: str,
        x_label: str,
        y_label: str,
        line_color: str = '#4bc0c0',
        system_info: dict[str, str] | None = None,
        figsize: tuple[int, int] = (10, 8) # Increase height
    ) -> plt.Figure:
        """
        Draw single line chart with system info display
        """
        fig = plt.figure(figsize=figsize)

        # Top title
        fig.text(0.5, 0.95, title, fontsize=FONT_CONFIG['title_size'] + 2,
                 fontweight='bold', ha='center', va='top')

        # Draw system info (if provided)
        chart_top = 0.85
        if system_info:
            chart_top = self._draw_system_info(fig, system_info, chart_top)

        # Draw chart
        # [left, bottom, width, height]
        # bottom starts at 0.15, height dynamically calculated to chart_top, leaving a small gap
        ax_height = chart_top - 0.20
        ax = fig.add_axes([0.1, 0.15, 0.85, ax_height])

        # Use equal-width categorical X-axis: use index as actual plot position

        # Use equal-width categorical X-axis: use index as actual plot position
        n_points = len(x_data)
        x_positions = list(range(1, n_points + 1))  # Start from 1, leave room for 0 position

        # Calculate Y-axis range
        y_max = self._get_nice_max_value(max(y_data)) if y_data else 100
        y_ticks = np.linspace(0, y_max, 5)

        # X-axis ticks: 0 + actual data labels
        x_tick_positions = [0] + x_positions
        x_tick_labels = ['0'] + [str(int(x)) if x == int(x) else str(x) for x in x_data]

        # Set axis range
        ax.set_xlim(-0.5, n_points + 0.8)  # Leave right space for X-axis labels
        ax.set_ylim(0, y_max * 1.05)

        # Draw grid lines
        ax.grid(True, linestyle='-', linewidth=1, color=COLORS['grid'], alpha=0.8)
        ax.set_axisbelow(True)  # Grid lines below data

        # Set axis style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_color(COLORS['axis'])
        ax.spines['bottom'].set_color(COLORS['axis'])

        # Draw lines
        ax.plot(x_positions, y_data, color=line_color, linewidth=3, zorder=2)

        # Draw data points using scatter (avoids axis aspect ratio issues)
        # Outer ring: white fill, colored border
        ax.scatter(x_positions, y_data, s=250, c='white', edgecolors=line_color,
                   linewidths=2, zorder=3)
        # Inner ring: solid fill
        ax.scatter(x_positions, y_data, s=60, c=line_color, zorder=4)

        # Data point value labels
        for i, (x_pos, y) in enumerate(zip(x_positions, y_data, strict=False)):
            value_text = smart_format_value(float(y), y_max) if isinstance(y, (int, float)) else str(y)
            ax.annotate(value_text, (x_pos, y), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=FONT_CONFIG['data_point_size'],
                       color=COLORS['text'], zorder=5)

        # Set title
        # Title already drawn at top, no need for set_title here

        # Set axis label - Y-axis label
        ax.set_ylabel(y_label, fontsize=FONT_CONFIG['label_size'], fontweight='bold',
                     labelpad=10)

        # Smart process X-axis tick labels to prevent overlap
        max_label_len = max([len(l) for l in x_tick_labels])
        need_sampling = n_points > 12 or (n_points > 6 and max_label_len >= 5)

        if need_sampling:
            if max_label_len >= 6:
                max_visible_ticks = min(6, n_points)
            elif max_label_len >= 5:
                max_visible_ticks = min(8, n_points)
            else:
                max_visible_ticks = min(10, n_points)

            tick_step = max(1, n_points // max_visible_ticks)
            visible_indices = {0}
            for i in range(0, n_points, tick_step):
                visible_indices.add(i)
            visible_indices.add(n_points - 1)

            filtered_positions = [x_tick_positions[i+1] for i in sorted(visible_indices)]
            filtered_labels = [x_tick_labels[i+1] for i in sorted(visible_indices)]
            ax.set_xticks([0] + filtered_positions)
            ax.set_xticklabels(['0'] + filtered_labels)
        else:
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels)

        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick_size'])

        # Label anti-overlap: rotate if many labels or long text
        needs_rotation = n_points > 6 or max_label_len > 4

        if needs_rotation:
            if max_label_len >= 6 or n_points > 15:
                rotation_angle = 45
            elif max_label_len >= 5 or n_points > 10:
                rotation_angle = 35
            else:
                rotation_angle = 25
            plt.setp(ax.get_xticklabels(), rotation=rotation_angle, ha="right", rotation_mode="anchor")

        # Use set_xlabel to add X-axis label
        ax.set_xlabel(x_label, fontsize=FONT_CONFIG['label_size'], fontweight='bold',
                      labelpad=10)

        # Format Y-axis tick labels
        ax.yaxis.set_major_formatter(plt.FuncFormatter(get_smart_formatter(y_max)))

        # plt.tight_layout()
        return fig

    def draw_dual_line_chart(
        self,
        x_data: list[float],
        y1_data: list[float],
        y2_data: list[float],
        title: str,
        x_label: str,
        y1_label: str,
        y2_label: str,
        y1_color: str = '#4bc0c0',
        y2_color: str = '#ff6384',
        system_info: dict[str, str] | None = None,
        figsize: tuple[int, int] = (10, 10) # Increase height to accommodate info
    ) -> plt.Figure:
        """
        Draw dual line chart (stacked vertically) with system info display
        """
        fig = plt.figure(figsize=figsize)

        # Top title
        fig.text(0.5, 0.95, title, fontsize=FONT_CONFIG['title_size'] + 4,
                 fontweight='bold', ha='center', va='top')

        # Draw system info (if provided)
        # Default chart start height slightly lowered to prevent title overlap
        chart_top = 0.82

        # If title contains newline (long title) and no system info, move down a bit
        if '\n' in title and not system_info:
             chart_top = 0.78

        if system_info:
            chart_top = self._draw_system_info(fig, system_info, chart_top)

        # Dynamically calculate chart height and position with sufficient spacing
        bottom_margin = 0.08
        inter_chart_gap = 0.18  # Increase spacing to prevent X-axis labels overlapping with the bottom title

        # Available height = top start - bottom margin - inter-chart gap
        available_height = chart_top - bottom_margin - inter_chart_gap
        chart_height = available_height / 2

        # Draw chart 1 (Top)
        # top_chart_bottom = chart_top - chart_height
        ax1 = fig.add_axes([0.1, chart_top - chart_height, 0.85, chart_height])
        self._draw_single_chart(ax1, x_data, y1_data, y1_label, x_label,
                                f"{y1_label} (token/s)", y1_color)

        # Draw chart 2 (Bottom)
        # bottom_chart_bottom = 0.08 (bottom_margin)
        ax2 = fig.add_axes([0.1, bottom_margin, 0.85, chart_height])
        self._draw_single_chart(ax2, x_data, y2_data, y2_label, x_label,
                                f"{y2_label} (token/s)", y2_color)

        return fig

    def _draw_single_chart(
        self,
        ax: plt.Axes,
        x_data: list[Any],
        y_data: list[float],
        title: str,
        x_label: str,
        y_label: str,
        line_color: str
    ):
        """Draw single line chart on given Axes (equal-width categorical X-axis)"""
        # Ensure all y_data are numeric
        y_data = [float(y) if y is not None else 0.0 for y in y_data]
        n_points = len(x_data)

        # Use equal-width categorical X-axis: use index as actual plot position
        x_positions = list(range(1, n_points + 1))  # Start from 1, leave room for 0 position

        y_max = self._get_nice_max_value(max(y_data)) if y_data and max(y_data) > 0 else 100
        y_ticks = np.linspace(0, y_max, 5)

        # Process X-axis labels
        def format_label(x):
            if isinstance(x, (int, float)):
                return str(int(x)) if x == int(x) else f"{x:.1f}"
            return str(x)

        # X-axis ticks: 0 + actual data labels
        # Note: for concurrency etc. non-zero starting data, keeping 0 as origin is good visual style
        x_tick_positions = [0] + x_positions
        x_tick_labels = ['0'] + [format_label(x) for x in x_data]

        # Set axis range
        ax.set_xlim(-0.5, n_points + 0.8)
        ax.set_ylim(0, y_max * 1.05)

        ax.grid(True, linestyle='-', linewidth=1, color=COLORS['grid'], alpha=0.8)
        ax.set_axisbelow(True)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # Draw lines
        ax.plot(x_positions, y_data, color=line_color, linewidth=3, zorder=2)

        # Draw data points using scatter
        # Data point size dynamically adjusted based on canvas width to prevent crowding
        s_outer = 250
        s_inner = 60
        if n_points > 15: # If many points, reduce point size
            s_outer = 150
            s_inner = 40

        # Outer ring: white fill, colored border
        ax.scatter(x_positions, y_data, s=s_outer, c='white', edgecolors=line_color,
                   linewidths=2, zorder=3)
        # Inner ring: solid fill
        ax.scatter(x_positions, y_data, s=s_inner, c=line_color, zorder=4)

        # Data point value labels (prevent overlap)
        # If many points, display every other or only max/min values
        skip_step = 1
        if n_points > 20:
             skip_step = 2 # Simple sampling

        for i, (x_pos, y) in enumerate(zip(x_positions, y_data, strict=False)):
            if i % skip_step == 0:
                value_text = smart_format_value(float(y), y_max)
                ax.annotate(value_text, (x_pos, y), textcoords="offset points",
                           xytext=(0, 12), ha='center', fontsize=FONT_CONFIG['data_point_size'] - 2,
                           color=COLORS['text'], zorder=5)

        ax.set_title(title, fontsize=FONT_CONFIG['title_size'], fontweight='bold', pad=15)
        ax.set_ylabel(y_label, fontsize=FONT_CONFIG['label_size'], fontweight='bold')

        # Smart process X-axis tick labels to prevent overlap
        # Determine tick sampling based on data point count and label length
        max_label_len = max([len(l) for l in x_tick_labels])

        # Estimate character width per label, determine if sampling needed
        # Empirical formula: when n_points * max_label_len > threshold, labels will overlap
        need_sampling = n_points > 12 or (n_points > 6 and max_label_len >= 5)

        if need_sampling:
            # Dynamically calculate max visible ticks based on label length
            if max_label_len >= 6:
                max_visible_ticks = min(6, n_points)  # 6-digit labels show max 6
            elif max_label_len >= 5:
                max_visible_ticks = min(8, n_points)  # 5-digit labels show max 8
            else:
                max_visible_ticks = min(10, n_points)

            tick_step = max(1, n_points // max_visible_ticks)
            visible_indices = {0}  # Always include first point
            for i in range(0, n_points, tick_step):
                visible_indices.add(i)
            visible_indices.add(n_points - 1)  # Always include last point

            # Only display sampled ticks
            filtered_positions = [x_tick_positions[i+1] for i in sorted(visible_indices)]
            filtered_labels = [x_tick_labels[i+1] for i in sorted(visible_indices)]
            # Add origin 0
            ax.set_xticks([0] + filtered_positions)
            ax.set_xticklabels(['0'] + filtered_labels)
        else:
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels)

        # Label anti-overlap: rotate if many labels or long text
        needs_rotation = n_points > 6 or max_label_len > 4

        if needs_rotation:
            # Rotation angle determined by label length and point count
            if max_label_len >= 6 or n_points > 15:
                rotation_angle = 45
            elif max_label_len >= 5 or n_points > 10:
                rotation_angle = 35
            else:
                rotation_angle = 25
            plt.setp(ax.get_xticklabels(), rotation=rotation_angle, ha="right", rotation_mode="anchor")

        # Use set_xlabel to add X-axis label, leveraging matplotlib auto-avoidance of tick label overlap
        ax.set_xlabel(x_label, fontsize=FONT_CONFIG['label_size'], fontweight='bold',
                      labelpad=10)

        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick_size'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(get_smart_formatter(y_max)))

    def create_performance_report_image(
        self,
        test_results: list[dict[str, Any]],
        system_info: dict[str, str],
        test_time: datetime | None = None,
        figsize: tuple[int, int] = (10, 12)
    ) -> plt.Figure:
        """
        Create complete performance test report image
        Contains: title, system info, prefill speed chart, output speed chart

        Args:
            test_results: Test results list, each item contains input_length, prefill_speed, output_speed
            system_info: System info dict containing processor, mainboard, memory, gpu, system, engine_name, model_name
            test_time: Test time
            figsize: Image size

        Returns:
            matplotlib Figure object
        """
        # Extract data
        input_lengths = [r.get('input_length', r.get('inputLength', 0)) for r in test_results]
        prefill_speeds = [float(r.get('prefill_speed', r.get('prefillSpeed', 0))) for r in test_results]
        output_speeds = [float(r.get('output_speed', r.get('outputSpeed', 0))) for r in test_results]

        # Create canvas
        fig = plt.figure(figsize=figsize)

        # Add title area
        title_height = 0.08
        info_height = 0.12
        (1 - title_height - info_height) / 2

        # Add main title
        fig.text(0.5, 1 - title_height/2, 'Local LLM Inference Performance Test Results',
                fontsize=FONT_CONFIG['title_size'] + 8, fontweight='bold',
                ha='center', va='center')

        # Add system info area
        # Restore to top layout, below title
        info_y_start = 0.88
        # Build info list containing only valid info
        raw_info = [
            ("Processor", system_info.get('processor')),
            ("Mainboard", system_info.get('mainboard')),
            ("Memory", system_info.get('memory')),
            ("GPU", system_info.get('gpu')),
            ("System", system_info.get('system')),
            ("Engine Name", system_info.get('engine_name')),
            ("Model Name", system_info.get('model_name')),
        ]

        valid_info_lines = []
        for label, value in raw_info:
            if value and value not in ['N/A', 'no', 'None', '']:
                valid_info_lines.append(f"{label}:{value}")

        if test_time:
            valid_info_lines.append(f"Test time:{test_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Dynamic layout: determine rows per column based on valid info count
        total_items = len(valid_info_lines)
        items_per_col = (total_items + 1) // 2

        left_col_x = 0.05
        right_col_x = 0.55

        for i, line in enumerate(valid_info_lines):
            if i < items_per_col:
                x_pos = left_col_x
                y_pos = info_y_start - i * 0.025
            else:
                x_pos = right_col_x
                y_pos = info_y_start - (i - items_per_col) * 0.025

            fig.text(x_pos, y_pos, line, fontsize=FONT_CONFIG['info_size'], ha='left', va='center')

        # Add prefill speed chart
        # Move chart down, leave space at top
        # [left, bottom, width, height]
        ax1 = fig.add_axes([0.1, 0.48, 0.85, 0.28])
        self._draw_single_chart(ax1, input_lengths, prefill_speeds,
                               'Prefill Speed', 'Input Length', 'Prefill Speed (token/s)',
                                COLORS['prefill'])

        # Add output speed chart
        ax2 = fig.add_axes([0.1, 0.08, 0.85, 0.28])
        self._draw_single_chart(ax2, input_lengths, output_speeds,
                               'Output Speed', 'Input Length', 'Output Speed (token/s)',
                                COLORS['output'])

        return fig

    def draw_multi_line_chart(
        self,
        x_data: list[float],
        datasets: list[dict[str, Any]],
        title: str,
        x_label: str,
        y_label: str,
        system_info: dict[str, str] | None = None,
        figsize: tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Draw a chart with multiple lines (e.g., for different concurrencies).

        Args:
            x_data: Shared X-axis data (List of float/int)
            datasets: List of dictionaries, each containing:
                - label: Legend label (str)
                - data: Y-axis data (List of float)
                - color: Color code (str)
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            system_info: System info dictionary
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)

        # Title
        fig.text(0.5, 0.95, title, fontsize=FONT_CONFIG['title_size'] + 2,
                 fontweight='bold', ha='center', va='top')

        # System Info
        chart_top = 0.85
        if system_info:
            chart_top = self._draw_system_info(fig, system_info, chart_top)

        # Chart area
        ax_height = chart_top - 0.20
        ax = fig.add_axes([0.1, 0.15, 0.85, ax_height])

        # X Axis setup
        n_points = len(x_data)
        x_positions = list(range(1, n_points + 1))

        # Calculate global Y Max
        all_y_values = []
        for ds in datasets:
            all_y_values.extend([v for v in ds['data'] if v is not None])

        if not all_y_values:
            all_y_values = [100]

        y_max_raw = max(all_y_values)
        y_max = self._get_nice_max_value(y_max_raw) if y_max_raw > 0 else 100
        y_ticks = np.linspace(0, y_max, 5)

        # X Ticks
        x_tick_positions = [0] + x_positions

        def format_label(x):
            if isinstance(x, (int, float)):
                if x >= 1000:
                    return f"{x/1000:.1f}k"
                return str(int(x)) if x == int(x) else f"{x:.1f}"
            return str(x)

        x_tick_labels = ['0'] + [format_label(x) for x in x_data]

        # Axes limits
        ax.set_xlim(-0.5, n_points + 0.8)
        ax.set_ylim(0, y_max * 1.05)

        # Grid
        ax.grid(True, linestyle='-', linewidth=1, color=COLORS['grid'], alpha=0.8)
        ax.set_axisbelow(True)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_color(COLORS['axis'])
        ax.spines['bottom'].set_color(COLORS['axis'])

        # Plot each line
        s_outer = 150 if len(datasets) > 3 else 200
        s_inner = 50 if len(datasets) > 3 else 60

        for ds in datasets:
            y_dataset = ds['data']
            line_color = ds.get('color', COLORS['primary'])
            label = ds.get('label', '')

            ax.plot(x_positions, y_dataset, color=line_color, linewidth=2.5, zorder=2, label=label)
            ax.scatter(x_positions, y_dataset, s=s_outer, c='white', edgecolors=line_color,
                       linewidths=2, zorder=3)
            ax.scatter(x_positions, y_dataset, s=s_inner, c=line_color, zorder=4)

        # Labels
        ax.set_ylabel(y_label, fontsize=FONT_CONFIG['label_size'], fontweight='bold', labelpad=10)
        ax.set_xlabel(x_label, fontsize=FONT_CONFIG['label_size'], fontweight='bold', labelpad=10)

        # Ticks
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick_size'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(get_smart_formatter(y_max)))

        # Legend
        ax.legend(loc='upper left', frameon=True, fontsize=12, ncol=len(datasets) // 4 + 1)

        return fig

    def _draw_system_info(self, fig, system_info: dict, chart_top: float) -> float:
        """
        Draw system info uniformly at the top of the figure.

        Args:
            fig: matplotlib Figure
            system_info: System info dictionary
            chart_top: Current chart top position

        Returns:
            Updated chart_top (after moving down)
        """
        raw_info = [
            ("Processor", system_info.get('processor')),
            ("Mainboard", system_info.get('mainboard')),
            ("Memory", system_info.get('memory')),
            ("GPU", system_info.get('gpu')),
            ("System", system_info.get('system')),
            ("Engine Name", system_info.get('engine_name')),
            ("Model Name", system_info.get('model_name')),
        ]

        valid_info_lines = []
        for label, value in raw_info:
            if value and value not in ['N/A', 'no', 'None', '']:
                if len(str(value)) > 40:
                    value = str(value)[:37] + "..."
                valid_info_lines.append(f"{label}:{value}")

        if valid_info_lines:
            items_per_col = (len(valid_info_lines) + 1) // 2
            info_y_start = 0.88
            left_col_x = 0.05
            right_col_x = 0.60

            for i, line in enumerate(valid_info_lines):
                if i < items_per_col:
                    x = left_col_x
                    y = info_y_start - i * 0.025
                else:
                    x = right_col_x
                    y = info_y_start - (i - items_per_col) * 0.025
                fig.text(x, y, line, fontsize=FONT_CONFIG['info_size'], ha='left', va='center')

            return 0.75  # After info display, chart start position moves down

        return chart_top

    def draw_quad_chart(
        self,
        x_data: list,
        charts: list[dict[str, Any]],
        title: str,
        x_label: str,
        system_info: dict[str, str] | None = None,
        figsize: tuple[int, int] = (14, 14)
    ) -> plt.Figure:
        """
        Draw unified 2x2 quad-panel chart.

        Each test type static analysis is unified as 4 sub-charts:
        Top-left: TTFT / Latency metrics
        Top-right: Speed / Throughput metrics
        Bottom-left: Input Throughput / Cache metrics
        Bottom-right: Output Throughput / TPOT metrics

        Args:
            x_data: X-axis data
            charts: 4 sub-chart config list, each dict contains:
                - y_data: Y-axis data (list[float])
                - title: Sub-chart title (str)
                - y_label: Y-axis label (str)
                - color: Line color (str)
                If None, skip this panel
            title: Main title
            x_label: X-axis label
            system_info: System info
            figsize: Image size

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)

        # Top title
        fig.text(0.5, 0.97, title, fontsize=FONT_CONFIG['title_size'] + 4,
                 fontweight='bold', ha='center', va='top')

        # System info
        chart_top = 0.88
        if system_info:
            chart_top = self._draw_system_info(fig, system_info, chart_top)

        # Calculate 2x2 sub-chart positions
        bottom_margin = 0.06
        h_gap = 0.12  # Horizontal gap
        v_gap = 0.12  # Vertical gap
        left_margin = 0.08
        right_margin = 0.04

        available_height = chart_top - bottom_margin - v_gap
        available_width = 1.0 - left_margin - right_margin - h_gap
        chart_h = available_height / 2
        chart_w = available_width / 2

        # Sub-chart positions: [left, bottom, width, height]
        positions = [
            [left_margin, chart_top - chart_h, chart_w, chart_h],                           # Top-left
            [left_margin + chart_w + h_gap, chart_top - chart_h, chart_w, chart_h],          # Top-right
            [left_margin, bottom_margin, chart_w, chart_h],                                  # Bottom-left
            [left_margin + chart_w + h_gap, bottom_margin, chart_w, chart_h],                # Bottom-right
        ]

        for i, chart_cfg in enumerate(charts):
            if chart_cfg is None or i >= 4:
                continue

            pos = positions[i]
            ax = fig.add_axes(pos)
            self._draw_single_chart(
                ax,
                x_data,
                chart_cfg['y_data'],
                chart_cfg['title'],
                x_label,
                chart_cfg['y_label'],
                chart_cfg['color']
            )

        return fig

    def save_figure_to_bytes(self, fig: plt.Figure, format: str = 'png') -> BytesIO:
        """
        Save chart to byte stream

        Args:
            fig: matplotlib Figure object
            format: Image format ('png', 'jpg', 'svg', 'pdf')

        Returns:
            BytesIO object
        """
        buf = BytesIO()
        fig.savefig(buf, format=format, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        return buf

    def save_figure_to_base64(self, fig: plt.Figure, format: str = 'png') -> str:
        """
        Save chart as Base64 encoded string

        Args:
            fig: matplotlib Figure object
            format: Image format

        Returns:
            Base64 encoded string
        """
        buf = self.save_figure_to_bytes(fig, format)
        return base64.b64encode(buf.read()).decode()

    def save_figure_to_file(self, fig: plt.Figure, filepath: str, format: str = 'png'):
        """
        Save chart to file

        Args:
            fig: matplotlib Figure object
            filepath: File path
            format: Image format
        """
        fig.savefig(filepath, format=format, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)


def create_benchmark_chart_from_dataframe(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str = '#4bc0c0'
) -> plt.Figure:
    """
    Quick function to create chart from DataFrame

    Args:
        df: DataFrame containing data
        x_col: X-axis column name
        y_col: Y-axis column name
        title: Chart Title
        x_label: X-axis label
        y_label: Y-axis label
        color: Line color

    Returns:
        matplotlib Figure object
    """
    generator = StaticChartGenerator()
    x_data = df[x_col].tolist()
    y_data = df[y_col].tolist()
    return generator.draw_line_chart(x_data, y_data, title, x_label, y_label, color)


def create_concurrency_static_chart(
    df: pd.DataFrame,
    metric: str = 'system_output_throughput',
    title: str = 'System Throughput vs Concurrency',
    color: str = '#4bc0c0'
) -> plt.Figure:
    """
    Create concurrency test static chart

    Args:
        df: DataFrame with concurrency and metric columns
        metric: Metric column name to plot
        title: Chart Title
        color: Line color

    Returns:
        matplotlib Figure object
    """
    generator = StaticChartGenerator()

    # Sort by concurrency
    df_sorted = df.sort_values('concurrency')
    x_data = df_sorted['concurrency'].tolist()
    y_data = df_sorted[metric].tolist()

    # Determine Y-axis label based on metric
    y_label_map = {
        'system_output_throughput': 'System Output Throughput (tokens/s)',
        'tps': 'TPS (tokens/s)',
        'ttft': 'TTFT (s)',
        'prefill_speed': 'Prefill Speed (tokens/s)',
    }
    y_label = y_label_map.get(metric, metric)

    return generator.draw_line_chart(x_data, y_data, title, 'Concurrency', y_label, color)


def create_prefill_static_chart(
    df: pd.DataFrame,
    metric: str = 'prefill_speed',
    title: str = 'Prefill Speed vs Input Length',
    color: str = '#4bc0c0'
) -> plt.Figure:
    """
    Create prefill test static chart

    Args:
        df: DataFrame with prefill_tokens and metric columns
        metric: Metric column name to plot
        title: Chart Title
        color: Line color

    Returns:
        matplotlib Figure object
    """
    generator = StaticChartGenerator()

    # Sort by input length
    df_sorted = df.sort_values('prefill_tokens')
    x_data = df_sorted['prefill_tokens'].tolist()
    y_data = df_sorted[metric].tolist()

    y_label_map = {
        'prefill_speed': 'Prefill Speed (tokens/s)',
        'ttft': 'TTFT (s)',
        'tps': 'TPS (tokens/s)',
    }
    y_label = y_label_map.get(metric, metric)

    return generator.draw_line_chart(x_data, y_data, title, 'Input Length (tokens)', y_label, color)


# ===== HTML Report Generation =====

def generate_static_html_report(
    test_results: list[dict[str, Any]],
    system_info: dict[str, str],
    test_time: datetime | None = None
) -> str:
    """
    Generate static HTML report (can be opened directly in browser)

    Args:
        test_results: Test results list
        system_info: System info dictionary
        test_time: Test time

    Returns:
        HTML string
    """
    generator = StaticChartGenerator()
    fig = generator.create_performance_report_image(test_results, system_info, test_time)
    img_base64 = generator.save_figure_to_base64(fig)

    if test_time is None:
        test_time = datetime.now()

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Performance Test Report</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-image {{
            width: 100%;
            max-width: 1000px;
            display: block;
            margin: 20px auto;
        }}
        .download-btn {{
            display: inline-block;
            padding: 12px 24px;
            background-color: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px;
            cursor: pointer;
        }}
        .download-btn:hover {{
            background-color: #0056b3;
        }}
        .buttons {{
            text-align: center;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <img src="data:image/png;base64,{img_base64}" class="chart-image" alt="Performance Test Report">
        <div class="buttons">
            <a href="data:image/png;base64,{img_base64}" download="llm_performance_report_{test_time.strftime('%Y%m%d_%H%M%S')}.png" class="download-btn">
                📷 Download Image
            </a>
        </div>
    </div>
</body>
</html>
"""
    return html

