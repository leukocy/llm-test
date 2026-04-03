"""
Styled table components for professional data visualization.

Provides styled pandas DataFrames with conditional formatting,
highlighting, and custom themes.
"""
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_styled_summary_table(df, highlight_cols=None, highlight_best=True):
    """
    Create professionally styled summary table.

    Args:
        df: DataFrame to style
        highlight_cols: List of columns to apply gradient coloring
        highlight_best: Whether to highlight best values (green background)

    Returns:
        Styled DataFrame
    """
    styler = df.style

    # 1. Apply gradient to specified columns
    if highlight_cols:
        def _get_custom_gradient(s):
            """Calculate gradient colors without matplotlib dependency."""
            if not pd.api.types.is_numeric_dtype(s):
                return ['' for _ in s]

            min_v, max_v = s.min(), s.max()
            if min_v == max_v or pd.isna(min_v) or pd.isna(max_v):
                return ['' for _ in s]

            rng = max_v - min_v
            styles = []

            for v in s:
                if pd.isna(v):
                    styles.append('')
                    continue

                # Normalize 0..1
                norm = (v - min_v) / rng

                # Red (Tomato) -> Yellow -> Green (MediumSeaGreen)
                # Red: (255, 99, 71)
                # Yellow: (255, 255, 0)
                # Green: (60, 179, 113)

                if norm < 0.5:
                    # Red to Yellow
                    local_norm = norm * 2
                    r = 255
                    g = int(99 + (255 - 99) * local_norm)
                    b = int(71 + (0 - 71) * local_norm)
                else:
                    # Yellow to Green
                    local_norm = (norm - 0.5) * 2
                    r = int(255 + (60 - 255) * local_norm)
                    g = int(255 + (179 - 255) * local_norm)
                    b = int(0 + (113 - 0) * local_norm)

                # Use black text for better contrast on light colors
                styles.append(f'background-color: rgb({r}, {g}, {b}); color: black')
            return styles

        for col in highlight_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    styler = styler.apply(_get_custom_gradient, subset=[col])
                except Exception:
                    pass


    # 2. Format numeric columns with smart decimal places
    def get_smart_format(col):
        """Smart format selection based on column max value"""
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            return None
        max_val = df[col].abs().max()
        if pd.isna(max_val):
            return '{:.2f}'
        if max_val > 100:
            return '{:.0f}'
        elif max_val > 10:
            return '{:.1f}'
        else:
            return '{:.2f}'

    format_dict = {}
    for col in df.columns:
        # Handle columns with unit suffixes like "(s)", "(ms)", "(tokens/s)"
        col_base = col.split('(')[0].strip()

        if 'TTFT' in col_base or 'time' in col_base.lower():
            # TTFT columns - always use 3 decimal places (usually small values in seconds)
            format_dict[col] = '{:.3f}'
        elif 'TPOT' in col_base:
            # TPOT in ms - smart format based on value
            format_dict[col] = get_smart_format(col) or '{:.2f}'
        elif 'TPS' in col_base or 'Speed' in col_base or 'Throughput' in col_base:
            # TPS/Speed/Throughput - smart format based on value
            format_dict[col] = get_smart_format(col) or '{:.2f}'
        elif 'Rate' in col_base:
            # Success_Rate - percentage
            format_dict[col] = '{:.1%}'
        elif 'Tokens' in col_base and pd.api.types.is_numeric_dtype(df.get(col, pd.Series(dtype=float))):
            format_dict[col] = '{:,.0f}'
        elif pd.api.types.is_float_dtype(df.get(col, pd.Series(dtype=float))):
            # Other float columns - smart format
            format_dict[col] = get_smart_format(col) or '{:.2f}'

    if format_dict:
        styler = styler.format(format_dict, na_rep='-')

    # 3. Table-wide styles
    styler = styler.set_table_styles([
        # Header style
        {'selector': 'thead th',
         'props': [
             ('background-color', '#007bff'),
             ('color', 'white'),
             ('font-weight', 'bold'),
             ('text-align', 'center'),
             ('padding', '12px 8px'),
             ('border', '1px solid #0056b3'),
             ('font-size', '13px')
         ]},
        # Cell style
        {'selector': 'tbody td',
         'props': [
             ('text-align', 'center'),
             ('padding', '10px 8px'),
             ('border', '1px solid #e6e9ef'),
             ('font-size', '12px')
         ]},
        # Hover effect
        {'selector': 'tbody tr:hover',
         'props': [('background-color', '#f1f8ff !important')]},
        # Alternating rows
        {'selector': 'tbody tr:nth-child(even)',
         'props': [('background-color', '#f8f9fa')]},
        # Table border
        {'selector': '',
         'props': [
             ('border-collapse', 'collapse'),
             ('width', '100%'),
             ('box-shadow', '0 2px 8px rgba(0,0,0,0.1)')
         ]}
    ])

    # 4. Highlight best values
    if highlight_best:
        def highlight_best_value(s):
            """Highlight best value in each numeric column."""
            if not pd.api.types.is_numeric_dtype(s):
                return ['' for _ in s]

            # Determine if smaller or larger is better
            if any(keyword in s.name for keyword in ['TTFT', 'TPOT', 'time', 'latency', 'delay']):
                # Smaller is better
                is_best = s == s.min()
                return ['background-color: #90EE90; font-weight: bold' if v else '' for v in is_best]
            elif any(keyword in s.name for keyword in ['TPS', 'Speed', 'Throughput', 'Rate']):
                # Larger is better
                is_best = s == s.max()
                return ['background-color: #90EE90; font-weight: bold' if v else '' for v in is_best]

            return ['' for _ in s]

        styler = styler.apply(highlight_best_value)

    # 5. Hide index if it's just sequential numbers
    if df.index.name is None and all(isinstance(i, (int, np.integer)) for i in df.index):
        styler = styler.hide(axis='index')

    return styler


def add_statistical_summary(df, metric_cols, group_col=None):
    """
    Add statistical summary rows to DataFrame.

    Args:
        df: Source DataFrame
        metric_cols: List of columns to calculate statistics for
        group_col: Optional grouping column (e.g., 'concurrency')

    Returns:
        DataFrame with summary rows appended
    """
    summary_rows = []

    # Define statistics to calculate
    stats_config = [
        ('Mean', 'mean', '{:.2f}'),
        ('Median', 'median', '{:.2f}'),
        ('P95', lambda x: x.quantile(0.95) if len(x) > 0 else np.nan, '{:.2f}'),
        ('P99', lambda x: x.quantile(0.99) if len(x) > 0 else np.nan, '{:.2f}'),
        ('Min', 'min', '{:.2f}'),
        ('Max', 'max', '{:.2f}'),
        ('StdDev', 'std', '{:.3f}')
    ]

    for stat_name, stat_func, _fmt in stats_config:
        row = {}

        # Fill group column with stat name
        if group_col and group_col in df.columns:
            row[group_col] = f'📊 {stat_name}'
        else:
            # Use first column for stat name
            row[df.columns[0]] = f'📊 {stat_name}'

        # Calculate statistics for metric columns
        for col in df.columns:
            if col in metric_cols and pd.api.types.is_numeric_dtype(df[col]):
                try:
                    if callable(stat_func):
                        value = stat_func(df[col].dropna())
                    else:
                        value = getattr(df[col].dropna(), stat_func)()
                    row[col] = value
                except Exception:
                    row[col] = np.nan
            elif col not in row:
                row[col] = ''

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Concatenate with original
    result = pd.concat([df, summary_df], ignore_index=True)

    return result


def create_comparison_table(df1, df2, labels=None, metrics=None):
    """
    Create side-by-side comparison table.

    Args:
        df1, df2: DataFrames to compare
        labels: Labels for each DataFrame
        metrics: List of metrics to compare (if None, use all numeric columns)

    Returns:
        Styled comparison DataFrame
    """
    if labels is None:
        labels = ['Run 1', 'Run 2']
    if metrics is None:
        metrics = df1.select_dtypes(include=[np.number]).columns.tolist()

    comparison_data = []

    for metric in metrics:
        if metric in df1.columns and metric in df2.columns:
            val1 = df1[metric].mean()
            val2 = df2[metric].mean()
            diff = val2 - val1
            diff_pct = (diff / val1 * 100) if val1 != 0 else 0

            comparison_data.append({
                'Metric': metric,
                labels[0]: val1,
                labels[1]: val2,
                'Difference': diff,
                'Change %': diff_pct
            })

    comp_df = pd.DataFrame(comparison_data)

    # Style the comparison
    styler = comp_df.style

    # Highlight improvements (green) and regressions (red)
    def highlight_change(val):
        if pd.isna(val) or val == 0:
            return ''
        color = '#90EE90' if val > 0 else '#FFB6C1'
        return f'background-color: {color}; font-weight: bold'

    styler = styler.applymap(highlight_change, subset=['Change %'])

    # Format columns
    styler = styler.format({
        labels[0]: '{:.2f}',
        labels[1]: '{:.2f}',
        'Difference': '{:+.2f}',
        'Change %': '{:+.1f}%'
    })

    # Apply table styles
    styler = styler.set_table_styles([
        {'selector': 'thead th',
         'props': [
             ('background-color', '#007bff'),
             ('color', 'white'),
             ('font-weight', 'bold'),
             ('text-align', 'center'),
             ('padding', '10px')
         ]},
        {'selector': 'tbody td',
         'props': [
             ('text-align', 'center'),
             ('padding', '8px'),
             ('border', '1px solid #ddd')
         ]}
    ])

    return styler


def format_large_numbers(df, columns=None):
    """
    Format large numbers with K/M suffixes.

    Args:
        df: DataFrame
        columns: Columns to format (if None, auto-detect)

    Returns:
        DataFrame with formatted columns
    """
    df_copy = df.copy()

    if columns is None:
        # Auto-detect columns with "Tokens" in name
        columns = [col for col in df.columns if 'Tokens' in col or 'tokens' in col]

    def format_number(num):
        if pd.isna(num):
            return '-'
        if abs(num) >= 1_000_000:
            return f'{num/1_000_000:.2f}M'
        elif abs(num) >= 1_000:
            return f'{num/1_000:.1f}K'
        else:
            return f'{num:.0f}'

    for col in columns:
        if col in df_copy.columns:
            df_copy[f'{col}_formatted'] = df_copy[col].apply(format_number)

    return df_copy
