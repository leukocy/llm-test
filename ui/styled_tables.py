"""
Styled table components for professional data visualization.

Provides styled pandas DataFrames with conditional formatting,
highlighting, and custom themes.
"""

import importlib.util

import numpy as np
import pandas as pd

from ui.reporting.theme import INK, SUBTLE

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


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
    # Prevent pyarrow JSON metadata serialization errors:
    # numpy int64/float64 scalars in DataFrame metadata (attrs, index
    # dtype descriptors) cannot be JSON-serialized by pyarrow's
    # pandas_compat.construct_metadata.  Rebuild the DataFrame with
    # Python-native types so no numpy scalars leak into Arrow metadata.
    df = df.reset_index(drop=True)
    clean = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            clean[col] = [
                (
                    float(x)
                    if isinstance(x, (np.floating, float))
                    else int(x) if isinstance(x, (np.integer, int)) else x
                )
                for x in series
            ]
        else:
            clean[col] = series.tolist()
    df = pd.DataFrame(clean)

    styler = df.style

    # 1. Apply gradient to specified columns
    if highlight_cols:

        def _get_custom_gradient(s):
            """Calculate gradient colors without matplotlib dependency."""
            if not pd.api.types.is_numeric_dtype(s):
                return ["" for _ in s]

            min_v, max_v = s.min(), s.max()
            if min_v == max_v or pd.isna(min_v) or pd.isna(max_v):
                return ["" for _ in s]

            rng = max_v - min_v
            styles = []

            for v in s:
                if pd.isna(v):
                    styles.append("")
                    continue

                # Normalize 0..1
                norm = (v - min_v) / rng

                # Quiet sequential tint keeps the table readable in dense reports.
                r = int(246 - 25 * norm)
                g = int(248 - 18 * norm)
                b = int(252 - 7 * norm)
                styles.append(f"background-color: rgb({r}, {g}, {b}); color: {INK}")
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
            return "{:.2f}"
        if max_val > 100:
            return "{:.0f}"
        elif max_val > 10:
            return "{:.1f}"
        else:
            return "{:.2f}"

    format_dict = {}
    for col in df.columns:
        # Handle columns with unit suffixes like "(s)", "(ms)", "(tokens/s)"
        col_base = col.split("(")[0].strip()

        if "TTFT" in col_base or "time" in col_base.lower():
            # TTFT columns - always use 3 decimal places (usually small values in seconds)
            format_dict[col] = "{:.3f}"
        elif "TPOT" in col_base:
            # TPOT in ms - smart format based on value
            format_dict[col] = get_smart_format(col) or "{:.2f}"
        elif "TPS" in col_base or "Speed" in col_base or "Throughput" in col_base:
            # TPS/Speed/Throughput - smart format based on value
            format_dict[col] = get_smart_format(col) or "{:.2f}"
        elif "Rate" in col_base:
            # Rate columns may be stored as fractions or already as 0-100 percentages.
            format_dict[col] = "{:.1f}%" if "(%)" in col else "{:.1%}"
        elif "Tokens" in col_base and pd.api.types.is_numeric_dtype(
            df.get(col, pd.Series(dtype=float))
        ):
            format_dict[col] = "{:,.0f}"
        elif pd.api.types.is_float_dtype(df.get(col, pd.Series(dtype=float))):
            # Other float columns - smart format
            format_dict[col] = get_smart_format(col) or "{:.2f}"

    if format_dict:
        styler = styler.format(format_dict, na_rep="-")

    # 3. Table-wide styles
    styler = styler.set_table_styles(
        [
            # Header style
            {
                "selector": "thead th",
                "props": [
                    ("background-color", INK),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "12px 8px"),
                    ("border", f"1px solid {INK}"),
                    ("font-size", "13px"),
                ],
            },
            # Cell style
            {
                "selector": "tbody td",
                "props": [
                    ("text-align", "center"),
                    ("padding", "10px 8px"),
                    ("border", f"1px solid {SUBTLE}"),
                    ("font-size", "12px"),
                ],
            },
            # Hover effect
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#eef2ff !important")],
            },
            # Alternating rows
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#fafbfe")],
            },
            # Table border
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("border", f"1px solid {SUBTLE}"),
                ],
            },
        ]
    )

    # 4. Highlight best values
    if highlight_best:

        def highlight_best_value(s):
            """Highlight best value in each numeric column."""
            if not pd.api.types.is_numeric_dtype(s):
                return ["" for _ in s]

            positive = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            positive = positive.where(positive > 0)
            if positive.notna().sum() < 2:
                return ["" for _ in s]

            # Determine if smaller or larger is better
            if any(keyword in s.name for keyword in ["TTFT", "TPOT", "time", "latency", "delay"]):
                is_best = positive == positive.min()
                return [
                    "background-color: #e8f1ff; font-weight: bold" if v else "" for v in is_best
                ]
            elif any(keyword in s.name for keyword in ["TPS", "Speed", "Throughput", "Rate"]):
                is_best = positive == positive.max()
                return [
                    "background-color: #e8f1ff; font-weight: bold" if v else "" for v in is_best
                ]

            return ["" for _ in s]

        styler = styler.apply(highlight_best_value)

    # 5. Hide index if it's just sequential numbers
    if df.index.name is None and all(isinstance(i, (int, np.integer)) for i in df.index):
        styler = styler.hide(axis="index")

    return styler
