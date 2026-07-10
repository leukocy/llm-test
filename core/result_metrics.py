"""Shared result metric helpers for benchmark summaries and UI rendering."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

_EMPTY_ERROR_STRINGS = frozenset({"", "none", "null", "nan", "nat", "<na>"})

PERFORMANCE_KEYWORDS = (
    "ttft",
    "tpot",
    "tps",
    "throughput",
    "prefill speed",
    "prefill_speed",
    "rps",
    "qpm",
    "latency",
)

NON_PERFORMANCE_KEYWORDS = (
    "success rate",
    "success_rate",
    "cache hit rate",
    "cache_hit_rate",
    "cache hit",
    "cache_hit",
    "tokens",
    "token",
    "requests",
    "num_requests",
    "concurrency",
    "round",
    "target",
)


def success_mask_from_error(error_values: pd.Series | Iterable[object]) -> pd.Series:
    """Return True for rows whose error value represents a successful request."""
    series = error_values if isinstance(error_values, pd.Series) else pd.Series(error_values)
    text_values = series.astype("string").str.strip().str.lower()
    return series.isna() | text_values.isin(_EMPTY_ERROR_STRINGS)


def count_successful_requests(error_values: pd.Series | Iterable[object]) -> int:
    """Count successful requests from an error column."""
    return int(success_mask_from_error(error_values).sum())


def calculate_success_rate(
    error_values: pd.Series | Iterable[object], *, percent: bool = False
) -> float:
    """Calculate request success rate from an error column."""
    mask = success_mask_from_error(error_values)
    if mask.empty:
        return 0.0

    rate = float(mask.mean())
    return rate * 100 if percent else rate


def calculate_success_rate_percent(error_values: pd.Series | Iterable[object]) -> float:
    """Calculate request success rate as a 0-100 percentage."""
    return calculate_success_rate(error_values, percent=True)


def _normalize_column_name(column: object) -> str:
    return str(column).lower().replace("_", " ")


def is_performance_metric_column(column: object) -> bool:
    """Return True for latency/throughput rate columns where zero means invalid."""
    name = _normalize_column_name(column)
    if any(keyword in name for keyword in NON_PERFORMANCE_KEYWORDS):
        return False
    return any(keyword.replace("_", " ") in name for keyword in PERFORMANCE_KEYWORDS)


def _resolve_columns(df: pd.DataFrame, columns: Iterable[object] | None) -> list[object]:
    candidates = df.columns if columns is None else columns
    return [
        column
        for column in candidates
        if column in df.columns and is_performance_metric_column(column)
    ]


def valid_performance_series(series: pd.Series) -> pd.Series:
    """Convert a metric series to numeric values and mask non-positive values."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric > 0, np.nan)


def sanitize_performance_metrics(
    df: pd.DataFrame,
    columns: Iterable[object] | None = None,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Replace zero or negative values in performance metric columns with NaN."""
    result = df.copy() if copy else df
    for column in _resolve_columns(result, columns):
        result[column] = valid_performance_series(result[column])
    return result


def positive_mean(series: pd.Series) -> float:
    return float(valid_performance_series(series).mean())


def positive_min(series: pd.Series) -> float:
    return float(valid_performance_series(series).min())


def positive_max(series: pd.Series) -> float:
    return float(valid_performance_series(series).max())


def positive_quantile(series: pd.Series, q: float) -> float:
    valid = valid_performance_series(series).dropna()
    return float(valid.quantile(q)) if not valid.empty else float(np.nan)


def summarize_metric_extreme(
    df: pd.DataFrame,
    group_cols: str | list[str],
    metric_col: str,
    output_col: str | None = None,
    how: str = "max",
) -> pd.DataFrame:
    """Summarize a metric by group, ignoring non-positive values."""
    if how not in {"min", "max"}:
        raise ValueError("how must be 'min' or 'max'")

    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    output_col = output_col or metric_col
    groups = df[group_cols].drop_duplicates().reset_index(drop=True)

    if metric_col not in df.columns:
        return groups.assign(**{output_col: np.nan})

    work = df[group_cols].copy()
    work[metric_col] = valid_performance_series(df[metric_col])
    agg = (
        work.groupby(group_cols, dropna=False)[metric_col]
        .agg(how)
        .reset_index()
        .rename(columns={metric_col: output_col})
    )
    return groups.merge(agg, on=group_cols, how="left")


def summarize_metric_row(
    df: pd.DataFrame,
    group_cols: str | list[str],
    metric_col: str,
    output_cols: dict[str, str] | None = None,
    how: str = "max",
) -> pd.DataFrame:
    """Pick the row with a group-wise metric extreme, ignoring invalid metric zeros."""
    if how not in {"min", "max"}:
        raise ValueError("how must be 'min' or 'max'")

    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    output_cols = output_cols or {metric_col: metric_col}
    groups = df[group_cols].drop_duplicates().reset_index(drop=True)

    missing = [col for col in output_cols if col not in df.columns]
    if metric_col not in df.columns or missing:
        return groups.assign(**dict.fromkeys(output_cols.values(), np.nan))

    keep_cols = list(dict.fromkeys(group_cols + [metric_col] + list(output_cols.keys())))
    work = df[keep_cols].copy()
    work[metric_col] = valid_performance_series(work[metric_col])
    work = work.dropna(subset=[metric_col])

    if work.empty:
        return groups.assign(**dict.fromkeys(output_cols.values(), np.nan))

    ascending = how == "min"
    selected = (
        work.sort_values(
            group_cols + [metric_col], ascending=[True] * len(group_cols) + [ascending]
        )
        .groupby(group_cols, as_index=False, dropna=False)
        .first()
    )
    selected = selected[group_cols + list(output_cols.keys())].rename(columns=output_cols)
    return groups.merge(selected, on=group_cols, how="left")


def fill_non_performance_na(df: pd.DataFrame, value: object = 0) -> pd.DataFrame:
    """Fill NaN in non-performance columns while preserving missing metric values."""
    result = df.copy()
    for column in result.columns:
        if not is_performance_metric_column(column):
            result[column] = result[column].fillna(value)
    return result


def safe_positive_max(values: pd.Series, multiplier: float = 1.0, fallback: float = 1.0) -> float:
    """Return a positive max for UI progress bounds."""
    max_value = valid_performance_series(values).max()
    if pd.isna(max_value) or max_value <= 0:
        return fallback
    return float(max_value) * multiplier
