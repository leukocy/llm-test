"""Safe, typed exports for data opened in spreadsheet applications."""

from __future__ import annotations

import re

import pandas as pd


def safe_spreadsheet_text(value):
    """Prefix formula-like user text so spreadsheet software keeps it literal."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def safe_csv_text(df: pd.DataFrame) -> str:
    """Render CSV without changing numeric types or the source DataFrame."""
    safe = df.copy()
    safe.columns = [safe_spreadsheet_text(str(column)) for column in safe.columns]
    for index in range(len(safe.columns)):
        series = safe.iloc[:, index]
        if not pd.api.types.is_numeric_dtype(series):
            safe.isetitem(index, series.map(safe_spreadsheet_text))
    return str(safe.to_csv(index=False))


def safe_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return spreadsheet-safe CSV with a UTF-8 BOM for multilingual labels."""
    return safe_csv_text(df).encode("utf-8-sig")


def safe_download_filename(name: str) -> str:
    """Keep download names within a single portable path component."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", str(name)).strip(" .")
    if len(cleaned) > 160 and "." in cleaned:
        stem, extension = cleaned.rsplit(".", 1)
        if len(extension) <= 10:
            cleaned = stem[: 159 - len(extension)] + "." + extension
    return cleaned[:160] or "report"
