"""Spreadsheet-safe review export for persisted quality failures."""

from __future__ import annotations

import csv
import io
from typing import Any

from utils.spreadsheet import safe_spreadsheet_text

FIELDS = (
    "dataset",
    "sample_id",
    "category",
    "failure_category",
    "question",
    "correct_answer",
    "predicted_answer",
    "prompt",
    "model_response",
    "evaluation_method",
    "answer_parse_method",
    "answer_parse_confidence",
    "failure_analysis",
    "error",
)


def quality_errors_csv(report: dict[str, Any]) -> str:
    """Export every failed sample, including its provenance dataset name."""
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(FIELDS)
    for dataset, result in report["datasets"].items():
        if not isinstance(result, dict) or not isinstance(result.get("details"), list):
            raise ValueError("Quality report samples are invalid")
        for sample in result["details"]:
            if not isinstance(sample, dict):
                raise ValueError("Quality report samples are invalid")
            if sample.get("is_correct") and not sample.get("error"):
                continue
            row = {"dataset": dataset, **sample}
            writer.writerow(
                [
                    safe_spreadsheet_text(row.get(field) if row.get(field) is not None else "")
                    for field in FIELDS
                ]
            )
    return output.getvalue()
