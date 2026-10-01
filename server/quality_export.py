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
    "execution_error",
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


def quality_samples_csv(report: dict[str, Any]) -> str:
    output = io.StringIO()
    output.write("\ufeff")
    fields = (
        *FIELDS,
        "is_correct",
        "is_judge_corrected",
        "judge_verdict",
        "input_tokens",
        "output_tokens",
        "ttft_ms",
        "tps",
        "total_time_ms",
        "measurement_provenance",
    )
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(fields)
    import json

    for name, result in report["datasets"].items():
        for sample in result.get("details", []):
            row = {"dataset": name, **sample}
            row["measurement_provenance"] = json.dumps(
                row.get("measurement_provenance") or {}, ensure_ascii=False, sort_keys=True
            )
            writer.writerow(
                [
                    safe_spreadsheet_text(row.get(field) if row.get(field) is not None else "")
                    for field in fields
                ]
            )
    return output.getvalue()


def quality_summary_csv(report: dict[str, Any]) -> str:
    from server.quality_analysis import METRICS, quality_analysis

    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output, lineterminator="\n")
    metrics = [
        f"{key}_{stat}"
        for key in METRICS
        for stat in ("count", "mean", "median", "p95", "p99", "min", "max", "total")
    ]
    fields = [
        "dataset",
        "model",
        "timestamp",
        "accuracy",
        "correct",
        "total",
        "standard_correct",
        "judge_corrected",
        "duration_seconds",
        "ci95_lower",
        "ci95_upper",
        "cache_count",
        "unknown_cache",
        "non_error_fraction",
        "methods",
        "parsers",
        *metrics,
    ]
    writer.writerow(fields)
    for name, summary in quality_analysis(report)["datasets"].items():
        row = {
            "dataset": name,
            **summary,
            "ci95_lower": summary["ci95"][0] if summary["ci95"] else None,
            "ci95_upper": summary["ci95"][1] if summary["ci95"] else None,
        }
        row.update(
            {
                f"{key}_{stat}": summary["metrics"][key][stat]
                for key in METRICS
                for stat in ("count", "mean", "median", "p95", "p99", "min", "max", "total")
            }
        )
        writer.writerow(
            [
                safe_spreadsheet_text(row.get(field) if row.get(field) is not None else "")
                for field in fields
            ]
        )
    return output.getvalue()
