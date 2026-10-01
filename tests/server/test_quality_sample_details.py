"""All-sample export retains persisted diagnostics without rerunning scoring."""

import csv
import io

from server.quality_export import quality_samples_csv


def test_diagnostic_export_preserves_missing_values_and_safe_multiline_reasoning():
    report = {
        "datasets": {
            "lab": {
                "details": [
                    {
                        "sample_id": "a",
                        "latency_ms": 12.5,
                        "reasoning_quality_overall": 7.2,
                        "reasoning_content": "=unsafe\nsecond line",
                    },
                    {"sample_id": "b", "latency_ms": None, "reasoning_quality_overall": None},
                    {"sample_id": "c", "reasoning_quality": 6},
                ]
            }
        }
    }
    rows = list(csv.DictReader(io.StringIO(quality_samples_csv(report).lstrip("\ufeff"))))
    assert rows[0]["latency_ms"] == "12.5"
    assert rows[0]["reasoning_quality_overall"] == "7.2"
    assert rows[0]["reasoning_content"] == "'=unsafe\nsecond line"
    assert rows[1]["latency_ms"] == rows[1]["reasoning_quality_overall"] == ""
    assert rows[2]["reasoning_quality"] == "6"
    assert [row["sample_id"] for row in rows] == ["a", "b", "c"]
