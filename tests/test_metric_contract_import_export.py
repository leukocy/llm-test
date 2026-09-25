import csv
from unittest.mock import MagicMock

import pytest

from core.database import db_manager
from core.models.test_result import TestResult as BenchmarkResult
from core.services.data_export import DataExportService


def test_csv_import_preserves_metric_contract_in_result_metadata():
    service = db_manager.importer

    result = service._row_to_result(
        {"session_id": "1", "tps": "2.5", "metric_contract_version": "decode-interval-v2"},
        run_id=1,
        index=0,
    )

    assert result.tps == 2.5
    assert result.extra_metrics["metric_contract_version"] == "decode-interval-v2"


def test_csv_import_reads_embedded_contract_and_rejects_conflicting_labels():
    service = db_manager.importer
    row = {"extra_metrics": '{"metric_contract_version": "decode-interval-v2"}'}

    assert (
        service._row_to_result(row, run_id=1, index=0).extra_metrics["metric_contract_version"]
        == "decode-interval-v2"
    )

    row["metric_contract_version"] = "legacy-unversioned"
    with pytest.raises(ValueError, match="conflicting metric contract"):
        service._row_to_result(row, run_id=1, index=0)


def test_csv_import_rejects_mixed_metric_contracts(tmp_path):
    source = tmp_path / "mixed.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["session_id", "metric_contract_version"])
        writer.writeheader()
        writer.writerow({"session_id": 1, "metric_contract_version": "decode-interval-v2"})
        writer.writerow({"session_id": 2, "metric_contract_version": "legacy-unversioned"})

    imported, errors = db_manager.importer.import_csv_file(str(source))

    assert imported == 0
    assert errors == ["CSV contains mixed metric contract versions"]


def test_csv_export_exposes_metric_contract_column(tmp_path):
    service = DataExportService.__new__(DataExportService)
    service.result_repo = MagicMock()
    service.result_repo.find_by_run_id.return_value = [
        BenchmarkResult(
            run_id=1,
            tps=2.5,
            extra_metrics={"metric_contract_version": "decode-interval-v2"},
        )
    ]
    export_path = tmp_path / "results.csv"

    assert service.export_run_to_csv(1, str(export_path)) == str(export_path)

    with export_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["metric_contract_version"] == "decode-interval-v2"
