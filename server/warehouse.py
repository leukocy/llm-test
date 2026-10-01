"""Bounded read model for the legacy warehouse's reusable run data."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, cast

from core.models.test_run import TestRun
from core.warehouse import (
    build_cross_matrix,
    build_hardware_inventory_rows,
    build_hm_test_rows,
    build_scaling_efficiency,
    export_template_csv,
    export_template_json,
    interpret_efficiency,
    project_run,
)

SCAN_LIMIT = 1000
MATRIX_METRICS = {"decode_tps", "effective_bandwidth_gbps"}
EXPORT_TEMPLATES = {"hwInventory", "hmTest"}


def _finite_values(value: Any) -> Any:
    """Represent non-finite legacy measurements as missing data in JSON and exports."""
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_values(item) for item in value]
    return value


class ExportScopeTooLarge(ValueError):
    """The export would silently omit rows outside the bounded read window."""


class WarehouseRunNotFound(LookupError):
    """No persisted measurement has this test ID."""


@dataclass(frozen=True)
class WarehouseSelection:
    model_id: str | None = None
    machine_id: str | None = None
    test_type: str | None = None
    status: str | None = None
    external_level: str | None = None
    search: str | None = None


@dataclass
class WarehouseWindow:
    runs: list[TestRun]
    projected: list[dict[str, Any]]
    matched_total: int
    scanned: int
    invalid_rows: int

    @property
    def truncated(self) -> bool:
        return self.matched_total > self.scanned


class WarehouseReader:
    def __init__(self, db_path: str | Path) -> None:
        self.path = str(db_path)

    @staticmethod
    def _where(selection: WarehouseSelection) -> tuple[str, list[str]]:
        clauses: list[str] = []
        params: list[str] = []
        for field in ("model_id", "machine_id", "test_type"):
            value = getattr(selection, field)
            if value:
                clauses.append(f"{field} = ?")
                params.append(value)
        if selection.status:
            clauses.append("COALESCE(NULLIF(status_detail, ''), status) = ?")
            params.append(selection.status)
        if selection.external_level:
            clauses.append("COALESCE(NULLIF(external_level, ''), 'internal') = ?")
            params.append(selection.external_level)
        if selection.search:
            clauses.append(
                "(model_id LIKE ? OR machine_id LIKE ? OR tester LIKE ? OR notes LIKE ?)"
            )
            pattern = f"%{selection.search}%"
            params.extend([pattern] * 4)
        return ("WHERE " + " AND ".join(clauses) if clauses else ""), params

    def window(self, selection: WarehouseSelection) -> WarehouseWindow:
        where, params = self._where(selection)
        with closing(sqlite3.connect(self.path, timeout=30)) as conn, conn:
            conn.row_factory = sqlite3.Row
            matched = int(
                conn.execute(f"SELECT COUNT(*) FROM test_runs {where}", params).fetchone()[0]
            )
            rows = conn.execute(
                f"SELECT * FROM test_runs {where} ORDER BY created_at DESC, id DESC LIMIT ?",
                (*params, SCAN_LIMIT),
            ).fetchall()
        runs: list[TestRun] = []
        projected: list[dict[str, Any]] = []
        invalid = 0
        for row in rows:
            try:
                run = TestRun.from_row(dict(row))
                projection = project_run(run)
            except (ValueError, TypeError, KeyError, AttributeError):
                invalid += 1
                continue
            runs.append(run)
            projected.append(projection)
        superseded = {run.supersedes_test_id for run in runs if run.supersedes_test_id}
        kept = [
            (run, item)
            for run, item in zip(runs, projected, strict=True)
            if run.test_id not in superseded
        ]
        return WarehouseWindow(
            runs=[run for run, _ in kept],
            projected=[item for _, item in kept],
            matched_total=matched,
            scanned=len(rows),
            invalid_rows=invalid,
        )

    def filter_options(self) -> dict[str, list[str]]:
        options: dict[str, list[str]] = {}
        with closing(sqlite3.connect(self.path, timeout=30)) as conn, conn:
            for field in ("model_id", "machine_id", "test_type", "external_level"):
                rows = conn.execute(
                    f"SELECT DISTINCT {field} FROM test_runs "
                    f"WHERE {field} IS NOT NULL AND {field} != '' ORDER BY {field} LIMIT 200"
                ).fetchall()
                options[field] = [str(row[0]) for row in rows]
            rows = conn.execute(
                """SELECT DISTINCT COALESCE(NULLIF(status_detail, ''), status) FROM test_runs
                   WHERE status IS NOT NULL ORDER BY 1 LIMIT 200"""
            ).fetchall()
            options["status"] = [str(row[0]) for row in rows]
        return options

    def detail(self, test_id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.path, timeout=30)) as conn, conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM test_runs WHERE test_id = ?", (test_id,)).fetchone()
        if row is None:
            raise WarehouseRunNotFound(test_id)
        try:
            run = TestRun.from_row(dict(row))
            fields = project_run(run)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError("Measurement metadata cannot be parsed") from exc
        return cast(
            dict[str, Any],
            _finite_values(
                {
                    "run_id": run.id,
                    "test_id": run.test_id,
                    "test_type": run.test_type,
                    "fields": fields,
                }
            ),
        )

    def dashboard(
        self,
        selection: WarehouseSelection,
        *,
        metric: str = "decode_tps",
        aggregate: str = "latest",
    ) -> dict[str, Any]:
        if metric not in MATRIX_METRICS or aggregate not in {"latest", "best"}:
            raise ValueError("Unsupported warehouse matrix option")
        window = self.window(selection)
        matrix = build_cross_matrix(window.runs, metric=metric, agg=aggregate)
        inventory = build_hardware_inventory_rows(window.runs)
        scaling = build_scaling_efficiency(window.runs)
        for row in scaling:
            row["interpretation"] = interpret_efficiency(row["efficiency"])
        rows = []
        for run, item in zip(window.runs, window.projected, strict=True):
            rows.append(
                {
                    "run_id": run.id,
                    "test_id": run.test_id,
                    "test_type": run.test_type,
                    **{
                        key: item.get(key)
                        for key in (
                            "date",
                            "machine_id",
                            "model_name",
                            "engine",
                            "parallel_strategy",
                            "concurrency",
                            "decode_tps",
                            "ttft_s",
                            "effective_bandwidth_gbps",
                            "gpu_vram_peak_gb",
                            "bottleneck",
                            "status",
                            "external_level",
                            "tester",
                            "config_hash",
                            "tags",
                        )
                    },
                }
            )
        return cast(
            dict[str, Any],
            _finite_values(
                {
                    "scope": {
                        "matched_total": window.matched_total,
                        "scanned": window.scanned,
                        "shown": len(rows),
                        "scan_limit": SCAN_LIMIT,
                        "truncated": window.truncated,
                        "invalid_rows": window.invalid_rows,
                    },
                    "kpis": {
                        "runs": len(rows),
                        "machines": len(
                            {item["machine_id"] for item in rows if item["machine_id"]}
                        ),
                        "models": len({item["model_name"] for item in rows if item["model_name"]}),
                        "completed": sum(item["status"] == "completed" for item in rows),
                        "publishable": sum(
                            item["external_level"] == "publishable" for item in rows
                        ),
                    },
                    "filters": self.filter_options(),
                    "rows": rows,
                    "matrix": asdict(matrix),
                    "inventory": [
                        {
                            key: row.get(key)
                            for key in (
                                "machine_id",
                                "cpu_model",
                                "memory_capacity_gb",
                                "gpu_model",
                                "gpu_count",
                                "gpu_vram_gb",
                                "gpu_bandwidth_gbps",
                                "os",
                                "driver",
                                "owner",
                            )
                        }
                        for row in inventory
                    ],
                    "scaling": scaling,
                }
            ),
        )

    def export(self, selection: WarehouseSelection, template: str, format: str) -> str:
        if template not in EXPORT_TEMPLATES or format not in {"csv", "json"}:
            raise ValueError("Unsupported warehouse export")
        window = self.window(selection)
        if window.truncated or window.invalid_rows:
            raise ExportScopeTooLarge("Narrow the filters or repair invalid rows before export")
        rows = (
            build_hardware_inventory_rows(window.runs)
            if template == "hwInventory"
            else build_hm_test_rows(window.runs)
        )
        clean_rows = _finite_values(rows)
        return (
            export_template_csv(template, clean_rows)
            if format == "csv"
            else export_template_json(template, clean_rows)
        )
