"""
Data export服务

支持willDatabaseinData exportis JSON、CSV、Excel 格式。
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from core.database.connection import Database, db
from core.repositories.report import ReportRepository
from core.repositories.test_result import TestResultRepository
from core.repositories.test_run import TestRunRepository

logger = logging.getLogger(__name__)


class DataExportService:
    """
    Data export服务

    功能：
    - ExportTest运行and其Result
    - 支持多种格式：JSON、CSV、Excel
    - Batch Export
    """

    def __init__(self, database: Database | None = None):
        self.db = database or db
        self.run_repo = TestRunRepository(self.db)
        self.result_repo = TestResultRepository(self.db)
        self.report_repo = ReportRepository(self.db)
        self.export_dir = Path("data/exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_run_to_json(
        self, run_id: int, output_path: str | None = None, include_results: bool = True
    ) -> str | None:
        """
        ExportTest运行到 JSON 文件

        Args:
            run_id: 运行 ID
            output_path: 输出路径（optional）
            include_results: is否包含Detailed Results

        Returns:
            ExportFile path，失败Return None
        """
        run = self.run_repo.find_by_id(run_id)
        if not run:
            logger.error(f"Not foundTest运行: {run_id}")
            return None

        data: dict[str, Any] = {
            "run": run.to_dict(),
            "exported_at": datetime.now().isoformat(),
        }

        if include_results:
            results = self.result_repo.find_by_run_id(run_id)
            data["results"] = [r.to_dict() for r in results]

        if output_path is None:
            final_path: Path = (
                self.export_dir / f"run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        else:
            final_path = Path(output_path)

        try:
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"已Export到: {final_path}")
            return str(final_path)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None

    def export_run_to_csv(self, run_id: int, output_path: str | None = None) -> str | None:
        """
        ExportTest Results到 CSV 文件

        Args:
            run_id: 运行 ID
            output_path: 输出路径（optional）

        Returns:
            ExportFile path，失败Return None
        """
        results = self.result_repo.find_by_run_id(run_id)
        if not results:
            logger.error(f"Not foundTest Results: {run_id}")
            return None

        if output_path is None:
            final_path: Path = (
                self.export_dir / f"results_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        else:
            final_path = Path(output_path)

        try:
            # 收集所has字段
            fieldnames: set[str] = set()
            for r in results:
                fieldnames.update(r.to_dict().keys())
            fieldnames.add("metric_contract_version")

            sorted_fieldnames = sorted(fieldnames)

            with open(final_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
                writer.writeheader()

                for r in results:
                    row = r.to_dict()
                    row["metric_contract_version"] = (
                        r.extra_metrics.get("metric_contract_version") or "legacy-unversioned"
                    )
                    # Process None 值
                    row = {k: (v if v is not None else "") for k, v in row.items()}
                    writer.writerow(row)

            logger.info(f"已Export到: {final_path}")
            return str(final_path)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None

    def export_run_to_excel(self, run_id: int, output_path: str | None = None) -> str | None:
        """
        ExportTest Results到 Excel 文件

        Args:
            run_id: 运行 ID
            output_path: 输出路径（optional）

        Returns:
            ExportFile path，失败Return None
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("need安装 pandas: pip install pandas openpyxl")
            return None

        run = self.run_repo.find_by_id(run_id)
        results = self.result_repo.find_by_run_id(run_id)

        if not run:
            logger.error(f"Not foundTest运行: {run_id}")
            return None

        if output_path is None:
            final_path: Path = (
                self.export_dir / f"report_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
        else:
            final_path = Path(output_path)

        try:
            with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
                # Sheet 1: Test运行摘要
                run_df = pd.DataFrame([run.to_dict()])
                run_df.to_excel(writer, sheet_name="运行摘要", index=False)

                # Sheet 2: Test Results
                if results:
                    results_data = [r.to_dict() for r in results]
                    results_df = pd.DataFrame(results_data)
                    results_df.to_excel(writer, sheet_name="Test Results", index=False)

                # Sheet 3: Statistics信息
                stats = self.result_repo.get_aggregate_metrics(run_id)
                stats_df = pd.DataFrame([stats])
                stats_df.to_excel(writer, sheet_name="Statistics信息", index=False)

            logger.info(f"已Export到: {final_path}")
            return str(final_path)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None


def export_to_json(run_id: int, output_path: str | None = None) -> str | None:
    """便捷函数：Export到 JSON"""
    service = DataExportService()
    return service.export_run_to_json(run_id, output_path)


def export_to_csv(run_id: int, output_path: str | None = None) -> str | None:
    """便捷函数：Export到 CSV"""
    service = DataExportService()
    return service.export_run_to_csv(run_id, output_path)


def export_to_excel(run_id: int, output_path: str | None = None) -> str | None:
    """便捷函数：Export到 Excel"""
    service = DataExportService()
    return service.export_run_to_excel(run_id, output_path)
