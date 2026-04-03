"""
Data export服务

支持willDatabaseinData exportis JSON、CSV、Excel 格式。
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.database.connection import Database, db
from core.repositories.test_run import TestRunRepository
from core.repositories.test_result import TestResultRepository
from core.repositories.report import ReportRepository


logger = logging.getLogger(__name__)


class DataExportService:
    """
    Data export服务

    功能：
    - ExportTest运行and其Result
    - 支持多种格式：JSON、CSV、Excel
    - Batch Export
    """

    def __init__(self, database: Database = None):
        self.db = database or db
        self.run_repo = TestRunRepository(self.db)
        self.result_repo = TestResultRepository(self.db)
        self.report_repo = ReportRepository(self.db)
        self.export_dir = Path("data/exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_run_to_json(
        self,
        run_id: int,
        output_path: str = None,
        include_results: bool = True
    ) -> Optional[str]:
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

        data = {
            "run": run.to_dict(),
            "exported_at": datetime.now().isoformat(),
        }

        if include_results:
            results = self.result_repo.find_by_run_id(run_id)
            data["results"] = [r.to_dict() for r in results]

        if output_path is None:
            output_path = self.export_dir / f"run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_path = Path(output_path)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"已Export到: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None

    def export_run_to_csv(
        self,
        run_id: int,
        output_path: str = None
    ) -> Optional[str]:
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
            output_path = self.export_dir / f"results_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            output_path = Path(output_path)

        try:
            # 收集所has字段
            fieldnames = set()
            for r in results:
                fieldnames.update(r.to_dict().keys())

            fieldnames = sorted(list(fieldnames))

            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for r in results:
                    row = r.to_dict()
                    # Process None 值
                    row = {k: (v if v is not None else '') for k, v in row.items()}
                    writer.writerow(row)

            logger.info(f"已Export到: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None

    def export_run_to_excel(
        self,
        run_id: int,
        output_path: str = None
    ) -> Optional[str]:
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
            output_path = self.export_dir / f"report_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        else:
            output_path = Path(output_path)

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Test运行摘要
                run_df = pd.DataFrame([run.to_dict()])
                run_df.to_excel(writer, sheet_name='运行摘要', index=False)

                # Sheet 2: Test Results
                if results:
                    results_data = [r.to_dict() for r in results]
                    results_df = pd.DataFrame(results_data)
                    results_df.to_excel(writer, sheet_name='Test Results', index=False)

                # Sheet 3: Statistics信息
                stats = self.result_repo.get_aggregate_metrics(run_id)
                stats_df = pd.DataFrame([stats])
                stats_df.to_excel(writer, sheet_name='Statistics信息', index=False)

            logger.info(f"已Export到: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None

    def export_model_history(
        self,
        model_id: str,
        output_path: str = None,
        format: str = "json"
    ) -> Optional[str]:
        """
        ExportModel所hasTest History

        Args:
            model_id: Model ID
            output_path: 输出路径（optional）
            format: 输出格式 (json/csv)

        Returns:
            ExportFile path，失败Return None
        """
        runs = self.run_repo.find_by_model(model_id, limit=1000)

        if not runs:
            logger.warning(f"Not foundModelTest记录: {model_id}")
            return None

        if output_path is None:
            safe_model_id = model_id.replace("/", "_").replace("\\", "_")
            output_path = self.export_dir / f"history_{safe_model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        else:
            output_path = Path(output_path)

        try:
            if format == "json":
                data = {
                    "model_id": model_id,
                    "exported_at": datetime.now().isoformat(),
                    "runs": [r.to_dict() for r in runs],
                }
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            elif format == "csv":
                runs_data = [r.to_dict() for r in runs]
                fieldnames = sorted(set(k for d in runs_data for k in d.keys()))

                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for d in runs_data:
                        row = {k: (v if v is not None else '') for k, v in d.items()}
                        writer.writerow(row)

            logger.info(f"已Export到: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None

    def get_export_history(self) -> List[Dict[str, Any]]:
        """GetExport历史"""
        exports = []
        for f in self.export_dir.glob("*"):
            if f.is_file():
                stat = f.stat()
                exports.append({
                    "path": str(f),
                    "name": f.name,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

        exports.sort(key=lambda x: x['created_at'], reverse=True)
        return exports


def export_to_json(run_id: int, output_path: str = None) -> Optional[str]:
    """便捷函数：Export到 JSON"""
    service = DataExportService()
    return service.export_run_to_json(run_id, output_path)


def export_to_csv(run_id: int, output_path: str = None) -> Optional[str]:
    """便捷函数：Export到 CSV"""
    service = DataExportService()
    return service.export_run_to_csv(run_id, output_path)


def export_to_excel(run_id: int, output_path: str = None) -> Optional[str]:
    """便捷函数：Export到 Excel"""
    service = DataExportService()
    return service.export_run_to_excel(run_id, output_path)
