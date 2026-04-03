"""
Data import服务

支持从 CSV 文件Import历史Data到Database。
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from core.database.connection import Database, db
from core.repositories.test_run import TestRunRepository
from core.repositories.test_result import TestResultRepository
from core.models.test_run import TestRun, TestRunStatus
from core.models.test_result import TestResult


logger = logging.getLogger(__name__)


class DataImportService:
    """
    Data import服务

    功能：
    - 从 CSV 文件ImportTest Results
    - 批量Import多文件
    - 进度Callback支持
    - ErrorProcessand跳过
    """

    def __init__(self, database: Database = None):
        self.db = database or db
        self.run_repo = TestRunRepository(self.db)
        self.result_repo = TestResultRepository(self.db)

    def import_csv_file(
        self,
        csv_path: str,
        model_id: str = None,
        test_type: str = None,
        provider: str = None,
        on_progress: callable = None
    ) -> Tuple[int, List[str]]:
        """
        Import单 CSV 文件

        Args:
            csv_path: CSV File path
            model_id: Model ID（optional，从Filename推断）
            test_type: Test Type（optional，从Filename推断）
            provider: Provider（optional）
            on_progress: 进度Callback函数 (current, total)

        Returns:
            (Import记录数, Error消息列表)
        """
        path = Path(csv_path)
        if not path.exists():
            return 0, [f"文件not存in: {csv_path}"]

        # 从Filename推断信息
        filename = path.stem
        if model_id is None:
            model_id = self._extract_model_id(filename)
        if test_type is None:
            test_type = self._extract_test_type(filename)

        errors = []
        imported = 0

        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return 0, ["文件is空ornohas效Data"]

            # CreateTest运行
            run = TestRun.create(
                test_type=test_type or "unknown",
                model_id=model_id or "unknown",
                provider=provider,
            )
            run.csv_path = str(path)
            run.total_requests = len(rows)
            run.status = TestRunStatus.COMPLETED.value

            run_id = self.run_repo.insert(run)

            # ImportResult
            results = []
            for i, row in enumerate(rows):
                try:
                    result = self._row_to_result(row, run_id, i)
                    results.append(result)
                    imported += 1

                    if on_progress and (i + 1) % 100 == 0:
                        on_progress(i + 1, len(rows))

                except Exception as e:
                    errors.append(f"行 {i + 1}: {str(e)}")

            # 批量Insert
            if results:
                self.result_repo.insert_batch(results)

            # Update运行Statistics
            stats = self.result_repo.get_aggregate_metrics(run_id)
            self.run_repo.update_statistics(run_id, stats)
            self.run_repo.complete(run_id, success=True)

        except Exception as e:
            errors.append(f"Failed to read file: {str(e)}")

        return imported, errors

    def import_directory(
        self,
        directory: str,
        recursive: bool = True,
        on_progress: callable = None
    ) -> Tuple[int, int, List[str]]:
        """
        Import目录in所has CSV 文件

        Args:
            directory: 目录路径
            recursive: is否递归查找
            on_progress: 进度Callback (filename, current, total)

        Returns:
            (succeeded文件数, 总Import记录数, Error消息列表)
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0, 0, [f"目录not存in: {directory}"]

        # 查找 CSV 文件
        if recursive:
            csv_files = list(dir_path.rglob("*.csv"))
        else:
            csv_files = list(dir_path.glob("*.csv"))

        if not csv_files:
            return 0, 0, ["Not found CSV 文件"]

        all_errors = []
        success_count = 0
        total_imported = 0

        for i, csv_file in enumerate(csv_files):
            if on_progress:
                on_progress(str(csv_file), i + 1, len(csv_files))

            imported, errors = self.import_csv_file(str(csv_file))
            if imported > 0:
                success_count += 1
                total_imported += imported
            all_errors.extend(errors)

        return success_count, total_imported, all_errors

    def _row_to_result(self, row: Dict[str, Any], run_id: int, index: int) -> TestResult:
        """will CSV 行Convertis TestResult"""
        def parse_float(value):
            if value is None or value == '':
                return None
            try:
                return float(value)
            except:
                return None

        def parse_int(value):
            if value is None or value == '':
                return None
            try:
                return int(float(value))
            except:
                return None

        return TestResult(
            run_id=run_id,
            request_index=index,
            session_id=parse_int(row.get('session_id')),
            round=parse_int(row.get('round')),
            concurrency_level=parse_int(row.get('concurrency')),
            input_tokens_target=parse_int(row.get('input_tokens_target')),
            context_length_target=parse_int(row.get('context_length_target')),
            ttft=parse_float(row.get('ttft')),
            tpot=parse_float(row.get('tpot')),
            tpot_p95=parse_float(row.get('tpot_p95')),
            tpot_p99=parse_float(row.get('tpot_p99')),
            total_time=parse_float(row.get('total_time')),
            decode_time=parse_float(row.get('decode_time')),
            prefill_speed=parse_float(row.get('prefill_speed')),
            tps=parse_float(row.get('tps')),
            system_throughput=parse_float(row.get('system_throughput')),
            system_input_throughput=parse_float(row.get('system_input_throughput')),
            system_output_throughput=parse_float(row.get('system_output_throughput')),
            system_total_throughput=parse_float(row.get('system_total_throughput')),
            rps=parse_float(row.get('rps')),
            prefill_tokens=parse_int(row.get('prefill_tokens')),
            decode_tokens=parse_int(row.get('decode_tokens')),
            cache_hit_tokens=parse_int(row.get('cache_hit_tokens')),
            api_prefill=parse_int(row.get('api_prefill')),
            api_decode=parse_int(row.get('api_decode')),
            effective_prefill_tokens=parse_int(row.get('effective_prefill_tokens')),
            effective_decode_tokens=parse_int(row.get('effective_decode_tokens')),
            token_source=row.get('token_source'),
            token_calc_method=row.get('token_calc_method'),
            cache_hit_source=row.get('cache_hit_source'),
            start_time=parse_float(row.get('start_time')),
            end_time=parse_float(row.get('end_time')),
            error=row.get('error') if row.get('error') and row.get('error') != 'None' else None,
            created_at=datetime.now(),
        )

    def _extract_model_id(self, filename: str) -> str:
        """从Filename提取Model ID"""
        # 格式: benchmark_results_{model_id}_{test_type}_{timestamp}.csv
        parts = filename.replace("benchmark_results_", "").split("_")
        if len(parts) >= 3:
            # Model ID 可能包含under划线，取倒数三到最后
            model_parts = parts[:-2]
            return "_".join(model_parts)
        return "unknown"

    def _extract_test_type(self, filename: str) -> str:
        """从Filename提取Test Type"""
        # 格式: benchmark_results_{model_id}_{test_type}_{timestamp}.csv
        parts = filename.replace("benchmark_results_", "").split("_")
        if len(parts) >= 2:
            return parts[-2]
        return "unknown"


def import_csv_to_database(
    csv_path: str,
    model_id: str = None,
    test_type: str = None
) -> Tuple[int, List[str]]:
    """
    便捷函数：Import CSV 文件到Database

    Args:
        csv_path: CSV File path
        model_id: Model ID（optional）
        test_type: Test Type（optional）

    Returns:
        (Import记录数, Error消息列表)
    """
    service = DataImportService()
    return service.import_csv_file(csv_path, model_id, test_type)
