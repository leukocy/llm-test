"""
Test Results Repository
"""

from typing import Optional, List, Dict, Any

from core.database.connection import Database, db
from core.repositories.base import BaseRepository
from core.models.test_result import TestResult


class TestResultRepository(BaseRepository[TestResult]):
    """Test Results Repository"""

    def __init__(self, database: Database = None):
        super().__init__(database)
        self._table_name = "test_results"

    def _from_row(self, row: Dict[str, Any]) -> TestResult:
        return TestResult.from_row(row)

    def insert(self, result: TestResult) -> int:
        """
        InsertTest Results

        Args:
            result: TestResult 实例

        Returns:
            新记录 ID
        """
        data = result.to_dict()
        columns = [k for k, v in data.items() if v is not None and k != 'id']
        placeholders = ", ".join(["?" for _ in columns])
        columns_str = ", ".join(columns)
        values = [v for k, v in data.items() if v is not None and k != 'id']

        sql = f"INSERT INTO test_results ({columns_str}) VALUES ({placeholders})"
        cursor = self.db.execute(sql, tuple(values))
        return cursor.lastrowid

    def insert_batch(self, results: List[TestResult]) -> int:
        """
        批量InsertTest Results

        Args:
            results: TestResult 列表

        Returns:
            Insert记录数
        """
        if not results:
            return 0

        # 准备Data
        sql = """
            INSERT INTO test_results (
                run_id, session_id, request_index, round, concurrency_level, batch_id,
                input_tokens_target, context_length_target,
                ttft, tpot, tpot_p95, tpot_p99, total_time, decode_time, prefill_speed,
                tps, system_throughput, system_input_throughput, system_output_throughput,
                system_total_throughput, rps,
                prefill_tokens, decode_tokens, cache_hit_tokens,
                api_prefill, api_decode, effective_prefill_tokens, effective_decode_tokens,
                token_source, token_calc_method, cache_hit_source,
                start_time, end_time, error, error_type, prompt_text, output_text, extra_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params_list = []
        for r in results:
            data = r.to_dict()
            params_list.append((
                data.get('run_id'),
                data.get('session_id'),
                data.get('request_index'),
                data.get('round'),
                data.get('concurrency_level'),
                data.get('batch_id'),
                data.get('input_tokens_target'),
                data.get('context_length_target'),
                data.get('ttft'),
                data.get('tpot'),
                data.get('tpot_p95'),
                data.get('tpot_p99'),
                data.get('total_time'),
                data.get('decode_time'),
                data.get('prefill_speed'),
                data.get('tps'),
                data.get('system_throughput'),
                data.get('system_input_throughput'),
                data.get('system_output_throughput'),
                data.get('system_total_throughput'),
                data.get('rps'),
                data.get('prefill_tokens'),
                data.get('decode_tokens'),
                data.get('cache_hit_tokens'),
                data.get('api_prefill'),
                data.get('api_decode'),
                data.get('effective_prefill_tokens'),
                data.get('effective_decode_tokens'),
                data.get('token_source'),
                data.get('token_calc_method'),
                data.get('cache_hit_source'),
                data.get('start_time'),
                data.get('end_time'),
                data.get('error'),
                data.get('error_type'),
                data.get('prompt_text'),
                data.get('output_text'),
                data.get('extra_metrics'),
            ))

        return self.db.execute_many(sql, params_list)

    def find_by_run_id(self, run_id: int, limit: int = 1000) -> List[TestResult]:
        """based on运行 ID 查找所hasResult"""
        return self.find_by("run_id = ?", (run_id,), limit)

    def find_errors(self, run_id: int = None, limit: int = 100) -> List[TestResult]:
        """查找ErrorResult"""
        if run_id:
            return self.find_by("run_id = ? AND error IS NOT NULL", (run_id,), limit)
        return self.find_by("error IS NOT NULL", (), limit)

    def count_by_run(self, run_id: int) -> Dict[str, int]:
        """Statistics运行Result数量"""
        sql = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN error IS NULL THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as failed
            FROM test_results
            WHERE run_id = ?
        """
        row = self.db.fetch_one(sql, (run_id,))
        return row if row else {"total": 0, "success": 0, "failed": 0}

    def get_aggregate_metrics(self, run_id: int) -> Dict[str, Any]:
        """GetAggregate指标"""
        sql = """
            SELECT
                AVG(ttft) as avg_ttft,
                MIN(ttft) as min_ttft,
                MAX(ttft) as max_ttft,
                AVG(tps) as avg_tps,
                MIN(tps) as min_tps,
                MAX(tps) as max_tps,
                AVG(tpot) as avg_tpot,
                AVG(prefill_speed) as avg_prefill_speed,
                SUM(prefill_tokens) as total_prefill_tokens,
                SUM(decode_tokens) as total_decode_tokens,
                SUM(cache_hit_tokens) as total_cache_hit_tokens,
                COUNT(*) as total_requests
            FROM test_results
            WHERE run_id = ? AND error IS NULL
        """
        row = self.db.fetch_one(sql, (run_id,))
        return row if row else {}

    def get_percentiles(self, run_id: int, column: str = "ttft") -> Dict[str, float]:
        """GetPercentile"""
        # SQLite not直接支持 PERCENTILE，use子Query模拟
        sql = f"""
            SELECT
                AVG({column}) as avg,
                (
                    SELECT {column} FROM test_results
                    WHERE run_id = ? AND {column} IS NOT NULL
                    ORDER BY {column} LIMIT 1 OFFSET (
                        SELECT CAST(COUNT(*) * 0.5 AS INT) FROM test_results
                        WHERE run_id = ? AND {column} IS NOT NULL
                    )
                ) as p50,
                (
                    SELECT {column} FROM test_results
                    WHERE run_id = ? AND {column} IS NOT NULL
                    ORDER BY {column} LIMIT 1 OFFSET (
                        SELECT CAST(COUNT(*) * 0.95 AS INT) FROM test_results
                        WHERE run_id = ? AND {column} IS NOT NULL
                    )
                ) as p95,
                (
                    SELECT {column} FROM test_results
                    WHERE run_id = ? AND {column} IS NOT NULL
                    ORDER BY {column} LIMIT 1 OFFSET (
                        SELECT CAST(COUNT(*) * 0.99 AS INT) FROM test_results
                        WHERE run_id = ? AND {column} IS NOT NULL
                    )
                ) as p99
            FROM test_results
            WHERE run_id = ? AND {column} IS NOT NULL
        """
        row = self.db.fetch_one(sql, (run_id, run_id, run_id, run_id, run_id, run_id, run_id))
        return row if row else {}

    def delete_by_run(self, run_id: int) -> int:
        """Delete运行所hasResult"""
        return self.delete_by("run_id = ?", (run_id,))
