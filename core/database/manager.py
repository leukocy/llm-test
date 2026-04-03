"""
Database管理器

统一Database访问入口，封装所has Repository and Service。
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from core.database.connection import Database, db
from core.database.backup import DatabaseBackup
from core.repositories import (
    TestRunRepository,
    TestResultRepository,
    ApiLogRepository,
    ExecLogRepository,
    ReportRepository,
)
from core.repositories.test_run import TestRunRepository
from core.repositories.test_result import TestResultRepository
from core.repositories.api_log import ApiLogRepository
from core.repositories.exec_log import ExecLogRepository
from core.repositories.report import ReportRepository
from core.models import TestRun, TestResult, ApiLog, ExecLog, Report
from core.services.data_import import DataImportService
from core.services.data_export import DataExportService


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database管理器

    提供统一Database访问接口，including:
    - 所has Repository 访问
    - 常用操作便捷方法
    - Transaction支持
    """

    _instance = None

    def __new__(cls, db_path: str = "data/benchmark.db"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(db_path)
        return cls._instance

    def _init(self, db_path: str):
        """Initialize管理器"""
        self._db = Database(db_path)

        # Initialize Repository
        self._run_repo = TestRunRepository(self._db)
        self._result_repo = TestResultRepository(self._db)
        self._api_log_repo = ApiLogRepository(self._db)
        self._exec_log_repo = ExecLogRepository(self._db)
        self._report_repo = ReportRepository(self._db)

        # Initialize Service
        self._import_service = DataImportService(self._db)
        self._export_service = DataExportService(self._db)
        self._backup = DatabaseBackup(db_path)

    @property
    def db(self) -> Database:
        """DatabaseConnect"""
        return self._db

    @property
    def runs(self) -> TestRunRepository:
        """Test运行 Repository"""
        return self._run_repo

    @property
    def results(self) -> TestResultRepository:
        """Test Results Repository"""
        return self._result_repo

    @property
    def api_logs(self) -> ApiLogRepository:
        """API Log Repository"""
        return self._api_log_repo

    @property
    def exec_logs(self) -> ExecLogRepository:
        """执行Log Repository"""
        return self._exec_log_repo

    @property
    def reports(self) -> ReportRepository:
        """报告 Repository"""
        return self._report_repo

    @property
    def importer(self) -> DataImportService:
        """Data import服务"""
        return self._import_service

    @property
    def exporter(self) -> DataExportService:
        """Data export服务"""
        return self._export_service

    @property
    def backup(self) -> DatabaseBackup:
        """Backup管理器"""
        return self._backup

    # ============================================
    # 便捷方法：Test运行
    # ============================================

    def start_test_run(
        self,
        test_type: str,
        model_id: str,
        provider: str = None,
        config: Dict = None,
        system_info: Dict = None,
    ) -> TestRun:
        """
        开始新Test运行

        Args:
            test_type: Test Type
            model_id: Model ID
            provider: Provider
            config: Test Configuration
            system_info: 系统信息

        Returns:
            TestRun 实例
        """
        run = TestRun.create(
            test_type=test_type,
            model_id=model_id,
            provider=provider,
        )
        if config:
            run.config = config
        if system_info:
            run.system_info = system_info

        run_id = self._run_repo.insert(run)
        run.id = run_id

        logger.info(f"Start Test运行: ID={run_id}, type={test_type}, model={model_id}")
        return run

    def save_result(self, run: TestRun, result_data: Dict[str, Any]) -> TestResult:
        """
        Save单Test Results

        Args:
            run: Test运行实例
            result_data: ResultData字典

        Returns:
            TestResult 实例
        """
        result = TestResult.from_api_result(run.id, result_data)
        result.id = self._result_repo.insert(result)
        return result

    def save_results_batch(self, run: TestRun, results_data: List[Dict]) -> int:
        """
        批量SaveTest Results

        Args:
            run: Test运行实例
            results_data: ResultData列表

        Returns:
            Insert记录数
        """
        results = [TestResult.from_api_result(run.id, d) for d in results_data]
        return self._result_repo.insert_batch(results)

    def complete_test_run(
        self,
        run: TestRun,
        success: bool = True,
        calculate_stats: bool = True
    ) -> bool:
        """
        完成Test运行

        Args:
            run: Test运行实例
            success: is否succeeded
            calculate_stats: is否CalculateStatistics信息

        Returns:
            is否succeeded
        """
        run.complete(success)

        if calculate_stats and run.id:
            # CalculateStatistics信息
            stats = self._result_repo.get_aggregate_metrics(run.id)
            percentiles = self._result_repo.get_percentiles(run.id, "ttft")

            stats.update({
                "p50_ttft": percentiles.get("p50"),
                "p95_ttft": percentiles.get("p95"),
                "p99_ttft": percentiles.get("p99"),
            })

            # 只保留 test_runs 表in实际存in列，避免 UPDATE 时报错
            valid_columns = {
                'avg_ttft', 'avg_tps', 'avg_tpot',
                'p50_ttft', 'p95_ttft', 'p99_ttft',
                'total_tokens', 'total_requests',
                'completed_requests', 'failed_requests', 'success_rate',
                'duration_seconds',
            }
            filtered_stats = {k: v for k, v in stats.items() if k in valid_columns}

            return self._run_repo.complete(run.id, success, filtered_stats)

        return self._run_repo.complete(run.id, success)

    def update_run_progress(
        self,
        run: TestRun,
        completed: int,
        total: int,
        failed: int = 0
    ):
        """UpdateTest进度"""
        run.update_progress(completed, total, failed)
        self._run_repo.update_progress(run.id, completed, total, failed)

    # ============================================
    # 便捷方法：Log
    # ============================================

    def log_api_request(
        self,
        session_id: str,
        test_type: str,
        provider: str,
        model_id: str,
        request: Dict,
        run_id: int = None
    ) -> ApiLog:
        """记录 API 请求"""
        log = ApiLog.create(
            session_id=session_id,
            test_type=test_type,
            provider=provider,
            model_id=model_id,
            request=request,
            run_id=run_id,
        )
        log.id = self._api_log_repo.insert(log)
        return log

    def log_execution(
        self,
        message: str,
        level: str = "INFO",
        run_id: int = None,
        session_id: str = None,
        metrics: Dict = None
    ) -> ExecLog:
        """记录执行Log"""
        log = ExecLog.create(
            message=message,
            level=level,
            run_id=run_id,
            session_id=session_id,
            metrics=metrics,
        )
        log.id = self._exec_log_repo.insert(log)
        return log

    # ============================================
    # 便捷方法：报告
    # ============================================

    def create_report(
        self,
        model_id: str,
        report_type: str = "standard",
        run_id: int = None,
        **kwargs
    ) -> Report:
        """Create报告"""
        report = Report.create(
            model_id=model_id,
            report_type=report_type,
            run_id=run_id,
            **kwargs
        )
        report.id = self._report_repo.insert(report)
        return report

    # ============================================
    # 便捷方法：Query
    # ============================================

    def get_recent_runs(self, limit: int = 20) -> List[TestRun]:
        """Get最近Test运行"""
        return self._run_repo.find_recent(limit)

    def get_runs_by_model(self, model_id: str, limit: int = 50) -> List[TestRun]:
        """Get指定ModelTest运行"""
        return self._run_repo.find_by_model(model_id, limit)

    def get_run_with_results(self, run_id: int) -> Optional[Dict]:
        """GetTest运行and其Result"""
        run = self._run_repo.find_by_id(run_id)
        if not run:
            return None

        results = self._result_repo.find_by_run_id(run_id)
        stats = self._result_repo.get_aggregate_metrics(run_id)

        return {
            "run": run,
            "results": results,
            "stats": stats,
        }

    def search_runs(self, query: str, limit: int = 50) -> List[TestRun]:
        """搜索Test运行"""
        return self._run_repo.search(query, limit)

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get仪表盘StatisticsData"""
        total_runs = self._run_repo.count()
        recent_runs = self._run_repo.find_recent(5)

        # 按ModelStatistics
        sql = """
            SELECT model_id, COUNT(*) as count
            FROM test_runs
            GROUP BY model_id
            ORDER BY count DESC
            LIMIT 10
        """
        model_stats = self._db.fetch_all(sql)

        # 按类型Statistics
        sql = """
            SELECT test_type, COUNT(*) as count
            FROM test_runs
            GROUP BY test_type
            ORDER BY count DESC
        """
        type_stats = self._db.fetch_all(sql)

        return {
            "total_runs": total_runs,
            "recent_runs": recent_runs,
            "by_model": model_stats,
            "by_type": type_stats,
            "db_size_bytes": self._db.get_database_size(),
        }

    # ============================================
    # 便捷方法：ImportExport
    # ============================================

    def import_csv(self, csv_path: str, model_id: str = None, test_type: str = None):
        """Import CSV 文件"""
        return self._import_service.import_csv_file(csv_path, model_id, test_type)

    def export_run_json(self, run_id: int, output_path: str = None):
        """Export运行到 JSON"""
        return self._export_service.export_run_to_json(run_id, output_path)

    def export_run_csv(self, run_id: int, output_path: str = None):
        """Export运行到 CSV"""
        return self._export_service.export_run_to_csv(run_id, output_path)

    def export_run_excel(self, run_id: int, output_path: str = None):
        """Export运行到 Excel"""
        return self._export_service.export_run_to_excel(run_id, output_path)

    # ============================================
    # 便捷方法：Backup
    # ============================================

    def create_backup(self, reason: str = "manual"):
        """CreateBackup"""
        return self._backup.create_backup(reason)

    def list_backups(self):
        """列出Backup"""
        return self._backup.list_backups()

    def restore_backup(self, backup_path):
        """RestoreBackup"""
        return self._backup.restore_backup(backup_path)

    # ============================================
    # Cleanup
    # ============================================

    def cleanup_old_data(self, days: int = 30):
        """Cleanup旧Data"""
        api_deleted = self._api_log_repo.cleanup_old_logs(days)
        exec_deleted = self._exec_log_repo.cleanup_old_logs(days)
        reports_deleted = self._report_repo.delete_old_reports(days)

        logger.info(f"Cleanup完成: api_logs={api_deleted}, exec_logs={exec_deleted}, reports={reports_deleted}")
        return {
            "api_logs": api_deleted,
            "exec_logs": exec_deleted,
            "reports": reports_deleted,
        }

    def close(self):
        """CloseDatabaseConnect"""
        self._db.close()


# 全局实例
db_manager = DatabaseManager()
