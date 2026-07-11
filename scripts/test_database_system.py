"""
Database系统Test脚本

Validate新Database系统各组件is否正常工作。
"""

import os
import sys
from pathlib import Path

# Set控制台编码
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add items目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_database_connection():
    """test data库Connect"""
    print("\n" + "=" * 50)
    print("Test 1: DatabaseConnect")
    print("=" * 50)

    from core.database import SCHEMA_VERSION, db

    print(f"Database路径: {db.path}")
    print(f"Schema Version: {SCHEMA_VERSION}")
    print(f"Database大小: {db.get_database_size()} bytes")

    # Validate表存in
    tables = [
        "test_runs",
        "test_results",
        "api_logs",
        "execution_logs",
        "reports",
        "db_meta",
    ]
    for table in tables:
        exists = db.table_exists(table)
        print(f"  表 {table}: {'[OK]' if exists else '[MISSING]'}")

    print("DatabaseConnectTest: via")
    return True


def test_test_run_crud():
    """Test TestRun CRUD 操作"""
    print("\n" + "=" * 50)
    print("Test 2: TestRun CRUD")
    print("=" * 50)

    from core.models import TestRun
    from core.repositories import TestRunRepository

    repo = TestRunRepository()

    # Create
    run = TestRun.create(
        test_type="concurrency",
        model_id="test-model",
        provider="test-provider",
        concurrency=4,
        max_tokens=512,
    )
    run.tags = ["test", "benchmark"]

    run_id = repo.insert(run)
    print(f"CreateTest运行: ID={run_id}")

    # 读取
    found = repo.find_by_id(run_id)
    assert found is not None
    assert found.model_id == "test-model"
    print(f"读取Test运行: model_id={found.model_id}, tags={found.tags}")

    # Update
    repo.update_status(run_id, "completed", 100.0)
    repo.update_progress(run_id, 100, 100, 0)
    print("UpdateTest运行: status=completed, progress=100%")

    # Query
    runs = repo.find_by_model("test-model")
    assert len(runs) > 0
    print(f"按ModelQuery: 找到 {len(runs)} 条记录")

    # Delete
    deleted = repo.delete_by_id(run_id)
    assert deleted
    print(f"DeleteTest运行: {'成功' if deleted else '失败'}")

    print("TestRun CRUD Test: via")
    return True


def test_test_result_crud():
    """Test TestResult CRUD 操作"""
    print("\n" + "=" * 50)
    print("Test 3: TestResult CRUD")
    print("=" * 50)

    from core.models import TestResult, TestRun
    from core.repositories import TestResultRepository, TestRunRepository

    run_repo = TestRunRepository()
    result_repo = TestResultRepository()

    # 先CreateTest运行
    run = TestRun.create(test_type="prefill", model_id="result-test-model")
    run_id = run_repo.insert(run)

    # CreateTestResult
    result = TestResult(
        run_id=run_id,
        session_id=1,
        round=1,
        ttft=0.5,
        tps=50.0,
        prefill_tokens=100,
        decode_tokens=200,
    )

    result_id = result_repo.insert(result)
    print(f"CreateTestResult: ID={result_id}")

    # 批量Insert
    results = []
    for i in range(10):
        r = TestResult(
            run_id=run_id,
            session_id=i + 2,
            request_index=i,
            ttft=0.3 + i * 0.1,
            tps=45.0 + i * 2,
        )
        results.append(r)

    inserted = result_repo.insert_batch(results)
    print(f"批量Insert: {inserted} 条记录")

    # Query
    found_results = result_repo.find_by_run_id(run_id)
    print(f"按运行IDQuery: 找到 {len(found_results)} 条记录")

    # Statistics
    stats = result_repo.count_by_run(run_id)
    print(
        f"Statistics: total={stats['total']}, success={stats['success']}, failed={stats['failed']}"
    )

    # Aggregate指标
    metrics = result_repo.get_aggregate_metrics(run_id)
    print(
        f"Aggregate指标: avg_ttft={metrics.get('avg_ttft'):.3f}, avg_tps={metrics.get('avg_tps'):.2f}"
    )

    # Cleanup
    result_repo.delete_by_run(run_id)
    run_repo.delete_by_id(run_id)

    print("TestResult CRUD Test: via")
    return True


def test_api_log():
    """Test API Log"""
    print("\n" + "=" * 50)
    print("Test 4: ApiLog")
    print("=" * 50)

    from core.models import ApiLog
    from core.repositories import ApiLogRepository

    repo = ApiLogRepository()

    # CreateLog
    log = ApiLog.create(
        session_id="test-123",
        test_type="concurrency",
        provider="openai",
        model_id="gpt-4",
        request={"prompt": "Hello", "max_tokens": 100},
    )
    log.mark_success({"text": "Hello!"}, ttft=0.3, total_time=1.5)

    log_id = repo.insert(log)
    print(f"Create API Log: ID={log_id}")

    # Query
    found = repo.find_by_id(log_id)
    print(f"读取Log: status={found.status}, ttft={found.ttft}")

    # Statistics
    stats = repo.get_statistics()
    print(f"Statistics: total={stats.get('total')}")

    # Cleanup
    repo.delete_by_id(log_id)

    print("ApiLog Test: via")
    return True


def test_exec_log():
    """Test执行Log"""
    print("\n" + "=" * 50)
    print("Test 5: ExecLog")
    print("=" * 50)

    from core.models import ExecLog
    from core.repositories import ExecLogRepository

    repo = ExecLogRepository()

    # Create各种级别Log
    logs = [
        ExecLog.info("Test开始", session_id="1"),
        ExecLog.success("请求完成", session_id="1", metrics={"ttft": 0.5}),
        ExecLog.warning("响应较慢", session_id="2"),
        ExecLog.error("请求失败", error="Timeout"),
    ]

    for log in logs:
        repo.insert(log)

    print(f"Create {len(logs)} 条执行Log")

    # Query
    errors = repo.find_errors(limit=10)
    print(f"Error Logs: {len(errors)} 条")

    # 级别Statistics
    counts = repo.get_level_counts()
    print(f"级别Statistics: {counts}")

    # Cleanup
    for log in logs:
        if log.id:
            repo.delete_by_id(log.id)

    print("ExecLog Test: via")
    return True


def test_report():
    """Test报告"""
    print("\n" + "=" * 50)
    print("Test 6: Report")
    print("=" * 50)

    from core.models import Report
    from core.repositories import ReportRepository

    repo = ReportRepository()

    # Create报告
    report = Report.create(
        model_id="test-model",
        report_type="standard",
    )
    report.model_info = {"provider": "openai", "version": "1.0"}
    report.results = {"accuracy": 0.95}
    report.aggregate = {"avg_ttft": 0.5, "avg_tps": 50}

    report_id = repo.insert(report)
    print(f"Create报告: ID={report_id}")

    # 读取
    found = repo.find_by_id(report_id)
    print(f"读取报告: type={found.report_type}, model_id={found.model_id}")

    # Export as JSON
    json_dict = found.to_json_dict()
    print(f"JSON Export: keys={list(json_dict.keys())}")

    # Cleanup
    repo.delete_by_id(report_id)

    print("Report Test: via")
    return True


def test_backup():
    """TestBackup功能"""
    print("\n" + "=" * 50)
    print("Test 7: DatabaseBackup")
    print("=" * 50)

    from core.database import DatabaseBackup

    backup = DatabaseBackup()

    # CreateBackup
    backup_path = backup.create_backup("test")
    if backup_path:
        print(f"CreateBackup: {backup_path}")

        # 列出Backup
        backups = backup.list_backups()
        print(f"Backup列表: {len(backups)} ")

        # Backup摘要
        summary = backup.get_backup_summary()
        print(f"Backup摘要: {summary}")

        # CleanupTestBackup
        if backup_path.exists():
            backup_path.unlink()
            print("CleanupTestBackup: 完成")

    print("BackupTest: via")
    return True


def test_data_export():
    """test dataExport"""
    print("\n" + "=" * 50)
    print("Test 8: Data export")
    print("=" * 50)

    from core.models import TestResult, TestRun
    from core.repositories import TestResultRepository, TestRunRepository
    from core.services import DataExportService

    # Createtest data
    run_repo = TestRunRepository()
    result_repo = TestResultRepository()

    run = TestRun.create(test_type="export-test", model_id="export-model")
    run_id = run_repo.insert(run)

    results = [
        TestResult(run_id=run_id, session_id=i, ttft=0.1 * i, tps=50 + i)
        for i in range(5)
    ]
    result_repo.insert_batch(results)

    # Export服务
    export_service = DataExportService()

    # Export到 JSON
    json_path = export_service.export_run_to_json(run_id)
    if json_path:
        print(f"JSON Export: {json_path}")
        os.unlink(json_path)

    # Export到 CSV
    csv_path = export_service.export_run_to_csv(run_id)
    if csv_path:
        print(f"CSV Export: {csv_path}")
        os.unlink(csv_path)

    # Cleanup
    result_repo.delete_by_run(run_id)
    run_repo.delete_by_id(run_id)

    print("Data exportTest: via")
    return True


def run_all_tests():
    """运行所hasTest"""
    print("\n" + "=" * 60)
    print("   Database系统Test套件")
    print("=" * 60)

    tests = [
        ("DatabaseConnect", test_database_connection),
        ("TestRun CRUD", test_test_run_crud),
        ("TestResult CRUD", test_test_result_crud),
        ("ApiLog", test_api_log),
        ("ExecLog", test_exec_log),
        ("Report", test_report),
        ("DatabaseBackup", test_backup),
        ("Data export", test_data_export),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\nTest失败: {name}")
            print(f"   Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"   TestResult: {passed} via, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
