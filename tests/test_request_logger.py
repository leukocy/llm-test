"""
Request Logs记录器单元Test
"""

import json
import os
import tempfile
import shutil
import pytest
from datetime import datetime

from core.request_logger import (
    RequestLogger,
    RequestLogEntry,
    LogStats,
    LogIndexEntry,
    LogIndex,
    TestType,
    LogStatus,
    init_request_logger,
    get_request_logger,
)


class TestEnums:
    """Test枚举类"""

    def test_test_type_values(self):
        """Test TestType 枚举值"""
        assert TestType.CONCURRENCY.value == "concurrency"
        assert TestType.PREFILL.value == "prefill"
        assert TestType.LONG_CONTEXT.value == "long_context"

    def test_log_status_values(self):
        """Test LogStatus 枚举值"""
        assert LogStatus.SUCCESS.value == "success"
        assert LogStatus.ERROR.value == "error"


class TestRequestLogEntry:
    """Test RequestLogEntry Data类"""

    def test_create_entry(self):
        """TestCreateLog条目"""
        entry = RequestLogEntry(
            log_id="1234567890_1",
            created_at="2026-02-12T14:30:25.123456",
            session_id="1",
            test_type="concurrency",
            status="success",
            config={"provider": "OpenAIProvider", "model": "gpt-4"},
            request={"url": "https://api.openai.com/v1/chat/completions"},
            response={"full_content": "Hello"},
            metrics={"ttft": 1.5},
            error=None,
        )
        assert entry.log_id == "1234567890_1"
        assert entry.test_type == "concurrency"
        assert entry.status == "success"
        assert entry.error is None


class TestLogIndexEntry:
    """Test LogIndexEntry Data类"""

    def test_create_index_entry(self):
        """TestCreateIndex条目"""
        entry = LogIndexEntry(
            log_id="1234567890_1",
            filepath="2026-02-12/143025_123_success_concurrency_OpenAI_gpt-4_1.json",
            created_at="2026-02-12T14:30:25.123456",
            date="2026-02-12",
            test_type="concurrency",
            status="success",
            session_id="1",
            provider="OpenAIProvider",
            model="gpt-4",
            ttft=1.5,
            total_time=2.5,
            error=None,
        )
        assert entry.log_id == "1234567890_1"
        assert entry.date == "2026-02-12"
        assert entry.test_type == "concurrency"


class TestLogIndex:
    """Test LogIndex 类"""

    def test_index_creation(self):
        """TestIndexCreate"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LogIndex(tmpdir)
            assert index._entries == {}

    def test_add_and_query(self):
        """TestAddandQuery"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LogIndex(tmpdir)

            entry = LogIndexEntry(
                log_id="1",
                filepath="2026-02-12/file1.json",
                created_at="2026-02-12T10:00:00",
                date="2026-02-12",
                test_type="concurrency",
                status="success",
                session_id="1",
                provider="OpenAI",
                model="gpt-4",
                ttft=1.0,
                total_time=2.0,
                error=None,
            )
            index.add(entry)

            results = index.query(test_type="concurrency")
            assert len(results) == 1
            assert results[0].log_id == "1"

    def test_query_with_filters(self):
        """Test带Filter条件Query"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LogIndex(tmpdir)

            # Add多条目
            for i in range(5):
                entry = LogIndexEntry(
                    log_id=str(i),
                    filepath=f"2026-02-12/file{i}.json",
                    created_at=f"2026-02-12T10:00:0{i}",
                    date="2026-02-12",
                    test_type="concurrency" if i < 3 else "prefill",
                    status="success" if i % 2 == 0 else "error",
                    session_id=str(i),
                    provider="OpenAI",
                    model="gpt-4",
                    ttft=1.0,
                    total_time=2.0,
                    error=None if i % 2 == 0 else "test error",
                )
                index.add(entry)

            # Test按Test TypeFilter
            results = index.query(test_type="concurrency")
            assert len(results) == 3

            # Test按StatusFilter
            results = index.query(status="error")
            assert len(results) == 2

            # Test组合Filter
            results = index.query(test_type="concurrency", status="success")
            assert len(results) == 2

    def test_get_stats(self):
        """TestGetStatistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = LogIndex(tmpdir)

            for i in range(3):
                entry = LogIndexEntry(
                    log_id=str(i),
                    filepath=f"2026-02-12/file{i}.json",
                    created_at=f"2026-02-12T10:00:0{i}",
                    date="2026-02-12",
                    test_type="concurrency",
                    status="success" if i < 2 else "error",
                    session_id=str(i),
                    provider="OpenAI",
                    model="gpt-4",
                    ttft=None,
                    total_time=None,
                    error=None,
                )
                index.add(entry)

            stats = index.get_stats()
            assert stats["total"] == 3
            assert stats["by_status"]["success"] == 2
            assert stats["by_status"]["error"] == 1


class TestRequestLogger:
    """Test RequestLogger 类"""

    def test_initialization(self):
        """TestInitialize"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, enabled=True)
            assert logger.enabled is True
            assert logger.log_dir == tmpdir
            assert os.path.exists(tmpdir)

    def test_disabled_logger(self):
        """Test禁用Status"""
        logger = RequestLogger(log_dir="nonexistent", enabled=False)
        assert logger.enabled is False
        result = logger.log_request(
            session_id="1",
            provider="Test",
            model_id="test",
            platform="test",
            api_base_url="https://test.com",
            headers={},
            payload={},
        )
        assert result is None

    def test_date_directory_structure(self):
        """Test按日期分目录存储"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, enabled=True)

            created_at = datetime(2026, 2, 12, 14, 30, 25).timestamp()
            filepath = logger.log_request(
                session_id="1",
                provider="OpenAI",
                model_id="gpt-4",
                platform="openai",
                api_base_url="https://api.openai.com/v1",
                headers={},
                payload={},
                test_type="concurrency",
                created_at=created_at,
            )

            assert filepath is not None
            # Check路径包含日期目录
            assert "2026-02-12" in filepath

    def test_filename_with_status_and_test_type(self):
        """TestFilename包含StatusandTest Type"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, enabled=True)

            # Test成功请求
            filepath = logger.log_request(
                session_id="1",
                provider="OpenAI",
                model_id="gpt-4",
                platform="openai",
                api_base_url="https://api.openai.com/v1",
                headers={},
                payload={},
                test_type="concurrency",
            )
            assert "success" in os.path.basename(filepath)
            assert "concurrency" in os.path.basename(filepath)

            # TestError请求
            filepath = logger.log_request(
                session_id="2",
                provider="OpenAI",
                model_id="gpt-4",
                platform="openai",
                api_base_url="https://api.openai.com/v1",
                headers={},
                payload={},
                test_type="prefill",
                error="Test error",
            )
            assert "error" in os.path.basename(filepath)
            assert "prefill" in os.path.basename(filepath)

    def test_log_content_includes_test_type_and_status(self):
        """TestLog Content包含Test TypeandStatus"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, enabled=True)

            filepath = logger.log_request(
                session_id="1",
                provider="OpenAI",
                model_id="gpt-4",
                platform="openai",
                api_base_url="https://api.openai.com/v1",
                headers={},
                payload={},
                test_type="long_context",
            )

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["test_type"] == "long_context"
            assert data["status"] == "success"

    def test_index_created(self):
        """TestIndex文件Create"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, enabled=True)

            logger.log_request(
                session_id="1",
                provider="OpenAI",
                model_id="gpt-4",
                platform="openai",
                api_base_url="https://api.openai.com/v1",
                headers={},
                payload={},
                test_type="concurrency",
            )

            index_file = os.path.join(tmpdir, "_index.json")
            assert os.path.exists(index_file)

            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            assert len(index_data["entries"]) == 1

    def test_query_logs(self):
        """TestLogQuery"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, enabled=True)

            # Create多Log
            for i in range(5):
                logger.log_request(
                    session_id=str(i),
                    provider="OpenAI",
                    model_id="gpt-4",
                    platform="openai",
                    api_base_url="https://api.openai.com/v1",
                    headers={},
                    payload={},
                    test_type="concurrency" if i < 3 else "prefill",
                    full_response_content=f"Response {i}",
                )

            # Query concurrency 类型
            results = logger.query(test_type="concurrency")
            assert len(results) == 3

            # Query prefill 类型
            results = logger.query(test_type="prefill")
            assert len(results) == 2

    def test_get_stats_with_breakdown(self):
        """Test带分类StatisticsStatistics信息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, enabled=True)

            # Create多Log
            for i in range(5):
                logger.log_request(
                    session_id=str(i),
                    provider="OpenAI",
                    model_id="gpt-4",
                    platform="openai",
                    api_base_url="https://api.openai.com/v1",
                    headers={},
                    payload={},
                    test_type="concurrency" if i < 3 else "prefill",
                    error=None if i % 2 == 0 else "error",
                )

            stats = logger.get_stats()
            assert stats.total_files == 5
            # Check分类Statistics
            assert stats.by_status.get("success", 0) == 3
            assert stats.by_status.get("error", 0) == 2
            assert stats.by_test_type.get("concurrency", 0) == 3
            assert stats.by_test_type.get("prefill", 0) == 2

    def test_rebuild_index(self):
        """Test重建Index"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, enabled=True)

            # CreateLog
            for i in range(3):
                logger.log_request(
                    session_id=str(i),
                    provider="OpenAI",
                    model_id="gpt-4",
                    platform="openai",
                    api_base_url="https://api.openai.com/v1",
                    headers={},
                    payload={},
                    test_type="concurrency",
                )

            # DeleteIndex文件
            index_file = os.path.join(tmpdir, "_index.json")
            os.remove(index_file)

            # 重建Index
            logger._index = LogIndex(tmpdir)
            count = logger.rebuild_index()
            assert count == 3

    def test_mask_api_key(self):
        """Test API 密钥隐藏"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, mask_api_key=True)
            headers = {
                "Authorization": "Bearer sk-1234567890",
                "api-key": "my-secret-key",
                "Content-Type": "application/json",
            }
            masked = logger._mask_headers(headers)
            assert masked["Authorization"] == "*****"
            assert masked["api-key"] == "*****"
            assert masked["Content-Type"] == "application/json"

    def test_size_limit_cleanup_with_date_dirs(self):
        """Test带日期目录大小限制Cleanup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set非常小限制 (10KB)
            logger = RequestLogger(
                log_dir=tmpdir, enabled=True, max_total_size_mb=10 / 1024
            )

            # Create多较大文件（分布innot同日期目录）
            for i in range(20):
                date_offset = i // 5  # 每5文件一日期
                created_at = datetime(2026, 2, 12 + date_offset, 14, 30, i).timestamp()
                logger.log_request(
                    session_id=str(i),
                    provider="Test",
                    model_id="model",
                    platform="test",
                    api_base_url="https://test.com",
                    headers={},
                    payload={},
                    test_type="concurrency",
                    full_response_content="x" * 2000,
                    created_at=created_at,
                )

            stats = logger.get_stats()
            # due to限制is 10KB，大部分旧文件应该被Delete
            assert stats.total_files < 20

    def test_concurrent_writes(self):
        """Test并发写入安全性"""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RequestLogger(log_dir=tmpdir, enabled=True)
            errors = []

            def log_request(i):
                try:
                    filepath = logger.log_request(
                        session_id=str(i),
                        provider="OpenAI",
                        model_id="gpt-4",
                        platform="openai",
                        api_base_url="https://api.openai.com/v1",
                        headers={},
                        payload={},
                        test_type="concurrency",
                        full_response_content=f"Response {i}",
                    )
                    assert filepath is not None
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=log_request, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            stats = logger.get_stats()
            assert stats.total_files == 10


class TestGlobalFunctions:
    """Test全局函数"""

    def test_init_and_get_logger(self):
        """TestInitializeandGet全局实例"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger1 = init_request_logger(log_dir=tmpdir, enabled=True)
            logger2 = get_request_logger()
            assert logger1 is logger2

    def test_get_logger_before_init(self):
        """Test未Initialize时Get实例"""
        # Reset全局实例
        import core.request_logger as module

        module._request_logger = None

        logger = get_request_logger()
        assert logger is None
