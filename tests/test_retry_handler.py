"""
单元Test: core/retry_handler.py

Test重试Process器功能，包括:
- 指数退避Latency calculation
- 可重试Error判断
- Async/Sync重试执行
- 重试Decorator
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from core.retry_handler import (
    RetryConfig,
    RetryHandler,
    RetryResult,
    retry_async,
    retry_sync,
)


class TestRetryConfig:
    """Test重试Configure"""

    def test_default_config(self):
        """defaultConfigure"""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert 429 in config.retryable_status_codes

    def test_custom_config(self):
        """Custom Configuration"""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0


class TestRetryHandler:
    """Test重试Process器"""

    def test_calculate_delay_exponential_backoff(self):
        """Test指数退避"""
        handler = RetryHandler(RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=100.0,
            jitter=False
        ))

        # 0次: 1 * 2^0 = 1
        assert handler._calculate_delay(0) == 1.0
        # 1次: 1 * 2^1 = 2
        assert handler._calculate_delay(1) == 2.0
        # 2次: 1 * 2^2 = 4
        assert handler._calculate_delay(2) == 4.0

    def test_calculate_delay_with_max(self):
        """Test最大延迟限制"""
        handler = RetryHandler(RetryConfig(
            base_delay=10.0,
            exponential_base=3.0,
            max_delay=20.0,
            jitter=False
        ))

        # 10 * 3^2 = 90 > 20，应该被限制is 20
        assert handler._calculate_delay(2) == 20.0

    def test_calculate_delay_with_retry_after(self):
        """Testuse Retry-After 值"""
        handler = RetryHandler(RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False
        ))

        # retry_after=10 > base_delay=1，应该use 10
        delay = handler._calculate_delay(0, retry_after=10.0)
        assert delay == 10.0

    def test_calculate_delay_with_jitter(self):
        """Test随机抖动"""
        handler = RetryHandler(RetryConfig(
            base_delay=1.0,
            jitter=True,
            jitter_range=(0.5, 1.5)
        ))

        delay = handler._calculate_delay(0)
        # 应该in 0.5 到 1.5 之间
        assert 0.5 <= delay <= 1.5

    def test_is_retryable_error_connection_error(self):
        """TestNetwork error可重试"""
        handler = RetryHandler()

        is_retryable, status_code, retry_after = handler._is_retryable_error(ConnectionError())
        assert is_retryable is True
        assert status_code is None

    def test_is_retryable_error_timeout(self):
        """Test超时可重试"""
        handler = RetryHandler()

        is_retryable, status_code, retry_after = handler._is_retryable_error(TimeoutError())
        assert is_retryable is True

    def test_is_retryable_error_status_code(self):
        """Test特定Status码可重试"""
        handler = RetryHandler()

        # Create一真实异常类
        class StatusCodeError(Exception):
            def __init__(self):
                self.status_code = 429
                super().__init__("Rate limit")

        error = StatusCodeError()

        is_retryable, status_code, retry_after = handler._is_retryable_error(error)
        assert is_retryable is True
        assert status_code == 429

    def test_is_retryable_error_non_retryable(self):
        """Testnot可重试Error"""
        handler = RetryHandler()

        error = Mock(status_code=400)

        is_retryable, status_code, retry_after = handler._is_retryable_error(error)
        assert is_retryable is False

    def test_is_retryable_error_with_message(self):
        """Test基于Error消息判断"""
        handler = RetryHandler()

        error = Exception("Rate limit exceeded")

        is_retryable, status_code, retry_after = handler._is_retryable_error(error)
        assert is_retryable is True
        assert status_code == 429

    def test_is_retryable_error_with_retry_after_header(self):
        """Test提取 Retry-After 头"""
        handler = RetryHandler()

        # Create一真实类来模拟响应
        class MockResponse:
            def __init__(self):
                self.headers = {"Retry-After": "5"}

        class MockError:
            def __init__(self):
                self.status_code = 429
                self.response = MockResponse()

        error = MockError()

        is_retryable, status_code, retry_after = handler._is_retryable_error(error)
        assert is_retryable is True
        assert retry_after == 5.0

    def test_is_retryable_error_with_http_date_retry_after(self):
        """Test提取 HTTP Date 格式 Retry-After 头"""
        handler = RetryHandler()

        # Mock response with HTTP Date
        # Use a future date to ensure positive delay
        from datetime import datetime, timedelta, timezone
        future_date = datetime.now(timezone.utc) + timedelta(seconds=100)
        http_date = future_date.strftime("%a, %d %b %Y %H:%M:%S GMT")

        class MockResponse:
            def __init__(self):
                self.headers = {"Retry-After": http_date}

        class MockError:
            def __init__(self):
                self.status_code = 429
                self.response = MockResponse()

        error = MockError()

        is_retryable, status_code, retry_after = handler._is_retryable_error(error)
        assert is_retryable is True
        # Allow some margin for execution time
        assert 90.0 <= retry_after <= 110.0

    def test_is_retryable_error_with_retry_delay_in_message(self):
        """Test从Error消息in提取 retryDelay"""
        handler = RetryHandler()

        # Create一带 status_code 异常，这样代码will进入 retry_after 提取分支
        class RetryDelayError(Exception):
            def __init__(self):
                self.status_code = 429
                super().__init__('Too many requests, retryDelay:10')

        error = RetryDelayError()

        is_retryable, status_code, retry_after = handler._is_retryable_error(error)
        assert is_retryable is True
        assert retry_after == 10.0


class TestRetryHandlerAsync:
    """TestAsync重试执行"""

    @pytest.mark.asyncio
    async def test_execute_async_success_on_first_try(self):
        """首次尝试成功"""
        handler = RetryHandler()

        async def success_func():
            return "success"

        result = await handler.execute_async(success_func)
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_execute_async_retry_then_success(self):
        """重试后成功"""
        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))

        attempts = []

        async def fail_then_succeed():
            attempts.append(1)
            if len(attempts) < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await handler.execute_async(fail_then_succeed)
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_execute_async_max_retries_exceeded(self):
        """超过最大Retry count"""
        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))

        async def always_fail():
            raise ConnectionError("Always fails")

        result = await handler.execute_async(always_fail)
        assert result.success is False
        assert result.attempts == 3  # 首次 + 2 retry
        assert isinstance(result.error, ConnectionError)

    @pytest.mark.asyncio
    async def test_execute_async_non_retryable_error(self):
        """not可重试Error立i.e.失败"""
        handler = RetryHandler()

        async def raise_400():
            error = Mock(status_code=400)
            raise error

        result = await handler.execute_async(raise_400)
        assert result.success is False
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_execute_async_with_retry_after(self):
        """Testuse Retry-After 延迟"""
        handler = RetryHandler(RetryConfig(max_retries=1, base_delay=0.01))

        # Create真实异常类
        class MockResponse:
            def __init__(self):
                self.headers = {"Retry-After": "0.01"}

        class RateLimitError(Exception):
            def __init__(self):
                self.status_code = 429
                self.response = MockResponse()
                super().__init__("Rate limited")

        async def fail_with_retry_after():
            raise RateLimitError()

        start = time.time()
        result = await handler.execute_async(fail_with_retry_after)
        elapsed = time.time() - start

        assert result.success is False
        # 应该至少etc.待 Retry-After 时间
        assert elapsed >= 0.01


class TestRetryHandlerSync:
    """TestSync重试执行"""

    def test_execute_sync_success_on_first_try(self):
        """首次尝试成功"""
        handler = RetryHandler()

        def success_func():
            return "success"

        result = handler.execute_sync(success_func)
        assert result.success is True
        assert result.result == "success"

    def test_execute_sync_retry_then_success(self):
        """重试后成功"""
        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))

        attempts = []

        def fail_then_succeed():
            attempts.append(1)
            if len(attempts) < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = handler.execute_sync(fail_then_succeed)
        assert result.success is True
        assert result.attempts == 2

    def test_execute_sync_max_retries_exceeded(self):
        """超过最大Retry count"""
        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))

        def always_fail():
            raise ConnectionError("Always fails")

        result = handler.execute_sync(always_fail)
        assert result.success is False
        assert result.attempts == 3


class TestRetryDecorator:
    """Test重试Decorator"""

    @pytest.mark.asyncio
    async def test_async_decorator_success(self):
        """AsyncDecorator成功"""
        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))

        @handler.retry_decorator
        async def my_func():
            return "decorated"

        result = await my_func()
        assert result == "decorated"

    @pytest.mark.asyncio
    async def test_async_decorator_retry(self):
        """AsyncDecorator重试"""
        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))

        attempts = []

        @handler.retry_decorator
        async def my_func():
            attempts.append(1)
            if len(attempts) < 2:
                raise ConnectionError()
            return "success"

        result = await my_func()
        assert result == "success"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_async_decorator_raises_on_failure(self):
        """AsyncDecorator失败时抛出异常"""
        handler = RetryHandler(RetryConfig(max_retries=1, base_delay=0.01))

        @handler.retry_decorator
        async def my_func():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await my_func()

    def test_sync_decorator_success(self):
        """SyncDecorator成功"""
        handler = RetryHandler()

        @handler.retry_decorator
        def my_func():
            return "decorated"

        result = my_func()
        assert result == "decorated"

    def test_sync_decorator_retry(self):
        """SyncDecorator重试"""
        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))

        attempts = []

        @handler.retry_decorator
        def my_func():
            attempts.append(1)
            if len(attempts) < 2:
                raise ConnectionError()
            return "success"

        result = my_func()
        assert result == "success"


class TestConvenienceFunctions:
    """Test便捷函数"""

    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """retry_async 成功"""
        async def my_func():
            return "result"

        result = await retry_async(my_func)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_retry_async_raises_on_failure(self):
        """retry_async 失败时抛出异常"""
        async def always_fail():
            raise ConnectionError("Fail")

        with pytest.raises(ConnectionError):
            await retry_async(always_fail)

    def test_retry_sync_success(self):
        """retry_sync 成功"""
        def my_func():
            return "result"

        result = retry_sync(my_func)
        assert result == "result"

    def test_retry_sync_raises_on_failure(self):
        """retry_sync 失败时抛出异常"""
        def always_fail():
            raise ConnectionError("Fail")

        with pytest.raises(ConnectionError):
            retry_sync(always_fail)


class TestRetryResult:
    """Test重试Result"""

    def test_retry_result_success(self):
        """成功Result"""
        result = RetryResult(
            success=True,
            result="data",
            attempts=1,
            total_delay=0.0
        )
        assert result.success is True
        assert result.result == "data"
        assert result.error is None

    def test_retry_result_failure(self):
        """失败Result"""
        error = Exception("Test error")
        result = RetryResult(
            success=False,
            error=error,
            attempts=3,
            total_delay=5.0,
            last_status_code=429
        )
        assert result.success is False
        assert result.error == error
        assert result.attempts == 3
        assert result.total_delay == 5.0
        assert result.last_status_code == 429
