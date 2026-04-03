"""
Phase 4: 指数退避重试Process器 (Retry Handler)

提供带指数退避重试机制，确保大规模Test稳定性。
支持：
- 可ConfigureRetry countandLatency
- 指数退避策略
- Retry-After 头Parse
- 可重试Error码识别
"""

import asyncio
import inspect
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Optional, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """重试Configure"""
    max_retries: int = 3  # 最大Retry count
    base_delay: float = 1.0  # 基础Latency（seconds）
    max_delay: float = 60.0  # 最大Latency（seconds）
    exponential_base: float = 2.0  # 指数基数
    jitter: bool = True  # is否Add随机抖动
    jitter_range: tuple[float, float] = (0.5, 1.5)  # 抖动范围倍数
    retryable_status_codes: set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})
    retryable_exceptions: tuple = (ConnectionError, TimeoutError)


@dataclass
class RetryResult:
    """重试Result"""
    success: bool
    result: Any = None
    error: Exception | None = None
    attempts: int = 0
    total_delay: float = 0.0
    last_status_code: int | None = None


class RetryHandler:
    """
    指数退避重试Process器

    Usage:
        handler = RetryHandler()

        # 方式1：包装Async函数
        result = await handler.execute_async(api_call, arg1, arg2)

        # 方式2：Sync执行
        result = handler.execute_sync(sync_function, arg1, arg2)

        # 方式3：作isDecorator
        @handler.retry_decorator
        async def my_api_call():
            ...
    """

    def __init__(self, config: RetryConfig | None = None):
        """
        Initialize重试Process器

        Args:
            config: 重试Configure，None 则usedefaultConfigure
        """
        self.config = config or RetryConfig()

    def _calculate_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """
        Calculate重试Latency

        Args:
            attempt: 当前尝试次数（0-based）
            retry_after: API Return Retry-After 值

        Returns:
            Latencyseconds数
        """
        # 指数退避
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)

        # ifhas Retry-After，use较大值
        if retry_after:
            delay = max(delay, retry_after)

        # 限制最大Latency
        delay = min(delay, self.config.max_delay)

        # Add抖动
        if self.config.jitter:
            jitter_factor = random.uniform(*self.config.jitter_range)
            delay *= jitter_factor

        return delay

    def _is_retryable_error(self, error: Exception) -> tuple[bool, int | None, float | None]:
        """
        判断is否is可重试Error

        Args:
            error: 异常对象

        Returns:
            (is否可重试, Status码, Retry-After值)
        """
        status_code = None
        retry_after = None

        # Checkis否is网络异常
        if isinstance(error, self.config.retryable_exceptions):
            return True, None, None

        # 尝试从异常in提取Status码
        if hasattr(error, 'status_code'):
            status_code = error.status_code
        elif hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            status_code = error.response.status_code

        # CheckStatus码is否可重试
        if status_code in self.config.retryable_status_codes:
            # 尝试提取 Retry-After
            if hasattr(error, 'response') and hasattr(error.response, 'headers'):
                retry_after_str = error.response.headers.get('Retry-After')
                if retry_after_str:
                    try:
                        # Try parsing as seconds first
                        retry_after = float(retry_after_str)
                    except ValueError:
                        # Try parsing as HTTP Date (RFC 7231)
                        try:
                            import email.utils
                            from datetime import datetime, timezone
                            
                            dt = email.utils.parsedate_to_datetime(retry_after_str)
                            if dt:
                                now = datetime.now(timezone.utc)
                                retry_after = (dt - now).total_seconds()
                                if retry_after < 0:
                                    retry_after = 0
                        except Exception:
                            pass

            # 也Check异常消息in retryDelay
            error_str = str(error)
            if 'retryDelay' in error_str:
                import re
                match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)s?", error_str)
                if match:
                    retry_after = float(match.group(1))

            return True, status_code, retry_after

        # CheckError消息inis否包含可重试关键词
        error_msg = str(error).lower()
        retryable_keywords = ['rate limit', 'too many requests', 'resource exhausted', 'temporarily unavailable']
        for keyword in retryable_keywords:
            if keyword in error_msg:
                return True, 429, None

        return False, status_code, None

    async def execute_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> RetryResult:
        """
        带重试执行Async函数

        Args:
            func: Async函数
            *args, **kwargs: 函数参数

        Returns:
            RetryResult: 执行Result
        """
        attempts = 0
        total_delay = 0.0
        last_error = None
        last_status_code = None

        while attempts <= self.config.max_retries:
            try:
                result = await func(*args, **kwargs)
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts + 1,
                    total_delay=total_delay
                )

            except Exception as e:
                last_error = e
                is_retryable, status_code, retry_after = self._is_retryable_error(e)
                last_status_code = status_code

                if not is_retryable or attempts >= self.config.max_retries:
                    logger.warning(f"Non-retryable error or max retries reached: {e}")
                    break

                delay = self._calculate_delay(attempts, retry_after)
                total_delay += delay

                logger.info(f"Retry {attempts + 1}/{self.config.max_retries}: "
                           f"status={status_code}, delay={delay:.1f}s")

                await asyncio.sleep(delay)
                attempts += 1

        return RetryResult(
            success=False,
            error=last_error,
            attempts=attempts + 1,
            total_delay=total_delay,
            last_status_code=last_status_code
        )

    def execute_sync(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> RetryResult:
        """
        带重试执行Sync函数

        Args:
            func: Sync函数
            *args, **kwargs: 函数参数

        Returns:
            RetryResult: 执行Result
        """
        attempts = 0
        total_delay = 0.0
        last_error = None
        last_status_code = None

        while attempts <= self.config.max_retries:
            try:
                result = func(*args, **kwargs)
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts + 1,
                    total_delay=total_delay
                )

            except Exception as e:
                last_error = e
                is_retryable, status_code, retry_after = self._is_retryable_error(e)
                last_status_code = status_code

                if not is_retryable or attempts >= self.config.max_retries:
                    logger.warning(f"Non-retryable error or max retries reached: {e}")
                    break

                delay = self._calculate_delay(attempts, retry_after)
                total_delay += delay

                logger.info(f"Retry {attempts + 1}/{self.config.max_retries}: "
                           f"status={status_code}, delay={delay:.1f}s")

                time.sleep(delay)
                attempts += 1

        return RetryResult(
            success=False,
            error=last_error,
            attempts=attempts + 1,
            total_delay=total_delay,
            last_status_code=last_status_code
        )

    def retry_decorator(self, func: Callable) -> Callable:
        """
        重试Decorator

        Usage:
            @handler.retry_decorator
            async def my_function():
                ...
        """
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await self.execute_async(func, *args, **kwargs)
                if result.success:
                    return result.result
                raise result.error
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = self.execute_sync(func, *args, **kwargs)
                if result.success:
                    return result.result
                raise result.error
            return sync_wrapper


# default全局实例
default_retry_handler = RetryHandler()


async def retry_async[T](func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
    """
    便捷函数：带重试执行Async函数

    Args:
        func: Async函数
        *args, **kwargs: 函数参数

    Returns:
        函数Return值

    Raises:
        原始异常（if所has重试都失败）
    """
    result = await default_retry_handler.execute_async(func, *args, **kwargs)
    if result.success:
        return result.result
    raise result.error


def retry_sync[T](func: Callable[..., T], *args, **kwargs) -> T:
    """
    便捷函数：带重试执行Sync函数

    Args:
        func: Sync函数
        *args, **kwargs: 函数参数

    Returns:
        函数Return值

    Raises:
        原始异常（if所has重试都失败）
    """
    result = default_retry_handler.execute_sync(func, *args, **kwargs)
    if result.success:
        return result.result
    raise result.error
