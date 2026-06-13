import asyncio
import threading
import time
from typing import Any
from weakref import WeakSet, WeakValueDictionary

import httpx

from ..error_messages import get_error_info
from ..thinking_params import build_thinking_params, detect_platform
from .base import LLMProvider, get_request_timeout_seconds
from .stream_parser import parse_openai_stream_line

# 取消/暂停标志已迁移至 core.cancel_state（进程级 threading.Event，UI 无关）
# 活跃流任务追踪
_active_streams: WeakSet = WeakSet()
_active_streams_lock = threading.Lock()
# 活跃的 httpx 客户端连接 - 用于强制中断
_active_clients: WeakValueDictionary = WeakValueDictionary()
_active_clients_lock = threading.Lock()
_client_counter = 0

# 并发同步屏障 - 用于让所有并发请求同时开始
_concurrent_barrier: asyncio.Event | None = None
_barrier_waiters = 0
_barrier_total = 0
_barrier_lock = threading.Lock()


def set_stop_requested(value: bool):
    """设置停止标志并取消所有活跃流。

    状态存于 core.cancel_state（进程级，UI 无关）。UI 层负责把信号镜像到 session_state 供显示。
    """
    from core import cancel_state

    if value:
        cancel_state.request_stop()
        # 关闭所有活跃的 httpx 客户端 - 这会立即中断正在进行的流式请求
        with _active_clients_lock:
            for client_id, client in list(_active_clients.items()):
                try:
                    # 在事件循环中调度关闭操作
                    if hasattr(client, 'aclose'):
                        # 尝试获取正在运行的事件循环
                        try:
                            loop = asyncio.get_running_loop()
                            loop.call_soon_threadsafe(
                                lambda c=client: asyncio.create_task(c.aclose())
                            )
                        except RuntimeError:
                            # 没有运行中的事件循环，尝试同步关闭
                            pass
                except Exception:
                    pass

        # 取消所有活跃的流式任务
        with _active_streams_lock:
            for stream_task in list(_active_streams):
                try:
                    if isinstance(stream_task, asyncio.Task) and not stream_task.done():
                        stream_task.cancel()
                except Exception:
                    pass
    else:
        cancel_state.clear_stop()


def is_stop_requested() -> bool:
    """检查是否请求停止（读 core.cancel_state，不再读 session_state）。"""
    from core import cancel_state

    return cancel_state.is_stop_requested()


def set_pause_requested(value: bool):
    """设置暂停标志（状态存于 core.cancel_state）。"""
    from core import cancel_state

    if value:
        cancel_state.request_pause()
    else:
        cancel_state.clear_pause()


def is_pause_requested() -> bool:
    """检查是否请求暂停（读 core.cancel_state）。"""
    from core import cancel_state

    return cancel_state.is_pause_requested()


def register_stream(stream_task):
    """注册活跃流任务"""
    with _active_streams_lock:
        _active_streams.add(stream_task)


def unregister_stream(stream_task):
    """取消注册流任务"""
    with _active_streams_lock:
        try:
            _active_streams.discard(stream_task)
        except Exception:
            pass


def register_client(client) -> int:
    """注册活跃的 httpx 客户端，返回客户端ID"""
    global _client_counter
    with _active_clients_lock:
        _client_counter += 1
        _active_clients[_client_counter] = client
        return _client_counter


def unregister_client(client_id: int):
    """取消注册客户端"""
    with _active_clients_lock:
        try:
            del _active_clients[client_id]
        except Exception:
            pass


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI-compatible APIs."""

    def __init__(self, api_base_url: str, api_key: str, model_id: str):
        super().__init__(api_base_url, api_key, model_id)
        self.platform = detect_platform(api_base_url, model_id)

    async def get_completion(self, client, session_id: int, prompt: str = "", max_tokens: int = 256,
                            log_callback=None, messages: list[dict] | None = None, **kwargs) -> dict[str, Any]:
        # 检查停止状态
        if is_stop_requested():
            raise asyncio.CancelledError("Test stopped by user.")

        created_at = time.time()
        # start_time 将在实际发送 HTTP 请求时记录，以准确测量并发请求的真实开始时间
        start_time = None
        first_token_time = None
        usage_info = None
        full_response_content = ""
        reasoning_content = ""
        raw_stream_chunks = []
        token_timestamps = []
        request_timeout = kwargs.pop('request_timeout', None)
        input_tokens_hint = kwargs.pop('input_tokens_hint', None)
        actual_messages = messages if messages else [{"role": "user", "content": prompt}]
        request_timeout_seconds = get_request_timeout_seconds(
            prompt=prompt,
            messages=messages,
            input_tokens=input_tokens_hint,
            request_timeout=request_timeout,
        )

        # 如果没有传入客户端，创建一个
        own_client = False
        if client is None:
            client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(
                    limits=httpx.Limits(max_connections=2048, max_keepalive_connections=256),
                ),
                timeout=request_timeout_seconds,
            )
            own_client = True

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            if self.platform == "mimo":
                headers["api-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_id,
            "messages": actual_messages,
            "stream": True,
        }

        if self.platform != "unknown":
            payload["stream_options"] = {"include_usage": True}

        if self.platform == "mimo":
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

        # 提取同步屏障（用于并发请求近乎同时发送）
        barrier = kwargs.pop('_barrier', None)

        # 提取推理相关参数
        thinking_enabled = kwargs.pop('thinking_enabled', None)
        thinking_budget = kwargs.pop('thinking_budget', None)
        reasoning_effort = kwargs.pop('reasoning_effort', None)

        if thinking_enabled is not None or thinking_budget or reasoning_effort:
            thinking_params = build_thinking_params(
                thinking_enabled,
                thinking_budget,
                reasoning_effort,
                self.platform
            )

            if "_extra_body_reasoning_split" in thinking_params:
                if "extra_body" not in payload:
                    payload["extra_body"] = {}
                payload["extra_body"]["reasoning_split"] = thinking_params.pop("_extra_body_reasoning_split")

            if "_extra_body_aliyun" in thinking_params:
                if "extra_body" not in payload:
                    payload["extra_body"] = {}
                payload["extra_body"].update(thinking_params.pop("_extra_body_aliyun"))

            if "_extra_body_deepseek" in thinking_params:
                if "extra_body" not in payload:
                    payload["extra_body"] = {}
                payload["extra_body"].update(thinking_params.pop("_extra_body_deepseek"))

            if "_extra_body_volcano" in thinking_params:
                if "extra_body" not in payload:
                    payload["extra_body"] = {}
                payload["extra_body"].update(thinking_params.pop("_extra_body_volcano"))

            payload.update(thinking_params)

        for k, v in kwargs.items():
            if v is not None:
                payload[k] = v

        # 注册客户端以便可以强制关闭
        client_id = register_client(client)

        try:
            # 创建流式请求任务
            async def stream_request():
                nonlocal first_token_time, usage_info, full_response_content, reasoning_content, raw_stream_chunks, token_timestamps, start_time

                # 等待同步屏障（如果存在），确保所有并发请求近乎同时发送
                if barrier is not None:
                    await barrier.wait()

                # 在实际发送 HTTP 请求之前记录开始时间，确保并发测试的时间准确性
                start_time = time.monotonic()

                async with client.stream(
                    "POST",
                    f"{self.api_base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=request_timeout_seconds
                ) as response:
                    if response.status_code != 200:
                        status_code = response.status_code
                        error_text = (await response.aread())[:200].decode('utf-8', errors='ignore')
                        if log_callback:
                            log_callback(f"API Error {status_code}: {error_text}")

                        if status_code == 401 or status_code == 429:
                            error_info = get_error_info(
                                Exception(f"HTTP {status_code}: {error_text}"),
                                context=f"Model: {self.model_id}",
                                language="zh"
                            )
                            raise Exception(f"HTTP {status_code}: {error_info['title']}: {error_info['details']}")
                        elif status_code >= 500:
                            error_info = get_error_info(
                                Exception(f"HTTP {status_code}: {error_text}"),
                                context=f"Model: {self.model_id}, URL: {self.api_base_url}",
                                language="zh"
                            )
                            raise Exception(f"HTTP {status_code}: {error_info['title']}: {error_info['details']}")
                        else:
                            raise Exception(f"HTTP {status_code}: {error_text}")

                    # 使用带超时的读取，以便能够响应取消请求
                    async for line_bytes in response.aiter_lines():
                        # 每次迭代检查停止标志
                        if is_stop_requested():
                            raise asyncio.CancelledError("Test stopped by user.")

                        if not line_bytes:
                            continue

                        line = line_bytes.strip()

                        event = parse_openai_stream_line(line)
                        if event is None:
                            continue
                        if event.done:
                            break

                        if event.usage:
                            usage_info = event.usage

                        if event.has_choice and event.raw_chunk and len(raw_stream_chunks) < 5:
                            raw_stream_chunks.append(event.raw_chunk)

                        if event.reasoning:
                            reasoning_content += event.reasoning

                        text_chunk = event.text
                        if text_chunk:
                            current_time = time.monotonic()

                            is_only_tag = text_chunk.strip() in ["<think", "<think\\n", "\\n<think"]

                            if first_token_time is None:
                                if not is_only_tag or is_only_tag and len(text_chunk) > 10:
                                    first_token_time = current_time

                            token_timestamps.append(current_time)
                            full_response_content += text_chunk

            # 注册流任务以便可以取消
            current_task = asyncio.current_task()
            register_stream(current_task)

            try:
                await stream_request()
            finally:
                unregister_stream(current_task)

            end_time = time.monotonic()

            if not full_response_content.strip():
                error_info = get_error_info(
                    Exception(f"Empty response received for {self.model_id}"),
                    context=f"URL: {self.api_base_url}",
                    language="zh"
                )
                return {
                    "error": f"Empty response received. {error_info['title']}: {error_info['details']}",
                    "error_info": error_info,
                    "debug_info": {
                        "url": self.api_base_url,
                        "model": self.model_id
                    }
                }

            if not usage_info:
                usage_info = {}

            # 记录请求日志
            from ..request_logger import get_request_logger
            req_logger = get_request_logger()
            if req_logger:
                req_logger.log_request(
                    session_id=str(session_id),
                    provider="OpenAIProvider",
                    model_id=self.model_id,
                    platform=self.platform,
                    api_base_url=self.api_base_url,
                    headers=headers,
                    payload=payload,
                    thinking_enabled=thinking_enabled,
                    thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort,
                    full_response_content=full_response_content,
                    reasoning_content=reasoning_content,
                    usage_info=usage_info,
                    raw_stream_chunks=raw_stream_chunks,
                    created_at=created_at,
                    start_time=start_time,
                    first_token_time=first_token_time,
                    end_time=end_time,
                    token_timestamps=token_timestamps,
                    error=None,
                )

            return {
                "created_at": created_at,
                "start_time": start_time,
                "first_token_time": first_token_time,
                "end_time": end_time,
                "full_response_content": full_response_content,
                "usage_info": usage_info,
                "token_timestamps": token_timestamps,
                "error": None
            }

        except asyncio.CancelledError:
            # 用户取消 - 直接传播
            raise
        except httpx.TimeoutException as e:
            end_time = time.monotonic()
            # 如果请求未发出就超时，使用当前时间作为 start_time
            if start_time is None:
                start_time = end_time
            error_msg = f"{str(e)}. Timeout"
            error_info = get_error_info(
                e,
                context=f"Model: {self.model_id}, URL: {self.api_base_url}",
                language="zh"
            )
            from ..request_logger import get_request_logger
            req_logger = get_request_logger()
            if req_logger:
                req_logger.log_request(
                    session_id=str(session_id),
                    provider="OpenAIProvider",
                    model_id=self.model_id,
                    platform=self.platform,
                    api_base_url=self.api_base_url,
                    headers=headers,
                    payload=payload,
                    thinking_enabled=thinking_enabled,
                    thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort,
                    full_response_content=full_response_content,
                    reasoning_content=reasoning_content,
                    usage_info=usage_info,
                    raw_stream_chunks=raw_stream_chunks,
                    created_at=created_at,
                    start_time=start_time,
                    first_token_time=first_token_time,
                    end_time=end_time,
                    token_timestamps=token_timestamps,
                    error=error_msg,
                )
            return {"error": f"{str(e)}. {error_info['title']}: {error_info['details']}", "error_info": error_info}
        except httpx.NetworkError as e:
            end_time = time.monotonic()
            # 如果请求未发出就失败，使用当前时间作为 start_time
            if start_time is None:
                start_time = end_time
            error_msg = f"{str(e)}. ConnectionError"
            error_info = get_error_info(
                e,
                context=f"Model: {self.model_id}, URL: {self.api_base_url}",
                language="zh"
            )
            from ..request_logger import get_request_logger
            req_logger = get_request_logger()
            if req_logger:
                req_logger.log_request(
                    session_id=str(session_id),
                    provider="OpenAIProvider",
                    model_id=self.model_id,
                    platform=self.platform,
                    api_base_url=self.api_base_url,
                    headers=headers,
                    payload=payload,
                    thinking_enabled=thinking_enabled,
                    thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort,
                    full_response_content=full_response_content,
                    reasoning_content=reasoning_content,
                    usage_info=usage_info,
                    raw_stream_chunks=raw_stream_chunks,
                    created_at=created_at,
                    start_time=start_time,
                    first_token_time=first_token_time,
                    end_time=end_time,
                    token_timestamps=token_timestamps,
                    error=error_msg,
                )
            return {"error": f"{str(e)}. {error_info['title']}: {error_info['details']}", "error_info": error_info}
        except Exception as e:
            end_time = time.monotonic()
            # 如果请求未发出就失败，使用当前时间作为 start_time
            if start_time is None:
                start_time = end_time
            if isinstance(e, asyncio.CancelledError):
                raise
            error_msg = f"{str(e)}. Exception"
            error_info = get_error_info(
                e,
                context=f"Model: {self.model_id}",
                language="zh"
            )
            from ..request_logger import get_request_logger
            req_logger = get_request_logger()
            if req_logger:
                req_logger.log_request(
                    session_id=str(session_id),
                    provider="OpenAIProvider",
                    model_id=self.model_id,
                    platform=self.platform,
                    api_base_url=self.api_base_url,
                    headers=headers,
                    payload=payload,
                    thinking_enabled=thinking_enabled,
                    thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort,
                    full_response_content=full_response_content,
                    reasoning_content=reasoning_content,
                    usage_info=usage_info,
                    raw_stream_chunks=raw_stream_chunks,
                    created_at=created_at,
                    start_time=start_time,
                    first_token_time=first_token_time,
                    end_time=end_time,
                    token_timestamps=token_timestamps,
                    error=error_msg,
                )
            return {"error": f"{str(e)}. {error_info['title']}: {error_info['original']}", "error_info": error_info}
        finally:
            # 取消注册客户端
            unregister_client(client_id)
            # 关闭自己创建的客户端
            if own_client:
                try:
                    await client.aclose()
                except Exception:
                    pass
