import asyncio
import json
import time
from typing import Any, Dict

import streamlit as st
import httpx

from ..thinking_params import build_thinking_params, detect_platform
from ..error_messages import get_error_info
from .base import LLMProvider
from .openai import is_stop_requested, register_stream, unregister_stream, register_client, unregister_client


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini API."""

    def __init__(self, api_base_url: str, api_key: str, model_id: str):
        super().__init__(api_base_url, api_key, model_id)
        self.platform = detect_platform(api_base_url, model_id)

    async def get_completion(self, client, session_id: int, prompt: str, max_tokens: int, log_callback=None, **kwargs) -> dict[str, Any]:
        # 检查停止状态
        if is_stop_requested():
            raise asyncio.CancelledError("Test stopped by user.")

        if log_callback:
            log_callback(f"Session {session_id} (Gemini): PROMPT: {prompt[:100]}...")

        # start_time 将在实际发送 HTTP 请求时记录，以准确测量并发请求的真实开始时间
        start_time = None
        first_token_time = None
        full_response_content = ""

        url = f"{self.api_base_url}/v1beta/models/{self.model_id}:streamGenerateContent?key={self.api_key}&alt=sse"

        generation_config = {
            "maxOutputTokens": max_tokens,
            "temperature": kwargs.get('temperature', 0.7)
        }

        # 提取同步屏障（用于并发请求近乎同时发送）
        barrier = kwargs.pop('_barrier', None)

        # 提取推理相关参数
        thinking_enabled = kwargs.pop('thinking_enabled', None)
        thinking_budget = kwargs.pop('thinking_budget', None)
        reasoning_effort = kwargs.pop('reasoning_effort', None)

        # 构建推理参数
        if thinking_enabled is not None or thinking_budget or reasoning_effort:
            thinking_params = build_thinking_params(
                thinking_enabled,
                thinking_budget,
                reasoning_effort,
                self.platform
            )

            # Gemini: thinkingConfig 放在 generationConfig 中
            if "_generation_config_gemini" in thinking_params:
                generation_config.update(thinking_params["_generation_config_gemini"])

        # 允许其他 Gemini 特定参数传递
        for k, v in kwargs.items():
            if k not in generation_config and k != 'temperature':
                 generation_config[k] = v

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config
        }
        headers = {"Content-Type": "application/json"}

        # 如果没有传入客户端，创建一个
        own_client = False
        if client is None:
            client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(), timeout=600.0)
            own_client = True

        # 注册客户端以便可以强制关闭
        client_id = register_client(client)

        try:
            # 注册当前任务以便取消
            current_task = asyncio.current_task()
            register_stream(current_task)

            # 等待同步屏障（如果存在），确保所有并发请求近乎同时发送
            if barrier is not None:
                await barrier.wait()

            # 在实际发送 HTTP 请求之前记录开始时间，确保并发测试的时间准确性
            start_time = time.time()

            async with client.stream("POST", url, json=payload, headers=headers, timeout=600.0) as response:
                # HTTP 错误检测
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    if status_code == 401:
                        error_info = get_error_info(
                            e,
                            context=f"Model: {self.model_id}",
                            language="zh"
                        )
                        return {"error": f"{str(e)}. {error_info['title']}: {error_info['details']}", "error_info": error_info}
                    elif status_code == 429:
                        error_info = get_error_info(
                            e,
                            context=f"Model: {self.model_id}",
                            language="zh"
                        )
                        return {"error": f"{str(e)}. {error_info['title']}: {error_info['details']}", "error_info": error_info}
                    elif status_code >= 500:
                        error_info = get_error_info(
                            e,
                            context=f"Model: {self.model_id}",
                            language="zh"
                        )
                        return {"error": f"{str(e)}. {error_info['title']}: {error_info['details']}", "error_info": error_info}
                    else:
                        raise

                async for line in response.aiter_lines():
                    # 每次迭代检查停止标志
                    if is_stop_requested():
                        raise asyncio.CancelledError("Test stopped by user.")

                    if line.strip().startswith('data: '):
                        line_data = line[len('data: '):].strip()
                        try:
                            chunk = json.loads(line_data)
                            if "candidates" in chunk and len(chunk["candidates"]) > 0:
                                parts = chunk["candidates"][0].get("content", {}).get("parts", [])
                                for part in parts:
                                    text = part.get("text") or ""
                                    thought = part.get("thought") or ""
                                    content = thought + text

                                    if content:
                                        if first_token_time is None:
                                            first_token_time = time.time()
                                            ttft_raw = first_token_time - start_time
                                            if log_callback:
                                                log_callback(f"Session {session_id} (Gemini): FIRST_TOKEN (TTFT: {ttft_raw:.3f}s)")
                                        full_response_content += content
                        except json.JSONDecodeError:
                            continue

            # 取消注册
            unregister_stream(current_task)

            end_time = time.time()

            if log_callback:
                log_callback(f"Session {session_id} (Gemini): RECV: {full_response_content[:100]}...")

            return {
                "start_time": start_time,
                "first_token_time": first_token_time,
                "end_time": end_time,
                "full_response_content": full_response_content,
                "usage_info": None,  # Gemini doesn't return usage in stream
                "error": None
            }

        except httpx.TimeoutException as e:
            if log_callback:
                log_callback(f"Session {session_id} (Gemini): ERROR: {str(e)}")
            error_info = get_error_info(
                e,
                context=f"Model: {self.model_id}",
                language="zh"
            )
            return {"error": f"{str(e)}. {error_info['title']}: {error_info['details']}", "error_info": error_info}
        except httpx.NetworkError as e:
            if log_callback:
                log_callback(f"Session {session_id} (Gemini): ERROR: {str(e)}")
            error_info = get_error_info(
                e,
                context=f"Model: {self.model_id}",
                language="zh"
            )
            return {"error": f"{str(e)}. {error_info['title']}: {error_info['details']}", "error_info": error_info}
        except httpx.HTTPStatusError as e:
            if log_callback:
                log_callback(f"Session {session_id} (Gemini): ERROR: {str(e)}")
            error_info = get_error_info(
                e,
                context=f"Model: {self.model_id}",
                language="zh"
            )
            return {"error": f"{str(e)}. {error_info['title']}: {error_info['details']}", "error_info": error_info}
        except asyncio.CancelledError:
            # 用户取消 - 传播给上层
            raise
        except Exception as e:
            if log_callback:
                log_callback(f"Session {session_id} (Gemini): ERROR: {str(e)}")
            error_info = get_error_info(
                e,
                context=f"Model: {self.model_id}",
                language="zh"
            )
            return {"error": f"{str(e)}. {error_info['title']}: {error_info['details']}", "error_info": error_info}
        finally:
            # 取消注册客户端
            unregister_client(client_id)
            # 关闭自己创建的客户端
            if own_client:
                try:
                    await client.aclose()
                except Exception:
                    pass
