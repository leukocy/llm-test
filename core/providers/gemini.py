import asyncio
import time
from typing import Any

import httpx

from utils.log_sanitizer import sanitize_api_key

from ..error_messages import get_error_info
from ..thinking_params import build_thinking_params, detect_platform
from .base import LLMProvider, get_request_timeout_seconds
from .gemini_stream import normalize_usage, split_parts, stream_chunks
from .openai import (
    is_stop_requested,
    register_client,
    register_stream,
    unregister_client,
    unregister_stream,
)


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini API."""

    def __init__(self, api_base_url: str, api_key: str, model_id: str):
        super().__init__(api_base_url, api_key, model_id)
        self.platform = detect_platform(api_base_url, model_id)

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-style messages to Gemini format.

        Returns:
            (system_instruction, contents) where system_instruction is the
            system message text (or None) and contents is the Gemini contents array.
        """
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = "\n\n".join(filter(None, [system_instruction, content]))
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})
        return system_instruction, contents

    async def get_completion(
        self,
        client,
        session_id: int,
        prompt: str = "",
        max_tokens: int = 256,
        log_callback=None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        # 检查停止状态
        if is_stop_requested():
            raise asyncio.CancelledError("Test stopped by user.")

        if log_callback:
            log_callback(f"Session {session_id} (Gemini): PROMPT: {prompt[:100]}...")

        # start_time 将在实际发送 HTTP 请求时记录，以准确测量并发请求的真实开始时间
        start_time = None
        first_token_time = None
        full_response_content = ""
        reasoning_content = ""
        first_reasoning_time = None
        usage_info = None
        finish_reason = None
        request_timeout = kwargs.pop("request_timeout", None)
        input_tokens_hint = kwargs.pop("input_tokens_hint", None)
        request_timeout_seconds = get_request_timeout_seconds(
            prompt=prompt,
            messages=messages,
            input_tokens=input_tokens_hint,
            request_timeout=request_timeout,
        )

        base = self.api_base_url.rstrip("/")
        if not base.endswith(("/v1", "/v1beta")):
            base += "/v1beta"
        model = self.model_id.removeprefix("models/")
        url = f"{base}/models/{model}:streamGenerateContent?alt=sse"

        generation_config: dict[str, Any] = {
            "maxOutputTokens": max_tokens,
        }

        # Sampling temperature — only sent when explicitly provided (default: API default)
        temperature = kwargs.pop("temperature", None)
        if temperature is not None:
            generation_config["temperature"] = temperature

        # 提取同步屏障（用于并发请求近乎同时发送）
        barrier = kwargs.pop("_barrier", None)

        # 提取推理相关参数
        thinking_enabled = kwargs.pop("thinking_enabled", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if thinking_enabled is False and (thinking_budget not in (None, 0) or reasoning_effort):
            return {"error": "Gemini thinking-off conflicts with a positive budget/effort"}
        if thinking_enabled is False and model.startswith(("gemini-3", "gemini-2.5-pro")):
            return {
                "error": "This Gemini model cannot disable thinking; select an effort/budget instead"
            }
        if model.startswith("gemini-2.5") and reasoning_effort:
            return {
                "error": "Gemini 2.5 requires thinkingBudget, not reasoning_effort/thinkingLevel"
            }

        # 构建推理参数
        if thinking_enabled is not None or thinking_budget or reasoning_effort:
            thinking_params = build_thinking_params(
                thinking_enabled, thinking_budget, reasoning_effort, self.platform
            )

            # Gemini: thinkingConfig 放在 generationConfig 中
            if "_generation_config_gemini" in thinking_params:
                generation_config.update(thinking_params["_generation_config_gemini"])
                if model.startswith("gemini-2.5"):
                    config = generation_config["thinkingConfig"]
                    if "thinkingLevel" in config:
                        config.pop("thinkingLevel")
                        config["thinkingBudget"] = -1

        # 允许其他 Gemini 特定参数传递
        # (Gemini has no extra_body concept; custom params merge into generationConfig)
        custom_extra_body = kwargs.pop("_custom_extra_body", None)
        if custom_extra_body:
            for k, v in custom_extra_body.items():
                if k not in generation_config:
                    generation_config[k] = v
        for k, v in kwargs.items():
            if k not in generation_config and k != "temperature":
                generation_config[k] = v
        count = generation_config.get("candidateCount", 1)
        if type(count) is not int or count != 1:
            return {"error": "Gemini benchmark requires a single response candidate"}

        # Build payload: use structured messages if provided, otherwise flat prompt
        if messages:
            system_instruction, gemini_contents = self._convert_messages(messages)
            payload = {
                "contents": gemini_contents,
                "generationConfig": generation_config,
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        else:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": generation_config,
            }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

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
            created_at = time.time()
            start_time = time.monotonic()

            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=request_timeout_seconds,
            ) as response:
                # HTTP 错误检测
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    if status_code == 401 or status_code == 429 or status_code >= 500:
                        error_info = get_error_info(
                            e, context=f"Model: {self.model_id}", language="zh"
                        )
                        return {
                            "error": sanitize_api_key(
                                f"{str(e)}. {error_info['title']}: {error_info['details']}"
                            ),
                            "error_info": error_info,
                        }
                    else:
                        raise

                async for chunk in stream_chunks(response):
                    # 每次迭代检查停止标志
                    if is_stop_requested():
                        raise asyncio.CancelledError("Test stopped by user.")

                    if not isinstance(chunk, dict) or "error" in chunk:
                        raise ValueError("Gemini stream returned an invalid/error event")
                    if chunk.get("promptFeedback", {}).get("blockReason"):
                        raise ValueError("Gemini prompt was blocked")
                    if "usageMetadata" in chunk:
                        usage_info = normalize_usage(chunk["usageMetadata"])
                    candidates = chunk.get("candidates", [])
                    if len(candidates) > 1:
                        raise ValueError("Gemini stream returned multiple candidates")
                    if not candidates:
                        continue
                    candidate = candidates[0]
                    content, thought = split_parts(candidate.get("content", {}).get("parts", []))
                    if finish_reason and (content or thought):
                        raise ValueError("Gemini stream returned text after termination")
                    if thought:
                        if first_reasoning_time is None:
                            first_reasoning_time = time.monotonic()
                        reasoning_content += thought
                    if content:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                            if log_callback:
                                log_callback(
                                    f"Session {session_id} (Gemini): FIRST_TOKEN (TTFT: {first_token_time - start_time:.3f}s)"
                                )
                        full_response_content += content
                    if candidate.get("finishReason"):
                        finish_reason = candidate["finishReason"]
                        if finish_reason not in {"STOP", "MAX_TOKENS"}:
                            raise ValueError(
                                "Gemini generation did not finish with a usable answer"
                            )

            if not finish_reason or not full_response_content.strip():
                raise ValueError("Gemini stream ended without a terminal answer")

            # 取消注册
            unregister_stream(current_task)

            end_time = time.monotonic()

            if log_callback:
                log_callback(
                    f"Session {session_id} (Gemini): RECV: {full_response_content[:100]}..."
                )

            from ..request_logger import get_request_logger

            logger = get_request_logger()
            if logger:
                logger.log_request(
                    session_id=str(session_id),
                    provider="Gemini",
                    model_id=self.model_id,
                    platform="gemini",
                    api_base_url=self.api_base_url,
                    headers=headers,
                    payload=payload,
                    thinking_enabled=thinking_enabled,
                    thinking_budget=thinking_budget,
                    reasoning_effort=reasoning_effort,
                    full_response_content=full_response_content,
                    reasoning_content=reasoning_content,
                    usage_info=usage_info,
                    created_at=created_at,
                    start_time=start_time,
                    first_token_time=first_token_time,
                    end_time=end_time,
                )

            return {
                "timing_clock": "client_monotonic",
                "created_at": created_at,
                "start_time": start_time,
                "first_token_time": first_token_time,
                "end_time": end_time,
                "full_response_content": full_response_content,
                "reasoning_content": reasoning_content,
                "first_reasoning_time": first_reasoning_time,
                "first_content_time": first_token_time,
                "ttft_scope": "first_answer_text",
                "output_token_scope": "response_candidates_excluding_thoughts",
                "finish_reason": finish_reason,
                "usage_info": usage_info,
                "error": None,
            }

        except httpx.TimeoutException as e:
            if log_callback:
                log_callback(f"Session {session_id} (Gemini): ERROR: {sanitize_api_key(str(e))}")
            error_info = get_error_info(e, context=f"Model: {self.model_id}", language="zh")
            return {
                "error": sanitize_api_key(
                    f"{str(e)}. {error_info['title']}: {error_info['details']}"
                ),
                "error_info": error_info,
            }
        except httpx.NetworkError as e:
            if log_callback:
                log_callback(f"Session {session_id} (Gemini): ERROR: {sanitize_api_key(str(e))}")
            error_info = get_error_info(e, context=f"Model: {self.model_id}", language="zh")
            return {
                "error": sanitize_api_key(
                    f"{str(e)}. {error_info['title']}: {error_info['details']}"
                ),
                "error_info": error_info,
            }
        except httpx.HTTPStatusError as e:
            if log_callback:
                log_callback(f"Session {session_id} (Gemini): ERROR: {sanitize_api_key(str(e))}")
            error_info = get_error_info(e, context=f"Model: {self.model_id}", language="zh")
            return {
                "error": sanitize_api_key(
                    f"{str(e)}. {error_info['title']}: {error_info['details']}"
                ),
                "error_info": error_info,
            }
        except asyncio.CancelledError:
            # 用户取消 - 传播给上层
            raise
        except Exception as e:
            if log_callback:
                log_callback(f"Session {session_id} (Gemini): ERROR: {sanitize_api_key(str(e))}")
            error_info = get_error_info(e, context=f"Model: {self.model_id}", language="zh")
            return {
                "error": sanitize_api_key(
                    f"{str(e)}. {error_info['title']}: {error_info['details']}"
                ),
                "error_info": error_info,
            }
        finally:
            unregister_stream(asyncio.current_task())
            # 取消注册客户端
            unregister_client(client_id)
            # 关闭自己创建的客户端
            if own_client:
                try:
                    await client.aclose()
                except Exception:
                    pass
