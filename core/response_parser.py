"""
Phase 2: 统一响应Parse器 (Unified Response Parser)

提供跨平台流式响应Parse能力，自动适配各平台推理内容字段。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .thinking_params import get_content_field, get_platform_features, get_reasoning_field


@dataclass
class ParsedChunk:
    """Parse后响应块"""
    content: str = ""  # 正文内容
    reasoning: str = ""  # 推理内容
    finish_reason: str | None = None  # 完成原因
    usage: dict[str, Any] | None = None  # Token use信息
    raw_chunk: dict[str, Any] | None = None  # 原始响应块（用于调试）


@dataclass
class ParsedResponse:
    """Parse后完整响应"""
    full_content: str = ""  # 完整正文
    full_reasoning: str = ""  # 完整推理内容
    total_chunks: int = 0  # 总块数
    reasoning_chunks: int = 0  # 包含推理块数
    content_chunks: int = 0  # 包含正文块数
    usage: dict[str, Any] | None = None  # 最终 Token use信息
    finish_reason: str | None = None  # 最终完成原因
    first_reasoning_chunk_index: int | None = None  # 一推理块Index
    first_content_chunk_index: int | None = None  # 一正文块Index
    raw_snapshots: list[dict[str, Any]] = field(default_factory=list)  # 原始响应快照


class UnifiedResponseParser:
    """
    统一响应Parse器

    支持多平台流式响应Parse，自动适配各平台推理内容字段。

    Usage:
        parser = UnifiedResponseParser("mimo")

        for chunk in stream:
            parsed = parser.parse_chunk(chunk)
            print(parsed.reasoning, parsed.content)

        result = parser.get_result()
        print(f"推理: {result.full_reasoning}")
        print(f"正文: {result.full_content}")
    """

    def __init__(self, platform: str, max_snapshots: int = 5):
        """
        InitializeParse器

        Args:
            platform: 平台标识 (mimo, deepseek, gemini, etc.)
            max_snapshots: Save原始响应快照最大数量
        """
        self.platform = platform
        self.features = get_platform_features(platform)
        self.reasoning_field = get_reasoning_field(platform)
        self.content_field = get_content_field(platform)
        self.max_snapshots = max_snapshots

        # 累积Status
        self._full_content = ""
        self._full_reasoning = ""
        self._total_chunks = 0
        self._reasoning_chunks = 0
        self._content_chunks = 0
        self._usage = None
        self._finish_reason = None
        self._first_reasoning_idx = None
        self._first_content_idx = None
        self._snapshots = []

    def parse_chunk(self, chunk: dict[str, Any]) -> ParsedChunk:
        """
        Parse单流式响应块

        Args:
            chunk: 原始响应块 (JSON Parse后字典)

        Returns:
            ParsedChunk: Parse后响应块
        """
        result = ParsedChunk(raw_chunk=chunk if self._total_chunks < self.max_snapshots else None)

        # Save快照
        if self._total_chunks < self.max_snapshots:
            self._snapshots.append(chunk)

        self._total_chunks += 1

        # 提取 usage
        if "usage" in chunk and chunk["usage"]:
            result.usage = chunk["usage"]
            self._usage = chunk["usage"]

        # Process choices
        if "choices" not in chunk or not chunk["choices"]:
            return result

        choice = chunk["choices"][0]
        delta = choice.get("delta", {})

        # 提取 finish_reason
        if "finish_reason" in choice and choice["finish_reason"]:
            result.finish_reason = choice["finish_reason"]
            self._finish_reason = choice["finish_reason"]

        # based on平台提取内容
        if self.platform == "gemini":
            # Gemini 特殊Process：parts 数组
            result = self._parse_gemini_chunk(delta, result)
        else:
            # 通用Process：delta.content / delta.reasoning_content
            result = self._parse_standard_chunk(delta, result)

        # Update累积Status
        if result.content:
            self._full_content += result.content
            self._content_chunks += 1
            if self._first_content_idx is None:
                self._first_content_idx = self._total_chunks - 1

        if result.reasoning:
            self._full_reasoning += result.reasoning
            self._reasoning_chunks += 1
            if self._first_reasoning_idx is None:
                self._first_reasoning_idx = self._total_chunks - 1

        return result

    def _parse_standard_chunk(self, delta: dict[str, Any], result: ParsedChunk) -> ParsedChunk:
        """Parse标准格式响应块 (OpenAI 兼容)"""
        # 提取正文
        content = delta.get(self.content_field, "") or ""
        result.content = content

        # 提取推理内容
        reasoning = delta.get(self.reasoning_field, "") or ""
        result.reasoning = reasoning

        return result

    def _parse_gemini_chunk(self, delta: dict[str, Any], result: ParsedChunk) -> ParsedChunk:
        """Parse Gemini 格式响应块"""
        # Gemini  delta 可能直接包含 parts，也可能is标准格式
        parts = delta.get("parts", [])

        if parts:
            for part in parts:
                # Gemini 用 text 表示正文，thought 表示推理
                text = part.get("text", "") or ""
                thought = part.get("thought", "") or ""

                result.content += text
                result.reasoning += thought
        else:
            # if没has parts，尝试标准格式
            result = self._parse_standard_chunk(delta, result)

        return result

    def get_result(self) -> ParsedResponse:
        """
        GetParse result

        Returns:
            ParsedResponse: Parse后完整响应
        """
        return ParsedResponse(
            full_content=self._full_content,
            full_reasoning=self._full_reasoning,
            total_chunks=self._total_chunks,
            reasoning_chunks=self._reasoning_chunks,
            content_chunks=self._content_chunks,
            usage=self._usage,
            finish_reason=self._finish_reason,
            first_reasoning_chunk_index=self._first_reasoning_idx,
            first_content_chunk_index=self._first_content_idx,
            raw_snapshots=self._snapshots
        )

    def reset(self):
        """ResetParse器Status"""
        self._full_content = ""
        self._full_reasoning = ""
        self._total_chunks = 0
        self._reasoning_chunks = 0
        self._content_chunks = 0
        self._usage = None
        self._finish_reason = None
        self._first_reasoning_idx = None
        self._first_content_idx = None
        self._snapshots = []

    @property
    def has_reasoning(self) -> bool:
        """is否包含推理内容"""
        return len(self._full_reasoning) > 0

    @property
    def reasoning_ratio(self) -> float:
        """推理内容占比 (字符数)"""
        total = len(self._full_content) + len(self._full_reasoning)
        if total == 0:
            return 0.0
        return len(self._full_reasoning) / total


def parse_stream_response(
    chunks: list[dict[str, Any]],
    platform: str
) -> ParsedResponse:
    """
    便捷函数：Parse完整流式响应

    Args:
        chunks: 响应块列表
        platform: 平台标识

    Returns:
        ParsedResponse: Parse result
    """
    parser = UnifiedResponseParser(platform)
    for chunk in chunks:
        parser.parse_chunk(chunk)
    return parser.get_result()
