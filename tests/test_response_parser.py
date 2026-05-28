"""
单元Test: core/response_parser.py

Test响应Parse器功能，包括:
- 标准格式响应Parse
- Gemini 格式响应Parse
- 推理内容提取
- 完整响应Aggregate
"""

import pytest

from core.response_parser import (
    ParsedChunk,
    ParsedResponse,
    UnifiedResponseParser,
    parse_stream_response,
)


class TestParsedChunk:
    """TestParse后响应块"""

    def test_default_values(self):
        """default值"""
        chunk = ParsedChunk()
        assert chunk.content == ""
        assert chunk.reasoning == ""
        assert chunk.finish_reason is None
        assert chunk.usage is None
        assert chunk.raw_chunk is None

    def test_with_values(self):
        """带值Initialize"""
        chunk = ParsedChunk(
            content="Hello",
            reasoning="Thinking...",
            finish_reason="stop",
            usage={"total_tokens": 100}
        )
        assert chunk.content == "Hello"
        assert chunk.reasoning == "Thinking..."
        assert chunk.finish_reason == "stop"
        assert chunk.usage["total_tokens"] == 100


class TestParsedResponse:
    """TestParse后完整响应"""

    def test_default_values(self):
        """default值"""
        response = ParsedResponse()
        assert response.full_content == ""
        assert response.full_reasoning == ""
        assert response.total_chunks == 0
        assert response.usage is None

    def test_with_values(self):
        """带值Initialize"""
        response = ParsedResponse(
            full_content="Hello world",
            full_reasoning="I thought about it",
            total_chunks=10,
            reasoning_chunks=5,
            content_chunks=5,
            finish_reason="stop"
        )
        assert response.full_content == "Hello world"
        assert response.full_reasoning == "I thought about it"
        assert response.total_chunks == 10


class TestUnifiedResponseParser:
    """Test统一响应Parse器"""

    def test_init_standard_platform(self):
        """Initialize标准平台"""
        parser = UnifiedResponseParser("mimo")
        assert parser.platform == "mimo"
        assert parser._total_chunks == 0

    def test_init_with_max_snapshots(self):
        """Custom快照数量"""
        parser = UnifiedResponseParser("deepseek", max_snapshots=10)
        assert parser.max_snapshots == 10

    def test_parse_chunk_empty(self):
        """Parse空块"""
        parser = UnifiedResponseParser("mimo")
        result = parser.parse_chunk({})
        assert result.content == ""
        assert result.reasoning == ""

    def test_parse_chunk_with_choices(self):
        """Parse带 choices 块"""
        parser = UnifiedResponseParser("mimo")
        chunk = {
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }
        result = parser.parse_chunk(chunk)
        assert result.content == "Hello"
        assert result.finish_reason is None

    def test_parse_chunk_with_reasoning(self):
        """Parse带推理内容块"""
        parser = UnifiedResponseParser("deepseek")
        chunk = {
            "choices": [{
                "delta": {
                    "content": "Answer",
                    "reasoning_content": "Let me think..."
                },
                "finish_reason": None
            }]
        }
        result = parser.parse_chunk(chunk)
        assert result.content == "Answer"
        assert result.reasoning == "Let me think..."

    def test_parse_chunk_with_usage(self):
        """Parse带 usage 块"""
        parser = UnifiedResponseParser("mimo")
        chunk = {
            "choices": [{
                "delta": {"content": "Hi"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        result = parser.parse_chunk(chunk)
        assert result.usage["total_tokens"] == 15
        assert result.finish_reason == "stop"

    def test_parse_chunk_no_choices(self):
        """Parseno choices 块"""
        parser = UnifiedResponseParser("mimo")
        result = parser.parse_chunk({"usage": {"total_tokens": 10}})
        assert result.usage is not None

    def test_parse_multiple_chunks(self):
        """Parse多块"""
        parser = UnifiedResponseParser("mimo")

        # 一块
        parser.parse_chunk({
            "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]
        })
        # 二块
        parser.parse_chunk({
            "choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]
        })

        response = parser.get_result()
        assert response.full_content == "Hello world"
        assert response.total_chunks == 2
        assert response.content_chunks == 2
        assert response.finish_reason == "stop"

    def test_parse_multiple_chunks_with_reasoning(self):
        """Parse多带推理块"""
        parser = UnifiedResponseParser("deepseek")

        # 推理块
        parser.parse_chunk({
            "choices": [{"delta": {"reasoning_content": "Thinking... "}}]
        })
        # 正文块
        parser.parse_chunk({
            "choices": [{"delta": {"content": "Answer"}}]
        })

        response = parser.get_result()
        assert response.full_reasoning == "Thinking... "
        assert response.full_content == "Answer"
        assert response.reasoning_chunks == 1
        assert response.content_chunks == 1

    def test_get_result_after_reset(self):
        """Reset后GetResult"""
        parser = UnifiedResponseParser("mimo")
        parser.parse_chunk({
            "choices": [{"delta": {"content": "Test"}}]
        })

        parser.reset()
        response = parser.get_result()

        assert response.full_content == ""
        assert response.total_chunks == 0

    def test_has_reasoning_property(self):
        """Test has_reasoning 属性"""
        parser = UnifiedResponseParser("deepseek")

        assert parser.has_reasoning is False

        parser.parse_chunk({
            "choices": [{"delta": {"reasoning_content": "Thinking"}}]
        })

        assert parser.has_reasoning is True

    def test_reasoning_ratio_property(self):
        """Test reasoning_ratio 属性"""
        parser = UnifiedResponseParser("deepseek")

        parser.parse_chunk({
            "choices": [{"delta": {"reasoning_content": "AAAA", "content": "BB"}}]
        })

        # 4字符推理 / 6字符总计 = 2/3
        assert parser.reasoning_ratio == pytest.approx(4/6)

    def test_reasoning_ratio_empty(self):
        """空响应推理比例"""
        parser = UnifiedResponseParser("mimo")
        assert parser.reasoning_ratio == 0.0

    def test_first_chunk_indices(self):
        """Test首块Index记录"""
        parser = UnifiedResponseParser("deepseek")

        # 推理块
        parser.parse_chunk({
            "choices": [{"delta": {"reasoning_content": "First"}}]
        })
        # 正文块
        parser.parse_chunk({
            "choices": [{"delta": {"content": "Second"}}]
        })

        response = parser.get_result()
        assert response.first_reasoning_chunk_index == 0
        assert response.first_content_chunk_index == 1


class TestGeminiParsing:
    """Test Gemini 格式Parse"""

    def test_parse_gemini_with_parts(self):
        """Parse Gemini parts 格式"""
        parser = UnifiedResponseParser("gemini")

        chunk = {
            "choices": [{
                "delta": {
                    "parts": [
                        {"thought": "Thinking process"},
                        {"text": "Answer text"}
                    ]
                },
                "finish_reason": None
            }]
        }

        result = parser.parse_chunk(chunk)
        assert result.reasoning == "Thinking process"
        assert result.content == "Answer text"

    def test_parse_gemini_without_parts(self):
        """Gemini no parts 时use标准格式"""
        parser = UnifiedResponseParser("gemini")

        # Gemini use text 字段作is内容
        chunk = {
            "choices": [{
                "delta": {"text": "Standard content"},
                "finish_reason": None
            }]
        }

        result = parser.parse_chunk(chunk)
        assert result.content == "Standard content"

    def test_parse_gemini_multiple_chunks(self):
        """Parse多 Gemini 块"""
        parser = UnifiedResponseParser("gemini")

        parser.parse_chunk({
            "choices": [{"delta": {"parts": [{"thought": "Thinking "}]}}]
        })
        parser.parse_chunk({
            "choices": [{"delta": {"parts": [{"thought": "more..."}]}}]
        })
        parser.parse_chunk({
            "choices": [{"delta": {"parts": [{"text": "Answer"}]}}]
        })

        response = parser.get_result()
        assert response.full_reasoning == "Thinking more..."
        assert response.full_content == "Answer"


class TestParseStreamResponse:
    """Test便捷函数 parse_stream_response"""

    def test_parse_standard_stream(self):
        """Parse标准平台流"""
        chunks = [
            {"choices": [{"delta": {"content": "Hello "}}]},
            {"choices": [{"delta": {"content": "world"}}]}
        ]

        result = parse_stream_response(chunks, "mimo")
        assert result.full_content == "Hello world"
        assert result.total_chunks == 2

    def test_parse_reasoning_stream(self):
        """Parse推理Model流"""
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Let me think"}}]},
            {"choices": [{"delta": {"content": "The answer is 42"}}]}
        ]

        result = parse_stream_response(chunks, "deepseek")
        assert result.full_reasoning == "Let me think"
        assert result.full_content == "The answer is 42"

    def test_parse_gemini_stream(self):
        """Parse Gemini 流"""
        chunks = [
            {"choices": [{"delta": {"parts": [{"thought": "Hmm"}]}}]},
            {"choices": [{"delta": {"parts": [{"text": "Got it"}]}}]}
        ]

        result = parse_stream_response(chunks, "gemini")
        assert result.full_reasoning == "Hmm"
        assert result.full_content == "Got it"

    def test_parse_with_usage(self):
        """Parse带 usage 流"""
        chunks = [
            {"choices": [{"delta": {"content": "Test"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 100}
            }
        ]

        result = parse_stream_response(chunks, "mimo")
        assert result.usage["total_tokens"] == 100
        assert result.finish_reason == "stop"

    def test_parse_empty_stream(self):
        """Parse空流"""
        result = parse_stream_response([], "mimo")
        assert result.full_content == ""
        assert result.total_chunks == 0


class TestRawSnapshots:
    """Test原始快照Save"""

    def test_snapshots_within_limit(self):
        """快照数量in限制内"""
        parser = UnifiedResponseParser("mimo", max_snapshots=3)

        for i in range(3):
            parser.parse_chunk({"choices": [{"delta": {"content": str(i)}}]})

        response = parser.get_result()
        assert len(response.raw_snapshots) == 3

    def test_snapshots_exceed_limit(self):
        """快照数量超过限制"""
        parser = UnifiedResponseParser("mimo", max_snapshots=2)

        for i in range(5):
            parser.parse_chunk({"choices": [{"delta": {"content": str(i)}}]})

        response = parser.get_result()
        # 只Save前2
        assert len(response.raw_snapshots) == 2

    def test_raw_chunk_in_parsed_result(self):
        """Parse result包含原始块"""
        parser = UnifiedResponseParser("mimo", max_snapshots=2)

        raw_chunk = {"choices": [{"delta": {"content": "test"}}]}
        parsed = parser.parse_chunk(raw_chunk)

        # 一块应该has raw_chunk
        assert parsed.raw_chunk == raw_chunk

        # 超过限制后not应该Save
        for i in range(10):
            parser.parse_chunk({"choices": [{"delta": {"content": str(i)}}]})

        parsed_late = parser.parse_chunk({"choices": [{"delta": {"content": "late"}}]})
        assert parsed_late.raw_chunk is None
