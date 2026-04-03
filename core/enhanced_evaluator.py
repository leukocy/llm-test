"""
增强型Evaluator (Enhanced Evaluator)

集成智能AnswerParse器and推理Quality Evaluator到评估流程in。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from .metrics import ThinkingMetrics, ThinkingMetricsResult
from .reasoning_evaluator import ReasoningEvaluationResult, ReasoningQualityEvaluator
from .response_parser import ParsedResponse, UnifiedResponseParser

# Import核心模块
from .smart_answer_parser import AnswerType, ParseResult, SmartAnswerParser, compare_answers


@dataclass
class EnhancedEvaluationResult:
    """增强型Evaluation result"""
    # 基础Result
    is_correct: bool
    predicted_answer: str
    correct_answer: str

    # Parse详情
    parse_result: ParseResult

    # 推理评估
    reasoning_eval: ReasoningEvaluationResult | None = None

    # Performance Metrics
    metrics: ThinkingMetricsResult | None = None

    # 原始内容
    reasoning_content: str = ""
    final_content: str = ""


class EnhancedEvaluator:
    """
    增强型Evaluator

    集成：
    1. 智能AnswerParse（规则 + LLM 兜底）
    2. Reasoning processQuality Assessment
    3. Performance Metrics收集
    4. 统一响应Parse

    Usage:
        evaluator = EnhancedEvaluator(platform="mimo")

        result = await evaluator.evaluate(
            question="What is 2+2?",
            response_chunks=stream_chunks,  # or full_response
            correct_answer="4",
            answer_type=AnswerType.NUMBER,
            llm_func=my_llm_call  # optional，用于 LLM 兜底
        )

        print(result.is_correct)
        print(result.reasoning_eval.quality_score.overall)
    """

    def __init__(
        self,
        platform: str = "unknown",
        enable_reasoning_eval: bool = True,
        enable_llm_fallback: bool = False,
        llm_fallback_threshold: float = 0.6
    ):
        """
        Initialize增强型Evaluator

        Args:
            platform: 平台标识（用于响应Parse）
            enable_reasoning_eval: is否启用推理Quality Assessment
            enable_llm_fallback: is否启用 LLM Parse兜底
            llm_fallback_threshold: LLM 兜底置信度阈值
        """
        self.platform = platform
        self.enable_reasoning_eval = enable_reasoning_eval
        self.enable_llm_fallback = enable_llm_fallback

        # Initialize组件
        self.answer_parser = SmartAnswerParser(llm_fallback_threshold)
        self.reasoning_evaluator = ReasoningQualityEvaluator()
        self.response_parser = UnifiedResponseParser(platform)
        self.metrics = ThinkingMetrics(platform=platform)

    async def evaluate(
        self,
        question: str,
        correct_answer: str,
        answer_type: AnswerType = AnswerType.NUMBER,
        response_chunks: list = None,
        full_response: str = None,
        reasoning_content: str = None,
        llm_func: Callable = None
    ) -> EnhancedEvaluationResult:
        """
        执行增强型评估

        Args:
            question: 问题
            correct_answer: Correct answer
            answer_type: Answer类型
            response_chunks: 流式响应块列表（optional）
            full_response: 完整响应（optional，与 response_chunks 二选一）
            reasoning_content: 推理内容（optional，if response_chunks innot包含）
            llm_func: LLM 调用函数（用于兜底Parse）

        Returns:
            EnhancedEvaluationResult
        """
        final_content = ""
        reasoning = reasoning_content or ""

        # 1. Parse响应
        if response_chunks:
            parsed = self._parse_stream_response(response_chunks)
            final_content = parsed.full_content
            reasoning = reasoning or parsed.full_reasoning
        elif full_response:
            final_content = full_response

        # 2. ParseAnswer
        if self.enable_llm_fallback and llm_func:
            parse_result = await self.answer_parser.parse_with_llm_fallback(
                final_content, answer_type, llm_func, correct_answer
            )
        else:
            parse_result = self.answer_parser.parse(final_content, answer_type, correct_answer)

        # 3. 比较Answer
        is_correct, similarity = compare_answers(
            parse_result.normalized_value,
            correct_answer,
            answer_type
        )

        # 4. 评估推理质量
        reasoning_eval = None
        if self.enable_reasoning_eval and reasoning:
            reasoning_eval = self.reasoning_evaluator.evaluate(
                question=question,
                reasoning=reasoning,
                final_answer=parse_result.extracted_answer,
                correct_answer=correct_answer,
                is_answer_correct=is_correct
            )

        # 5. 收集Performance Metrics
        metrics_result = self.metrics.calculate() if hasattr(self.metrics, '_request_start') and self.metrics._request_start else None

        return EnhancedEvaluationResult(
            is_correct=is_correct,
            predicted_answer=parse_result.extracted_answer,
            correct_answer=correct_answer,
            parse_result=parse_result,
            reasoning_eval=reasoning_eval,
            metrics=metrics_result,
            reasoning_content=reasoning,
            final_content=final_content
        )

    def _parse_stream_response(self, chunks: list) -> ParsedResponse:
        """Parse流式响应"""
        self.response_parser.reset()
        for chunk in chunks:
            self.response_parser.parse_chunk(chunk)
        return self.response_parser.get_result()

    def evaluate_sync(
        self,
        question: str,
        correct_answer: str,
        answer_type: AnswerType = AnswerType.NUMBER,
        full_response: str = None,
        reasoning_content: str = None
    ) -> EnhancedEvaluationResult:
        """
        SyncVersion评估（notuse LLM 兜底）

        Args:
            question: 问题
            correct_answer: Correct answer
            answer_type: Answer类型
            full_response: 完整响应
            reasoning_content: 推理内容

        Returns:
            EnhancedEvaluationResult
        """
        final_content = full_response or ""
        reasoning = reasoning_content or ""

        # ParseAnswer
        parse_result = self.answer_parser.parse(final_content, answer_type, correct_answer)

        # 比较Answer
        is_correct, _ = compare_answers(
            parse_result.normalized_value,
            correct_answer,
            answer_type
        )

        # 评估推理质量
        reasoning_eval = None
        if self.enable_reasoning_eval and reasoning:
            reasoning_eval = self.reasoning_evaluator.evaluate(
                question=question,
                reasoning=reasoning,
                final_answer=parse_result.extracted_answer,
                correct_answer=correct_answer,
                is_answer_correct=is_correct
            )

        return EnhancedEvaluationResult(
            is_correct=is_correct,
            predicted_answer=parse_result.extracted_answer,
            correct_answer=correct_answer,
            parse_result=parse_result,
            reasoning_eval=reasoning_eval,
            metrics=None,
            reasoning_content=reasoning,
            final_content=final_content
        )

    def record_request_start(self):
        """记录请求开始（用于Performance Metrics）"""
        self.metrics.record_request_start()

    def record_request_end(self):
        """记录请求结束"""
        self.metrics.record_request_end()

    def record_reasoning_chunk(self, content: str):
        """记录推理内容块"""
        self.metrics.record_reasoning_chunk(content)

    def record_content_chunk(self, content: str):
        """记录正文内容块"""
        self.metrics.record_content_chunk(content)


def create_enhanced_evaluator(
    platform: str,
    enable_reasoning_eval: bool = True,
    enable_llm_fallback: bool = False
) -> EnhancedEvaluator:
    """
    Factory函数：Create增强型Evaluator

    Args:
        platform: 平台标识
        enable_reasoning_eval: is否启用推理评估
        enable_llm_fallback: is否启用 LLM 兜底

    Returns:
        EnhancedEvaluator
    """
    return EnhancedEvaluator(
        platform=platform,
        enable_reasoning_eval=enable_reasoning_eval,
        enable_llm_fallback=enable_llm_fallback
    )
