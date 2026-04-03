"""
LLM-as-Judge 评估系统

use强大 LLM 作is评判者，评估开放式问题回答质量。
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from core.providers.factory import get_provider


class JudgeCriteria(Enum):
    """Evaluation Criteria"""
    HELPFULNESS = "helpfulness"      # is否hasHelp
    RELEVANCE = "relevance"          # is否相关
    ACCURACY = "accuracy"            # 准确性
    COHERENCE = "coherence"          # 连贯性
    COMPLETENESS = "completeness"    # 完整性
    CLARITY = "clarity"              # 清晰度


@dataclass
class JudgeRequest:
    """裁判请求"""
    question: str                      # 问题
    answer: str                         # 回答
    reference_answer: Optional[str] = None  # 参考Answer（optional）
    criteria: List[JudgeCriteria] = field(default_factory=lambda: [
        JudgeCriteria.HELPFULNESS,
        JudgeCriteria.RELEVANCE,
        JudgeCriteria.ACCURACY,
        JudgeCriteria.COHERENCE
    ])
    context: Optional[str] = None      # 额外onunder文
    max_score: int = 10                 # 最大分数


@dataclass
class JudgeResult:
    """裁判Result"""
    score: float                        # 总分 (0-max_score)
    reasoning: str                      # Score理由
    category_scores: Dict[str, float] = field(default_factory=dict)  # 各维度分数
    confidence: float = 0.8             # 置信度 (0-1)
    suggestion: Optional[str] = None     # 改进Suggestion
    timestamp: datetime = field(default_factory=datetime.now)


class LLMJudge:
    """
    LLM 裁判器

    use强大 LLM (如 GPT-4o) 作is评判者，评估回答质量。
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o",
        judge_api_base: str = "https://api.openai.com/v1",
        judge_api_key: str = "",
        temperature: float = 0.3  # 低温度以保持一致性
    ):
        """
        Initialize LLM 裁判器

        Args:
            judge_model: 裁判Model ID
            judge_api_base: API Base URL
            judge_api_key: API 密钥
            temperature: Generate温度
        """
        self.judge_model = judge_model
        self.judge_api_base = judge_api_base
        self.judge_api_key = judge_api_key
        self.temperature = temperature

        # Create裁判用 provider
        self.provider = get_provider(
            "OpenAI",  # 假设use OpenAI 兼容接口
            judge_api_base,
            judge_api_key,
            judge_model
        )

    def _build_judge_prompt(self, request: JudgeRequest) -> str:
        """
        Build裁判Tip词

        Args:
            request: 裁判请求

        Returns:
            FormatTip词
        """
        criteria_desc = {
            JudgeCriteria.HELPFULNESS: "回答is否直接Help解决问题",
            JudgeCriteria.RELEVANCE: "回答is否与问题相关，没has偏离主题",
            JudgeCriteria.ACCURACY: "回答in事实信息is否准确no误",
            JudgeCriteria.COHERENCE: "回答is否逻辑连贯，前后一致",
            JudgeCriteria.COMPLETENESS: "回答is否完整，没has遗漏重要内容",
            JudgeCriteria.CLARITY: "回答is否清晰易懂，表达准确"
        }

        criteria_list = [c.value for c in request.criteria]

        prompt_parts = [
            "你is一专业内容Quality Assessment专家。请based on以under标准评估回答质量：\n",
            "Evaluation Criteria：\n"
        ]

        for criterion in request.criteria:
            prompt_parts.append(f"- **{criterion.value}**: {criteria_desc[criterion]}\n")

        prompt_parts.append(f"\nScore范围: 0-{request.max_score} 分\n")

        prompt_parts.append("\n**问题**:\n")
        prompt_parts.append(request.question)

        if request.context:
            prompt_parts.append("\n\n**onunder文**:\n")
            prompt_parts.append(request.context)

        if request.reference_answer:
            prompt_parts.append("\n\n**参考Answer**:\n")
            prompt_parts.append(request.reference_answer)

        prompt_parts.append("\n\n**待评估回答**:\n")
        prompt_parts.append(request.answer)

        prompt_parts.append("\n\n---")
        prompt_parts.append(f"\n请以 JSON 格式ReturnEvaluation result，包含以under字段：")
        prompt_parts.append("```json")
        prompt_parts.append("{")
        prompt_parts.append(f'  "total_score": <总分, 0-{request.max_score}>,')
        prompt_parts.append('  "reasoning": "<Score理由，解释is什么给这分数>",')
        prompt_parts.append('  "category_scores": {')
        for i, criterion in enumerate(criteria_list):
            if i < len(criteria_list) - 1:
                prompt_parts.append(f'    "{criterion}": <分数>,')
            else:
                prompt_parts.append(f'    "{criterion}": <分数>')
        prompt_parts.append('  },')
        prompt_parts.append('  "suggestion": "<改进Suggestion，optional>",')
        prompt_parts.append('  "confidence": <置信度 0-1>')
        prompt_parts.append("}")
        prompt_parts.append("```")

        return "".join(prompt_parts)

    def _parse_judge_response(self, response_text: str, max_score: int) -> JudgeResult:
        """
        Parse裁判响应

        Args:
            response_text: LLM Return文本
            max_score: 最大分数

        Returns:
            裁判Result
        """
        try:
            # 尝试提取 JSON
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "{" in response_text:
                # 找到 JSON 对象（从一 { 开始）
                # use括号匹配找到完整 JSON 对象
                start = response_text.find("{")
                # 找到匹配右括号
                depth = 0
                for i, char in enumerate(response_text[start:], start=start):
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = response_text[start:i+1]
                            break
                else:
                    # No matching braces found
                    raise ValueError("No matching braces found")
            else:
                raise ValueError("No JSON format found in response")

            data = json.loads(json_str)

            return JudgeResult(
                score=float(data.get("total_score", 0)),
                reasoning=data.get("reasoning", ""),
                category_scores={
                    k: float(v) for k, v in data.get("category_scores", {}).items()
                },
                confidence=float(data.get("confidence", 0.8)),
                suggestion=data.get("suggestion")
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Parse failed，ReturndefaultResult
            return JudgeResult(
                score=0.0,
                reasoning=f"Parse failed: {str(e)}\nRaw response: {response_text[:200]}",
                category_scores={},
                confidence=0.0
            )

    async def evaluate(
        self,
        request: JudgeRequest,
        log_callback: Optional[callable] = None
    ) -> JudgeResult:
        """
        评估回答质量

        Args:
            request: 裁判请求
            log_callback: LogCallback函数

        Returns:
            裁判Result
        """
        # BuildTip词
        prompt = self._build_judge_prompt(request)

        if log_callback:
            log_callback(f"LLM Judge: currently评估回答...")

        try:
            # 调用 LLM
            result = await self.provider.get_completion(
                client=None,  # use provider 内部 client
                session_id="judge",
                prompt=prompt,
                max_tokens=1000,
                log_callback=log_callback
            )

            if result.get("error"):
                raise Exception(result["error"])

            response_content = result.get("full_response_content", "")

            # Parse result
            judge_result = self._parse_judge_response(response_content, request.max_score)

            if log_callback:
                log_callback(f"LLM Judge: 评估完成，Score: {judge_result.score}/{request.max_score}")

            return judge_result

        except Exception as e:
            if log_callback:
                log_callback(f"LLM Judge: 评估失败 - {str(e)}")

            # Return失败Result
            return JudgeResult(
                score=0.0,
                reasoning=f"评估失败: {str(e)}",
                category_scores={},
                confidence=0.0
            )

    async def evaluate_batch(
        self,
        requests: List[JudgeRequest],
        log_callback: Optional[callable] = None
    ) -> List[JudgeResult]:
        """
        批量评估

        Args:
            requests: 裁判请求列表
            log_callback: LogCallback函数

        Returns:
            裁判Result列表
        """
        results = []

        for i, request in enumerate(requests):
            if log_callback:
                log_callback(f"LLM Judge: 评估进度 {i+1}/{len(requests)}")

            result = await self.evaluate(request, log_callback)
            results.append(result)

        return results

    def compare_answers(
        self,
        question: str,
        answers: List[str],
        criteria: Optional[List[JudgeCriteria]] = None
    ) -> List[Tuple[int, str, JudgeResult]]:
        """
        比较多回答质量

        Args:
            question: 问题
            answers: 回答列表
            criteria: Evaluation Criteria（optional）

        Returns:
            Sort后 [(Index, 回答, Evaluation result)]，按分数降序
        """
        # is每回答Create请求
        requests = [
            JudgeRequest(
                question=question,
                answer=answer,
                criteria=criteria or list(JudgeCriteria)[:4]
            )
            for answer in answers
        ]

        # 执行评估 (needinAsynconunder文in调用 evaluate_batch)
        return requests


# 便捷函数
async def judge_answer(
    question: str,
    answer: str,
    judge_model: str = "gpt-4o",
    api_key: str = ""
) -> JudgeResult:
    """
    便捷函数：评估单回答

    Args:
        question: 问题
        answer: 回答
        judge_model: 裁判Model
        api_key: API 密钥

    Returns:
        Evaluation result
    """
    judge = LLMJudge(
        judge_model=judge_model,
        judge_api_key=api_key
    )

    request = JudgeRequest(
        question=question,
        answer=answer
    )

    return await judge.evaluate(request)


async def judge_answers_batch(
    question: str,
    answers: List[str],
    judge_model: str = "gpt-4o",
    api_key: str = ""
) -> List[JudgeResult]:
    """
    便捷函数：批量评估多回答

    Args:
        question: 问题
        answers: 回答列表
        judge_model: 裁判Model
        api_key: API 密钥

    Returns:
        Evaluation result列表
    """
    judge = LLMJudge(
        judge_model=judge_model,
        judge_api_key=api_key
    )

    requests = [
        JudgeRequest(question=question, answer=answer)
        for answer in answers
    ]

    return await judge.evaluate_batch(requests)
