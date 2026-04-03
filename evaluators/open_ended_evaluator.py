"""
开放式问题Evaluator

use LLM-as-Judge 评估no法用简单标准Answer评估问题。
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

import pandas as pd

from core.llm_judge import LLMJudge, JudgeCriteria, JudgeRequest, JudgeResult


@dataclass
class EvaluationConfig:
    """评估Configure"""
    judge_model: str = "gpt-4o"
    judge_api_base: str = "https://api.openai.com/v1"
    judge_api_key: str = ""
    temperature: float = 0.3
    max_score: int = 10
    criteria: List[JudgeCriteria] = field(default_factory=list)


class OpenEndedEvaluator:
    """
    开放式问题Evaluator

    用于评估没has标准Answer问题回答质量，如：
    - 创意写作
    - 推理问题
    - 主观性问题
    - 代码审查
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        InitializeEvaluator

        Args:
            config: 评估Configure
        """
        self.config = config or EvaluationConfig()

        # if没has指定标准，usedefault标准
        if not self.config.criteria:
            self.config.criteria = [
                JudgeCriteria.HELPFULNESS,
                JudgeCriteria.RELEVANCE,
                JudgeCriteria.ACCURACY,
                JudgeCriteria.COHERENCE
            ]

        # Create裁判
        self.judge = LLMJudge(
            judge_model=self.config.judge_model,
            judge_api_base=self.config.judge_api_base,
            judge_api_key=self.config.judge_api_key,
            temperature=self.config.temperature
        )

    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        reference_answer: Optional[str] = None,
        context: Optional[str] = None,
        log_callback: Optional[Callable] = None
    ) -> JudgeResult:
        """
        评估单回答

        Args:
            question: 问题
            answer: 回答
            reference_answer: 参考Answer（optional）
            context: 额外onunder文（optional）
            log_callback: LogCallback

        Returns:
            Evaluation result
        """
        request = JudgeRequest(
            question=question,
            answer=answer,
            reference_answer=reference_answer,
            context=context,
            criteria=self.config.criteria,
            max_score=self.config.max_score
        )

        return await self.judge.evaluate(request, log_callback)

    async def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        reference_answers: Optional[List[str]] = None,
        log_callback: Optional[Callable] = None
    ) -> List[JudgeResult]:
        """
        批量评估

        Args:
            questions: 问题列表
            answers: 回答列表
            reference_answers: 参考Answer列表（optional）
            log_callback: LogCallback

        Returns:
            Evaluation result列表
        """
        if len(questions) != len(answers):
            raise ValueError("问题and回答数量not匹配")

        if reference_answers and len(reference_answers) != len(questions):
            raise ValueError("参考Answer数量与问题数量not匹配")

        requests = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            request = JudgeRequest(
                question=question,
                answer=answer,
                reference_answer=reference_answers[i] if reference_answers else None,
                criteria=self.config.criteria,
                max_score=self.config.max_score
            )
            requests.append(request)

        return await self.judge.evaluate_batch(requests, log_callback)

    async def compare_models(
        self,
        questions: List[str],
        model_responses: Dict[str, List[str]],
        log_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        比较not同Model回答质量

        Args:
            questions: 问题列表
            model_responses: {Model名: [回答1, 回答2, ...]}
            log_callback: LogCallback

        Returns:
            比较Result DataFrame，列包括Model、问题、分数etc.
        """
        results = []

        for model_name, answers in model_responses.items():
            if len(answers) != len(questions):
                if log_callback:
                    log_callback(f"Warning: {model_name} 回答数量与问题数量not匹配")
                continue

            for i, (question, answer) in enumerate(zip(questions, answers)):
                result = await self.evaluate_answer(question, answer, log_callback=log_callback)

                results.append({
                    "model": model_name,
                    "question_index": i,
                    "question": question[:100] + "..." if len(question) > 100 else question,
                    "answer": answer[:100] + "..." if len(answer) > 100 else answer,
                    "score": result.score,
                    "reasoning": result.reasoning[:200] + "..." if len(result.reasoning) > 200 else result.reasoning,
                    "confidence": result.confidence,
                    "category_scores": result.category_scores
                })

        df = pd.DataFrame(results)
        return df

    def generate_report(self, results: pd.DataFrame) -> str:
        """
        Generate评估报告

        Args:
            results: Evaluation result DataFrame

        Returns:
            Markdown 格式报告
        """
        if results.empty:
            return "# 评估报告\n\nNo results"

        report = ["# LLM-as-Judge 评估报告\n"]
        report.append(f"Generate时间: {pd.Timestamp.now()}\n")

        # 总体Statistics
        report.append("## 总体Statistics\n")
        report.append(f"- 评估Sample count: {len(results)}\n")

        if "model" in results.columns:
            report.append("\n## 按ModelStatistics\n")
            model_stats = results.groupby("model")["score"].agg(["mean", "std", "count"])
            for model, row in model_stats.iterrows():
                report.append(f"\n### {model}\n")
                report.append(f"- 平均分: {row['mean']:.2f}\n")
                report.append(f"- Standard deviation: {row['std']:.2f}\n")
                report.append(f"- Sample count: {row['count']}\n")

        # 最佳and最差回答
        report.append("\n## 最佳回答 (Top 5)\n")
        top_results = results.nlargest(5, "score")
        for i, row in enumerate(top_results.itertuples(), 1):
            model = row.model if "model" in results.columns else "Unknown"
            report.append(f"\n{i}. **{model}** - {row.score:.1f}分\n")
            report.append(f"   问题: {row.question}\n")
            report.append(f"   理由: {row.reasoning}\n")

        # 分数分布
        report.append("\n## 分数分布\n")
        score_bins = [0, 3, 5, 7, 9, 10]
        results["score_range"] = pd.cut(results["score"], bins=score_bins)
        distribution = results["score_range"].value_counts().sort_index()

        for range_val, count in distribution.items():
            percentage = count / len(results) * 100
            report.append(f"- {range_val}: {count} ({percentage:.1f}%)\n")

        return "".join(report)


# 便捷函数
async def evaluate_open_ended(
    question: str,
    answer: str,
    judge_api_key: str = "",
    judge_model: str = "gpt-4o"
) -> JudgeResult:
    """
    便捷函数：评估开放式问题回答

    Args:
        question: 问题
        answer: 回答
        judge_api_key: 裁判 API 密钥
        judge_model: 裁判Model

    Returns:
        Evaluation result
    """
    config = EvaluationConfig(
        judge_model=judge_model,
        judge_api_key=judge_api_key
    )

    evaluator = OpenEndedEvaluator(config)

    return await evaluator.evaluate_answer(question, answer)
