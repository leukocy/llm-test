"""
一致性Test系统 (Consistency Testing System)

评估Model对同一问题回答一致性，检测Answer稳定性。
"""

import asyncio
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class ConsistencyRunResult:
    """单次运行Result"""
    run_id: int
    predicted_answer: str
    is_correct: bool
    response_time_ms: float
    reasoning_content: str = ""
    full_response: str = ""


@dataclass
class ConsistencyTestResult:
    """一致性Test Results"""
    sample_id: str
    question: str
    correct_answer: str

    # 运行Result
    runs: list[ConsistencyRunResult] = field(default_factory=list)

    # 一致性指标
    consistency_rate: float = 0.0          # Answer一致率 (0-1)
    accuracy_rate: float = 0.0             # Accuracy (0-1)
    majority_answer: str = ""              # 多数Answer
    majority_is_correct: bool = False      # 多数Answeris否正确
    answer_variants: dict[str, int] = field(default_factory=dict)  # Answer分布

    # 稳定性指标
    response_time_mean: float = 0.0
    response_time_std: float = 0.0

    # 诊断
    is_stable: bool = False                # is否稳定（一致率 > 阈值）
    instability_note: str = ""


@dataclass
class ConsistencyReport:
    """一致性Test Report"""
    model_id: str
    test_timestamp: str
    total_samples: int
    runs_per_sample: int

    # 汇总指标
    overall_consistency: float = 0.0       # 整体一致率
    overall_accuracy: float = 0.0          # 整体Accuracy
    stable_sample_count: int = 0           # 稳定Sample count
    unstable_sample_count: int = 0         # not稳定Sample count

    # Detailed Results
    results: list[ConsistencyTestResult] = field(default_factory=list)

    # 分析
    most_unstable_samples: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class ConsistencyTester:
    """
    Consistency Tester

    对同一问题进行多次Test，评估Model answer稳定性。

    Usage:
        tester = ConsistencyTester(runs_per_sample=5)

        # Test单 samples
        result = await tester.test_single(
            sample_id="001",
            question="What is 2 + 2?",
            correct_answer="4",
            get_response_func=my_api_call
        )

        print(result.consistency_rate)
        print(result.majority_answer)

        # Batch Test
        report = await tester.test_batch(samples, get_response_func)
    """

    def __init__(
        self,
        runs_per_sample: int = 3,
        stability_threshold: float = 0.8,
        temperature_variation: bool = False
    ):
        """
        InitializeConsistency Tester

        Args:
            runs_per_sample: 每 samplesTest次数
            stability_threshold: 稳定性阈值（一致率高于此值视is稳定）
            temperature_variation: is否innot同Runningusenot同 temperature
        """
        self.runs_per_sample = runs_per_sample
        self.stability_threshold = stability_threshold
        self.temperature_variation = temperature_variation

    async def test_single(
        self,
        sample_id: str,
        question: str,
        correct_answer: str,
        get_response_func: Callable,
        answer_parser: Callable = None
    ) -> ConsistencyTestResult:
        """
        Test单 samples一致性

        Args:
            sample_id: 样本 ID
            question: 问题
            correct_answer: Correct answer
            get_response_func: Get响应Async函数 async (prompt) -> dict/str
            answer_parser: AnswerParse函数 (response) -> str

        Returns:
            ConsistencyTestResult
        """
        result = ConsistencyTestResult(
            sample_id=sample_id,
            question=question,
            correct_answer=correct_answer
        )

        runs = []
        answers = []
        response_times = []

        # 多次运行
        for run_id in range(self.runs_per_sample):
            try:
                import time
                start_time = time.time()

                response_data = await get_response_func(question)

                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)

                # Parse响应
                if isinstance(response_data, dict):
                    full_response = response_data.get('content', '')
                    reasoning = response_data.get('reasoning_content', '')
                else:
                    full_response = str(response_data)
                    reasoning = ""

                # ParseAnswer
                if answer_parser:
                    predicted = answer_parser(full_response)
                else:
                    predicted = self._simple_answer_extract(full_response)

                # Check正确性
                is_correct = self._check_answer(predicted, correct_answer)

                answers.append(predicted)
                runs.append(ConsistencyRunResult(
                    run_id=run_id,
                    predicted_answer=predicted,
                    is_correct=is_correct,
                    response_time_ms=response_time,
                    reasoning_content=reasoning,
                    full_response=full_response[:1000]  # 限制长度
                ))

            except Exception as e:
                runs.append(ConsistencyRunResult(
                    run_id=run_id,
                    predicted_answer="",
                    is_correct=False,
                    response_time_ms=0,
                    full_response=f"Error: {str(e)}"
                ))
                answers.append("")

        result.runs = runs

        # Calculate一致性指标
        self._calculate_consistency_metrics(result, answers, response_times)

        return result

    def _calculate_consistency_metrics(
        self,
        result: ConsistencyTestResult,
        answers: list[str],
        response_times: list[float]
    ):
        """Calculate一致性指标"""
        if not answers:
            return

        # Answer分布
        from collections import Counter
        answer_counts = Counter(answers)
        result.answer_variants = dict(answer_counts)

        # 多数Answer
        most_common = answer_counts.most_common(1)
        if most_common:
            result.majority_answer = most_common[0][0]
            majority_count = most_common[0][1]
            result.consistency_rate = majority_count / len(answers)

        # 多数Answeris否正确
        result.majority_is_correct = self._check_answer(
            result.majority_answer, result.correct_answer
        )

        # Accuracy
        correct_count = sum(1 for run in result.runs if run.is_correct)
        result.accuracy_rate = correct_count / len(result.runs) if result.runs else 0

        # 响应时间Statistics
        if response_times:
            result.response_time_mean = statistics.mean(response_times)
            if len(response_times) > 1:
                result.response_time_std = statistics.stdev(response_times)

        # 稳定性判断
        result.is_stable = result.consistency_rate >= self.stability_threshold

        if not result.is_stable:
            result.instability_note = self._diagnose_instability(result)

    def _diagnose_instability(self, result: ConsistencyTestResult) -> str:
        """诊断not稳定原因"""
        variants = len(result.answer_variants)

        if variants == len(result.runs):
            return "每次运行都产生not同Answer，Model高度not稳定"
        elif variants > len(result.runs) / 2:
            return f"Answer变体过多（{variants}种），Model决策not够确定"
        elif result.accuracy_rate > 0 and result.accuracy_rate < 1:
            return "部分运行正确、部分Error，说明Model对此问题理解not稳定"
        else:
            return "Answer一致but可能not够稳定"

    def _simple_answer_extract(self, response: str) -> str:
        """简单Answer提取"""
        import re

        # 尝试提取数字
        numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        # 尝试提取Options
        choices = re.findall(r'\b([A-E])\b', response.upper())
        if choices:
            return choices[0]

        # Return响应最后一部分
        lines = response.strip().split('\n')
        return lines[-1][:100] if lines else ""

    def _check_answer(self, predicted: str, correct: str) -> bool:
        """CheckAnsweris否正确"""
        if not predicted or not correct:
            return False

        pred_clean = predicted.strip().lower().replace(',', '')
        correct_clean = correct.strip().lower().replace(',', '')

        # 直接比较
        if pred_clean == correct_clean:
            return True

        # 数值比较
        try:
            pred_num = float(pred_clean)
            correct_num = float(correct_clean)
            return abs(pred_num - correct_num) < 0.01
        except:
            pass

        return False

    async def test_batch(
        self,
        samples: list[dict[str, Any]],
        get_response_func: Callable,
        answer_parser: Callable = None,
        concurrency: int = 4,
        progress_callback: Callable = None
    ) -> ConsistencyReport:
        """
        批量一致性Test

        Args:
            samples: 样本列表，每 samples应包含 sample_id, question, correct_answer
            get_response_func: Get响应函数
            answer_parser: AnswerParse函数
            concurrency: Concurrency
            progress_callback: 进度Callback (current, total)

        Returns:
            ConsistencyReport
        """
        report = ConsistencyReport(
            model_id="",
            test_timestamp=datetime.now().isoformat(),
            total_samples=len(samples),
            runs_per_sample=self.runs_per_sample
        )

        semaphore = asyncio.Semaphore(concurrency)
        completed = 0

        async def test_with_semaphore(sample: dict) -> ConsistencyTestResult:
            nonlocal completed
            async with semaphore:
                result = await self.test_single(
                    sample_id=sample.get('sample_id', str(completed)),
                    question=sample.get('question', ''),
                    correct_answer=sample.get('correct_answer', ''),
                    get_response_func=get_response_func,
                    answer_parser=answer_parser
                )
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(samples))
                return result

        tasks = [test_with_semaphore(s) for s in samples]
        results = await asyncio.gather(*tasks)

        report.results = list(results)

        # Calculate汇总指标
        self._calculate_report_metrics(report)

        return report

    def _calculate_report_metrics(self, report: ConsistencyReport):
        """Calculate报告汇总指标"""
        if not report.results:
            return

        # 整体一致率
        consistency_rates = [r.consistency_rate for r in report.results]
        report.overall_consistency = statistics.mean(consistency_rates)

        # 整体Accuracy
        accuracy_rates = [r.accuracy_rate for r in report.results]
        report.overall_accuracy = statistics.mean(accuracy_rates)

        # 稳定/not稳定样本
        report.stable_sample_count = sum(1 for r in report.results if r.is_stable)
        report.unstable_sample_count = len(report.results) - report.stable_sample_count

        # 最not稳定样本
        unstable = sorted(
            [r for r in report.results if not r.is_stable],
            key=lambda x: x.consistency_rate
        )
        report.most_unstable_samples = [r.sample_id for r in unstable[:5]]

        # GenerateSuggestion
        report.recommendations = self._generate_recommendations(report)

    def _generate_recommendations(self, report: ConsistencyReport) -> list[str]:
        """Generate改进Suggestion"""
        recommendations = []

        if report.overall_consistency < 0.7:
            recommendations.append("整体一致性较低，Suggestion降低 temperature 参数")

        if report.unstable_sample_count > report.total_samples * 0.3:
            recommendations.append("超过 30% 样本not稳定，SuggestionCheck prompt 设计")

        if report.overall_accuracy < 0.5:
            recommendations.append("Accuracy较低，need优化 prompt or更换Model")

        if report.overall_consistency > 0.9 and report.overall_accuracy < 0.7:
            recommendations.append("Model一致地给出ErrorAnswer，可能存in系统性偏差")

        if not recommendations:
            recommendations.append("Model一致性andAccuracy表现Good")

        return recommendations


def create_consistency_tester(
    runs: int = 3,
    threshold: float = 0.8
) -> ConsistencyTester:
    """Factory函数：CreateConsistency Tester"""
    return ConsistencyTester(
        runs_per_sample=runs,
        stability_threshold=threshold
    )
