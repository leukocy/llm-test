"""
统一Test运行器 (Unified Test Runner)

集成所hasEvaluation模块，提供完整Test执行流程。
支持从 YAML Configure启动Test，自动Generate报告。
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .failure_analyzer import FailureAnalysisReport, FailureAnalyzer
from .reasoning_evaluator import ReasoningQualityEvaluator
from .retry_handler import RetryHandler
from .smart_answer_parser import AnswerType, SmartAnswerParser

# Import核心模块
from .test_config import TestConfig, TestConfigLoader


@dataclass
class TestRunResult:
    """单次Test运行Result"""
    sample_id: str
    question: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool

    # 性能
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    ttut_ms: float = 0.0

    # Token
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    # 推理质量
    reasoning_quality: float = 0.0
    answer_confidence: float = 0.0

    # 原始内容
    reasoning_content: str = ""
    full_response: str = ""

    # Error
    error: str | None = None


@dataclass
class ModelTestResult:
    """单 modelsTest Results"""
    model_id: str
    platform: str

    # 总体指标
    accuracy: float = 0.0
    total_samples: int = 0
    correct_samples: int = 0
    error_samples: int = 0

    # 性能Statistics
    avg_latency_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    avg_ttut_ms: float = 0.0
    avg_tps: float = 0.0

    # Token Statistics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    avg_reasoning_ratio: float = 0.0

    # 推理质量
    avg_reasoning_quality: float = 0.0

    # 成本估算
    estimated_cost_usd: float = 0.0

    # Detailed Results
    results: list[TestRunResult] = field(default_factory=list)

    # 失败分析
    failure_report: FailureAnalysisReport | None = None


@dataclass
class TestSuiteResult:
    """完整Test套件Result"""
    test_name: str
    test_config: TestConfig
    start_time: str
    end_time: str
    duration_seconds: float

    # 各ModelResult
    model_results: dict[str, ModelTestResult] = field(default_factory=dict)

    # 比较Result
    best_accuracy_model: str = ""
    best_latency_model: str = ""
    best_quality_model: str = ""

    # 汇总
    summary: dict[str, Any] = field(default_factory=dict)


class UnifiedTestRunner:
    """
    统一Test运行器

    集成所hasEvaluation模块，提供完整Test执行流程。

    Usage:
        # 从 YAML Configure运行
        runner = UnifiedTestRunner()
        result = await runner.run_from_config("tests/config/gsm8k_comparison.yaml")

        # 手动Configure运行
        runner = UnifiedTestRunner()
        result = await runner.run(
            config=my_config,
            samples=my_samples,
            get_response_funcs={
                "mimo": mimo_api_call,
                "deepseek": deepseek_api_call
            }
        )
    """

    def __init__(
        self,
        config_dir: str = "tests/config",
        report_dir: str = "reports"
    ):
        self.config_loader = TestConfigLoader(config_dir)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Initialize组件
        self.answer_parser = SmartAnswerParser()
        self.reasoning_evaluator = ReasoningQualityEvaluator()
        self.failure_analyzer = FailureAnalyzer()
        self.retry_handler = RetryHandler()

    async def run_from_config(
        self,
        config_path: str,
        samples: list[dict[str, Any]],
        get_response_funcs: dict[str, Callable],
        progress_callback: Callable | None = None
    ) -> TestSuiteResult:
        """
        从Configure文件Run Test

        Args:
            config_path: ConfigureFile path
            samples: Test样本列表
            get_response_funcs: 各Model响应函数 {model_id: async func}
            progress_callback: 进度Callback

        Returns:
            TestSuiteResult
        """
        config = self.config_loader.load(config_path)
        return await self.run(config, samples, get_response_funcs, progress_callback)

    async def run(
        self,
        config: TestConfig,
        samples: list[dict[str, Any]],
        get_response_funcs: dict[str, Callable],
        progress_callback: Callable | None = None
    ) -> TestSuiteResult:
        """
        Execute test

        Args:
            config: Test Configuration
            samples: 样本列表，每 samples应包含 sample_id, question, correct_answer
            get_response_funcs: 各Model响应函数
            progress_callback: 进度Callback (current, total, message)

        Returns:
            TestSuiteResult
        """
        start_time = datetime.now()

        result = TestSuiteResult(
            test_name=config.name,
            test_config=config,
            start_time=start_time.isoformat(),
            end_time="",
            duration_seconds=0.0
        )

        # 限制Sample count
        if config.dataset.samples > 0:
            samples = samples[:config.dataset.samples]

        total_tasks = len(samples) * len(config.models)
        current_task = 0

        # Test每 models
        for model_config in config.models:
            model_key = f"{model_config.platform}:{model_config.model_id}"

            if progress_callback:
                progress_callback(current_task, total_tasks, f"Testing {model_config.model_id}...")

            # Get响应函数
            response_func = get_response_funcs.get(model_config.model_id) or \
                           get_response_funcs.get(model_config.platform)

            if not response_func:
                print(f"Warning: No response function for {model_key}")
                continue

            # Run Test
            model_result = await self._test_model(
                model_config=model_config,
                samples=samples,
                get_response_func=response_func,
                config=config,
                progress_callback=lambda c, t: progress_callback(
                    current_task + c, total_tasks, f"{model_config.model_id}: {c}/{t}"
                ) if progress_callback else None
            )

            result.model_results[model_key] = model_result
            current_task += len(samples)

        # Calculate比较Result
        self._calculate_comparison(result)

        # 完成
        end_time = datetime.now()
        result.end_time = end_time.isoformat()
        result.duration_seconds = (end_time - start_time).total_seconds()

        # Generate报告
        await self._generate_reports(result, config)

        return result

    async def _test_model(
        self,
        model_config,
        samples: list[dict[str, Any]],
        get_response_func: Callable,
        config: TestConfig,
        progress_callback: Callable | None = None
    ) -> ModelTestResult:
        """Test单 models"""
        model_result = ModelTestResult(
            model_id=model_config.model_id,
            platform=model_config.platform,
            total_samples=len(samples)
        )

        semaphore = asyncio.Semaphore(config.concurrency)
        completed = 0

        async def test_sample(sample: dict) -> TestRunResult:
            nonlocal completed
            async with semaphore:
                result = await self._evaluate_single(
                    sample=sample,
                    get_response_func=get_response_func,
                    config=config
                )
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(samples))
                return result

        tasks = [test_sample(s) for s in samples]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ProcessResult
        for r in results:
            if isinstance(r, Exception):
                model_result.results.append(TestRunResult(
                    sample_id="error",
                    question="",
                    correct_answer="",
                    predicted_answer="",
                    is_correct=False,
                    error=str(r)
                ))
                model_result.error_samples += 1
            else:
                model_result.results.append(r)
                if r.is_correct:
                    model_result.correct_samples += 1
                if r.error:
                    model_result.error_samples += 1

        # CalculateStatistics
        self._calculate_model_stats(model_result)

        # 失败分析
        if config.metrics.reasoning_quality:
            failed_samples = [
                {
                    "sample_id": r.sample_id,
                    "question": r.question,
                    "correct_answer": r.correct_answer,
                    "predicted_answer": r.predicted_answer,
                    "model_response": r.full_response,
                    "reasoning_content": r.reasoning_content,
                    "error": r.error
                }
                for r in model_result.results if not r.is_correct
            ]
            if failed_samples:
                model_result.failure_report = self.failure_analyzer.analyze_batch(
                    failed_samples, len(samples)
                )

        return model_result

    async def _evaluate_single(
        self,
        sample: dict[str, Any],
        get_response_func: Callable,
        config: TestConfig
    ) -> TestRunResult:
        """评估单 samples"""
        sample_id = sample.get('sample_id', '')
        question = sample.get('question', '')
        correct_answer = sample.get('correct_answer', '')

        result = TestRunResult(
            sample_id=sample_id,
            question=question,
            correct_answer=correct_answer,
            predicted_answer="",
            is_correct=False
        )

        try:
            # 计时
            start_time = time.time()

            # 调用 API（带重试）
            response_data = await self.retry_handler.execute_async(
                get_response_func, question
            )

            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms

            # Parse响应
            if response_data.success:
                data = response_data.result
                if isinstance(data, dict):
                    result.full_response = data.get('content', '')
                    result.reasoning_content = data.get('reasoning_content', '')
                    result.ttft_ms = data.get('ttft_ms', 0)
                    result.ttut_ms = data.get('ttut_ms', 0)
                    result.input_tokens = data.get('input_tokens', 0)
                    result.output_tokens = data.get('output_tokens', 0)
                    result.reasoning_tokens = data.get('reasoning_tokens', 0)
                else:
                    result.full_response = str(data)
            else:
                result.error = response_data.error
                return result

            # ParseAnswer
            parse_result = self.answer_parser.parse(
                result.full_response,
                AnswerType.NUMBER,
                correct_answer
            )
            result.predicted_answer = parse_result.extracted_answer
            result.answer_confidence = parse_result.confidence

            # Check正确性
            from .smart_answer_parser import compare_answers
            result.is_correct, _ = compare_answers(
                parse_result.normalized_value,
                correct_answer,
                AnswerType.NUMBER
            )

            # 推理Quality Assessment
            if config.metrics.reasoning_quality and result.reasoning_content:
                reasoning_eval = self.reasoning_evaluator.evaluate(
                    question=question,
                    reasoning=result.reasoning_content,
                    final_answer=result.predicted_answer,
                    correct_answer=correct_answer,
                    is_answer_correct=result.is_correct
                )
                result.reasoning_quality = reasoning_eval.quality_score.overall

        except Exception as e:
            result.error = str(e)

        return result

    def _calculate_model_stats(self, model_result: ModelTestResult):
        """CalculateModelStatistics指标"""
        valid_results = [r for r in model_result.results if not r.error]

        if not valid_results:
            return

        # Accuracy
        model_result.accuracy = model_result.correct_samples / model_result.total_samples

        # Latency
        latencies = [r.latency_ms for r in valid_results if r.latency_ms > 0]
        if latencies:
            model_result.avg_latency_ms = sum(latencies) / len(latencies)

        # TTFT
        ttfts = [r.ttft_ms for r in valid_results if r.ttft_ms > 0]
        if ttfts:
            model_result.avg_ttft_ms = sum(ttfts) / len(ttfts)

        # TTUT
        ttuts = [r.ttut_ms for r in valid_results if r.ttut_ms > 0]
        if ttuts:
            model_result.avg_ttut_ms = sum(ttuts) / len(ttuts)

        # Tokens
        model_result.total_input_tokens = sum(r.input_tokens for r in valid_results)
        model_result.total_output_tokens = sum(r.output_tokens for r in valid_results)
        model_result.total_reasoning_tokens = sum(r.reasoning_tokens for r in valid_results)

        total_output = model_result.total_output_tokens + model_result.total_reasoning_tokens
        if total_output > 0:
            model_result.avg_reasoning_ratio = model_result.total_reasoning_tokens / total_output

        # TPS
        tps_values = []
        for r in valid_results:
            if r.latency_ms > 0 and r.output_tokens > 0:
                tps_values.append(r.output_tokens / (r.latency_ms / 1000))
        if tps_values:
            model_result.avg_tps = sum(tps_values) / len(tps_values)

        # 推理质量
        quality_scores = [r.reasoning_quality for r in valid_results if r.reasoning_quality > 0]
        if quality_scores:
            model_result.avg_reasoning_quality = sum(quality_scores) / len(quality_scores)

    def _calculate_comparison(self, result: TestSuiteResult):
        """CalculateModel比较Result"""
        if not result.model_results:
            return

        models = list(result.model_results.values())

        # BestAccuracy
        best_acc = max(models, key=lambda m: m.accuracy)
        result.best_accuracy_model = f"{best_acc.platform}:{best_acc.model_id}"

        # BestLatency
        valid_latency = [m for m in models if m.avg_latency_ms > 0]
        if valid_latency:
            best_lat = min(valid_latency, key=lambda m: m.avg_latency_ms)
            result.best_latency_model = f"{best_lat.platform}:{best_lat.model_id}"

        # Best推理质量
        valid_quality = [m for m in models if m.avg_reasoning_quality > 0]
        if valid_quality:
            best_qual = max(valid_quality, key=lambda m: m.avg_reasoning_quality)
            result.best_quality_model = f"{best_qual.platform}:{best_qual.model_id}"

        # 汇总
        result.summary = {
            "total_models": len(models),
            "best_accuracy": {
                "model": result.best_accuracy_model,
                "value": best_acc.accuracy
            },
            "accuracy_ranking": sorted(
                [(f"{m.platform}:{m.model_id}", m.accuracy) for m in models],
                key=lambda x: x[1],
                reverse=True
            )
        }

    async def _generate_reports(self, result: TestSuiteResult, config: TestConfig):
        """Generate报告"""
        output_dir = self.report_dir / config.output.report_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{config.name.replace(' ', '_')}_{timestamp}"

        # JSON 报告
        if config.output.json_export:
            import json
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self._result_to_dict(result, config), f, ensure_ascii=False, indent=2)

        # Markdown 报告
        if config.output.markdown_export:
            md_path = output_dir / f"{base_name}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_markdown_report(result))

    def _result_to_dict(self, result: TestSuiteResult, config: TestConfig) -> dict:
        """willResultConvertis字典"""
        return {
            "test_name": result.test_name,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "duration_seconds": result.duration_seconds,
            "summary": result.summary,
            "models": {
                k: {
                    "model_id": v.model_id,
                    "platform": v.platform,
                    "accuracy": v.accuracy,
                    "total_samples": v.total_samples,
                    "correct_samples": v.correct_samples,
                    "avg_latency_ms": v.avg_latency_ms,
                    "avg_reasoning_quality": v.avg_reasoning_quality
                }
                for k, v in result.model_results.items()
            }
        }

    def _generate_markdown_report(self, result: TestSuiteResult) -> str:
        """Generate Markdown 报告"""
        lines = [
            f"# {result.test_name}",
            "",
            f"**Test时间**: {result.start_time}",
            f"**Duration**: {result.duration_seconds:.1f} seconds",
            "",
            "## Result汇总",
            "",
            "| Model | Accuracy | AverageLatency | 推理质量 |",
            "|------|--------|----------|----------|"
        ]

        for _key, m in result.model_results.items():
            lines.append(
                f"| {m.model_id} | {m.accuracy*100:.1f}% | {m.avg_latency_ms:.0f}ms | {m.avg_reasoning_quality:.1f}/10 |"
            )

        lines.extend([
            "",
            "## Best表现",
            "",
            f"- **HighestAccuracy**: {result.best_accuracy_model}",
            f"- **LowestLatency**: {result.best_latency_model}",
            f"- **Best推理质量**: {result.best_quality_model}",
        ])

        return "\n".join(lines)


async def run_test_from_config(
    config_path: str,
    samples: list[dict],
    response_funcs: dict[str, Callable]
) -> TestSuiteResult:
    """便捷函数：从ConfigureRun Test"""
    runner = UnifiedTestRunner()
    return await runner.run_from_config(config_path, samples, response_funcs)
