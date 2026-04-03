"""
Quality Evaluator Engine
Quality Assessment引擎 - 管理Dataset评估核心模块
"""

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from core.providers.factory import get_provider
from core.failure_analyzer import analyze_failures
from evaluators.base_evaluator import BaseEvaluator, EvaluationResult
from utils.logger import LogLevel


@dataclass
class QualityTestConfig:
    """质量Test Configuration"""
    datasets: list[str] = field(default_factory=lambda: ["mmlu"])
    num_shots: int = 5
    max_samples: int | None = None  # None = 全部
    temperature: float = 0.0
    max_tokens: int = 256
    concurrency: int = 4
    subsets: dict[str, list[str]] | None = None  # {dataset: [subsets]}

    # ModelConfigure
    model_type: str = "standard"  # standard, thinking, code

    # 思考Model特定Configure
    thinking_enabled: bool = False
    thinking_budget: int = 1024  # 思考 token 预算
    reasoning_effort: str = "medium"  # low, medium, high (for o1/o3-mini like models)

    # 评判Configure
    use_llm_judge: bool = False  # is否启用Model自评 (针对误判进行二次Confirm)

    # 缓存Configure
    use_cache: bool = True  # is否启用响应缓存
    cache_ttl_hours: int = 168  # 缓存has效期 (hours)，default7天
    resume_from_checkpoint: str | None = None  # 从断点RestoreCheck点名称
    save_checkpoint: bool = True  # is否自动SaveCheck点

    # Data集重写Configure (每Dataset可覆盖全局Configure)
    # 格式: {"mmlu": {"max_tokens": 1024, "temperature": 0.5}, ...}
    dataset_overrides: dict[str, dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasets": self.datasets,
            "num_shots": self.num_shots,
            "max_samples": self.max_samples,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "concurrency": self.concurrency,
            "subsets": self.subsets,
            "model_type": self.model_type,
            "thinking_enabled": self.thinking_enabled,
            "thinking_budget": self.thinking_budget,
            "reasoning_effort": self.reasoning_effort,
            "dataset_overrides": self.dataset_overrides,
            "use_cache": self.use_cache,
            "cache_ttl_hours": self.cache_ttl_hours
        }


class QualityEvaluator:
    """
    LLM Quality Assessment引擎

    负责:
    1. 管理多Dataset评估
    2. 调用 LLM API Get响应
    3. 汇总Evaluation result
    """

    # Data集路径映射
    DATASET_PATHS = {
        "mmlu": "datasets/mmlu",
        "gsm8k": "datasets/gsm8k",
        "math500": "datasets/math500",
        "humaneval": "datasets/humaneval",
        "ceval": "datasets/ceval",
        "arc": "datasets/arc",
        "truthfulqa": "datasets/truthfulqa",
        "gpqa": "datasets/gpqa",
        "hellaswag": "datasets/hellaswag",
        "winogrande": "datasets/winogrande",
        "mbpp": "datasets/mbpp",
        "longbench": "datasets/longbench",
        "swebench_lite": "datasets/swebench_lite",
        "needle_haystack": "datasets/needle_haystack",
        "custom_needle": "needle_haystack_data",  # Custom needle test directory
    }



    # Evaluator类映射 (willin各Evaluator实现后Pad)
    EVALUATOR_CLASSES: dict[str, type[BaseEvaluator]] = {}

    def __init__(
        self,
        api_base_url: str,
        model_id: str,
        api_key: str = "",
        provider: str = "OpenAI 兼容",
        output_dir: str = "quality_results",
        log_callback: Callable[[str, LogLevel], None] | None = None,
        tokenizer_model_id: str | None = None,
        enable_cache: bool = True
    ):
        """
        InitializeQuality Assessment引擎

        Args:
            api_base_url: API Base URL
            model_id: Model ID
            api_key: API 密钥
            provider: Provider名称
            output_dir: 输出目录
            log_callback: LogCallback函数
            tokenizer_model_id: 用于Calculate token Model ID
            enable_cache: is否启用响应缓存
        """
        self.api_base_url = api_base_url
        self.model_id = model_id
        self.api_key = api_key
        self.provider_name = provider
        self.output_dir = output_dir
        self.log_callback = log_callback

        # Initialize Provider (use model_id)
        self.provider = get_provider(provider, api_base_url, api_key, model_id)

        # Result存储
        self.results: dict[str, EvaluationResult] = {}

        # Status
        self.is_running = False
        self.should_stop = False

        # Initialize Tokenizer
        self.tokenizer = None
        self._init_tokenizer(tokenizer_model_id or model_id)

        # 确保输出目录存in
        os.makedirs(output_dir, exist_ok=True)

        # Initialize响应缓存
        self.cache = None
        self._cache_enabled = enable_cache
        self._cache_stats = {"hits": 0, "misses": 0}
        if enable_cache:
            try:
                from core.response_cache import get_cache
                self.cache = get_cache(cache_dir=os.path.join(output_dir, "cache"))
                self._log("响应缓存已启用")
            except ImportError:
                self._log("Response Cache Modulenot可用", LogLevel.WARNING)

    def _init_tokenizer(self, model_id: str):
        """Initialize tokenizer，自带 Fallback 策略"""
        try:
            from config.settings import HF_MODEL_MAPPING
            from core.tokenizer_utils import get_cached_tokenizer

            # 1. 尝试直接Load
            self.tokenizer = get_cached_tokenizer(model_id)
            if self.tokenizer:
                self._log(f"已Load tokenizer: {model_id}")
                return

            # 2. 智能回退策略 (基于 settings.py Configure)
            fallback_id = None
            model_id_lower = model_id.lower()

            # 优先匹配 config/settings.py in映射
            for key, target in HF_MODEL_MAPPING.items():
                if key in model_id_lower:
                    fallback_id = target
                    break

            if fallback_id:
                self._log(f"no法Load {model_id}，尝试回退到本地/通用 Tokenizer: {fallback_id} ...", LogLevel.INFO)
                self.tokenizer = get_cached_tokenizer(fallback_id)
                if self.tokenizer:
                    self._log(f"已Load回退 tokenizer: {fallback_id}")
                    return

            # 3. 最终失败
            self._log(f"no法Load tokenizer: {model_id}，willuse字符估算", LogLevel.WARNING)

        except Exception as e:
            self._log(f"Initialize tokenizer 失败: {e}，willuse字符估算", LogLevel.WARNING)
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Calculate文本 token 数"""
        if not text:
            return 0

        if self.tokenizer:
            try:
                if hasattr(self.tokenizer, 'encode'):
                    return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass

        # 回退到字符估算 (in英文混合约 2.5 字符/token)
        return max(1, len(text) // 3)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get缓存Statistics信息"""
        if not self.cache:
            return {"enabled": False}

        try:
            cache_stats = self.cache.get_stats()
            return {
                "enabled": True,
                "session_hits": self._cache_stats["hits"],
                "session_misses": self._cache_stats["misses"],
                "session_hit_rate": self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"])
                    if (self._cache_stats["hits"] + self._cache_stats["misses"]) > 0 else 0,
                "total_entries": cache_stats.total_entries,
                "total_bytes": cache_stats.total_bytes,
                "total_bytes_mb": cache_stats.total_bytes / (1024 * 1024) if cache_stats.total_bytes else 0
            }
        except Exception:
            return {"enabled": True, "error": "no法GetStatistics"}

    def clear_cache(self, model_only: bool = True):
        """
        清除缓存

        Args:
            model_only: True 只清除当前Model缓存，False 清除所has
        """
        if self.cache:
            if model_only:
                self.cache.clear(model_id=self.model_id)
                self._log(f"已清除Model {self.model_id} 缓存")
            else:
                self.cache.clear()
                self._log("已清除所has缓存")
            self._cache_stats = {"hits": 0, "misses": 0}


    def _log(self, message: str, level: LogLevel = LogLevel.INFO):
        """输出Log"""
        if self.log_callback:
            self.log_callback(message, level)
        print(f"[{level.name}] {message}")

    async def _get_response_with_metrics(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        use_cache: bool = True,
        **kwargs
    ) -> dict[str, Any]:
        """
        调用 LLM Get响应andPerformance Metrics
        """
        # Check缓存
        if use_cache and self.cache and self._cache_enabled:
            cached_response = self.cache.get(prompt, model_id=self.model_id)
            if cached_response:
                self._cache_stats["hits"] += 1
                return {
                    "content": cached_response,
                    "error": None,
                    "input_tokens": self.count_tokens(prompt) if prompt else 0,
                    "output_tokens": self.count_tokens(cached_response) if cached_response else 0,
                    "ttft_ms": 0,
                    "tps": 0,
                    "total_time_ms": 0,
                    "from_cache": True
                }
            else:
                self._cache_stats["misses"] += 1

        # Add重试机制
        retries = kwargs.get('retries', 3)
        backoff = 1.0

        for attempt in range(retries + 1):
            try:
                # use provider  get_completion 接口
                result = await self.provider.get_completion(
                    client=None,
                    session_id=0,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )

                # CheckError
                if result.get('error'):
                    raise Exception(result['error'])

                # 提取Performance Metrics
                content = result.get('full_response_content', '').strip()
                start_time = result.get('start_time', 0)
                first_token_time = result.get('first_token_time', 0)
                end_time = result.get('end_time', 0)
                usage_info = result.get('usage_info', {}) or {}

                # Calculate TTFT (毫seconds)
                ttft_ms = 0
                if first_token_time and start_time:
                    ttft_ms = (first_token_time - start_time) * 1000

                # Calculate总时间 (毫seconds)
                total_time_ms = 0
                if end_time and start_time:
                    total_time_ms = (end_time - start_time) * 1000

                # Get token 数
                input_tokens = usage_info.get('prompt_tokens', 0)
                output_tokens = usage_info.get('completion_tokens', 0)

                if input_tokens == 0 and prompt:
                    input_tokens = self.count_tokens(prompt)

                if output_tokens == 0 and content:
                    output_tokens = self.count_tokens(content)

                # Calculate TPS
                tps = 0
                decode_time_ms = total_time_ms - ttft_ms if ttft_ms > 0 else total_time_ms
                if decode_time_ms > 0 and output_tokens > 0:
                    tps = output_tokens / (decode_time_ms / 1000)

                # 缓存succeeded响应
                if use_cache and self.cache and self._cache_enabled and content:
                    try:
                        self.cache.set(
                            prompt,
                            content,
                            model_id=self.model_id,
                            metadata={"temperature": temperature, "max_tokens": max_tokens}
                        )
                    except Exception:
                        pass

                return {
                    "content": content,
                    "error": None,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "ttft_ms": ttft_ms,
                    "tps": tps,
                    "total_time_ms": total_time_ms,
                    "from_cache": False
                }

            except Exception as e:
                if attempt < retries:
                    wait_time = backoff * (2 ** attempt)
                    self._log(f"API 调用失败 (尝试 {attempt+1}/{retries+1}): {e}, {wait_time:.1f}s 后重试...", LogLevel.WARNING)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    self._log(f"API 调用失败 (最终): {e}", LogLevel.ERROR)
                    return {
                        "content": "",
                        "error": str(e),
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "ttft_ms": 0,
                        "tps": 0,
                        "total_time_ms": 0,
                        "from_cache": False
                    }

    async def _get_response(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256
    ) -> str:
        """简化版响应Get (Backward compatibility)"""
        result = await self._get_response_with_metrics(prompt, temperature, max_tokens)
        if result.get('error'):
            raise Exception(result['error'])
        return result.get('content', '')


    def register_evaluator(self, dataset_name: str, evaluator_class: type[BaseEvaluator]):
        """RegisterEvaluator类"""
        self.EVALUATOR_CLASSES[dataset_name] = evaluator_class

    def get_evaluator(
        self,
        dataset_name: str,
        config: QualityTestConfig,
        test_filter: str | None = None
    ) -> BaseEvaluator | None:
        """
        Get指定DatasetEvaluator实例

        Args:
            dataset_name: Dataset名称
            config: Test Configuration
            test_filter: TestFilter器 (用于Custom大海捞针Test)

        Returns:
            Evaluator实例
        """
        # Process带Filter器Dataset名称 (如 custom_needle_frankenstein)
        actual_dataset_name = dataset_name
        if dataset_name.startswith("custom_needle_"):
            # 提取Filter器
            test_filter = dataset_name.replace("custom_needle_", "")
            actual_dataset_name = "custom_needle"

        if actual_dataset_name not in self.EVALUATOR_CLASSES:
            self._log(f"Not foundDataset '{actual_dataset_name}' Evaluator", LogLevel.WARNING)
            return None

        dataset_path = self.DATASET_PATHS.get(actual_dataset_name, f"datasets/{actual_dataset_name}")

        evaluator_class = self.EVALUATOR_CLASSES[actual_dataset_name]

        # is CustomNeedleEvaluator 传递 test_filter 参数
        if actual_dataset_name == "custom_needle" and test_filter:
            evaluator = evaluator_class(
                dataset_name=dataset_name,  # 保留原始名称用于Result标识
                dataset_path=dataset_path,
                num_shots=config.num_shots,
                max_samples=config.max_samples,
                seed=42,
                test_filter=test_filter  # 传递Filter器
            )
        else:
            evaluator = evaluator_class(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                num_shots=config.num_shots,
                max_samples=config.max_samples,
                seed=42
            )

        # 注入 LLM Judge Configure
        use_judge = getattr(config, 'use_llm_judge', False)
        evaluator.use_llm_judge = use_judge
        self._log(f"DEBUG: get_evaluator for {dataset_name} -> config.use_llm_judge={use_judge}, set evaluator.use_llm_judge={evaluator.use_llm_judge}")

        return evaluator

    async def evaluate_dataset(
        self,
        dataset_name: str,
        config: QualityTestConfig,
        subset: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None
    ) -> EvaluationResult | None:
        """
        评估单Dataset

        Args:
            dataset_name: Dataset名称
            config: Test Configuration
            subset: 子集名称
            progress_callback: 进度Callback (current, total, message)

        Returns:
            Evaluation result
        """
        self._log(f"开始评估Dataset: {dataset_name}" + (f" (子集: {subset})" if subset else ""))

        evaluator = self.get_evaluator(dataset_name, config)
        if not evaluator:
            return None

        try:
            # LoadDataset
            samples = evaluator.load_dataset(subset=subset)
            if not samples:
                self._log(f"Dataset '{dataset_name}' Load failedoris空", LogLevel.ERROR)
                return None

            self._log(f"已Load {len(samples)}  samples")

            # 准备请求参数
            req_temperature = config.temperature
            req_max_tokens = config.max_tokens
            req_extra_params = {}

            # 1. ApplyDataset覆盖Configure
            if config.dataset_overrides and dataset_name in config.dataset_overrides:
                overrides = config.dataset_overrides[dataset_name]
                if "temperature" in overrides:
                    req_temperature = overrides["temperature"]
                if "max_tokens" in overrides:
                    req_max_tokens = overrides["max_tokens"]
                # Process其它可能参数

            # 2. Process思考Model Configuration
            if config.thinking_enabled:
                # 传递所has推理相关参数
                req_extra_params["thinking_enabled"] = config.thinking_enabled
                req_extra_params["reasoning_effort"] = config.reasoning_effort
                if hasattr(config, "thinking_budget") and config.thinking_budget:
                    req_extra_params["thinking_budget"] = config.thinking_budget

            # Create响应Get函数 (Return包含Performance Metrics字典)
            async def get_response_func(prompt: str) -> dict[str, Any]:
                # Check both internal flag and global stop signal
                if self.should_stop:
                    raise asyncio.CancelledError("评估已停止")

                # Also check global stop flag from session_state
                try:
                    import streamlit as st
                    if st.session_state.get('stop_requested', False):
                        self.should_stop = True
                        raise asyncio.CancelledError("评估已停止")
                except Exception:
                    pass

                # 传递参数
                return await self._get_response_with_metrics(
                    prompt,
                    temperature=req_temperature,
                    max_tokens=req_max_tokens,
                    **req_extra_params
                )


            # 用于追踪样本Result列表（用于实时Statistics）
            live_sample_results = []

            # 进度Callback包装 - 增强版，输出更多信息
            def internal_progress(current: int, total: int):
                # Calculate当前正确率
                if live_sample_results:
                    correct_count = sum(1 for r in live_sample_results if r.is_correct)
                    current_accuracy = correct_count / len(live_sample_results) * 100
                    status_emoji = "✅" if live_sample_results[-1].is_correct else "❌"

                    # 输出Verbose Logging
                    last_result = live_sample_results[-1]
                    short_answer = (last_result.predicted_answer or "")[:50]
                    self._log(
                        f"{status_emoji} 样本 {current}/{total} | "
                        f"正确率: {current_accuracy:.1f}% | "
                        f"Answer: {short_answer}..."
                    )

                if progress_callback:
                    progress_callback(
                        current,
                        total,
                        f"评估 {dataset_name}: {current}/{total}"
                    )

            # ResultCallback - 收集每 samplesResult用于Real-time logging
            def on_result_complete(result):
                live_sample_results.append(result)

            # 执行批量评估
            start_time = time.time()
            sample_results = await evaluator.evaluate_batch(
                samples=samples,
                get_response_func=get_response_func,
                concurrency=config.concurrency,
                progress_callback=internal_progress,
                result_callback=on_result_complete
            )
            duration = time.time() - start_time


            # Calculate指标
            accuracy, by_category = evaluator.compute_metrics(sample_results)

            # BuildResult
            result = EvaluationResult(
                dataset_name=dataset_name + (f"_{subset}" if subset else ""),
                model_id=self.model_id,
                accuracy=accuracy,
                total_samples=len(sample_results),
                correct_samples=sum(1 for r in sample_results if r.is_correct),
                by_category=by_category,
                details=sample_results,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                duration_seconds=duration,
                config=config.to_dict()
            )

            # Calculate性能Statistics
            result.compute_performance_stats()

            # --- 新增: 自动失败分析 ---
            try:
                # 收集失败样本
                failed_dicts = [
                    s.to_dict() for s in sample_results 
                    if not s.is_correct and not s.error  # 排除系统Error，只分析逻辑Error
                ]
                
                if failed_dicts:
                    self._log(f"currently分析 {len(failed_dicts)} 失败案例...")
                    failure_report = analyze_failures(failed_dicts, total=len(sample_results))
                    
                    # will分析报告摘要存入 extended_metrics
                    result.extended_metrics["failure_analysis"] = {
                        "failure_rate": failure_report.failure_rate,
                        "category_distribution": failure_report.category_distribution,
                        "top_issues": failure_report.top_issues,
                        "suggestions": failure_report.improvement_suggestions
                    }
                    
                    self._log(f"分析完成: 主要问题 - {', '.join(failure_report.top_issues[:2])}")
            except Exception as e:
                self._log(f"失败分析出错: {e}", LogLevel.WARNING)
            # --------------------------

            # Log输出
            stats = result.performance_stats
            self._log(
                f"Dataset '{dataset_name}' 评估完成: "
                f"Accuracy = {accuracy:.2%} ({result.correct_samples}/{result.total_samples})"
            )
            if stats:
                self._log(
                    f"Performance Metrics: Avg TTFT={stats.get('avg_ttft_ms', 0):.0f}ms, "
                    f"Avg TPS={stats.get('avg_tps', 0):.1f}, "
                    f"Total tokens={stats.get('total_input_tokens', 0)}+{stats.get('total_output_tokens', 0)}"
                )

            return result


        except asyncio.CancelledError:
            self._log("评估被Cancel", LogLevel.WARNING)
            return None
        except Exception as e:
            self._log(f"评估出错: {e}", LogLevel.ERROR)
            import traceback
            traceback.print_exc()
            return None

    async def run_evaluation(
        self,
        config: QualityTestConfig,
        progress_callback: Callable[[int, int, str], None] | None = None
    ) -> dict[str, EvaluationResult]:
        """
        运行完整Quality Assessment

        Args:
            config: Test Configuration
            progress_callback: 进度Callback (current, total, message)

        Returns:
            所hasDatasetEvaluation result
        """
        self.is_running = True
        self.should_stop = False
        self.results = {}

        # Calculate总Sample count用于统一进度追踪
        total_samples_all = 0
        completed_samples_all = 0

        # 预估每DatasetSample count
        dataset_sample_counts = {}
        for dataset_name in config.datasets:
            # default估算值，实际Load后willUpdate
            estimated = config.max_samples if config.max_samples else 100
            dataset_sample_counts[dataset_name] = estimated
            total_samples_all += estimated

        try:
            total_datasets = len(config.datasets)

            for i, dataset_name in enumerate(config.datasets):
                # Check both internal flag and global stop signal
                if self.should_stop:
                    break
                try:
                    import streamlit as st
                    if st.session_state.get('stop_requested', False):
                        self.should_stop = True
                        break
                except Exception:
                    pass

                self._log(f"=== 评估Dataset [{i+1}/{total_datasets}]: {dataset_name} ===")

                # Checkis否has指定子集
                subsets = None
                if config.subsets and dataset_name in config.subsets:
                    subsets = config.subsets[dataset_name]

                # Create带has全局进度Callback包装器
                def create_global_progress_callback(ds_name, ds_start_offset):
                    def global_progress(current, total, message):
                        nonlocal completed_samples_all, total_samples_all, dataset_sample_counts

                        # Update该Dataset实际Sample count（首次调用时）
                        if dataset_sample_counts.get(ds_name, 0) != total:
                            old_estimate = dataset_sample_counts.get(ds_name, 0)
                            dataset_sample_counts[ds_name] = total
                            total_samples_all = total_samples_all - old_estimate + total

                        # Calculate全局进度
                        global_current = ds_start_offset + current

                        if progress_callback:
                            progress_callback(
                                global_current,
                                total_samples_all,
                                f"[{ds_name}] {current}/{total} - {message}"
                            )
                    return global_progress

                if subsets:
                    # 评估每子集
                    for subset in subsets:
                        if self.should_stop:
                            break

                        subset_callback = create_global_progress_callback(
                            f"{dataset_name}_{subset}",
                            completed_samples_all
                        )

                        result = await self.evaluate_dataset(
                            dataset_name,
                            config,
                            subset=subset,
                            progress_callback=subset_callback
                        )
                        if result:
                            self.results[f"{dataset_name}_{subset}"] = result
                            completed_samples_all += result.total_samples
                else:
                    # 评估整Dataset
                    dataset_callback = create_global_progress_callback(
                        dataset_name,
                        completed_samples_all
                    )

                    result = await self.evaluate_dataset(
                        dataset_name,
                        config,
                        progress_callback=dataset_callback
                    )
                    if result:
                        self.results[dataset_name] = result
                        completed_samples_all += result.total_samples

            # SaveResult
            self._save_results()

            # 输出缓存Statistics
            cache_stats = self.get_cache_stats()
            if cache_stats.get("enabled"):
                hits = cache_stats.get("session_hits", 0)
                misses = cache_stats.get("session_misses", 0)
                hit_rate = cache_stats.get("session_hit_rate", 0)
                self._log(
                    f"📦 缓存Statistics: 命in {hits}, 未命in {misses}, "
                    f"命in率 {hit_rate*100:.1f}%"
                )

            return self.results

        finally:
            self.is_running = False

    def stop(self):
        """停止评估"""
        self.should_stop = True
        self._log("Stopping评估...", LogLevel.WARNING)

    def _save_results(self):
        """SaveEvaluation result"""
        if not self.results:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save JSON Detailed Results
        for name, result in self.results.items():
            filepath = os.path.join(
                self.output_dir,
                self.model_id,
                f"{name}_{timestamp}.json"
            )
            result.save_to_json(filepath)
            self._log(f"ResultSaved: {filepath}")

        # Save汇总 CSV
        summary_data = []
        for name, result in self.results.items():
            # 从result.configin提取Thinking modeConfigure
            config = result.config or {}
            thinking_enabled = config.get('thinking_enabled', False)
            thinking_budget = config.get('thinking_budget', 0)
            reasoning_effort = config.get('reasoning_effort', 'N/A')

            # BuildThinking mode描述
            if thinking_enabled:
                thinking_mode = f"Enabled (Budget: {thinking_budget}, Effort: {reasoning_effort})"
            else:
                thinking_mode = "Disabled"

            summary_data.append({
                "Dataset": name,
                "Model": result.model_id,
                "Thinking Mode": thinking_mode,
                "Thinking Budget": thinking_budget if thinking_enabled else "N/A",
                "Reasoning Effort": reasoning_effort if thinking_enabled else "N/A",
                "Accuracy": f"{result.accuracy:.2%}",
                "Correct": result.correct_samples,
                "Total": result.total_samples,
                "Duration (s)": f"{result.duration_seconds:.1f}",
                "Timestamp": result.timestamp
            })

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(
                self.output_dir,
                self.model_id,
                f"summary_{timestamp}.csv"
            )
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            self._log(f"汇总Saved: {csv_path}")

        # GenerateStandardize报告
        self._export_standard_reports(timestamp)

    def _export_standard_reports(self, timestamp: str):
        """ExportStandardize格式报告"""
        try:
            from core.standard_report import ModelInfo, ReportExporter, StandardReport

            # CreateModel信息
            model_info = ModelInfo(
                model_id=self.model_id,
                provider=self.provider_name,
                api_base_url=self.api_base_url
            )

            # 从所hasResultCreate报告
            results_list = list(self.results.values())
            if not results_list:
                return

            report = StandardReport.from_multiple_results(
                results_list,
                model_info=model_info,
                config={
                    "num_shots": getattr(self, 'num_shots', 0),
                    "max_samples": getattr(self, 'max_samples', None),
                    "use_cache": getattr(self, 'enable_cache', False)
                }
            )

            exporter = ReportExporter(report)

            # Export目录
            report_dir = os.path.join(self.output_dir, self.model_id, "reports")
            os.makedirs(report_dir, exist_ok=True)

            # Export多种格式
            json_path = exporter.to_json(
                os.path.join(report_dir, f"standard_{timestamp}.json")
            )
            self._log(f"标准报告Saved: {json_path}")

            lm_eval_path = exporter.to_lm_eval_format(
                os.path.join(report_dir, f"lm_eval_{timestamp}.json")
            )
            self._log(f"lm-eval 格式Saved: {lm_eval_path}")

            md_path = exporter.to_markdown(
                os.path.join(report_dir, f"report_{timestamp}.md")
            )
            self._log(f"Markdown 报告Saved: {md_path}")

        except ImportError:
            self._log("标准Reports Modulenot可用，跳过Export", LogLevel.WARNING)
        except Exception as e:
            self._log(f"Export标准Report failed: {e}", LogLevel.ERROR)

    def get_summary_df(self) -> pd.DataFrame:
        """Get汇总 DataFrame"""
        if not self.results:
            return pd.DataFrame()

        data = []
        for name, result in self.results.items():
            # 从result.configin提取Thinking modeConfigure
            config = result.config or {}
            thinking_enabled = config.get('thinking_enabled', False)
            thinking_budget = config.get('thinking_budget', 0)
            reasoning_effort = config.get('reasoning_effort', 'N/A')

            # BuildThinking mode描述
            if thinking_enabled:
                thinking_mode = f"✅ {reasoning_effort.upper()}"
                budget_display = f"{thinking_budget} tokens"
            else:
                thinking_mode = "❌ Close"
                budget_display = "N/A"

            data.append({
                "Dataset": name,
                "Model": result.model_id,
                "Thinking mode": thinking_mode,
                "Thinking budget": budget_display,
                "Accuracy": result.accuracy,
                "正确数": result.correct_samples,
                "总数": result.total_samples,
                "耗时(seconds)": result.duration_seconds
            })

        return pd.DataFrame(data)

    def get_category_breakdown(self, dataset_name: str) -> pd.DataFrame:
        """Get某Dataset分类别Statistics"""
        if dataset_name not in self.results:
            return pd.DataFrame()

        result = self.results[dataset_name]
        data = []
        for cat, metrics in result.by_category.items():
            data.append({
                "类别": cat,
                "Accuracy": metrics.get("accuracy", 0),
                "Sample count": metrics.get("count", 0)
            })

        return pd.DataFrame(data).sort_values("Accuracy", ascending=False)


# ============================================
# 便捷函数
# ============================================

async def quick_evaluate(
    api_base_url: str,
    model_id: str,
    datasets: list[str],
    api_key: str = "",
    num_shots: int = 5,
    max_samples: int = 100
) -> dict[str, EvaluationResult]:
    """
    快速评估函数

    示例:
    ```python
    results = await quick_evaluate(
        api_base_url="http://localhost:8000/v1",
        model_id="qwen-7b",
        datasets=["mmlu", "gsm8k"],
        max_samples=100
    )
    ```
    """
    config = QualityTestConfig(
        datasets=datasets,
        num_shots=num_shots,
        max_samples=max_samples
    )

    evaluator = QualityEvaluator(
        api_base_url=api_base_url,
        model_id=model_id,
        api_key=api_key
    )

    return await evaluator.run_evaluation(config)
