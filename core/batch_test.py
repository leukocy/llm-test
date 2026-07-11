"""
Batch Test Module

支持Batch Execution多Test Configuration：
- 多Model ComparisonTest
- 多Configure对比Test
- Batch Test调度
- 批量Test Results汇总

启动优化：
- BenchmarkRunner useLatencyImport
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable

import pandas as pd

from core.cancel_state import is_batch_stop_requested
from core.result_metrics import success_mask_from_error

# LatencyImport BenchmarkRunner（仅inTest执行时Load）
_BenchmarkRunner = None


def _get_benchmark_runner():
    """LatencyGet BenchmarkRunner 类"""
    global _BenchmarkRunner
    if _BenchmarkRunner is None:
        from core.benchmark_runner import BenchmarkRunner

        _BenchmarkRunner = BenchmarkRunner
    return _BenchmarkRunner


# ============================================================================
# Mock UI 组件 (用于Batch Testno UI 环境)
# ============================================================================


class _MockPlaceholder:
    """Mock placeholder for batch testing (no UI)"""

    def container(self):
        return self

    def empty(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _MockProgressBar:
    """Mock progress bar for batch testing"""

    def progress(self, value):
        pass


class _MockStatusText:
    """Mock status text for batch testing"""

    def info(self, msg):
        pass

    def success(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


# ============================================================================
# 批量Test Configuration
# ============================================================================


@dataclass
class BatchTestItem:
    """单Batch Test items"""

    name: str  # Test名称
    api_base_url: str  # API 地址
    model_id: str  # Model ID
    api_key: str  # API 密钥

    # Test Parameters
    test_type: str = "concurrency"  # Test类型
    concurrency: int = 1
    max_tokens: int = 512
    temperature: float = 0.0
    thinking_enabled: bool = False
    thinking_budget: int = 0
    reasoning_effort: str = "medium"

    # 额外参数
    extra_params: dict[str, Any] = field(default_factory=dict)

    # Status
    enabled: bool = True  # is否启用此Test
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: Any | None = None
    error: str | None = None  # 失败时的错误信息

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        return {
            "name": self.name,
            "api_base_url": self.api_base_url,
            "model_id": self.model_id,
            "api_key": self.api_key,
            "test_type": self.test_type,
            "concurrency": self.concurrency,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking_enabled": self.thinking_enabled,
            "thinking_budget": self.thinking_budget,
            "reasoning_effort": self.reasoning_effort,
            "extra_params": self.extra_params,
            "enabled": self.enabled,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchTestItem":
        """从字典Create"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BatchTestConfig:
    """批量Test Configuration"""

    name: str  # Batch Test名称
    description: str = ""
    items: list[BatchTestItem] = field(default_factory=list)

    # 执行Options
    parallel: bool = False  # is否并行执行
    max_parallel: int = 2  # 最大并行数
    stop_on_error: bool = False  # 遇到Erroris否停止
    save_intermediate: bool = True  # is否Savein间Result

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        return {
            "name": self.name,
            "description": self.description,
            "items": [item.to_dict() for item in self.items],
            "parallel": self.parallel,
            "max_parallel": self.max_parallel,
            "stop_on_error": self.stop_on_error,
            "save_intermediate": self.save_intermediate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchTestConfig":
        """从字典Create"""
        items = [
            BatchTestItem.from_dict(item_data) for item_data in data.get("items", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            items=items,
            parallel=data.get("parallel", False),
            max_parallel=data.get("max_parallel", 2),
            stop_on_error=data.get("stop_on_error", False),
            save_intermediate=data.get("save_intermediate", True),
        )


# ============================================================================
# Batch Test进度
# ============================================================================


@dataclass
class BatchTestProgress:
    """Batch Test进度"""

    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    current_item: str | None = None
    start_time: float | None = None
    end_time: float | None = None

    @property
    def progress_percentage(self) -> float:
        """Get进度百分比"""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def elapsed_time(self) -> float:
        """GetElapsed Time"""
        if not self.start_time:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def estimated_remaining_time(self) -> float | None:
        """估算Remaining时间"""
        if self.completed_items == 0 or self.total_items == 0:
            return None
        rate = self.elapsed_time / self.completed_items
        remaining = self.total_items - self.completed_items
        return rate * remaining

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        return {
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "current_item": self.current_item,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "progress_percentage": self.progress_percentage,
            "elapsed_time": self.elapsed_time,
        }


# ============================================================================
# 批量Test Results
# ============================================================================


@dataclass
class BatchTestResult:
    """批量Test Results"""

    batch_name: str
    start_time: str
    end_time: str
    duration_seconds: float

    # 各 itemsTest Results
    item_results: list[dict[str, Any]] = field(default_factory=list)

    # 汇总Statistics
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0

    # 对比Data
    comparison_data: pd.DataFrame | None = None

    def get_summary_df(self) -> pd.DataFrame:
        """Get汇总 DataFrame"""
        data = []
        for item_result in self.item_results:
            data.append(
                {
                    "Test名称": item_result.get("name", "未知"),
                    "Model": item_result.get("model_id", "未知"),
                    "Status": item_result.get("status", "未知"),
                    "Accuracy": item_result.get("accuracy", 0),
                    "AverageLatency": item_result.get("avg_latency_ms", 0),
                    "AverageTPS": item_result.get("avg_tps", 0),
                    "Error": item_result.get("error", ""),
                }
            )

        return pd.DataFrame(data)

    def get_comparison_df(self) -> pd.DataFrame:
        """Get对比 DataFrame"""
        if self.comparison_data is not None:
            return self.comparison_data

        # if没has对比Data，从 item_results Generate
        data = []
        for item_result in self.item_results:
            if item_result.get("status") == "completed":
                data.append(
                    {
                        "Test名称": item_result.get("name", "未知"),
                        "Model": item_result.get("model_id", "未知"),
                        "Accuracy": item_result.get("accuracy", 0),
                        "AverageLatency": item_result.get("avg_latency_ms", 0),
                        "AverageTPS": item_result.get("avg_tps", 0),
                        "输入Tokens": item_result.get("total_input_tokens", 0),
                        "输出Tokens": item_result.get("total_output_tokens", 0),
                    }
                )

        return pd.DataFrame(data) if data else pd.DataFrame()


# ============================================================================
# 辅助函数
# ============================================================================


def _create_batch_csv_filename(item: BatchTestItem) -> str:
    """GenerateBatch Test唯一 CSV Filename"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in item.name)
    temp_dir = Path(gettempdir()) / "llm_batch_tests"
    temp_dir.mkdir(exist_ok=True)
    return str(temp_dir / f"batch_{safe_name}_{timestamp}.csv")


def _extract_metrics_from_dataframe(
    df: pd.DataFrame, item: BatchTestItem
) -> dict[str, Any]:
    """从Test Results DataFrame in提取关键指标"""
    if df.empty:
        return {
            "name": item.name,
            "model_id": item.model_id,
            "status": "completed",
            "error": "Empty results",
            "avg_latency_ms": 0,
            "avg_tps": 0,
            "accuracy": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    result = {
        "name": item.name,
        "model_id": item.model_id,
        "status": "completed",
        "avg_latency_ms": 0.0,
        "avg_tps": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "accuracy": 1.0,
    }

    # 提取 TTFT (Time To First Token) - Convertis毫seconds
    if "ttft" in df.columns:
        result["avg_latency_ms"] = float(df["ttft"].mean() * 1000)

    # 提取 TPS (Tokens Per Second)
    if "tps" in df.columns:
        result["avg_tps"] = float(df["tps"].mean())

    # 提取输入/输出 token 总数
    if "prefill_tokens" in df.columns:
        result["total_input_tokens"] = int(df["prefill_tokens"].sum())

    if "decode_tokens" in df.columns:
        result["total_output_tokens"] = int(df["decode_tokens"].sum())

    # Checkis否hasError
    if "error" in df.columns:
        error_count = int((~success_mask_from_error(df["error"])).sum())
        if error_count > 0:
            result["status"] = "partial_failure"
            result["error"] = f"{error_count} requests failed"

    return result


# ============================================================================
# Batch Test调度器
# ============================================================================


class BatchTestScheduler:
    """Batch Test调度器"""

    def __init__(
        self,
        config: BatchTestConfig,
        test_function: Callable,
        progress_callback: Callable[[BatchTestProgress], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.test_function = test_function
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.should_stop = False

    def stop(self):
        """停止Batch Test"""
        self.should_stop = True

    async def run(self) -> BatchTestResult:
        """
        运行Batch Test

        Returns:
            BatchTestResult: 批量Test Results
        """
        start_time = time.time()
        start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Filter启用Test items
        enabled_items = [item for item in self.config.items if item.enabled]

        # Initialize进度
        progress = BatchTestProgress(total_items=len(enabled_items))
        progress.start_time = start_time

        # Result存储
        item_results = []

        self._log(f"开始Batch Test: {self.config.name}")
        self._log(f"共 {len(enabled_items)}  tests items")

        if self.config.parallel:
            # 并行执行
            item_results = await self._run_parallel(enabled_items, progress)
        else:
            # 串行执行
            item_results = await self._run_sequential(enabled_items, progress)

        # CalculateStatistics
        end_time = time.time()
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - start_time

        progress.end_time = end_time

        # CreateResult
        result = BatchTestResult(
            batch_name=self.config.name,
            start_time=start_time_str,
            end_time=end_time_str,
            duration_seconds=duration,
            item_results=item_results,
            total_items=len(enabled_items),
            completed_items=progress.completed_items,
            failed_items=progress.failed_items,
        )

        # Generate对比Data
        result.comparison_data = result.get_comparison_df()

        self._log(f"批量Test completed: {self.config.name}")
        self._log(f"完成: {progress.completed_items}/{len(enabled_items)}")

        return result

    async def _run_sequential(
        self, items: list[BatchTestItem], progress: BatchTestProgress
    ) -> list[dict[str, Any]]:
        """串行执行Test"""
        results = []

        for item in items:
            # Check both internal flag and global stop signal
            if self.should_stop:
                item.status = "skipped"
                progress.skipped_items += 1
                results.append(item.to_dict())
                continue

            # Also check global batch stop flag (进程级 cancel_state)
            if is_batch_stop_requested():
                self.should_stop = True
                item.status = "skipped"
                progress.skipped_items += 1
                results.append(item.to_dict())
                continue

            progress.current_item = item.name
            item.status = "running"

            self._log(f"执行Test: {item.name} ({item.model_id})")

            try:
                item_result = await self._run_single_test(item)
                item.status = "completed"
                item.result = item_result
                results.append(item.to_dict())
                progress.completed_items += 1

            except Exception as e:
                item.status = "failed"
                item.error = str(e)
                results.append(item.to_dict())
                progress.failed_items += 1

                self._log(f"Test failed: {item.name} - {e}")

                if self.config.stop_on_error:
                    self._log("遇到Error，停止Batch Test")
                    break

            # Update进度
            if self.progress_callback:
                self.progress_callback(progress)

        return results

    async def _run_parallel(
        self, items: list[BatchTestItem], progress: BatchTestProgress
    ) -> list[dict[str, Any]]:
        """并行执行Test"""
        results = []

        # 分批执行
        max_parallel = min(self.config.max_parallel, len(items))
        self._log(f"并行执行: 最大并行数 {max_parallel}")

        for i in range(0, len(items), max_parallel):
            batch = items[i : i + max_parallel]

            # Check stop signal before starting batch
            if self.should_stop:
                break
            if is_batch_stop_requested():
                self.should_stop = True
                break

            # Create任务
            tasks = []
            for item in batch:
                if not item.enabled:
                    continue

                if self.should_stop:
                    item.status = "skipped"
                    results.append(item.to_dict())
                    progress.skipped_items += 1
                    continue

                item.status = "running"
                progress.current_item = f"批次 {i // max_parallel + 1}"

                tasks.append(self._run_single_test_wrapper(item))

            # 执行批次
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # ProcessResult
            for item, batch_result in zip(batch, batch_results, strict=True):
                if isinstance(batch_result, Exception):
                    item.status = "failed"
                    item.error = str(batch_result)
                    progress.failed_items += 1
                    self._log(f"Test failed: {item.name} - {batch_result}")
                else:
                    item.status = "completed"
                    item.result = batch_result
                    progress.completed_items += 1

                results.append(item.to_dict())

            # Update进度
            if self.progress_callback:
                self.progress_callback(progress)

        return results

    async def _run_single_test_wrapper(self, item: BatchTestItem) -> dict[str, Any]:
        """包装单 tests执行"""
        try:
            return await self._run_single_test(item)
        except Exception as e:
            raise

    async def _run_single_test(self, item: BatchTestItem) -> dict[str, Any]:
        """
        运行单 tests

        Args:
            item: Test itemsConfigure

        Returns:
            Test Results字典
        """
        # 1. Create mock UI 组件
        placeholder = _MockPlaceholder()
        progress_bar = _MockProgressBar()
        status_text = _MockStatusText()
        log_placeholder = _MockStatusText()
        output_placeholder = _MockPlaceholder()

        # 2. Create临时 CSV 文件
        csv_filename = _create_batch_csv_filename(item)

        try:
            # 3. Create BenchmarkRunner 实例（LatencyImport）
            BenchmarkRunner = _get_benchmark_runner()
            runner = BenchmarkRunner(
                placeholder=placeholder,
                progress_bar=progress_bar,
                status_text=status_text,
                api_base_url=item.api_base_url,
                model_id=item.model_id,
                tokenizer_option="API (usage field)",
                csv_filename=csv_filename,
                api_key=item.api_key,
                log_placeholder=log_placeholder,
                provider="OpenAI",
                dashboard=None,
                output_placeholder=output_placeholder,
                thinking_enabled=(
                    item.thinking_enabled if item.thinking_enabled else None
                ),
                thinking_budget=(
                    item.thinking_budget if item.thinking_budget > 0 else None
                ),
                reasoning_effort=item.reasoning_effort or None,
            )

            # 4. based on test_type 执行对应Test
            result_df = pd.DataFrame()

            if item.test_type == "concurrency":
                result_df = await runner.run_concurrency_test(
                    selected_concurrencies=[item.concurrency],
                    rounds_per_level=1,
                    max_tokens=item.max_tokens,
                    input_tokens_target=item.extra_params.get("input_tokens_target", 0),
                )

            elif item.test_type == "prefill":
                token_levels = item.extra_params.get(
                    "token_levels", [512, 1024, 2048, 4096]
                )
                result_df = await runner.run_prefill_test(
                    token_levels=token_levels,
                    requests_per_level=1,
                    max_tokens=item.max_tokens,
                )

            elif item.test_type == "long_context":
                context_lengths = item.extra_params.get(
                    "context_lengths", [1024, 2048, 4096, 8192]
                )
                result_df = await runner.run_long_context_test(
                    context_lengths=context_lengths,
                    rounds_per_level=1,
                    max_tokens=item.max_tokens,
                )

            else:
                raise ValueError(f"Unsupported test type: {item.test_type}")

            # 5. 提取指标
            result = _extract_metrics_from_dataframe(result_df, item)

            # 6. Cleanup临时文件
            try:
                if os.path.exists(csv_filename):
                    os.remove(csv_filename)
            except Exception as e:
                self._log(f"Warning: Could not delete temp file: {e}")

            return result

        except Exception as e:
            return {
                "name": item.name,
                "model_id": item.model_id,
                "status": "failed",
                "error": str(e),
                "avg_latency_ms": 0,
                "avg_tps": 0,
                "accuracy": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }

    def _log(self, message: str):
        """记录Log"""
        if self.log_callback:
            self.log_callback(message)


# ============================================================================
# Batch Test管理器
# ============================================================================


class BatchTestManager:
    """Batch Test管理器"""

    def __init__(self, save_dir: str = "batch_tests"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save_config(self, config: BatchTestConfig) -> bool:
        """Save批量Test Configuration"""
        try:
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in config.name
            )
            filename = safe_name.lower().replace(" ", "_") + ".json"
            filepath = self.save_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"Save Config失败: {e}")
            return False

    def load_config(self, name: str) -> BatchTestConfig | None:
        """Load批量Test Configuration"""
        try:
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name
            )
            pattern = safe_name.lower().replace(" ", "_") + ".json"
            matching_files = list(self.save_dir.glob(pattern))

            if not matching_files:
                return None

            with open(matching_files[0], encoding="utf-8") as f:
                data = json.load(f)

            return BatchTestConfig.from_dict(data)
        except Exception as e:
            print(f"Load Config失败: {e}")
            return None

    def list_configs(self) -> list[dict[str, Any]]:
        """列出所has批量Test Configuration"""
        configs = []
        for config_file in self.save_dir.glob("*.json"):
            try:
                with open(config_file, encoding="utf-8") as f:
                    data = json.load(f)

                configs.append(
                    {
                        "name": data.get("name", config_file.stem),
                        "description": data.get("description", ""),
                        "test_count": len(data.get("items", [])),
                        "file_time": datetime.fromtimestamp(
                            config_file.stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            except Exception:
                continue

        return sorted(configs, key=lambda x: x["file_time"], reverse=True)

    def save_result(self, result: BatchTestResult) -> bool:
        """Save批量Test Results"""
        try:
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_"
                for c in result.batch_name
            )
            filename = f"{safe_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.save_dir / "results" / filename
            filepath.parent.mkdir(exist_ok=True)

            # SaveResult
            result_data = {
                "batch_name": result.batch_name,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "duration_seconds": result.duration_seconds,
                "item_results": result.item_results,
                "total_items": result.total_items,
                "completed_items": result.completed_items,
                "failed_items": result.failed_items,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            # Save对比CSV
            csv_path = (
                self.save_dir
                / "results"
                / filepath.stem.replace(".json", "_comparison.csv")
            )
            result.get_comparison_df().to_csv(csv_path, index=False, encoding="utf-8")

            return True
        except Exception as e:
            print(f"SaveResult失败: {e}")
            return False

    def list_results(self) -> list[dict[str, Any]]:
        """列出所has批量Test Results"""
        results_dir = self.save_dir / "results"
        if not results_dir.exists():
            return []

        results = []
        for result_file in results_dir.glob("*.json"):
            try:
                with open(result_file, encoding="utf-8") as f:
                    data = json.load(f)

                results.append(
                    {
                        "batch_name": data.get("batch_name", result_file.stem),
                        "start_time": data.get("start_time", ""),
                        "end_time": data.get("end_time", ""),
                        "duration_seconds": data.get("duration_seconds", 0),
                        "total_items": data.get("total_items", 0),
                        "completed_items": data.get("completed_items", 0),
                        "failed_items": data.get("failed_items", 0),
                    }
                )
            except Exception:
                continue

        return sorted(results, key=lambda x: x["start_time"], reverse=True)


# 全局Batch Test管理器实例
batch_test_manager = BatchTestManager()
