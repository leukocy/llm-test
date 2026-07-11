"""
Result Comparator Module
Result对比模块 - 用于Loadand对比多Test Results JSON 文件

功能:
1. Load多Test Results JSON 文件
2. 并排Display关键指标（Accuracy、耗时、token use）
3. Generate对比图表（use matplotlib）
4. Export对比报告

use方式:
    from core.result_comparator import ResultComparator

    # Create对比器
    comparator = ResultComparator()

    # AddResult文件
    comparator.add_result("result/model_a_mmlu_20240101_120000.json", label="Model A")
    comparator.add_result("result/model_b_mmlu_20240102_120000.json", label="Model B")

    # Generate对比报告
    report = comparator.generate_comparison_report()

    # Generate图表
    comparator.plot_comparison(output_path="comparison.png")
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from evaluators.base_evaluator import EvaluationResult

# ============================================
# Data结构
# ============================================


@dataclass
class ResultMetadata:
    """Result文件元Data"""

    filepath: str
    label: str
    model_id: str = ""
    dataset_name: str = ""
    timestamp: str = ""
    file_size: int = 0

    # 关键指标
    accuracy: float = 0.0
    correct_samples: int = 0
    total_samples: int = 0
    duration_seconds: float = 0.0

    # Performance Metrics
    avg_ttft_ms: float = 0.0
    avg_tps: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Config信息
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_result(
        cls, result: EvaluationResult, filepath: str, label: str
    ) -> "ResultMetadata":
        """从 EvaluationResult Create元Data"""
        # Get性能Statistics
        perf_stats = result.performance_stats or {}

        return cls(
            filepath=filepath,
            label=label,
            model_id=result.model_id,
            dataset_name=result.dataset_name,
            timestamp=result.timestamp,
            file_size=os.path.getsize(filepath) if os.path.exists(filepath) else 0,
            accuracy=result.accuracy,
            correct_samples=result.correct_samples,
            total_samples=result.total_samples,
            duration_seconds=result.duration_seconds,
            avg_ttft_ms=perf_stats.get("avg_ttft_ms", 0.0),
            avg_tps=perf_stats.get("avg_tps", 0.0),
            total_input_tokens=perf_stats.get("total_input_tokens", 0),
            total_output_tokens=perf_stats.get("total_output_tokens", 0),
            config=result.config or {},
        )


@dataclass
class ComparisonEntry:
    """单对比条目"""

    metric_name: str
    values: dict[str, float | int | str] = field(default_factory=dict)  # label -> value
    unit: str = ""
    higher_is_better: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "metric_name": self.metric_name,
            "values": self.values,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "description": self.description,
            "best": self.get_best(),
            "winner": self.get_winner(),
            "difference_percent": self.get_difference_percent(),
        }
        return result

    def get_best(self) -> float | int | str | None:
        """GetBest值"""
        if not self.values:
            return None

        # Filter掉 None 值
        valid_values = {k: v for k, v in self.values.items() if v is not None}
        if not valid_values:
            return None

        # 对于字符串类型or higher_is_better is None，Return一值
        if self.higher_is_better is None or any(
            not isinstance(v, (int, float)) for v in valid_values.values()
        ):
            return next(iter(valid_values.values()))

        if self.higher_is_better:
            return max(valid_values.values())
        else:
            # 对于 higher_is_better=False 的指标（如 TTFT、TPOT 等延迟指标）
            # 需要过滤掉 0 值，因为 0 通常表示测试失败或数据无效
            positive_values = {k: v for k, v in valid_values.items() if v > 0}
            if not positive_values:
                return None  # 所有值都是 0 或负数，没有有效最佳值
            return min(positive_values.values())

    def get_winner(self) -> str | None:
        """Get获胜者"""
        if not self.values:
            return None
        best = self.get_best()
        for label, value in self.values.items():
            if value == best:
                return label
        return None

    def get_difference_percent(self) -> dict[str, float] | None:
        """Get与Best值差异百分比"""
        if len(self.values) < 2:
            return None
        best = self.get_best()
        if best is None or best == 0:
            return None
        return {
            label: (
                ((value - best) / best * 100) if isinstance(best, (int, float)) else 0
            )
            for label, value in self.values.items()
        }


@dataclass
class ComparisonReport:
    """对比报告"""

    comparison_id: str = ""
    created_at: str = ""

    # Result文件信息
    results: list[ResultMetadata] = field(default_factory=list)

    # 对比条目
    comparisons: list[ComparisonEntry] = field(default_factory=list)

    # 汇总信息
    summary: dict[str, Any] = field(default_factory=dict)

    # Raw data
    raw_results: dict[str, EvaluationResult] = field(default_factory=dict)

    def __post_init__(self):
        if not self.comparison_id:
            import hashlib

            self.comparison_id = hashlib.md5(
                datetime.now().isoformat().encode()
            ).hexdigest()[:12]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "created_at": self.created_at,
            "results": [r.to_dict() for r in self.results],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "summary": self.summary,
        }


# ============================================
# Result对比器
# ============================================


class ResultComparator:
    """
    Result对比器

    支持Loadand对比多Test Results JSON 文件
    """

    # 定义Comparison Metrics
    METRIC_DEFINITIONS = [
        {
            "name": "Accuracy",
            "key": "accuracy",
            "unit": "%",
            "higher_is_better": True,
            "description": "Model answer正确比例",
            "format": lambda x: f"{x * 100:.2f}%",
        },
        {
            "name": "正确Sample count",
            "key": "correct_samples",
            "unit": "",
            "higher_is_better": True,
            "description": "回答正确Sample count量",
            "format": lambda x: f"{int(x)}",
        },
        {
            "name": "总Sample count",
            "key": "total_samples",
            "unit": "",
            "higher_is_better": True,
            "description": "总TestSample count量",
            "format": lambda x: f"{int(x)}",
        },
        {
            "name": "评估耗时",
            "key": "duration_seconds",
            "unit": "seconds",
            "higher_is_better": False,
            "description": "完成评估所花费总时间",
            "format": lambda x: f"{x:.2f}s",
        },
        {
            "name": "Average TTFT",
            "key": "avg_ttft_ms",
            "unit": "ms",
            "higher_is_better": False,
            "description": "Time To First Token - Average首 token Latency",
            "format": lambda x: f"{x:.1f}ms" if x > 0 else "N/A",
        },
        {
            "name": "Average TPS",
            "key": "avg_tps",
            "unit": "tokens/s",
            "higher_is_better": True,
            "description": "Tokens Per Second - AverageGeneration speed",
            "format": lambda x: f"{x:.1f}" if x > 0 else "N/A",
        },
        {
            "name": "输入 Token 总数",
            "key": "total_input_tokens",
            "unit": "tokens",
            "higher_is_better": False,
            "description": "所has请求输入 token 总数",
            "format": lambda x: f"{int(x)}",
        },
        {
            "name": "输出 Token 总数",
            "key": "total_output_tokens",
            "unit": "tokens",
            "higher_is_better": False,
            "description": "所has响应输出 token 总数",
            "format": lambda x: f"{int(x)}",
        },
        {
            "name": "Average Token use",
            "key": "avg_tokens_per_sample",
            "unit": "tokens",
            "higher_is_better": False,
            "description": "每 samplesAverage token use量",
            "format": lambda x: f"{x:.0f}" if x > 0 else "N/A",
            "compute": lambda r: (
                (r.total_input_tokens + r.total_output_tokens) / r.total_samples
                if r.total_samples > 0
                else 0
            ),
        },
        {
            "name": "Thinking mode",
            "key": "thinking_enabled",
            "unit": "",
            "higher_is_better": None,
            "description": "is否启用Thinking mode",
            "format": lambda x: "启用" if x else "Close",
        },
        {
            "name": "Thinking budget",
            "key": "thinking_budget",
            "unit": "tokens",
            "higher_is_better": None,
            "description": "思考 token 预算",
            "format": lambda x: f"{int(x)}" if x else "N/A",
        },
        {
            "name": "推理努力",
            "key": "reasoning_effort",
            "unit": "",
            "higher_is_better": None,
            "description": "推理努力级别",
            "format": lambda x: str(x) if x else "N/A",
        },
    ]

    def __init__(self, log_callback: Callable[[str], None] | None = None):
        """
        InitializeResult对比器

        Args:
            log_callback: LogCallback函数
        """
        self.log_callback = log_callback
        self.results: dict[str, EvaluationResult] = {}
        self.metadatas: dict[str, ResultMetadata] = {}
        self.labels: dict[str, str] = {}  # filepath -> label
        self.report: ComparisonReport | None = None

    def _log(self, message: str):
        """输出Log"""
        if self.log_callback:
            self.log_callback(message)
        print(f"[ResultComparator] {message}")

    def add_result(self, filepath: str, label: str | None = None) -> bool:
        """
        AddResult文件

        Args:
            filepath: JSON ResultFile path
            label: optionalDisplayLabel

        Returns:
            is否succeededLoad
        """
        if not os.path.exists(filepath):
            self._log(f"文件not存in: {filepath}")
            return False

        try:
            # LoadResult
            result = EvaluationResult.load_from_json(filepath)

            # GenerateLabel
            if label is None:
                label = f"{result.model_id} - {result.dataset_name}"
            else:
                label = label

            # 存储Result
            self.results[filepath] = result
            self.labels[filepath] = label

            # Create元Data
            metadata = ResultMetadata.from_result(result, filepath, label)
            self.metadatas[filepath] = metadata

            self._log(f"已Load: {label} (Accuracy: {result.accuracy:.2%})")
            return True

        except Exception as e:
            self._log(f"Load failed {filepath}: {e}")
            return False

    def add_results_from_dir(
        self, directory: str, pattern: str = "*.json", label_prefix: str = ""
    ) -> int:
        """
        从目录批量AddResult文件

        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            label_prefix: Label前缀

        Returns:
            succeededLoad文件数量
        """
        count = 0
        for filepath in Path(directory).rglob(pattern):
            # 跳过 summary 文件
            if "summary" in filepath.name.lower():
                continue

            # GenerateLabel
            stem = filepath.stem
            label = f"{label_prefix}{stem}" if label_prefix else stem

            if self.add_result(str(filepath), label):
                count += 1

        self._log(f"从目录 {directory} Load {count} Result文件")
        return count

    def compare_results(self) -> ComparisonReport:
        """
        对比所has已LoadResult

        Returns:
            ComparisonReport 对象
        """
        if len(self.results) < 2:
            self._log("Warning: 至少need 2 Result进行对比")
            # but仍然Generate报告

        # Create报告
        self.report = ComparisonReport(results=list(self.metadatas.values()))

        # 存储原始Result
        self.report.raw_results = self.results.copy()

        # Checkis否can对比（确保Dataset相同）
        datasets = {m.dataset_name for m in self.metadatas.values()}
        if len(datasets) > 1:
            self._log(f"Warning: 检测到not同Dataset: {datasets}")

        # Generate对比条目
        self._generate_comparisons()

        # Generate汇总
        self._generate_summary()

        return self.report

    def _generate_comparisons(self):
        """Generate对比条目"""
        if not self.report:
            return

        for metric_def in self.METRIC_DEFINITIONS:
            entry = ComparisonEntry(
                metric_name=metric_def["name"],
                unit=metric_def["unit"],
                higher_is_better=metric_def.get("higher_is_better", True),
                description=metric_def.get("description", ""),
            )

            # 收集各Result指标值
            for filepath, metadata in self.metadatas.items():
                label = self.labels[filepath]

                # Get指标值
                compute_fn = metric_def.get("compute")
                if compute_fn:
                    value = compute_fn(metadata)
                else:
                    value = getattr(metadata, metric_def["key"], None)

                # ifisConfigure items，从 config inGet
                if value is None and metric_def["key"] in metadata.config:
                    value = metadata.config[metric_def["key"]]

                entry.values[label] = value

            self.report.comparisons.append(entry)

    def _generate_summary(self):
        """Generate汇总信息"""
        if not self.report:
            return

        self.report.summary = {
            "total_results": len(self.results),
            "unique_datasets": len({m.dataset_name for m in self.metadatas.values()}),
            "unique_models": len({m.model_id for m in self.metadatas.values()}),
            "best_accuracy": 0.0,
            "best_accuracy_label": "",
            "fastest_duration": float("inf"),
            "fastest_duration_label": "",
        }

        # 找出BestAccuracy
        for metadata in self.metadatas.values():
            if metadata.accuracy > self.report.summary["best_accuracy"]:
                self.report.summary["best_accuracy"] = metadata.accuracy
                self.report.summary["best_accuracy_label"] = metadata.label

            if metadata.duration_seconds < self.report.summary["fastest_duration"]:
                self.report.summary["fastest_duration"] = metadata.duration_seconds
                self.report.summary["fastest_duration_label"] = metadata.label

    def generate_comparison_report(self) -> str:
        """
        Generate可读对比报告

        Returns:
            Format报告字符串
        """
        if self.report is None:
            self.compare_results()

        if not self.report:
            return "no对比Data"

        lines = [
            "=" * 80,
            "Test Results对比报告",
            "=" * 80,
            f"对比 ID: {self.report.comparison_id}",
            f"Generate时间: {self.report.created_at}",
            f"Comparison Results数: {self.report.summary.get('total_results', 0)}",
            "",
            "=" * 80,
            "关键指标对比",
            "=" * 80,
            "",
        ]

        # Generate表格
        for comparison in self.report.comparisons:
            lines.append(f"【{comparison.metric_name}】")
            lines.append(f"  描述: {comparison.description}")

            # 按值Sort
            sorted_values = sorted(
                comparison.values.items(),
                key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                reverse=comparison.higher_is_better,
            )

            for i, (label, value) in enumerate(sorted_values):
                # Checkis否is数值类型
                if isinstance(value, (int, float)):
                    if comparison.higher_is_better:
                        rank_emoji = (
                            "🥇"
                            if i == 0
                            else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                        )
                    else:
                        rank_emoji = (
                            "🥇"
                            if i == 0
                            else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                        )

                    # Format值
                    format_fn = None
                    for metric_def in self.METRIC_DEFINITIONS:
                        if metric_def["name"] == comparison.metric_name:
                            format_fn = metric_def.get("format")
                            break

                    if format_fn:
                        display_value = format_fn(value)
                    else:
                        display_value = f"{value}{comparison.unit}"

                    lines.append(f"  {rank_emoji} {label}: {display_value}")
                else:
                    lines.append(f"    {label}: {value}")

            # Display获胜者
            winner = comparison.get_winner()
            if winner:
                lines.append(f"  → 获胜者: {winner}")

            lines.append("")

        # 汇总部分
        lines.extend(
            [
                "=" * 80,
                "汇总信息",
                "=" * 80,
                f"BestAccuracy: {self.report.summary.get('best_accuracy', 0):.2%}",
                f"  - {self.report.summary.get('best_accuracy_label', 'N/A')}",
                f"最快评估速度: {self.report.summary.get('fastest_duration', 0):.2f}s",
                f"  - {self.report.summary.get('fastest_duration_label', 'N/A')}",
                "",
            ]
        )

        # Result文件详情
        lines.extend(["=" * 80, "Result文件详情", "=" * 80, ""])

        for metadata in self.report.results:
            lines.extend(
                [
                    f"文件: {metadata.label}",
                    f"  路径: {metadata.filepath}",
                    f"  Model: {metadata.model_id}",
                    f"  Dataset: {metadata.dataset_name}",
                    f"  时间: {metadata.timestamp}",
                    f"  大小: {metadata.file_size / 1024:.1f} KB",
                    "",
                ]
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def get_comparison_df(self) -> pd.DataFrame:
        """
        Get对比 DataFrame

        Returns:
            包含对比Data DataFrame
        """
        if self.report is None:
            self.compare_results()

        if not self.report:
            return pd.DataFrame()

        data = []
        for comparison in self.report.comparisons:
            row = {"指标": comparison.metric_name}
            row.update(comparison.values)
            data.append(row)

        return pd.DataFrame(data)

    def plot_comparison(
        self,
        output_path: str | None = None,
        metrics: list[str] | None = None,
        figsize: tuple[int, int] = (12, 8),
        dpi: int = 100,
    ) -> str | None:
        """
        Generate对比图表

        Args:
            output_path: 输出File path (None = DisplaybutnotSave)
            metrics: 要绘制指标列表 (None = 自动选择)
            figsize: 图表大小
            dpi: 图表分辨率

        Returns:
            SaveFile path，if output_path is None 则Return None
        """
        if self.report is None:
            self.compare_results()

        if not self.report or not self.report.comparisons:
            self._log("No data可绘制")
            return None

        try:
            import matplotlib

            matplotlib.use("Agg")  # use非交互式后端
            import matplotlib.pyplot as plt

            # Setin文字体
            plt.rcParams["font.sans-serif"] = [
                "SimHei",
                "Microsoft YaHei",
                "Arial Unicode MS",
            ]
            plt.rcParams["axes.unicode_minus"] = False

        except ImportError:
            self._log("matplotlib 未安装，no法Generate图表")
            return None

        # 自动选择要绘制指标
        if metrics is None:
            # 只绘制数值型且notisThinking mode指标
            metrics = [
                comp.metric_name
                for comp in self.report.comparisons
                if comp.higher_is_better is not None
                and any(isinstance(v, (int, float)) for v in comp.values.values())
            ]

        # Filter出has效对比条目
        valid_comparisons = [
            comp
            for comp in self.report.comparisons
            if comp.metric_name in metrics
            and any(
                isinstance(v, (int, float)) and v != 0 for v in comp.values.values()
            )
        ]

        if not valid_comparisons:
            self._log("No valid数值型指标可绘制")
            return None

        # Calculate子图布局
        n_plots = len(valid_comparisons)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        # 绘制每指标
        for i, comp in enumerate(valid_comparisons):
            ax = axes[i]

            # 提取Labeland值（Filter非数值）
            labels = []
            values = []
            for label, value in comp.values.items():
                if isinstance(value, (int, float)):
                    labels.append(label)
                    values.append(value)

            if not values:
                continue

            # CreateBar Chart
            colors = plt.cm.Set3(range(len(labels)))
            bars = ax.bar(labels, values, color=colors)

            # Set标题andLabel
            ax.set_title(comp.metric_name, fontsize=12, fontweight="bold")
            ax.set_ylabel(comp.unit if comp.unit else "值")
            ax.tick_params(axis="x", rotation=45)

            # Add数值Label
            for bar, value in zip(bars, values, strict=True):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.2f}" if isinstance(value, float) else f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # AddGrid
            ax.grid(axis="y", alpha=0.3)

        # 隐藏多余子图
        for i in range(len(valid_comparisons), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # SaveorDisplay
        if output_path:
            # 确保目录存in
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            self._log(f"图表Saved: {output_path}")
            plt.close()
            return output_path
        else:
            # Display图表（in非交互式环境in可能not工作）
            try:
                plt.show()
            except Exception:
                pass
            plt.close()
            return None

    def save_report(self, output_path: str, format: str = "json") -> str:
        """
        Save报告

        Args:
            output_path: 输出File path
            format: 格式 (json, txt, csv)

        Returns:
            SaveFile path
        """
        if self.report is None:
            self.compare_results()

        # 确保目录存in
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        if format == "json":
            # Save JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.report.to_dict(), f, ensure_ascii=False, indent=2)

        elif format == "txt":
            # Save文本报告
            report_text = self.generate_comparison_report()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)

        elif format == "csv":
            # Save CSV
            df = self.get_comparison_df()
            df.to_csv(output_path, index=False, encoding="utf-8")

        else:
            raise ValueError(f"Not supported格式: {format}")

        self._log(f"报告Saved: {output_path}")
        return output_path

    def clear(self):
        """清除所hasResult"""
        self.results.clear()
        self.metadatas.clear()
        self.labels.clear()
        self.report = None


# ============================================
# 便捷函数
# ============================================


def compare_files(
    filepaths: list[str],
    labels: list[str] | None = None,
    output_path: str | None = None,
    plot_path: str | None = None,
) -> ComparisonReport:
    """
    快速对比多Result文件

    Args:
        filepaths: ResultFile path列表
        labels: optionalLabel列表
        output_path: optional报告输出路径
        plot_path: optional图表输出路径

    Returns:
        ComparisonReport 对象
    """
    comparator = ResultComparator()

    # AddResult
    if labels is None:
        labels = [None] * len(filepaths)

    for filepath, label in zip(filepaths, labels, strict=False):
        comparator.add_result(filepath, label)

    # Generate对比
    report = comparator.compare_results()

    # Save报告
    if output_path:
        ext = os.path.splitext(output_path)[1].lstrip(".")
        if ext in ["json", "txt", "csv"]:
            comparator.save_report(output_path, format=ext)
        else:
            comparator.save_report(output_path, format="json")

    # Generate图表
    if plot_path:
        comparator.plot_comparison(plot_path)

    return report


def load_comparison(filepath: str) -> ComparisonReport:
    """
    从文件Load对比报告

    Args:
        filepath: JSON 报告File path

    Returns:
        ComparisonReport 对象
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    report = ComparisonReport(
        comparison_id=data.get("comparison_id", ""),
        created_at=data.get("created_at", ""),
    )

    # 重建Result元Data
    for r_data in data.get("results", []):
        metadata = ResultMetadata(**r_data)
        report.results.append(metadata)

    # 重建对比条目
    for c_data in data.get("comparisons", []):
        entry = ComparisonEntry(
            metric_name=c_data["metric_name"],
            values=c_data["values"],
            unit=c_data.get("unit", ""),
            higher_is_better=c_data.get("higher_is_better", True),
            description=c_data.get("description", ""),
        )
        report.comparisons.append(entry)

    report.summary = data.get("summary", {})

    return report


def print_comparison_summary(report: ComparisonReport):
    """
    打印对比摘要

    Args:
        report: ComparisonReport 对象
    """
    print("\n" + "=" * 60)
    print("Test Results对比摘要")
    print("=" * 60)

    for comp in report.comparisons:
        print(f"\n【{comp.metric_name}】")
        for label, value in comp.values.items():
            if isinstance(value, (int, float)):
                print(f"  {label}: {value}{comp.unit}")
            else:
                print(f"  {label}: {value}")
        winner = comp.get_winner()
        if winner:
            print(f"  → Best: {winner}")

    print("\n" + "=" * 60)
