"""
Standardize报告格式 (Standard Report Format)

Generate符合行业标准Evaluation报告，支持与other框架Result对比。

支持格式:
1. 标准 JSON 报告 (本 items目格式)
2. lm-evaluation-harness 兼容格式
3. OpenCompass 兼容格式
4. 详细失败案例分析报告

use方式:
    from core.standard_report import (
        StandardReport,
        ReportExporter,
        export_lm_eval_format,
        generate_failure_analysis
    )

    # 从Evaluation resultGenerate报告
    report = StandardReport.from_evaluation_result(result)

    # Exportisnot同格式
    exporter = ReportExporter(report)
    exporter.to_json("report.json")
    exporter.to_lm_eval_format("lm_eval_results.json")
    exporter.to_markdown("report.md")
"""

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# ============================================
# Data结构
# ============================================

@dataclass
class ModelInfo:
    """Model信息"""
    model_id: str
    model_type: str = "unknown"  # standard, thinking, code
    provider: str = ""
    api_base_url: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentInfo:
    """运行环境信息"""
    python_version: str = ""
    os_name: str = ""
    os_version: str = ""
    hostname: str = ""
    git_hash: str = ""
    library_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> 'EnvironmentInfo':
        """捕获当前环境信息"""
        info = cls()

        # Python Version
        info.python_version = platform.python_version()

        # 操作系统
        info.os_name = platform.system()
        info.os_version = platform.release()

        # 主机名
        try:
            info.hostname = platform.node()
        except:
            pass

        # Git hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info.git_hash = result.stdout.strip()[:8]
        except:
            pass

        # 关键库Version
        try:
            import torch
            info.library_versions["torch"] = torch.__version__
        except ImportError:
            pass

        try:
            import transformers
            info.library_versions["transformers"] = transformers.__version__
        except ImportError:
            pass

        try:
            import datasets
            info.library_versions["datasets"] = datasets.__version__
        except ImportError:
            pass

        return info

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetMetrics:
    """单Dataset指标"""
    dataset_name: str
    alias: str = ""

    # 核心指标
    accuracy: float = 0.0
    accuracy_stderr: float = 0.0

    # 样本Statistics
    total_samples: int = 0
    correct_samples: int = 0

    # Confidence Interval
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    # 额外指标
    normalized_accuracy: float = 0.0
    f1_score: float = 0.0

    # 分类别Statistics
    per_category: dict[str, float] = field(default_factory=dict)

    # Performance Metrics
    avg_ttft_ms: float = 0.0
    avg_tps: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0

    # 时间
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureCase:
    """失败案例"""
    sample_id: int
    question: str
    expected_answer: str
    predicted_answer: str
    model_response: str = ""
    category: str = ""
    failure_type: str = ""  # parse_error, wrong_answer, empty_response
    analysis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question[:200] + "..." if len(self.question) > 200 else self.question,
            "expected": self.expected_answer,
            "predicted": self.predicted_answer,
            "category": self.category,
            "failure_type": self.failure_type,
            "analysis": self.analysis
        }


@dataclass
class StandardReport:
    """
    StandardizeEvaluation报告

    兼容 lm-evaluation-harness Result格式
    """
    # 元信息
    report_id: str = ""
    created_at: str = ""
    version: str = "1.0"

    # Model信息
    model: ModelInfo = field(default_factory=lambda: ModelInfo("unknown"))

    # 环境信息
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)

    # Config
    config: dict[str, Any] = field(default_factory=dict)

    # Data集Result
    results: dict[str, DatasetMetrics] = field(default_factory=dict)

    # 汇总指标
    aggregate: dict[str, float] = field(default_factory=dict)

    # 失败案例分析
    failure_analysis: dict[str, list[FailureCase]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.report_id:
            self.report_id = hashlib.md5(
                f"{self.model.model_id}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @classmethod
    def from_evaluation_result(
        cls,
        result,  # EvaluationResult
        model_info: ModelInfo | None = None,
        config: dict[str, Any] | None = None
    ) -> 'StandardReport':
        """从 EvaluationResult Create标准报告"""
        report = cls()

        # Model信息
        if model_info:
            report.model = model_info
        else:
            report.model = ModelInfo(model_id=result.model_id)

        # 环境信息
        report.environment = EnvironmentInfo.capture()

        # Config
        report.config = config or result.config

        # Data集Result
        metrics = DatasetMetrics(
            dataset_name=result.dataset_name,
            accuracy=result.accuracy,
            total_samples=result.total_samples,
            correct_samples=result.correct_samples,
            per_category=result.by_category,
            duration_seconds=result.duration_seconds
        )

        # 扩展指标
        if hasattr(result, 'extended_metrics') and result.extended_metrics:
            metrics.accuracy_stderr = result.extended_metrics.get('stderr', 0)
            metrics.ci_lower = result.extended_metrics.get('ci_lower', 0)
            metrics.ci_upper = result.extended_metrics.get('ci_upper', 0)

        # Performance Metrics
        if hasattr(result, 'performance_stats') and result.performance_stats:
            metrics.avg_ttft_ms = result.performance_stats.get('avg_ttft_ms', 0)
            metrics.avg_tps = result.performance_stats.get('avg_tps', 0)
            metrics.avg_latency_ms = result.performance_stats.get('avg_latency_ms', 0)
            metrics.total_tokens = (
                result.performance_stats.get('total_input_tokens', 0) +
                result.performance_stats.get('total_output_tokens', 0)
            )

        report.results[result.dataset_name] = metrics

        # 失败案例分析
        if hasattr(result, 'details'):
            failures = []
            for detail in result.details:
                if not detail.is_correct and not detail.error:
                    failure = FailureCase(
                        sample_id=detail.sample_id,
                        question=detail.prompt[:500] if detail.prompt else "",
                        expected_answer=detail.expected,
                        predicted_answer=detail.predicted,
                        model_response=detail.response[:500] if detail.response else "",
                        category=detail.category if hasattr(detail, 'category') else "",
                        failure_type=cls._classify_failure(detail)
                    )
                    failures.append(failure)

            if failures:
                report.failure_analysis[result.dataset_name] = failures[:50]  # 限制数量

        # 汇总
        report._compute_aggregate()

        return report

    @classmethod
    def from_multiple_results(
        cls,
        results: list,  # List[EvaluationResult]
        model_info: ModelInfo | None = None,
        config: dict[str, Any] | None = None
    ) -> 'StandardReport':
        """从多Evaluation resultCreate汇总报告"""
        report = cls()

        if model_info:
            report.model = model_info
        elif results:
            report.model = ModelInfo(model_id=results[0].model_id)

        report.environment = EnvironmentInfo.capture()
        report.config = config or {}

        for result in results:
            single_report = cls.from_evaluation_result(result)
            report.results.update(single_report.results)
            report.failure_analysis.update(single_report.failure_analysis)

        report._compute_aggregate()

        return report

    @staticmethod
    def _classify_failure(detail) -> str:
        """分类失败类型"""
        if not detail.predicted:
            return "empty_response"
        if detail.predicted == "PARSE_ERROR" or detail.predicted == "ERROR":
            return "parse_error"
        return "wrong_answer"

    def _compute_aggregate(self):
        """Calculate汇总指标"""
        if not self.results:
            return

        accuracies = [r.accuracy for r in self.results.values()]
        self.aggregate = {
            "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "total_samples": sum(r.total_samples for r in self.results.values()),
            "total_correct": sum(r.correct_samples for r in self.results.values()),
            "num_datasets": len(self.results)
        }

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        return {
            "report_id": self.report_id,
            "created_at": self.created_at,
            "version": self.version,
            "model": self.model.to_dict(),
            "environment": self.environment.to_dict(),
            "config": self.config,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "aggregate": self.aggregate,
            "failure_analysis": {
                k: [f.to_dict() for f in v]
                for k, v in self.failure_analysis.items()
            }
        }


# ============================================
# 报告Export器
# ============================================

class ReportExporter:
    """
    报告Export器

    支持多种输出格式
    """

    def __init__(self, report: StandardReport):
        self.report = report

    def to_json(self, filepath: str, indent: int = 2) -> str:
        """Exportis标准 JSON 格式"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.report.to_dict(), f, ensure_ascii=False, indent=indent)

        return filepath

    def to_lm_eval_format(self, filepath: str) -> str:
        """
        Exportis lm-evaluation-harness 兼容格式

        格式:
        {
            "results": {
                "task_name": {
                    "acc": 0.85,
                    "acc_stderr": 0.02,
                    "alias": "Task Name"
                }
            },
            "configs": {...},
            "versions": {...},
            "n-shot": {...}
        }
        """
        lm_eval_result = {
            "results": {},
            "configs": {},
            "versions": {},
            "n-shot": {},
            "config": {
                "model": self.report.model.model_id,
                "model_args": self.report.model.parameters
            },
            "git_hash": self.report.environment.git_hash
        }

        for name, metrics in self.report.results.items():
            # Convert指标名称is lm-eval 格式
            lm_eval_result["results"][name] = {
                "acc": metrics.accuracy,
                "acc_stderr": metrics.accuracy_stderr,
                "acc,none": metrics.accuracy,  # lm-eval 新格式
                "acc_stderr,none": metrics.accuracy_stderr,
                "alias": metrics.alias or name
            }

            # Add额外指标
            if metrics.normalized_accuracy > 0:
                lm_eval_result["results"][name]["acc_norm"] = metrics.normalized_accuracy

            # Version信息
            lm_eval_result["versions"][name] = 1

            # N-shot 信息
            lm_eval_result["n-shot"][name] = self.report.config.get("num_shots", 0)

        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(lm_eval_result, f, ensure_ascii=False, indent=2)

        return filepath

    def to_opencompass_format(self, filepath: str) -> str:
        """
        Exportis OpenCompass 兼容格式
        """
        oc_result = {
            "model": self.report.model.model_id,
            "time": self.report.created_at,
            "results": {}
        }

        for name, metrics in self.report.results.items():
            oc_result["results"][name] = {
                "accuracy": metrics.accuracy * 100,  # OpenCompass use百分比
                "score": metrics.accuracy * 100,
                "details": {
                    "total": metrics.total_samples,
                    "correct": metrics.correct_samples
                }
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(oc_result, f, ensure_ascii=False, indent=2)

        return filepath

    def to_markdown(self, filepath: str) -> str:
        """Export as Markdown 格式报告"""
        lines = []

        # 标题
        lines.append("# ModelEvaluation报告")
        lines.append("")
        lines.append(f"**报告 ID**: {self.report.report_id}")
        lines.append(f"**Generate时间**: {self.report.created_at}")
        lines.append("")

        # Model信息
        lines.append("## Model信息")
        lines.append("")
        lines.append(f"- **Model ID**: {self.report.model.model_id}")
        lines.append(f"- **Model类型**: {self.report.model.model_type}")
        lines.append(f"- **Provider**: {self.report.model.provider}")
        lines.append("")

        # EvaluationResult
        lines.append("## EvaluationResult")
        lines.append("")
        lines.append("| Dataset | Accuracy | 标准误差 | Sample count | 正确数 |")
        lines.append("|--------|--------|----------|--------|--------|")

        for name, metrics in self.report.results.items():
            stderr_str = f"±{metrics.accuracy_stderr:.4f}" if metrics.accuracy_stderr > 0 else "-"
            lines.append(
                f"| {name} | {metrics.accuracy:.4f} | {stderr_str} | "
                f"{metrics.total_samples} | {metrics.correct_samples} |"
            )

        lines.append("")

        # 汇总
        if self.report.aggregate:
            lines.append("### 汇总")
            lines.append("")
            lines.append(f"- **AverageAccuracy**: {self.report.aggregate.get('avg_accuracy', 0):.4f}")
            lines.append(f"- **总Sample count**: {self.report.aggregate.get('total_samples', 0)}")
            lines.append(f"- **Dataset数量**: {self.report.aggregate.get('num_datasets', 0)}")
            lines.append("")

        # 失败案例分析
        if self.report.failure_analysis:
            lines.append("## 失败案例分析")
            lines.append("")

            for dataset, failures in self.report.failure_analysis.items():
                lines.append(f"### {dataset}")
                lines.append("")

                # 按失败类型Statistics
                failure_types = {}
                for f in failures:
                    ft = f.failure_type or "unknown"
                    failure_types[ft] = failure_types.get(ft, 0) + 1

                lines.append("**失败类型分布:**")
                for ft, count in failure_types.items():
                    lines.append(f"- {ft}: {count}")
                lines.append("")

                # 示例
                lines.append("**示例 (前5):**")
                lines.append("")
                for i, f in enumerate(failures[:5]):
                    lines.append(f"**案例 {i+1}** (ID: {f.sample_id})")
                    lines.append(f"- 期望: `{f.expected_answer}`")
                    lines.append(f"- 预测: `{f.predicted_answer}`")
                    lines.append(f"- 类型: {f.failure_type}")
                    lines.append("")

        # 环境信息
        lines.append("## 环境信息")
        lines.append("")
        lines.append(f"- Python: {self.report.environment.python_version}")
        lines.append(f"- OS: {self.report.environment.os_name} {self.report.environment.os_version}")
        if self.report.environment.git_hash:
            lines.append(f"- Git: {self.report.environment.git_hash}")
        lines.append("")

        content = "\n".join(lines)

        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return filepath

    def to_csv(self, filepath: str) -> str:
        """Export as CSV 格式 (简易表格)"""
        import csv

        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 表头
            writer.writerow([
                "dataset", "accuracy", "stderr", "ci_lower", "ci_upper",
                "total_samples", "correct_samples", "avg_ttft_ms", "avg_tps"
            ])

            # Data
            for name, metrics in self.report.results.items():
                writer.writerow([
                    name,
                    f"{metrics.accuracy:.6f}",
                    f"{metrics.accuracy_stderr:.6f}",
                    f"{metrics.ci_lower:.6f}",
                    f"{metrics.ci_upper:.6f}",
                    metrics.total_samples,
                    metrics.correct_samples,
                    f"{metrics.avg_ttft_ms:.2f}",
                    f"{metrics.avg_tps:.2f}"
                ])

        return filepath


# ============================================
# 失败分析工具
# ============================================

def generate_failure_analysis(
    failures: list[FailureCase],
    top_n: int = 10
) -> dict[str, Any]:
    """
    Generate详细失败分析报告

    Args:
        failures: 失败案例列表
        top_n: Return详细案例数量

    Returns:
        分析报告字典
    """
    if not failures:
        return {"total_failures": 0}

    # 按失败类型Group
    by_type: dict[str, list[FailureCase]] = {}
    for f in failures:
        ft = f.failure_type or "unknown"
        if ft not in by_type:
            by_type[ft] = []
        by_type[ft].append(f)

    # 按类别Group
    by_category: dict[str, int] = {}
    for f in failures:
        cat = f.category or "unknown"
        by_category[cat] = by_category.get(cat, 0) + 1

    # Generate分析
    analysis = {
        "total_failures": len(failures),
        "by_failure_type": {k: len(v) for k, v in by_type.items()},
        "by_category": by_category,
        "top_cases": [f.to_dict() for f in failures[:top_n]],
        "recommendations": []
    }

    # GenerateSuggestion
    if by_type.get("parse_error", 0) > len(failures) * 0.2:
        analysis["recommendations"].append(
            "ParseError rate较高，SuggestionCheckAnswer提取逻辑or启用增强Parse器"
        )

    if by_type.get("empty_response", 0) > 0:
        analysis["recommendations"].append(
            "存in空响应，可能is API Erroror超时问题"
        )

    return analysis


# ============================================
# 便捷函数
# ============================================

def export_lm_eval_format(result, filepath: str) -> str:
    """快速Export lm-eval 格式"""
    report = StandardReport.from_evaluation_result(result)
    exporter = ReportExporter(report)
    return exporter.to_lm_eval_format(filepath)


def export_all_formats(result, output_dir: str, prefix: str = "report") -> dict[str, str]:
    """Export所has格式"""
    report = StandardReport.from_evaluation_result(result)
    exporter = ReportExporter(report)

    os.makedirs(output_dir, exist_ok=True)

    return {
        "json": exporter.to_json(os.path.join(output_dir, f"{prefix}.json")),
        "lm_eval": exporter.to_lm_eval_format(os.path.join(output_dir, f"{prefix}_lm_eval.json")),
        "markdown": exporter.to_markdown(os.path.join(output_dir, f"{prefix}.md")),
        "csv": exporter.to_csv(os.path.join(output_dir, f"{prefix}.csv"))
    }


def load_report(filepath: str) -> StandardReport:
    """从 JSON 文件Load报告"""
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    report = StandardReport()
    report.report_id = data.get("report_id", "")
    report.created_at = data.get("created_at", "")
    report.version = data.get("version", "1.0")
    report.config = data.get("config", {})
    report.aggregate = data.get("aggregate", {})

    # Model信息
    model_data = data.get("model", {})
    report.model = ModelInfo(**model_data) if model_data else ModelInfo("unknown")

    # 环境信息
    env_data = data.get("environment", {})
    if env_data:
        report.environment = EnvironmentInfo(**env_data)

    # Result
    for name, metrics_data in data.get("results", {}).items():
        report.results[name] = DatasetMetrics(**metrics_data)

    return report


def compare_reports(report1: StandardReport, report2: StandardReport) -> dict[str, Any]:
    """比较两报告"""
    comparison = {
        "report1_id": report1.report_id,
        "report2_id": report2.report_id,
        "model1": report1.model.model_id,
        "model2": report2.model.model_id,
        "datasets": {},
        "summary": {}
    }

    # 找出共同Dataset
    common_datasets = set(report1.results.keys()) & set(report2.results.keys())

    for ds in common_datasets:
        m1 = report1.results[ds]
        m2 = report2.results[ds]

        diff = m2.accuracy - m1.accuracy
        comparison["datasets"][ds] = {
            "report1_accuracy": m1.accuracy,
            "report2_accuracy": m2.accuracy,
            "difference": diff,
            "better": "report2" if diff > 0 else "report1" if diff < 0 else "equal"
        }

    # 汇总
    if common_datasets:
        avg_diff = sum(
            comparison["datasets"][ds]["difference"]
            for ds in common_datasets
        ) / len(common_datasets)

        comparison["summary"] = {
            "common_datasets": len(common_datasets),
            "avg_difference": avg_diff,
            "overall_better": "report2" if avg_diff > 0 else "report1" if avg_diff < 0 else "equal"
        }

    return comparison
