"""
A/B Model Comparison系统 (Model Comparison)

支持in相同Test seton对比多 models性能，并进行Statistics显著性分析。

功能:
1. 多Model并行评估
2. 逐样本Comparative Analysis
3. Statistics显著性检验 (McNemar, Paired t-test)
4. 对比Visualization报告

use方式:
    from core.model_comparator import ModelComparator, ComparisonResult

    # Create对比器
    comparator = ModelComparator()

    # AddModel Configuration
    comparator.add_model("model_a", config_a)
    comparator.add_model("model_b", config_b)

    # 运行对比评估
    results = await comparator.run_comparison(datasets=["gsm8k", "mmlu"])

    # Generate对比报告
    report = comparator.generate_report()
"""

import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, cast

from utils.logger import LogLevel

# ============================================
# Data结构
# ============================================


@dataclass
class ModelConfig:
    """Model Configuration"""

    model_id: str
    api_base_url: str
    api_key: str = ""
    provider: str = "openai"
    label: str = ""  # 用于DisplayLabel
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.label:
            self.label = self.model_id


@dataclass
class SampleComparison:
    """单 samplesComparison Results"""

    sample_id: str
    question: str = ""
    expected_answer: str = ""

    # 各Model预测and正确性
    predictions: dict[str, str] = field(default_factory=dict)  # model_id -> prediction
    correctness: dict[str, bool] = field(default_factory=dict)  # model_id -> is_correct

    # 差异分析
    all_correct: bool = False
    all_wrong: bool = False
    disagreement: bool = False  # Model之间has分歧

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetComparison:
    """单DatasetComparison Results"""

    dataset_name: str
    total_samples: int = 0

    # 各ModelAccuracy
    accuracies: dict[str, float] = field(default_factory=dict)
    correct_counts: dict[str, int] = field(default_factory=dict)

    # 对比Statistics
    both_correct: int = 0  # 所hasModel都对
    both_wrong: int = 0  # 所hasModel都错
    disagreements: int = 0  # has分歧Sample count

    # Statistics检验Result
    statistical_tests: dict[str, Any] = field(default_factory=dict)

    # 逐样本对比
    sample_comparisons: list[SampleComparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "accuracies": self.accuracies,
            "correct_counts": self.correct_counts,
            "both_correct": self.both_correct,
            "both_wrong": self.both_wrong,
            "disagreements": self.disagreements,
            "statistical_tests": self.statistical_tests,
            "sample_count": len(self.sample_comparisons),
        }
        return result


@dataclass
class ComparisonResult:
    """完整Comparison Results"""

    comparison_id: str = ""
    created_at: str = ""

    # 参与对比Model
    models: list[str] = field(default_factory=list)
    model_labels: dict[str, str] = field(default_factory=dict)

    # 各DatasetComparison Results
    datasets: dict[str, DatasetComparison] = field(default_factory=dict)

    # 汇总
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.comparison_id:
            import hashlib

            # usedforsecurity=False: 非密码学用途, 仅对比 ID(审查 #12)
            self.comparison_id = hashlib.md5(
                f"{self.models}_{datetime.now().isoformat()}".encode(),
                usedforsecurity=False,
            ).hexdigest()[:12]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "created_at": self.created_at,
            "models": self.models,
            "model_labels": self.model_labels,
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "summary": self.summary,
        }


# ============================================
# Statistics检验
# ============================================


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict[str, Any]:
    """
    McNemar 检验 - 检验两 modelsAccuracy差异is否显著

    适用于配对二分类Data

    Args:
        correct_a: Model A 正确性列表
        correct_b: Model B 正确性列表

    Returns:
        {"statistic": ..., "p_value": ..., "significant": bool}
    """
    if len(correct_a) != len(correct_b):
        return {"error": "长度not匹配"}

    # Build 2x2 列联表
    # b01: A 对 B 错
    # b10: A 错 B 对
    b01 = sum(1 for a, b in zip(correct_a, correct_b, strict=False) if a and not b)
    b10 = sum(1 for a, b in zip(correct_a, correct_b, strict=False) if not a and b)

    n = b01 + b10

    if n == 0:
        return {
            "statistic": 0,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "两Model表现完全一致",
        }

    # McNemar 检验Statistics量 (带连续性校正)
    statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10) if n > 0 else 0

    # 近似 p 值 (use卡方分布)
    # 简化Calculate，not依赖 scipy
    p_value = _chi2_p_value(statistic, df=1)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "b01_count": b01,  # A 对 B 错
        "b10_count": b10,  # A 错 B 对
        "interpretation": _interpret_mcnemar(b01, b10, p_value),
    }


def _chi2_p_value(x: float, df: int = 1) -> float:
    """简化卡方分布 p 值Calculate"""
    if x <= 0:
        return 1.0

    # use近似公式
    # P(X > x) ≈ exp(-x/2) for df=1 when x is large
    if x > 10:
        return math.exp(-x / 2)

    # 对于较小 x，use查表近似
    # 卡方分布 df=1 临界值: 3.84 (α=0.05), 6.63 (α=0.01)
    if x >= 6.63:
        return 0.01
    elif x >= 5.02:
        return 0.025
    elif x >= 3.84:
        return 0.05
    elif x >= 2.71:
        return 0.10
    else:
        return 0.5


def _interpret_mcnemar(b01: int, b10: int, p_value: float) -> str:
    """解释 McNemar 检验Result"""
    if p_value >= 0.05:
        return "两Model表现no显著差异"

    if b01 > b10:
        return f"Model A 显著优于Model B (p={p_value:.4f})"
    else:
        return f"Model B 显著优于Model A (p={p_value:.4f})"


# ============================================
# 对比器
# ============================================


class ModelComparator:
    """
    Model Comparison器

    支持in相同Test seton对比多 models性能
    """

    def __init__(
        self,
        output_dir: str = "quality_results/comparisons",
        log_callback: Callable[[str], None] | None = None,
    ):
        self.output_dir = output_dir
        self.log_callback = log_callback

        self.models: dict[str, ModelConfig] = {}
        self.evaluators: dict[str, Any] = {}  # QualityEvaluator instances
        self.results: dict[str, dict[str, Any]] = {}  # model_id -> {dataset -> EvaluationResult}

        self.comparison_result: ComparisonResult | None = None

    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)
        print(f"[Comparator] {message}")

    def add_model(self, model_id: str, config: ModelConfig):
        """Add要对比Model"""
        self.models[model_id] = config
        self._log(f"AddModel: {config.label} ({model_id})")

    async def run_comparison(
        self,
        datasets: list[str],
        max_samples: int | None = None,
        num_shots: int = 0,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ComparisonResult:
        """
        运行对比评估

        Args:
            datasets: 要评估Dataset列表
            max_samples: 每Dataset最大Sample count
            num_shots: Few-shot 示例数
            progress_callback: 进度Callback

        Returns:
            ComparisonResult
        """
        if len(self.models) < 2:
            raise ValueError("对比need至少 2  models")
        if not datasets:
            raise ValueError("Select at least one benchmark dataset for comparison")

        self.comparison_result = None
        self.results = {}

        self._log(f"开始对比评估: {len(self.models)}  models, {len(datasets)} Dataset")

        total_steps = len(self.models) * len(datasets)
        current_step = 0

        # 对每 models运行评估
        for model_id, model_config in self.models.items():
            self._log(f"评估Model: {model_config.label}")
            self.results[model_id] = {}

            try:
                from core.quality_evaluator import QualityEvaluator, QualityTestConfig

                # CreateEvaluator
                evaluator = QualityEvaluator(
                    model_id=model_config.model_id,
                    api_base_url=model_config.api_base_url,
                    api_key=model_config.api_key,
                    provider=model_config.provider,
                    output_dir=os.path.join(self.output_dir, "raw_results"),
                    log_callback=cast("Callable[[str, LogLevel], None] | None", self.log_callback),
                )

                # Config
                config = QualityTestConfig(
                    datasets=datasets,
                    max_samples=max_samples,
                    num_shots=num_shots,
                    concurrency=5,
                )

                # 运行评估
                results = await evaluator.run_evaluation(config)
                self.results[model_id] = results

                for ds in datasets:
                    current_step += 1
                    if progress_callback:
                        progress_callback(
                            current_step / total_steps, f"{model_config.label} - {ds}"
                        )

            except Exception as e:
                self._log(f"Model {model_id} 评估失败: {e}")
                raise RuntimeError(f"Model {model_id} evaluation failed: {e}") from e

        # GenerateComparative Analysis
        self.comparison_result = self._analyze_comparison(datasets)

        # SaveResult
        self._save_comparison()

        return self.comparison_result

    def _analyze_comparison(self, datasets: list[str]) -> ComparisonResult:
        """分析Comparison Results"""
        result = ComparisonResult(
            models=list(self.models.keys()),
            model_labels={m: c.label for m, c in self.models.items()},
        )

        for dataset in datasets:
            ds_comparison = self._compare_dataset(dataset)
            if ds_comparison:
                result.datasets[dataset] = ds_comparison

        # Calculate汇总
        result.summary = self._compute_summary(result)

        return result

    def _compare_dataset(self, dataset: str) -> DatasetComparison | None:
        """对比单DatasetResult"""
        # 收集各ModelResult
        model_results = {}
        for model_id in self.models:
            if model_id in self.results and dataset in self.results[model_id]:
                model_results[model_id] = self.results[model_id][dataset]

        if len(model_results) != len(self.models):
            raise ValueError(f"Dataset '{dataset}' is missing results for one or more models")

        fingerprints = set()
        expected_ids: set[str] | None = None
        for model_id, result in model_results.items():
            provenance = (result.config or {}).get("dataset_provenance", {})
            if provenance.get("source") != "configured_dataset":
                raise ValueError(
                    f"Dataset '{dataset}' for {model_id} lacks verified benchmark provenance"
                )
            sample_hash = provenance.get("sample_sha256")
            few_shot_hash = provenance.get("few_shot_sha256")
            if not sample_hash or not few_shot_hash:
                raise ValueError(f"Dataset '{dataset}' for {model_id} has no sample fingerprint")
            fingerprints.add((sample_hash, few_shot_hash, provenance.get("selection_seed")))
            if result.total_samples <= 0 or result.total_samples != provenance.get("sample_count"):
                raise ValueError(f"Dataset '{dataset}' for {model_id} has an invalid sample count")
            ids = [str(detail.sample_id) for detail in result.details]
            if len(ids) != result.total_samples or len(set(ids)) != len(ids):
                raise ValueError(
                    f"Dataset '{dataset}' for {model_id} has incomplete sample details"
                )
            if expected_ids is None:
                expected_ids = set(ids)
            elif set(ids) != expected_ids:
                raise ValueError(f"Dataset '{dataset}' has different sample IDs across models")
        if len(fingerprints) != 1:
            raise ValueError(f"Dataset '{dataset}' uses different sample sets across models")

        # CreateComparison Results
        comparison = DatasetComparison(dataset_name=dataset)

        # 收集Accuracy
        for model_id, result in model_results.items():
            comparison.accuracies[model_id] = result.accuracy
            comparison.correct_counts[model_id] = result.correct_samples

        # Get样本详情
        first_result = list(model_results.values())[0]
        comparison.total_samples = first_result.total_samples

        # 逐样本对比
        if hasattr(first_result, "details") and first_result.details:
            sample_map: dict[Any, dict[str, Any]] = {}  # sample_id -> {model_id: detail}

            for model_id, result in model_results.items():
                if hasattr(result, "details"):
                    for detail in result.details:
                        sid = str(detail.sample_id)
                        if sid not in sample_map:
                            sample_map[sid] = {}
                        sample_map[sid][model_id] = detail

            # 分析每 samples
            correctness_lists: dict[str, list[bool]] = {m: [] for m in self.models}

            for sid, model_details in sample_map.items():
                sample_comp = SampleComparison(sample_id=sid)

                if model_details:
                    first_detail = list(model_details.values())[0]
                    sample_comp.question = first_detail.prompt[:200] if first_detail.prompt else ""
                    sample_comp.expected_answer = first_detail.correct_answer

                all_correct = True
                all_wrong = True

                for model_id in self.models:
                    if model_id in model_details:
                        detail = model_details[model_id]
                        is_correct = detail.is_correct
                        sample_comp.predictions[model_id] = detail.predicted_answer
                        sample_comp.correctness[model_id] = is_correct
                        correctness_lists[model_id].append(is_correct)

                        if is_correct:
                            all_wrong = False
                        else:
                            all_correct = False

                sample_comp.all_correct = all_correct
                sample_comp.all_wrong = all_wrong
                sample_comp.disagreement = not all_correct and not all_wrong

                comparison.sample_comparisons.append(sample_comp)

                if all_correct:
                    comparison.both_correct += 1
                elif all_wrong:
                    comparison.both_wrong += 1
                else:
                    comparison.disagreements += 1

            # Statistics检验 (成对比较)
            model_ids = list(self.models.keys())
            if len(model_ids) >= 2:
                for i, model_a in enumerate(model_ids):
                    for model_b in model_ids[i + 1 :]:
                        if model_a in correctness_lists and model_b in correctness_lists:
                            test_result = mcnemar_test(
                                correctness_lists[model_a], correctness_lists[model_b]
                            )
                            comparison.statistical_tests[f"{model_a}_vs_{model_b}"] = test_result

        return comparison

    def _compute_summary(self, result: ComparisonResult) -> dict[str, Any]:
        """Calculate汇总Statistics"""
        summary: dict[str, Any] = {
            "num_models": len(result.models),
            "num_datasets": len(result.datasets),
            "model_rankings": {},
            "overall_winner": None,
        }

        # Calculate各ModelAverageAccuracy排名
        model_avg_acc = {}
        for model_id in result.models:
            accs = [ds.accuracies.get(model_id, 0) for ds in result.datasets.values()]
            model_avg_acc[model_id] = sum(accs) / len(accs) if accs else 0

        # Sort
        sorted_models = sorted(model_avg_acc.items(), key=lambda x: x[1], reverse=True)

        for rank, (model_id, avg_acc) in enumerate(sorted_models, 1):
            summary["model_rankings"][model_id] = {
                "rank": rank,
                "avg_accuracy": avg_acc,
                "label": result.model_labels.get(model_id, model_id),
            }

        if sorted_models:
            summary["overall_winner"] = sorted_models[0][0]

        return summary

    def _save_comparison(self):
        """SaveComparison Results"""
        if not self.comparison_result:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"comparison_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.comparison_result.to_dict(), f, ensure_ascii=False, indent=2)

        self._log(f"Comparison ResultsSaved: {filepath}")


# ============================================
# 便捷函数
# ============================================


def load_comparison(filepath: str) -> ComparisonResult:
    """LoadComparison Results"""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    result = ComparisonResult(
        comparison_id=data.get("comparison_id", ""),
        created_at=data.get("created_at", ""),
        models=data.get("models", []),
        model_labels=data.get("model_labels", {}),
        summary=data.get("summary", {}),
    )

    for ds_name, ds_data in data.get("datasets", {}).items():
        result.datasets[ds_name] = DatasetComparison(
            dataset_name=ds_name,
            total_samples=ds_data.get("total_samples", 0),
            accuracies=ds_data.get("accuracies", {}),
            correct_counts=ds_data.get("correct_counts", {}),
            both_correct=ds_data.get("both_correct", 0),
            both_wrong=ds_data.get("both_wrong", 0),
            disagreements=ds_data.get("disagreements", 0),
            statistical_tests=ds_data.get("statistical_tests", {}),
        )

    return result
