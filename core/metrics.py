"""
Phase 3: 推理Model专用指标 (Thinking Metrics)

提供推理Model专用EvaluationMetric calculation，including:
- TTUT (Time To User Text): 从请求到正文开始时间
- Reasoning Token Ratio: 推理 Token 占比
- Quality/Cost Score: 质量/成本Score
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ThinkingMetricsResult:
    """推理Metric calculationResult"""
    # Latency指标
    ttft_ms: float | None = None  # Time To First Token (ms)
    ttut_ms: float | None = None  # Time To User Text (ms) - 首正文 Token
    ttr_ms: float | None = None   # Time To Reasoning (ms) - 首推理 Token
    total_time_ms: float | None = None  # 总耗时 (ms)
    reasoning_time_ms: float | None = None  # 推理阶段耗时 (ms)

    # Token 指标
    reasoning_tokens: int = 0  # 推理 Token 数
    content_tokens: int = 0    # 正文 Token 数
    total_tokens: int = 0      # 总 Token 数
    reasoning_ratio: float = 0.0  # 推理 Token 占比

    # 字符指标
    reasoning_chars: int = 0  # 推理字符数
    content_chars: int = 0    # 正文字符数
    reasoning_density: float = 0.0  # 推理密度 (推理字符/正文字符)

    # 成本指标
    estimated_cost_usd: float | None = None  # 预估成本 (USD)
    quality_per_dollar: float | None = None  # 质量/美元 (need外部传入质量分)

    # 元信息
    platform: str = ""
    model_id: str = ""


class ThinkingMetrics:
    """
    推理ModelMetric calculation器

    Usage:
        metrics = ThinkingMetrics(platform="mimo", model_id="mimo-v2-flash")

        metrics.record_request_start()

        for chunk in stream:
            if is_reasoning_chunk:
                metrics.record_reasoning_chunk(chunk)
            if is_content_chunk:
                metrics.record_content_chunk(chunk)

        metrics.record_request_end()
        metrics.set_usage(usage_dict)

        result = metrics.calculate()
        print(f"TTUT: {result.ttut_ms}ms")
    """

    # 各平台定价 (USD per 1M tokens)
    PRICING = {
        "mimo": {"input": 0.0, "output": 0.0},  # MiMo 免费
        "deepseek": {"input": 0.14, "output": 0.28},  # DeepSeek V3
        "zhipu": {"input": 0.5, "output": 0.5},  # 智谱 GLM-4
        "gemini": {"input": 0.075, "output": 0.3},  # Gemini Flash
        "volcano": {"input": 0.5, "output": 0.5},  # 火山引擎
        "aliyun": {"input": 0.5, "output": 0.5},  # 阿里百炼
        "siliconflow": {"input": 0.3, "output": 0.6},  # 硅基流动
        "openai": {"input": 15.0, "output": 60.0},  # o1
        "openrouter": {"input": 0.5, "output": 1.0},  # Average估计
    }

    def __init__(self, platform: str = "", model_id: str = ""):
        """
        InitializeMetric calculation器

        Args:
            platform: 平台标识
            model_id: ModelID
        """
        self.platform = platform
        self.model_id = model_id

        # 时间戳
        self._request_start: float | None = None
        self._request_end: float | None = None
        self._first_token_time: float | None = None
        self._first_reasoning_time: float | None = None
        self._first_content_time: float | None = None
        self._last_reasoning_time: float | None = None

        # 累积量
        self._reasoning_chars = 0
        self._content_chars = 0
        self._reasoning_tokens = 0
        self._content_tokens = 0
        self._total_tokens = 0

        # Usage 信息
        self._usage: dict[str, Any] | None = None

    def record_request_start(self):
        """记录请求Start time"""
        self._request_start = time.time()

    def record_request_end(self):
        """记录请求End time"""
        self._request_end = time.time()

    def record_first_token(self):
        """记录首 Token 到达时间"""
        if self._first_token_time is None:
            self._first_token_time = time.time()

    def record_reasoning_chunk(self, content: str):
        """
        记录推理内容块

        Args:
            content: 推理内容
        """
        if not content:
            return

        now = time.time()

        if self._first_reasoning_time is None:
            self._first_reasoning_time = now
            # ifis一 Token，也记录 TTFT
            if self._first_token_time is None:
                self._first_token_time = now

        self._last_reasoning_time = now
        self._reasoning_chars += len(content)

    def record_content_chunk(self, content: str):
        """
        记录正文内容块

        Args:
            content: 正文内容
        """
        if not content:
            return

        now = time.time()

        if self._first_content_time is None:
            self._first_content_time = now
            # ifis一 Token，也记录 TTFT
            if self._first_token_time is None:
                self._first_token_time = now

        self._content_chars += len(content)

    def set_usage(self, usage: dict[str, Any] | None):
        """
        Set Token use信息

        Args:
            usage: API Return usage 字典
        """
        self._usage = usage

        if usage:
            # 尝试提取各种可能字段
            self._total_tokens = usage.get("total_tokens", 0) or 0

            # 推理 Token (not同平台字段名not同)
            self._reasoning_tokens = (
                usage.get("reasoning_tokens", 0) or
                usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or
                0
            )

            # 正文 Token
            completion_tokens = usage.get("completion_tokens", 0) or 0
            self._content_tokens = completion_tokens - self._reasoning_tokens

            # if没has reasoning_tokens，尝试按字符比例估算
            if self._reasoning_tokens == 0 and self._reasoning_chars > 0:
                total_chars = self._reasoning_chars + self._content_chars
                if total_chars > 0:
                    self._reasoning_tokens = int(completion_tokens * (self._reasoning_chars / total_chars))
                    self._content_tokens = completion_tokens - self._reasoning_tokens

    def calculate(self, quality_score: float | None = None) -> ThinkingMetricsResult:
        """
        Calculate所has指标

        Args:
            quality_score: 外部传入质量分数 (0-10)，用于Calculate质量/成本比

        Returns:
            ThinkingMetricsResult: CalculateResult
        """
        result = ThinkingMetricsResult(
            platform=self.platform,
            model_id=self.model_id
        )

        # Latency指标
        if self._request_start:
            if self._first_token_time:
                result.ttft_ms = (self._first_token_time - self._request_start) * 1000

            if self._first_content_time:
                result.ttut_ms = (self._first_content_time - self._request_start) * 1000

            if self._first_reasoning_time:
                result.ttr_ms = (self._first_reasoning_time - self._request_start) * 1000

            if self._request_end:
                result.total_time_ms = (self._request_end - self._request_start) * 1000

        # 推理阶段耗时
        if self._first_reasoning_time and self._last_reasoning_time:
            result.reasoning_time_ms = (self._last_reasoning_time - self._first_reasoning_time) * 1000

        # Token 指标
        result.reasoning_tokens = self._reasoning_tokens
        result.content_tokens = self._content_tokens
        result.total_tokens = self._total_tokens

        if self._total_tokens > 0:
            result.reasoning_ratio = self._reasoning_tokens / self._total_tokens

        # 字符指标
        result.reasoning_chars = self._reasoning_chars
        result.content_chars = self._content_chars

        if self._content_chars > 0:
            result.reasoning_density = self._reasoning_chars / self._content_chars

        # 成本估算
        pricing = self.PRICING.get(self.platform, {"input": 0.5, "output": 0.5})
        prompt_tokens = self._usage.get("prompt_tokens", 0) if self._usage else 0
        completion_tokens = self._usage.get("completion_tokens", 0) if self._usage else 0

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        result.estimated_cost_usd = input_cost + output_cost

        # 质量/成本比
        if quality_score is not None and result.estimated_cost_usd and result.estimated_cost_usd > 0:
            result.quality_per_dollar = quality_score / result.estimated_cost_usd

        return result

    def reset(self):
        """Reset所hasStatus"""
        self._request_start = None
        self._request_end = None
        self._first_token_time = None
        self._first_reasoning_time = None
        self._first_content_time = None
        self._last_reasoning_time = None
        self._reasoning_chars = 0
        self._content_chars = 0
        self._reasoning_tokens = 0
        self._content_tokens = 0
        self._total_tokens = 0
        self._usage = None


def format_metrics_report(metrics: ThinkingMetricsResult) -> str:
    """
    Format指标报告

    Args:
        metrics: Metric calculationResult

    Returns:
        Format报告字符串
    """
    lines = [
        "===== 推理Model指标报告 =====",
        f"平台: {metrics.platform}",
        f"Model: {metrics.model_id}",
        "",
        "【Latency指标】",
        f"  TTFT (首Token): {metrics.ttft_ms:.0f}ms" if metrics.ttft_ms else "  TTFT: N/A",
        f"  TTR (首推理): {metrics.ttr_ms:.0f}ms" if metrics.ttr_ms else "  TTR: N/A",
        f"  TTUT (首正文): {metrics.ttut_ms:.0f}ms" if metrics.ttut_ms else "  TTUT: N/A",
        f"  总耗时: {metrics.total_time_ms:.0f}ms" if metrics.total_time_ms else "  总耗时: N/A",
        f"  推理阶段: {metrics.reasoning_time_ms:.0f}ms" if metrics.reasoning_time_ms else "  推理阶段: N/A",
        "",
        "【Token 指标】",
        f"  推理 Token: {metrics.reasoning_tokens}",
        f"  正文 Token: {metrics.content_tokens}",
        f"  总 Token: {metrics.total_tokens}",
        f"  推理占比: {metrics.reasoning_ratio:.1%}",
        "",
        "【字符指标】",
        f"  推理字符: {metrics.reasoning_chars}",
        f"  正文字符: {metrics.content_chars}",
        f"  推理密度: {metrics.reasoning_density:.2f}",
        "",
        "【成本指标】",
        f"  预估成本: ${metrics.estimated_cost_usd:.6f}" if metrics.estimated_cost_usd else "  预估成本: N/A",
        f"  质量/$ : {metrics.quality_per_dollar:.2f}" if metrics.quality_per_dollar else "  质量/$: N/A",
    ]

    return "\n".join(lines)


# ============================================
# 评估指标系统 (Evaluation Metrics)
# ============================================

import math
import random
import re
import statistics
from collections import Counter
from collections.abc import Callable
from typing import Tuple


@dataclass
class EvalMetricResult:
    """单评估指标CalculateResult"""
    name: str
    value: float
    count: int = 0
    stderr: float | None = None
    confidence_interval: tuple[float, float] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {"name": self.name, "value": self.value, "count": self.count}
        if self.stderr is not None:
            result["stderr"] = self.stderr
        if self.confidence_interval is not None:
            result["ci_lower"], result["ci_upper"] = self.confidence_interval
        if self.details:
            result["details"] = self.details
        return result


# ============================================
# 精确匹配指标
# ============================================

def exact_match(predictions: list[str], references: list[str]) -> float:
    """精确匹配率"""
    if not predictions or not references:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references, strict=False) if str(p).strip() == str(r).strip())
    return correct / len(predictions)


def accuracy(predictions: list[Any], references: list[Any]) -> float:
    """Accuracy (支持数值and字符串比较)"""
    if not predictions or not references:
        return 0.0

    correct = 0
    for pred, ref in zip(predictions, references, strict=False):
        pred_str = str(pred).strip().lower()
        ref_str = str(ref).strip().lower()

        if pred_str == ref_str:
            correct += 1
        else:
            # 尝试数值比较
            try:
                if abs(float(pred) - float(ref)) < 1e-6:
                    correct += 1
            except (ValueError, TypeError):
                pass

    return correct / len(predictions)


def normalized_accuracy(predictions: list[str], references: list[str], num_choices: int = 4) -> float:
    """NormalizeAccuracy (考虑随机猜测基准)"""
    acc = accuracy(predictions, references)
    random_baseline = 1.0 / num_choices
    if acc <= random_baseline:
        return 0.0
    return (acc - random_baseline) / (1 - random_baseline)


# ============================================
# 文本相似度指标
# ============================================

def _tokenize(text: str) -> list[str]:
    """简单分词"""
    if not text:
        return []
    return re.findall(r'\w+', text.lower())


def f1_score(prediction: str, reference: str) -> float:
    """Token 级别 F1 分数"""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def f1_score_batch(predictions: list[str], references: list[str]) -> float:
    """批量 F1 Average值"""
    scores = [f1_score(p, r) for p, r in zip(predictions, references, strict=False)]
    return sum(scores) / len(scores) if scores else 0.0


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Get n-gram"""
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def bleu_score(prediction: str, reference: str, max_n: int = 4) -> float:
    """BLEU 分数 (简化版)"""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        if not pred_ngrams:
            continue

        matches = sum((pred_ngrams & ref_ngrams).values())
        precision = matches / sum(pred_ngrams.values())
        precisions.append(precision)

    if not precisions or any(p == 0 for p in precisions):
        return 0.0

    avg_log_precision = sum(math.log(p) for p in precisions) / len(precisions)

    # 长度惩罚
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(pred_tokens))

    return bp * math.exp(avg_log_precision)


def bleu_score_batch(predictions: list[str], references: list[str]) -> float:
    """批量 BLEU Average值"""
    scores = [bleu_score(p, r) for p, r in zip(predictions, references, strict=False)]
    return sum(scores) / len(scores) if scores else 0.0


def _lcs_length(a: list[str], b: list[str]) -> int:
    """最长公共子序列长度"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-L 分数 (基于最长公共子序列)"""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = _lcs_length(pred_tokens, ref_tokens)

    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l_batch(predictions: list[str], references: list[str]) -> float:
    """批量 ROUGE-L F1 Average值"""
    scores = [rouge_l(p, r)["f1"] for p, r in zip(predictions, references, strict=False)]
    return sum(scores) / len(scores) if scores else 0.0


# ============================================
# 代码Generate指标
# ============================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k

    Args:
        n: 总Sample count
        c: viaTestSample count
        k: k 值
    """
    if n - c < k:
        return 1.0

    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)

    return 1.0 - result


def estimate_pass_at_k(num_samples: list[int], num_correct: list[int], k: int) -> float:
    """估计整体 pass@k"""
    total = sum(pass_at_k(n, c, k) for n, c in zip(num_samples, num_correct, strict=False) if n >= k)
    count = sum(1 for n in num_samples if n >= k)
    return total / count if count > 0 else 0.0


# ============================================
# 数值比较指标
# ============================================

def mean_absolute_error(predictions: list[float], references: list[float]) -> float:
    """Average绝对误差 (MAE)"""
    if not predictions or not references:
        return 0.0
    errors = [abs(float(p) - float(r)) for p, r in zip(predictions, references, strict=False)]
    return sum(errors) / len(errors)


def root_mean_squared_error(predictions: list[float], references: list[float]) -> float:
    """均方根误差 (RMSE)"""
    if not predictions or not references:
        return 0.0
    squared_errors = [(float(p) - float(r)) ** 2 for p, r in zip(predictions, references, strict=False)]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


# ============================================
# Statistical analysis函数
# ============================================

def bootstrap_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> tuple[float, float]:
    """Bootstrap Confidence Interval"""
    if not values:
        return (0.0, 0.0)

    random.seed(seed)
    n = len(values)

    bootstrap_means = []
    for _ in range(n_bootstraps):
        sample = [random.choice(values) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstraps)
    upper_idx = int((1 - alpha / 2) * n_bootstraps)

    return (bootstrap_means[lower_idx], bootstrap_means[upper_idx])


def standard_error(values: list[float]) -> float:
    """标准误差"""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson 分数区间 (适用于二 items分布)"""
    if total == 0:
        return (0.0, 1.0)

    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    p = successes / total

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

    return (max(0, center - spread), min(1, center + spread))


def per_category_accuracy(
    predictions: list[str],
    references: list[str],
    categories: list[str]
) -> dict[str, float]:
    """分类别Accuracy"""
    category_results: dict[str, dict[str, int]] = {}

    for pred, ref, cat in zip(predictions, references, categories, strict=False):
        if cat not in category_results:
            category_results[cat] = {"correct": 0, "total": 0}

        category_results[cat]["total"] += 1
        if str(pred).strip().lower() == str(ref).strip().lower():
            category_results[cat]["correct"] += 1

    return {
        cat: data["correct"] / data["total"] if data["total"] > 0 else 0.0
        for cat, data in category_results.items()
    }


# ============================================
# Metric calculation器
# ============================================

class EvalMetricsCalculator:
    """
    统一评估Metric calculation器

    Usage:
        calculator = EvalMetricsCalculator()
        results = calculator.compute(
            predictions=["A", "B", "C"],
            references=["A", "B", "D"],
            metrics=["accuracy", "f1"]
        )
    """

    METRICS = {
        "exact_match": lambda p, r: exact_match(p, r),
        "accuracy": lambda p, r: accuracy(p, r),
        "normalized_accuracy": lambda p, r: normalized_accuracy(p, r),
        "f1": lambda p, r: f1_score_batch(p, r),
        "bleu": lambda p, r: bleu_score_batch(p, r),
        "rouge_l": lambda p, r: rouge_l_batch(p, r),
    }

    def __init__(self, compute_ci: bool = True, ci_confidence: float = 0.95):
        self.compute_ci = compute_ci
        self.ci_confidence = ci_confidence

    def compute(
        self,
        predictions: list[Any],
        references: list[Any],
        metrics: list[str] = None,
        categories: list[str] | None = None
    ) -> dict[str, EvalMetricResult]:
        """Calculate指定指标"""
        if metrics is None:
            metrics = ["accuracy"]
        results = {}

        is_correct = [
            1.0 if str(p).strip().lower() == str(r).strip().lower() else 0.0
            for p, r in zip(predictions, references, strict=False)
        ]

        for metric_name in metrics:
            if metric_name not in self.METRICS:
                continue

            try:
                value = self.METRICS[metric_name](predictions, references)
                result = EvalMetricResult(name=metric_name, value=value, count=len(predictions))

                if self.compute_ci and metric_name in ["accuracy", "exact_match"]:
                    result.stderr = standard_error(is_correct)
                    result.confidence_interval = bootstrap_confidence_interval(is_correct, self.ci_confidence)

                    correct_count = int(sum(is_correct))
                    result.details["wilson_ci"] = wilson_score_interval(correct_count, len(predictions), self.ci_confidence)

                results[metric_name] = result

            except Exception as e:
                results[metric_name] = EvalMetricResult(
                    name=metric_name, value=0.0, count=len(predictions),
                    details={"error": str(e)}
                )

        if categories and "accuracy" in results:
            results["accuracy"].details["per_category"] = per_category_accuracy(predictions, references, categories)

        return results

    def compute_pass_at_k(
        self,
        num_samples: list[int],
        num_correct: list[int],
        k_values: list[int] = None
    ) -> dict[str, float]:
        """Calculate pass@k 指标"""
        if k_values is None:
            k_values = [1, 5, 10]
        return {f"pass@{k}": estimate_pass_at_k(num_samples, num_correct, k) for k in k_values}


# ============================================
# 便捷函数
# ============================================

def get_metric(name: str) -> Callable:
    """Get指标函数"""
    if name in EvalMetricsCalculator.METRICS:
        return EvalMetricsCalculator.METRICS[name]
    raise ValueError(f"Unknown metric: {name}")


def compute_metrics(
    predictions: list[Any],
    references: list[Any],
    metrics: list[str] = None
) -> dict[str, float]:
    """快速Calculated metrics"""
    if metrics is None:
        metrics = ["accuracy"]
    calculator = EvalMetricsCalculator(compute_ci=False)
    results = calculator.compute(predictions, references, metrics)
    return {name: m.value for name, m in results.items()}


def compute_accuracy_with_ci(
    predictions: list[Any],
    references: list[Any],
    confidence: float = 0.95
) -> dict[str, Any]:
    """CalculateAccuracyand其Confidence Interval"""
    calculator = EvalMetricsCalculator(compute_ci=True, ci_confidence=confidence)
    results = calculator.compute(predictions, references, ["accuracy"])

    acc_result = results.get("accuracy")
    if not acc_result:
        return {"accuracy": 0.0, "stderr": 0.0, "ci_lower": 0.0, "ci_upper": 1.0}

    return {
        "accuracy": acc_result.value,
        "stderr": acc_result.stderr or 0.0,
        "ci_lower": acc_result.confidence_interval[0] if acc_result.confidence_interval else 0.0,
        "ci_upper": acc_result.confidence_interval[1] if acc_result.confidence_interval else 1.0
    }

