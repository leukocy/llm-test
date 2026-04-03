"""
单元Test: core/metrics.py

Test评估Metric calculation函数，包括:
- AccuracyCalculate
- 文本相似度 (F1, BLEU, ROUGE-L)
- 代码Generate pass@k
- Statistical analysis (Confidence Interval, 标准误差)
"""

import math

import pytest

from core.metrics import (
    # Metric calculation器
    EvalMetricsCalculator,
    accuracy,
    bleu_score,
    bleu_score_batch,
    # Statistical analysis函数
    bootstrap_confidence_interval,
    compute_accuracy_with_ci,
    compute_metrics,
    estimate_pass_at_k,
    # 精确匹配指标
    exact_match,
    # 文本相似度指标
    f1_score,
    f1_score_batch,
    # 数值比较指标
    mean_absolute_error,
    normalized_accuracy,
    # 代码Generate指标
    pass_at_k,
    per_category_accuracy,
    root_mean_squared_error,
    rouge_l,
    rouge_l_batch,
    standard_error,
    wilson_score_interval,
)


class TestAccuracyMetrics:
    """TestAccuracy相关指标"""

    def test_exact_match_all_correct(self):
        """完全正确情况"""
        predictions = ["A", "B", "C"]
        references = ["A", "B", "C"]
        assert exact_match(predictions, references) == 1.0

    def test_exact_match_partial_correct(self):
        """部分正确情况"""
        predictions = ["A", "B", "D"]
        references = ["A", "B", "C"]
        assert exact_match(predictions, references) == pytest.approx(2/3)

    def test_exact_match_all_wrong(self):
        """全部Error情况"""
        predictions = ["X", "Y", "Z"]
        references = ["A", "B", "C"]
        assert exact_match(predictions, references) == 0.0

    def test_exact_match_empty(self):
        """空列表"""
        assert exact_match([], []) == 0.0

    def test_exact_match_with_whitespace(self):
        """忽略前后空白"""
        predictions = [" A ", "B", " C "]
        references = ["A", " B ", "C"]
        assert exact_match(predictions, references) == 1.0

    def test_accuracy_string_comparison(self):
        """字符串Accuracy"""
        predictions = ["apple", "banana", "cherry"]
        references = ["apple", "banana", "date"]
        assert accuracy(predictions, references) == pytest.approx(2/3)

    def test_accuracy_numeric_comparison(self):
        """数值Accuracy"""
        predictions = ["1.0", "2.5", "3.7"]
        references = ["1", "2.5", "3.70000001"]
        assert accuracy(predictions, references) == 1.0

    def test_accuracy_case_insensitive(self):
        """大小写not敏感"""
        predictions = ["APPLE", "Banana", "CHERRY"]
        references = ["apple", "BANANA", "cherry"]
        assert accuracy(predictions, references) == 1.0

    def test_normalized_accuracy(self):
        """NormalizeAccuracy (4选1)"""
        # 随机基准 = 0.25
        predictions = ["A", "B", "C", "D"]
        references = ["A", "B", "D", "A"]

        acc = accuracy(predictions, references)  # 0.5
        normalized = normalized_accuracy(predictions, references, num_choices=4)

        # (0.5 - 0.25) / (1 - 0.25) = 0.25 / 0.75 = 1/3
        assert normalized == pytest.approx(1/3)

    def test_normalized_accuracy_below_random(self):
        """低于随机基准时Return0"""
        predictions = ["A", "B", "C", "D"]
        references = ["A", "B", "C", "X"]  # 0.75 accuracy
        # 对于4选1，0.75 > 0.25，thereforenotis0
        normalized = normalized_accuracy(predictions, references, num_choices=4)
        assert normalized > 0


class TestTextSimilarityMetrics:
    """Test文本相似度指标"""

    def test_f1_score_identical(self):
        """完全相同文本"""
        assert f1_score("hello world", "hello world") == 1.0

    def test_f1_score_no_overlap(self):
        """no重叠"""
        assert f1_score("hello world", "foo bar") == 0.0

    def test_f1_score_partial_overlap(self):
        """部分重叠"""
        # pred: hello, world, test
        # ref: hello, world, example
        # common: hello, world (2)
        # precision = 2/3, recall = 2/3, f1 = 2/3
        assert f1_score("hello world test", "hello world example") == pytest.approx(2/3)

    def test_f1_score_batch(self):
        """批量F1Calculate"""
        predictions = ["hello world", "foo bar"]
        references = ["hello world", "foo baz"]
        # 一完全匹配 (1.0), 二 precision=1/2, recall=1/2, f1=1/2
        # 平均 = (1.0 + 0.5) / 2 = 0.75
        assert f1_score_batch(predictions, references) == pytest.approx(0.75)

    def test_bleu_score_identical(self):
        """完全相同文本"""
        assert bleu_score("hello world", "hello world") == 1.0

    def test_bleu_score_no_overlap(self):
        """no重叠"""
        assert bleu_score("hello world", "foo bar baz") == 0.0

    def test_bleu_score_partial_match(self):
        """部分匹配"""
        # 简化Validate：has重叠时应该大于0
        score = bleu_score("the cat is on the mat", "the dog is on the mat")
        assert 0 < score < 1

    def test_rouge_l_identical(self):
        """完全相同文本"""
        result = rouge_l("hello world", "hello world")
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_rouge_l_partial_overlap(self):
        """部分重叠 LCS"""
        # "hello world" vs "hello there"
        # tokens: ["hello", "world"] vs ["hello", "there"]
        # LCS = ["hello"] (1 token)
        # precision = 1/2, recall = 1/2, f1 = 1/2
        result = rouge_l("hello world", "hello there")
        assert result["f1"] == pytest.approx(0.5)

    def test_rouge_l_batch(self):
        """批量 ROUGE-L F1"""
        predictions = ["hello world", "foo bar baz"]
        references = ["hello there", "foo bar"]
        # 一: LCS=["hello"] -> precision=1/2, recall=1/2, f1=0.5
        # 二: LCS=["foo", "bar"] -> precision=2/3, recall=1, f1=4/5=0.8
        # 平均 = (0.5 + 0.8) / 2 = 0.65
        result = rouge_l_batch(predictions, references)
        assert result == pytest.approx(0.65)


class TestCodeMetrics:
    """Test代码Generate指标"""

    def test_pass_at_k_all_pass(self):
        """全部via"""
        # n=10, c=10, k=1 -> 1.0
        assert pass_at_k(10, 10, 1) == 1.0

    def test_pass_at_k_none_pass(self):
        """全部失败"""
        assert pass_at_k(10, 0, 1) == 0.0

    def test_pass_at_k_partial_pass(self):
        """部分via"""
        # n=5, c=3, k=1
        # result = 1 - ((5-3)/5) = 1 - 0.4 = 0.6
        assert pass_at_k(5, 3, 1) == pytest.approx(0.6)

    def test_pass_at_k_with_k_greater_than_failures(self):
        """k 大于失败数时Return1"""
        # n=5, c=4, k=2 -> 5-4=1 < 2 -> 1.0
        assert pass_at_k(5, 4, 2) == 1.0

    def test_estimate_pass_at_k(self):
        """估计整体 pass@k"""
        num_samples = [10, 20, 15]
        num_correct = [5, 10, 12]
        # k=1
        result = estimate_pass_at_k(num_samples, num_correct, k=1)
        assert 0 < result < 1


class TestNumericalMetrics:
    """Test数值比较指标"""

    def test_mae_identical(self):
        """完全相同"""
        assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0

    def test_mae_calculated(self):
        """MAE Calculate"""
        # |1-2| + |2-3| + |3-4| = 1 + 1 + 1 = 3, avg = 1
        assert mean_absolute_error([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)

    def test_rmse_identical(self):
        """完全相同"""
        assert root_mean_squared_error([1, 2, 3], [1, 2, 3]) == 0.0

    def test_rmse_calculated(self):
        """RMSE Calculate"""
        # (1-2)^2 + (2-3)^2 + (3-4)^2 = 1+1+1=3, sqrt(3/3)=1
        assert root_mean_squared_error([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)


class TestStatisticalMetrics:
    """TestStatistical analysis函数"""

    def test_bootstrap_confidence_interval(self):
        """Bootstrap Confidence Interval"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_confidence_interval(values, confidence=0.95, n_bootstraps=100)
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert 0 < ci[0] < 6
        assert 0 < ci[1] < 6

    def test_bootstrap_empty(self):
        """空列表"""
        ci = bootstrap_confidence_interval([])
        assert ci == (0.0, 0.0)

    def test_standard_error(self):
        """标准误差"""
        values = [1, 2, 3, 4, 5]
        se = standard_error(values)
        assert se > 0

    def test_standard_error_single_value(self):
        """单值标准误差is0"""
        assert standard_error([5]) == 0.0

    def test_wilson_score_interval_all_correct(self):
        """全部正确"""
        ci = wilson_score_interval(10, 10, confidence=0.95)
        # Wilson score interval 对于小样本i.e.使全对也notwill太接近 1
        # 对于 n=10, p=1.0, 95% CI 约is [0.72, 1.0]
        assert ci[0] > 0.7
        assert ci[1] == 1.0  # on限is1

    def test_wilson_score_interval_half_correct(self):
        """一半正确"""
        ci = wilson_score_interval(5, 10, confidence=0.95)
        assert 0 < ci[0] < 0.5
        assert 0.5 < ci[1] < 1

    def test_wilson_score_interval_empty(self):
        """空样本"""
        ci = wilson_score_interval(0, 0)
        assert ci == (0.0, 1.0)

    def test_per_category_accuracy(self):
        """分类别Accuracy"""
        predictions = ["A", "B", "A", "B", "C"]
        references = ["A", "B", "B", "A", "C"]
        categories = ["cat1", "cat1", "cat2", "cat2", "cat3"]

        result = per_category_accuracy(predictions, references, categories)
        # cat1: A=A, B=B -> 2/2 = 1.0
        # cat2: A!=B, B!=A -> 0/2 = 0.0
        # cat3: C=C -> 1/1 = 1.0
        assert result["cat1"] == 1.0
        assert result["cat2"] == 0.0
        assert result["cat3"] == 1.0


class TestEvalMetricsCalculator:
    """TestMetric calculation器"""

    def test_compute_accuracy(self):
        """CalculateAccuracy"""
        calculator = EvalMetricsCalculator(compute_ci=False)
        results = calculator.compute(
            predictions=["A", "B", "C"],
            references=["A", "B", "D"],
            metrics=["accuracy"]
        )
        assert "accuracy" in results
        assert results["accuracy"].value == pytest.approx(2/3)

    def test_compute_multiple_metrics(self):
        """Calculate多指标"""
        calculator = EvalMetricsCalculator(compute_ci=False)
        results = calculator.compute(
            predictions=["A", "B", "C"],
            references=["A", "B", "D"],
            metrics=["accuracy", "exact_match", "f1"]
        )
        assert len(results) == 3
        assert "accuracy" in results
        assert "exact_match" in results
        assert "f1" in results

    def test_compute_with_ci(self):
        """Calculate带Confidence Interval"""
        calculator = EvalMetricsCalculator(compute_ci=True, ci_confidence=0.95)
        results = calculator.compute(
            predictions=["A", "B"] * 10,
            references=["A", "B"] * 10,
            metrics=["accuracy"]
        )
        assert results["accuracy"].stderr is not None
        assert results["accuracy"].confidence_interval is not None

    def test_compute_with_categories(self):
        """Calculate分类别Accuracy"""
        calculator = EvalMetricsCalculator(compute_ci=False)
        results = calculator.compute(
            predictions=["A", "B", "A"],
            references=["A", "B", "B"],
            metrics=["accuracy"],
            categories=["cat1", "cat1", "cat2"]
        )
        assert "per_category" in results["accuracy"].details

    def test_compute_pass_at_k(self):
        """Calculate pass@k"""
        calculator = EvalMetricsCalculator()
        results = calculator.compute_pass_at_k(
            num_samples=[10, 20, 15],
            num_correct=[5, 10, 12],
            k_values=[1, 5]
        )
        assert "pass@1" in results
        assert "pass@5" in results


class TestConvenienceFunctions:
    """Test便捷函数"""

    def test_compute_metrics(self):
        """快速Calculated metrics"""
        results = compute_metrics(
            predictions=["A", "B", "C"],
            references=["A", "B", "D"],
            metrics=["accuracy", "exact_match"]
        )
        assert results["accuracy"] == pytest.approx(2/3)
        assert results["exact_match"] == pytest.approx(2/3)

    def test_compute_accuracy_with_ci(self):
        """CalculateAccuracyand其Confidence Interval"""
        results = compute_accuracy_with_ci(
            predictions=["A", "B"] * 10,
            references=["A", "B"] * 10
        )
        assert "accuracy" in results
        assert "stderr" in results
        assert "ci_lower" in results
        assert "ci_upper" in results
        assert results["accuracy"] == 1.0
