"""
Test Result Comparator 模块
TestResult对比功能单元Test
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.result_comparator import (
    ComparisonEntry,
    ComparisonReport,
    ResultComparator,
    ResultMetadata,
    compare_files,
    load_comparison,
    print_comparison_summary,
)
from evaluators.base_evaluator import EvaluationResult, SampleResult


class TestResultMetadata(unittest.TestCase):
    """Test ResultMetadata 类"""

    def test_from_result(self):
        """Test从 EvaluationResult Create元Data"""
        # Create一 tests用 EvaluationResult
        sample = SampleResult(
            sample_id="1",
            question="Test question",
            correct_answer="A",
            model_response="A",
            predicted_answer="A",
            is_correct=True,
            input_tokens=100,
            output_tokens=50,
            ttft_ms=100.0,
            tps=20.0,
        )

        result = EvaluationResult(
            dataset_name="test_dataset",
            model_id="test_model",
            accuracy=0.85,
            total_samples=100,
            correct_samples=85,
            details=[sample],
            timestamp="2024-01-01 12:00:00",
            duration_seconds=120.0,
        )
        result.compute_performance_stats()

        # Create临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
            result.save_to_json(filepath)

        try:
            # Create元Data
            metadata = ResultMetadata.from_result(result, filepath, "Test Label")

            # Validate
            self.assertEqual(metadata.label, "Test Label")
            self.assertEqual(metadata.model_id, "test_model")
            self.assertEqual(metadata.dataset_name, "test_dataset")
            self.assertEqual(metadata.accuracy, 0.85)
            self.assertEqual(metadata.correct_samples, 85)
            self.assertEqual(metadata.total_samples, 100)
            self.assertEqual(metadata.duration_seconds, 120.0)
            self.assertGreater(metadata.file_size, 0)

        finally:
            os.unlink(filepath)

    def test_to_dict(self):
        """TestConvertis字典"""
        metadata = ResultMetadata(
            filepath="test.json",
            label="Test",
            model_id="model",
            dataset_name="dataset",
            accuracy=0.9,
        )

        data = metadata.to_dict()
        self.assertEqual(data["label"], "Test")
        self.assertEqual(data["accuracy"], 0.9)


class TestComparisonEntry(unittest.TestCase):
    """Test ComparisonEntry 类"""

    def test_get_best_higher_is_better(self):
        """TestGet最佳值（higher is better）"""
        entry = ComparisonEntry(
            metric_name="Accuracy",
            values={"Model A": 0.85, "Model B": 0.90, "Model C": 0.80},
            higher_is_better=True
        )

        self.assertEqual(entry.get_best(), 0.90)
        self.assertEqual(entry.get_winner(), "Model B")

    def test_get_best_lower_is_better(self):
        """TestGet最佳值（lower is better）"""
        entry = ComparisonEntry(
            metric_name="Latency",
            values={"Model A": 100, "Model B": 80, "Model C": 120},
            unit="ms",
            higher_is_better=False
        )

        self.assertEqual(entry.get_best(), 80)
        self.assertEqual(entry.get_winner(), "Model B")

    def test_get_difference_percent(self):
        """TestCalculate差异百分比"""
        entry = ComparisonEntry(
            metric_name="Accuracy",
            values={"Model A": 0.80, "Model B": 0.90},
            higher_is_better=True
        )

        diff = entry.get_difference_percent()
        self.assertIsNotNone(diff)
        self.assertAlmostEqual(diff["Model A"], -11.11111111111111, places=10)  # (0.8 - 0.9) / 0.9 * 100
        self.assertEqual(diff["Model B"], 0.0)

    def test_to_dict(self):
        """TestConvertis字典"""
        entry = ComparisonEntry(
            metric_name="Test",
            values={"A": 1.0, "B": 2.0},
            higher_is_better=True
        )

        data = entry.to_dict()
        self.assertEqual(data["metric_name"], "Test")
        self.assertEqual(data["best"], 2.0)
        self.assertEqual(data["winner"], "B")
        self.assertIn("difference_percent", data)


class TestResultComparator(unittest.TestCase):
    """Test ResultComparator 类"""

    def setUp(self):
        """SetTest Environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.comparator = ResultComparator()

        # CreateTestResult文件
        self.test_files = []
        for i, (model, dataset, accuracy) in enumerate([
            ("model_a", "mmlu", 0.75),
            ("model_b", "mmlu", 0.82),
            ("model_c", "mmlu", 0.68),
        ]):
            result = self._create_test_result(model, dataset, accuracy)
            filepath = os.path.join(self.temp_dir, f"result_{i}.json")
            result.save_to_json(filepath)
            self.test_files.append((filepath, f"{model}_{dataset}"))

    def tearDown(self):
        """CleanupTest Environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_test_result(self, model_id: str, dataset_name: str, accuracy: float) -> EvaluationResult:
        """CreateTest用 EvaluationResult"""
        total_samples = 100
        correct_samples = int(total_samples * accuracy)

        details = []
        for i in range(total_samples):
            is_correct = i < correct_samples
            sample = SampleResult(
                sample_id=str(i),
                question=f"Question {i}",
                correct_answer="A" if i % 4 == 0 else "B" if i % 4 == 1 else "C" if i % 4 == 2 else "D",
                model_response="A" if is_correct else "B",
                predicted_answer="A" if is_correct else "B",
                is_correct=is_correct,
                category="test" if i % 2 == 0 else "train",
                input_tokens=100,
                output_tokens=50,
                ttft_ms=100.0 + i,
                tps=20.0,
            )
            details.append(sample)

        result = EvaluationResult(
            dataset_name=dataset_name,
            model_id=model_id,
            accuracy=accuracy,
            total_samples=total_samples,
            correct_samples=correct_samples,
            details=details,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=120.0,
            config={
                "temperature": 0.0,
                "max_tokens": 256,
                "thinking_enabled": i % 2 == 0,  # 模拟not同Configure
            }
        )
        result.compute_performance_stats()
        return result

    def test_add_result(self):
        """TestAddResult文件"""
        filepath, label = self.test_files[0]

        result = self.comparator.add_result(filepath, label)

        self.assertTrue(result)
        self.assertEqual(len(self.comparator.results), 1)
        self.assertEqual(len(self.comparator.metadatas), 1)
        self.assertIn(filepath, self.comparator.labels)
        self.assertEqual(self.comparator.labels[filepath], label)

    def test_add_result_file_not_found(self):
        """TestAddnot存in文件"""
        result = self.comparator.add_result("nonexistent.json")
        self.assertFalse(result)

    def test_add_results_from_dir(self):
        """Test从目录批量AddResult"""
        count = self.comparator.add_results_from_dir(self.temp_dir)

        self.assertEqual(count, 3)
        self.assertEqual(len(self.comparator.results), 3)

    def test_compare_results(self):
        """TestComparison Results"""
        # Add所hasResult
        for filepath, label in self.test_files:
            self.comparator.add_result(filepath, label)

        # Generate对比
        report = self.comparator.compare_results()

        # Validate报告
        self.assertIsNotNone(report)
        self.assertEqual(len(report.results), 3)
        self.assertGreater(len(report.comparisons), 0)
        self.assertIsNotNone(report.summary)

        # Validate汇总信息
        self.assertEqual(report.summary["total_results"], 3)
        self.assertEqual(report.summary["best_accuracy"], 0.82)
        self.assertIn("best_accuracy_label", report.summary)

    def test_compare_results_single(self):
        """Test单Result对比"""
        filepath, label = self.test_files[0]
        self.comparator.add_result(filepath, label)

        # i.e.使只has一Result也应该能Generate报告
        report = self.comparator.compare_results()
        self.assertIsNotNone(report)
        self.assertEqual(len(report.results), 1)

    def test_generate_comparison_report(self):
        """TestGenerate可读报告"""
        # AddResult
        for filepath, label in self.test_files:
            self.comparator.add_result(filepath, label)

        # Generate报告
        report_text = self.comparator.generate_comparison_report()

        # Validate报告内容
        self.assertIn("TestResult对比报告", report_text)
        self.assertIn("Accuracy", report_text)
        self.assertIn("model_a_mmlu", report_text)
        self.assertIn("model_b_mmlu", report_text)
        self.assertIn("model_c_mmlu", report_text)

    def test_get_comparison_df(self):
        """TestGet对比 DataFrame"""
        # AddResult
        for filepath, label in self.test_files:
            self.comparator.add_result(filepath, label)

        # Get DataFrame
        df = self.comparator.get_comparison_df()

        # Validate DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("指标", df.columns)
        self.assertGreater(len(df), 0)
        self.assertIn("model_a_mmlu", df.columns)
        self.assertIn("model_b_mmlu", df.columns)
        self.assertIn("model_c_mmlu", df.columns)

    def test_save_report_json(self):
        """TestSave JSON 报告"""
        # AddResult
        for filepath, label in self.test_files:
            self.comparator.add_result(filepath, label)

        # Generate对比
        self.comparator.compare_results()

        # Save报告
        output_path = os.path.join(self.temp_dir, "report.json")
        result_path = self.comparator.save_report(output_path, format="json")

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))

        # ValidatecanLoad
        with open(output_path, encoding='utf-8') as f:
            data = json.load(f)
        self.assertIn("comparison_id", data)
        self.assertIn("results", data)
        self.assertIn("comparisons", data)

    def test_save_report_txt(self):
        """TestSave文本报告"""
        # AddResult
        for filepath, label in self.test_files:
            self.comparator.add_result(filepath, label)

        # Generate对比
        self.comparator.compare_results()

        # Save报告
        output_path = os.path.join(self.temp_dir, "report.txt")
        result_path = self.comparator.save_report(output_path, format="txt")

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))

        # Validate内容
        with open(output_path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn("TestResult对比报告", content)

    def test_save_report_csv(self):
        """TestSave CSV 报告"""
        # AddResult
        for filepath, label in self.test_files:
            self.comparator.add_result(filepath, label)

        # Generate对比
        self.comparator.compare_results()

        # Save报告
        output_path = os.path.join(self.temp_dir, "report.csv")
        result_path = self.comparator.save_report(output_path, format="csv")

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))

        # ValidatecanLoad
        df = pd.read_csv(output_path)
        self.assertIn("指标", df.columns)

    def test_plot_comparison(self):
        """TestGenerate对比图表"""
        try:
            import matplotlib
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False

        if not matplotlib_available:
            self.skipTest("matplotlib 未安装")

        # AddResult
        for filepath, label in self.test_files:
            self.comparator.add_result(filepath, label)

        # Generate对比
        self.comparator.compare_results()

        # Generate图表
        output_path = os.path.join(self.temp_dir, "plot.png")
        result_path = self.comparator.plot_comparison(output_path=output_path)

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_plot_comparison_without_saving(self):
        """TestGenerate图表butnotSave"""
        try:
            import matplotlib
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False

        if not matplotlib_available:
            self.skipTest("matplotlib 未安装")

        # AddResult
        for filepath, label in self.test_files:
            self.comparator.add_result(filepath, label)

        # Generate对比
        self.comparator.compare_results()

        # Generate图表butnotSave
        result_path = self.comparator.plot_comparison()
        self.assertIsNone(result_path)

    def test_clear(self):
        """Test清除Result"""
        # AddResult
        for filepath, label in self.test_files:
            self.comparator.add_result(filepath, label)

        self.assertEqual(len(self.comparator.results), 3)

        # 清除
        self.comparator.clear()

        self.assertEqual(len(self.comparator.results), 0)
        self.assertEqual(len(self.comparator.metadatas), 0)
        self.assertEqual(len(self.comparator.labels), 0)
        self.assertIsNone(self.comparator.report)


class TestConvenienceFunctions(unittest.TestCase):
    """Test便捷函数"""

    def setUp(self):
        """SetTest Environment"""
        self.temp_dir = tempfile.mkdtemp()

        # CreateTestResult文件
        self.test_files = []
        for i, (model, dataset, accuracy) in enumerate([
            ("model_a", "mmlu", 0.75),
            ("model_b", "mmlu", 0.82),
        ]):
            result = self._create_test_result(model, dataset, accuracy)
            filepath = os.path.join(self.temp_dir, f"result_{i}.json")
            result.save_to_json(filepath)
            self.test_files.append(filepath)

    def tearDown(self):
        """CleanupTest Environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_test_result(self, model_id: str, dataset_name: str, accuracy: float) -> EvaluationResult:
        """CreateTest用 EvaluationResult"""
        total_samples = 100
        correct_samples = int(total_samples * accuracy)

        details = []
        for i in range(total_samples):
            is_correct = i < correct_samples
            sample = SampleResult(
                sample_id=str(i),
                question=f"Question {i}",
                correct_answer="A",
                model_response="A" if is_correct else "B",
                predicted_answer="A" if is_correct else "B",
                is_correct=is_correct,
                input_tokens=100,
                output_tokens=50,
            )
            details.append(sample)

        result = EvaluationResult(
            dataset_name=dataset_name,
            model_id=model_id,
            accuracy=accuracy,
            total_samples=total_samples,
            correct_samples=correct_samples,
            details=details,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=120.0,
        )
        result.compute_performance_stats()
        return result

    def test_compare_files(self):
        """Test compare_files 函数"""
        report = compare_files(
            self.test_files,
            labels=["Model A", "Model B"]
        )

        self.assertIsNotNone(report)
        self.assertEqual(len(report.results), 2)
        self.assertEqual(report.summary["total_results"], 2)

    def test_compare_files_with_output(self):
        """Test compare_files 函数并Save输出"""
        output_path = os.path.join(self.temp_dir, "report.json")

        report = compare_files(
            self.test_files,
            labels=["Model A", "Model B"],
            output_path=output_path
        )

        self.assertTrue(os.path.exists(output_path))

    def test_load_comparison(self):
        """TestLoad对比报告"""
        # 先Create一报告
        output_path = os.path.join(self.temp_dir, "report.json")
        compare_files(self.test_files, output_path=output_path)

        # Load报告
        report = load_comparison(output_path)

        self.assertIsNotNone(report)
        self.assertEqual(len(report.results), 2)
        self.assertGreater(len(report.comparisons), 0)

    def test_print_comparison_summary(self, capsys=None):
        """Test打印对比摘要"""
        # Create报告
        report = compare_files(self.test_files)

        # 打印摘要
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            print_comparison_summary(report)
            output = buffer.getvalue()

            # Validate输出
            self.assertIn("TestResult对比摘要", output)
            self.assertIn("Accuracy", output)
        finally:
            sys.stdout = old_stdout


class TestEdgeCases(unittest.TestCase):
    """Test边界情况"""

    def test_empty_comparator(self):
        """Test空对比器"""
        comparator = ResultComparator()

        report = comparator.compare_results()

        self.assertIsNotNone(report)
        self.assertEqual(len(report.results), 0)

    def test_comparison_with_string_values(self):
        """Test包含字符串值对比条目"""
        entry = ComparisonEntry(
            metric_name="Thinking Mode",
            values={"Model A": "启用", "Model B": "Close"},
            higher_is_better=None  # 字符串值没hashigher is better
        )

        # 字符串值Return一值作is"最佳"
        self.assertEqual(entry.get_best(), "启用")
        self.assertEqual(entry.get_winner(), "Model A")


if __name__ == "__main__":
    unittest.main()
