"""
Custom Needle-in-a-Haystack Evaluator

Evaluates long-context retrieval ability using pre-made test files:
- Frankenstein series (20K/40K)
- Gatsby series (20K/40K)

Difficulty levels:
- 1-needle (basic retrieval): Single fact extraction
- 2-needle (cross-paragraph): Multi-fact association
- 3-needle (reasoning chain): Multi-hop reasoning
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, SampleResult, normalize_text


class CustomNeedleEvaluator(BaseEvaluator):
    """
    Custom Needle-in-a-Haystack Evaluator

    Evaluates using pre-made test files with specific prompts and expected answers.
    Supports keyword matching and numerical calculation verification.
    """

    # defaultTest文件目录
    DEFAULT_TEST_DIR = "needle_haystack_data"
    CONFIG_FILE = "needle_tests_config.json"

    def __init__(
        self,
        dataset_name: str = "custom_needle",
        dataset_path: str = "needle_haystack_data",
        num_shots: int = 0,
        max_samples: int | None = None,
        seed: int = 42,
        test_filter: str | None = None,  # FilterTest: "frankenstein", "gatsby", "1needle", "2needle", "3needle"
        context_filter: str | None = None,  # Filteronunder文大小: "20K", "40K"
        difficulty_filter: str | None = None  # Filter难度: "easy", "medium", "hard"
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_shots=num_shots,
            max_samples=max_samples,
            seed=seed
        )

        self.test_filter = test_filter
        self.context_filter = context_filter
        self.difficulty_filter = difficulty_filter
        self.test_configs: list[dict[str, Any]] = []

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """Load预制大海捞针test data"""
        samples = []

        # LoadTestConfigure
        config_path = os.path.join(self.dataset_path, self.CONFIG_FILE)
        if not os.path.exists(config_path):
            # 尝试从default目录Load
            config_path = os.path.join(self.DEFAULT_TEST_DIR, self.CONFIG_FILE)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"TestConfigure文件Not found: {config_path}")

        with open(config_path, encoding='utf-8') as f:
            config_data = json.load(f)

        self.test_configs = config_data.get('tests', [])

        # based onFilter条件筛选Test
        for test in self.test_configs:
            # ApplyFilter器
            if self.test_filter:
                if self.test_filter in ['frankenstein', 'gatsby']:
                    if test.get('series') != self.test_filter:
                        continue
                elif 'needle' in self.test_filter:
                    needle_count = int(self.test_filter.replace('needle', ''))
                    if test.get('needle_count') != needle_count:
                        continue

            if self.context_filter and test.get('context_size') != self.context_filter:
                continue

            if self.difficulty_filter:
                if test.get('difficulty') != self.difficulty_filter:
                    continue

            # LoadTest文件内容
            test_file = test.get('file', '')
            file_path = os.path.join(self.dataset_path, test_file)
            if not os.path.exists(file_path):
                file_path = os.path.join(self.DEFAULT_TEST_DIR, test_file)

            if not os.path.exists(file_path):
                print(f"Warning: Test文件Not found - {test_file}")
                continue

            try:
                with open(file_path, encoding='utf-8') as f:
                    context = f.read()
            except Exception as e:
                print(f"LoadTest文件失败 {test_file}: {e}")
                continue

            # Build样本
            sample = {
                'id': test.get('id', ''),
                'name': test.get('name', ''),
                'context': context,
                'question': test.get('prompt', ''),
                'expected_answer': test.get('expected_answer', {}),
                'keywords': test.get('keywords', []),
                'required_keywords': test.get('required_keywords', 1),
                'series': test.get('series', ''),
                'difficulty': test.get('difficulty', ''),
                'needle_count': test.get('needle_count', 1),
                'context_size': test.get('context_size', ''),
                'calc_base': test.get('calc_base'),
                'calc_multiplier': test.get('calc_multiplier')
            }
            samples.append(sample)

        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """FormatTest prompt"""
        context = sample.get('context', '')
        question = sample.get('question', '')

        # Build完整 prompt
        prompt = f"""Please read the following text carefully and answer the question.

<TEXT>
{context}
</TEXT>

{question}

Please provide your answer based strictly on the information in the text above."""

        if include_answer:
            answer = json.dumps(sample.get('expected_answer', {}), ensure_ascii=False)
            prompt += f"\n\nExpected Answer: {answer}"

        return prompt

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt (no few-shot)"""
        return self.format_prompt(sample, include_answer=False)

    def parse_response(self, response: str) -> str:
        """ParseModel响应"""
        return response.strip()

    def check_answer(self, predicted: str, correct: str) -> bool:
        """
        CheckAnswer正确性

        采用关键词匹配策略：
        1. ifisCalculate题，ValidateCalculateResult
        2. otherwiseuse关键词匹配
        """
        if not predicted:
            return False

        predicted_lower = predicted.lower()

        # 从当前样本Get关键词 (needin evaluate_single inSet)
        keywords = getattr(self, '_current_keywords', [])
        required_count = getattr(self, '_current_required', 1)
        calc_base = getattr(self, '_current_calc_base', None)
        calc_multiplier = getattr(self, '_current_calc_multiplier', None)

        # Calculate题Validate
        if calc_base and calc_multiplier:
            expected_result = calc_base * calc_multiplier
            # Checkis否包含正确CalculateResult
            if str(expected_result) in predicted or f"{expected_result:,}" in predicted:
                return True
            # Checkis否Display正确Calculate过程
            if str(calc_base) in predicted and str(calc_multiplier) in predicted:
                # 同时提到基数and倍数，可能推理正确
                return True

        # 关键词匹配
        found_count = 0
        for keyword in keywords:
            if keyword.lower() in predicted_lower:
                found_count += 1

        return found_count >= required_count

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """GetCorrect answer字符串表示"""
        expected = sample.get('expected_answer', {})
        if isinstance(expected, dict):
            return json.dumps(expected, ensure_ascii=False)
        return str(expected)

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get样本类别"""
        series = sample.get('series', 'unknown')
        needle_count = sample.get('needle_count', 1)
        context_size = sample.get('context_size', '')
        return f"{series}_{context_size}_{needle_count}needle"

    async def evaluate_single(
        self,
        sample: dict[str, Any],
        get_response_func,
        sample_index: int = 0
    ) -> SampleResult:
        """
        评估单 samples

        重写父类方法以传递关键词信息到 check_answer
        """
        # Set当前样本评估参数
        self._current_keywords = sample.get('keywords', [])
        self._current_required = sample.get('required_keywords', 1)
        self._current_calc_base = sample.get('calc_base')
        self._current_calc_multiplier = sample.get('calc_multiplier')

        # 调用父类方法
        result = await super().evaluate_single(sample, get_response_func, sample_index)

        # Cleanup临时变量
        self._current_keywords = []
        self._current_required = 1
        self._current_calc_base = None
        self._current_calc_multiplier = None

        return result


class NeedleTestRunner:
    """
    Needle-in-a-Haystack Test Runner

    Convenience test runner with support for filtering by category/difficulty.
    """

    def __init__(self, test_dir: str = "needle_haystack_data"):
        self.test_dir = test_dir
        self.evaluator = None

    def get_available_tests(self) -> list[dict[str, str]]:
        """Get所has可用Test列表"""
        config_path = os.path.join(self.test_dir, "needle_tests_config.json")
        if not os.path.exists(config_path):
            return []

        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)

        tests = []
        for test in config.get('tests', []):
            tests.append({
                'id': test.get('id', ''),
                'name': test.get('name', ''),
                'series': test.get('series', ''),
                'difficulty': test.get('difficulty', ''),
                'needle_count': test.get('needle_count', 1),
                'context_size': test.get('context_size', '')
            })
        return tests

    def create_evaluator(
        self,
        test_filter: str | None = None,
        context_filter: str | None = None,
        difficulty_filter: str | None = None,
        max_samples: int | None = None
    ) -> CustomNeedleEvaluator:
        """CreateEvaluator实例"""
        self.evaluator = CustomNeedleEvaluator(
            dataset_path=self.test_dir,
            test_filter=test_filter,
            context_filter=context_filter,
            difficulty_filter=difficulty_filter,
            max_samples=max_samples
        )
        return self.evaluator

    def run_single_test(
        self,
        test_id: str,
        model_response: str
    ) -> dict[str, Any]:
        """
        运行单 tests快捷方法

        Args:
            test_id: TestID
            model_response: Model对Test prompt 响应

        Returns:
            TestResult字典
        """
        config_path = os.path.join(self.test_dir, "needle_tests_config.json")
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)

        test_config = None
        for test in config.get('tests', []):
            if test.get('id') == test_id:
                test_config = test
                break

        if not test_config:
            return {'error': f'Test not found: {test_id}'}

        # 关键词匹配评估
        predicted_lower = model_response.lower()
        keywords = test_config.get('keywords', [])
        required = test_config.get('required_keywords', 1)

        found_keywords = []
        for kw in keywords:
            if kw.lower() in predicted_lower:
                found_keywords.append(kw)

        is_correct = len(found_keywords) >= required

        # Calculate题额外Validate
        calc_base = test_config.get('calc_base')
        calc_multiplier = test_config.get('calc_multiplier')
        if calc_base and calc_multiplier:
            expected_result = calc_base * calc_multiplier
            if str(expected_result) in model_response or f"{expected_result:,}" in model_response:
                is_correct = True

        return {
            'test_id': test_id,
            'test_name': test_config.get('name', ''),
            'is_correct': is_correct,
            'found_keywords': found_keywords,
            'required_keywords': required,
            'expected_answer': test_config.get('expected_answer', {}),
            'difficulty': test_config.get('difficulty', ''),
            'needle_count': test_config.get('needle_count', 1)
        }


# 便捷函数
def get_needle_test_prompt(test_dir: str, test_id: str) -> str | None:
    """Get指定Test完整 prompt (包含onunder文)"""
    runner = NeedleTestRunner(test_dir)
    evaluator = runner.create_evaluator()
    evaluator.load_dataset()

    for sample in evaluator.samples:
        if sample.get('id') == test_id:
            return evaluator.build_full_prompt(sample)

    return None


def list_needle_tests(test_dir: str = "needle_haystack_data") -> list[dict[str, str]]:
    """List all available needle-in-a-haystack tests."""
    runner = NeedleTestRunner(test_dir)
    return runner.get_available_tests()
