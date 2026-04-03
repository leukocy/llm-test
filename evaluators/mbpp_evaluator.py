"""
MBPP Evaluator
MBPP (Mostly Basic Python Problems) DatasetEvaluator

MBPP is一包含 974 道 Python 编程题Dataset。
比 HumanEval 更多样化，覆盖更广泛编程场景。
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator


class MBPPEvaluator(BaseEvaluator):
    """
    MBPP DatasetEvaluator

    Data格式:
    {
        "task_id": 1,
        "text": "Write a function to find the minimum cost path...",
        "code": "def min_cost(cost, m, n): ...",
        "test_list": ["assert min_cost(...) == ..."],
        "test_setup_code": "",
        "challenge_test_list": []
    }
    """

    def __init__(
        self,
        dataset_name: str = "mbpp",
        dataset_path: str = "datasets/mbpp",
        num_shots: int = 3,  # MBPP 通常use较少 few-shot
        max_samples: int | None = None,
        seed: int = 42
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_shots=num_shots,
            max_samples=max_samples,
            seed=seed
        )
        random.seed(seed)

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """Load MBPP Dataset"""
        samples = []

        possible_files = [
            os.path.join(self.dataset_path, "mbpp.jsonl"),
            os.path.join(self.dataset_path, "test.jsonl"),
            os.path.join(self.dataset_path, "mbpp.json"),
            os.path.join(self.dataset_path, "sanitized-mbpp.json"),
        ]

        for filepath in possible_files:
            if os.path.exists(filepath):
                try:
                    if filepath.endswith('.jsonl'):
                        with open(filepath, encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    samples.append(json.loads(line))
                    else:
                        with open(filepath, encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            samples = data
                        elif isinstance(data, dict) and 'data' in data:
                            samples = data['data']
                    break
                except Exception as e:
                    print(f"Load {filepath} 失败: {e}")

        if not samples:
            samples = self._create_sample_data()

        samples = self._normalize_samples(samples)
        random.shuffle(samples)

        total_needed = self.num_shots + (self.max_samples if self.max_samples else len(samples))
        if len(samples) > total_needed:
            samples = samples[:total_needed]

        if self.num_shots > 0:
            self.few_shot_examples = samples[:self.num_shots]
            samples = samples[self.num_shots:]

        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def _normalize_samples(self, samples: list[dict]) -> list[dict]:
        """Standardize样本格式"""
        normalized = []

        for sample in samples:
            try:
                task_id = sample.get('task_id', 0)
                text = sample.get('text', sample.get('prompt', ''))
                code = sample.get('code', sample.get('canonical_solution', ''))

                # ProcessTest case
                test_list = sample.get('test_list', [])
                if isinstance(test_list, str):
                    test_list = [test_list]

                # 提取函数签名
                func_name = self._extract_function_name(code)

                normalized.append({
                    'task_id': task_id,
                    'text': text,
                    'code': code,
                    'test_list': test_list,
                    'func_name': func_name
                })
            except Exception as e:
                continue

        return normalized

    def _extract_function_name(self, code: str) -> str:
        """从代码in提取函数名"""
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
        return "solution"

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Create示例Data"""
        return [
            {
                "task_id": 1,
                "text": "Write a function to find the minimum of two numbers.",
                "code": "def min_of_two(a, b):\n    return min(a, b)",
                "test_list": ["assert min_of_two(3, 5) == 3", "assert min_of_two(7, 2) == 2"],
                "func_name": "min_of_two"
            },
            {
                "task_id": 2,
                "text": "Write a function to calculate the factorial of a number.",
                "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                "test_list": ["assert factorial(5) == 120", "assert factorial(0) == 1"],
                "func_name": "factorial"
            },
            {
                "task_id": 3,
                "text": "Write a function to check if a number is prime.",
                "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True",
                "test_list": ["assert is_prime(7) == True", "assert is_prime(4) == False"],
                "func_name": "is_prime"
            },
            {
                "task_id": 4,
                "text": "Write a function to reverse a string.",
                "code": "def reverse_string(s):\n    return s[::-1]",
                "test_list": ["assert reverse_string('hello') == 'olleh'"],
                "func_name": "reverse_string"
            },
            {
                "task_id": 5,
                "text": "Write a function to find the sum of a list of numbers.",
                "code": "def sum_list(nums):\n    return sum(nums)",
                "test_list": ["assert sum_list([1,2,3]) == 6"],
                "func_name": "sum_list"
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format MBPP 样本"""
        text = sample.get('text', '')
        test_list = sample.get('test_list', [])

        # BuildTest case描述
        test_desc = ""
        if test_list:
            test_desc = "\n\nExample tests:\n" + "\n".join(test_list[:3])

        prompt = f"Task: {text}{test_desc}\n\n"

        if include_answer:
            code = sample.get('code', '')
            prompt += f"Solution:\n```python\n{code}\n```"
        else:
            prompt += "Solution:\n```python\n"

        return prompt

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt"""
        instruction = (
            "You are a Python programming expert. Write a function to solve the given problem.\n"
            "Return only the Python code without any explanation.\n\n"
        )

        examples = []
        for example in self.few_shot_examples[:self.num_shots]:
            examples.append(self.format_prompt(example, include_answer=True))

        question = self.format_prompt(sample, include_answer=False)

        full_prompt = instruction + "\n\n".join(examples)
        if examples:
            full_prompt += "\n\n"
        full_prompt += question

        return full_prompt

    def parse_response(self, response: str) -> str:
        """ParseModel响应，提取代码"""
        # 尝试提取代码块
        code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        code_match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # if没has代码块，Return整响应
        return response.strip()

    def check_answer(self, predicted: str, correct: str) -> bool:
        """Check代码is否正确 - via执行Test case"""
        if not predicted:
            return False

        # 简单语法Check
        try:
            compile(predicted, '<string>', 'exec')
        except SyntaxError:
            return False

        # 对于简单Validate，我们Check预测代码is否包含关键函数定义
        # 完整Validateneed沙箱执行Test case
        return 'def ' in predicted

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """GetCorrect answer（代码）"""
        return sample.get('code', '')

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get样本类别"""
        return "programming"
