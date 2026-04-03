"""
SWE-Bench Lite Evaluator
SWE-Bench Lite 软件工程任务Evaluator

SWE-Bench Lite is SWE-Bench 精简版，包含 300 精选
GitHub Issue 修复任务，用于TestModel解决真实软件问题能力。
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator


class SWEBenchLiteEvaluator(BaseEvaluator):
    """
    SWE-Bench Lite DatasetEvaluator

    Data格式:
    {
        "instance_id": "django__django-11099",
        "repo": "django/django",
        "base_commit": "abc123...",
        "problem_statement": "Issue description...",
        "hints_text": "",
        "created_at": "2019-01-01",
        "patch": "diff --git a/...",  # 正确修复补丁
        "test_patch": "...",
        "version": "3.0",
        "FAIL_TO_PASS": ["test1", "test2"],
        "PASS_TO_PASS": ["test3"]
    }

    注意: 完整 SWE-Bench 评估need实际Run Test，
    这里我们简化is代码GenerateQuality Assessment。
    """

    def __init__(
        self,
        dataset_name: str = "swebench_lite",
        dataset_path: str = "datasets/swebench_lite",
        num_shots: int = 0,  # SWE-Bench 通常use 0-shot
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
        """Load SWE-Bench Lite Dataset"""
        samples = []

        possible_files = [
            os.path.join(self.dataset_path, "swe-bench-lite.json"),
            os.path.join(self.dataset_path, "swebench_lite.json"),
            os.path.join(self.dataset_path, "test.json"),
            os.path.join(self.dataset_path, "swe-bench-lite.jsonl"),
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
                normalized.append({
                    'instance_id': sample.get('instance_id', ''),
                    'repo': sample.get('repo', ''),
                    'problem_statement': sample.get('problem_statement', ''),
                    'hints': sample.get('hints_text', ''),
                    'patch': sample.get('patch', ''),
                    'fail_tests': sample.get('FAIL_TO_PASS', []),
                    'pass_tests': sample.get('PASS_TO_PASS', [])
                })
            except Exception as e:
                continue

        return normalized

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Create示例Data"""
        return [
            {
                "instance_id": "example__project-001",
                "repo": "example/project",
                "problem_statement": """
Bug: Division by zero error in calculate_average function

When the input list is empty, the calculate_average function raises
a ZeroDivisionError. The function should handle empty lists gracefully
and return 0 or raise a meaningful exception.

Current code:
```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
```

Expected behavior: Return 0 for empty lists or raise ValueError.
                """,
                "hints": "Check for empty list before division",
                "patch": """
--- a/utils.py
+++ b/utils.py
@@ -1,2 +1,4 @@
 def calculate_average(numbers):
+    if not numbers:
+        return 0
     return sum(numbers) / len(numbers)
                """,
                "fail_tests": ["test_empty_list"],
                "pass_tests": ["test_normal_average"]
            },
            {
                "instance_id": "example__project-002",
                "repo": "example/project",
                "problem_statement": """
Bug: String concatenation performance issue

The build_message function uses string concatenation in a loop which
is very slow for large inputs. Should use join() instead.

Current code:
```python
def build_message(parts):
    result = ""
    for part in parts:
        result += part + " "
    return result.strip()
```
                """,
                "hints": "Use str.join() for better performance",
                "patch": """
--- a/utils.py
+++ b/utils.py
@@ -1,5 +1,2 @@
 def build_message(parts):
-    result = ""
-    for part in parts:
-        result += part + " "
-    return result.strip()
+    return " ".join(parts)
                """,
                "fail_tests": ["test_performance"],
                "pass_tests": ["test_build_message"]
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format SWE-Bench 样本"""
        problem = sample.get('problem_statement', '')
        hints = sample.get('hints', '')

        prompt = f"Issue:\n{problem}"
        if hints:
            prompt += f"\n\nHints:\n{hints}"

        if include_answer:
            patch = sample.get('patch', '')
            prompt += f"\n\nPatch:\n```diff\n{patch}\n```"
        else:
            prompt += "\n\nGenerate a patch to fix this issue:"

        return prompt

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt"""
        instruction = (
            "You are a software engineer. Your task is to fix the bug described in the issue.\n"
            "Provide a git diff patch that fixes the problem.\n\n"
        )

        return instruction + self.format_prompt(sample, include_answer=False)

    def parse_response(self, response: str) -> str:
        """Parse响应，提取 diff 补丁"""
        # 尝试提取代码块in diff
        diff_match = re.search(r'```(?:diff)?\s*(.*?)```', response, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()

        # 查找 diff 格式内容
        if '---' in response and '+++' in response:
            lines = response.split('\n')
            diff_lines = []
            in_diff = False
            for line in lines:
                if line.startswith('---') or line.startswith('+++'):
                    in_diff = True
                if in_diff:
                    diff_lines.append(line)
            if diff_lines:
                return '\n'.join(diff_lines)

        return response.strip()

    def check_answer(self, predicted: str, correct: str) -> bool:
        """
        Check补丁质量
        简化Version：Checkis否包含关键修改
        完整Versionneed实际Run Test
        """
        if not predicted:
            return False

        # 简单Check：is否包含 diff 格式
        has_diff_format = ('---' in predicted or '+++' in predicted or
                          '+' in predicted or '-' in predicted)

        # Checkis否has代码修改
        has_code_change = bool(re.search(r'[\+\-]\s*\w+', predicted))

        return has_diff_format and has_code_change

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """Get正确补丁"""
        return sample.get('patch', '')

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get仓库名称作is类别"""
        return sample.get('repo', 'unknown')
