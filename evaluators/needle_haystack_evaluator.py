"""
Needle-in-a-Haystack Evaluator
大海捞针TestEvaluator

Needle-in-a-Haystack is一 testsModel长onunder文检索能力Test。
in大量no关文本（干草堆）inInsert一关键信息（针），
TestModel能否准确找到并回答About这信息问题。
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, normalize_text


class NeedleHaystackEvaluator(BaseEvaluator):
    """
    Needle-in-a-Haystack TestEvaluator

    Test方式:
    1. Generate指定长度Pad文本（干草堆）
    2. in特定位置Insert关键信息（针）
    3. 询问ModelAbout该信息问题
    4. 评估Modelis否能正确回答

    可Configure参数:
    - context_length: onunder文总长度
    - needle_depth: 针位置（0.0=开头, 1.0=结尾）
    """

    # default干草堆文本（Paul Graham 文章）
    HAYSTACK_TEXT = """
The best thing about San Francisco is the weather. It never gets too hot or too cold,
and the skies are usually clear. The city is also home to many great restaurants,
museums, and parks. One of the most popular tourist attractions is the Golden Gate Bridge,
which spans the Golden Gate strait and connects San Francisco to Marin County.

In the world of technology, San Francisco has become a hub for innovation and startups.
Many of the world's largest tech companies, such as Google, Facebook, and Twitter, have
offices in the Bay Area. The city's proximity to Silicon Valley has made it a magnet
for entrepreneurs and venture capitalists.

The history of San Francisco is rich and varied. The city was founded in 1776 by Spanish
colonists, and it grew rapidly during the California Gold Rush of 1849. Today, San Francisco
is known for its diverse population, progressive politics, and vibrant arts scene.

Transportation in the city includes the famous cable cars, which have been running since 1873.
The city also has an extensive public transit system, including buses, light rail, and the BART
subway system that connects San Francisco to other cities in the Bay Area.
"""

    # default针（关键信息）
    DEFAULT_NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."

    def __init__(
        self,
        dataset_name: str = "needle_haystack",
        dataset_path: str = "datasets/needle_haystack",
        num_shots: int = 0,
        max_samples: int | None = None,
        seed: int = 42,
        context_lengths: list[int] = None,  # Testonunder文长度列表
        needle_depths: list[float] = None   # 针深度位置列表
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_shots=num_shots,
            max_samples=max_samples,
            seed=seed
        )
        random.seed(seed)

        # defaultTestConfigure
        self.context_lengths = context_lengths or [1000, 2000, 4000, 8000, 16000, 32000]
        self.needle_depths = needle_depths or [0.0, 0.25, 0.5, 0.75, 1.0]

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """LoadorGenerate Needle-in-a-Haystack test data"""
        samples = []

        # 首先尝试Load预GenerateData
        possible_files = [
            os.path.join(self.dataset_path, "needle_haystack.json"),
            os.path.join(self.dataset_path, "test.json"),
        ]

        for filepath in possible_files:
            if os.path.exists(filepath):
                try:
                    with open(filepath, encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    break
                except Exception as e:
                    print(f"Load {filepath} 失败: {e}")

        # if没has预GenerateData，动态Generate
        if not samples:
            samples = self._generate_test_cases()

        random.shuffle(samples)

        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def _generate_test_cases(self) -> list[dict[str, Any]]:
        """GenerateTest case"""
        samples = []

        for context_length in self.context_lengths:
            for needle_depth in self.needle_depths:
                sample = self._create_single_test(context_length, needle_depth)
                samples.append(sample)

        return samples

    def _create_single_test(self, context_length: int, needle_depth: float) -> dict[str, Any]:
        """Create单Test case"""
        needle = self.DEFAULT_NEEDLE
        question = "What is the best thing to do in San Francisco?"
        answer = "eat a sandwich and sit in Dolores Park on a sunny day"

        # Generate足够长干草堆
        haystack = self._generate_haystack(context_length, needle, needle_depth)

        return {
            'context': haystack,
            'question': question,
            'answer': answer,
            'needle': needle,
            'context_length': context_length,
            'needle_depth': needle_depth
        }

    def _generate_haystack(self, target_length: int, needle: str, depth: float) -> str:
        """Generate包含针干草堆"""
        # 重复基础文本直到达到目标长度
        base_text = self.HAYSTACK_TEXT.strip()

        # Calculateneed多少份基础文本
        repeats = (target_length // len(base_text)) + 2
        haystack_parts = [base_text] * repeats

        # Calculate针Insert位置
        total_parts = len(haystack_parts)
        insert_pos = int(total_parts * depth)
        insert_pos = max(0, min(insert_pos, total_parts - 1))

        # Insert针
        haystack_parts.insert(insert_pos, f"\n\n{needle}\n\n")

        # Merge并Truncate到目标长度
        full_text = "\n\n".join(haystack_parts)

        # 保留针位置，从两端Truncate
        if len(full_text) > target_length:
            needle_start = full_text.find(needle)
            if needle_start != -1:
                # 确保针inTruncate后仍然存in
                start = max(0, needle_start - target_length // 2)
                end = start + target_length
                full_text = full_text[start:end]

        return full_text

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """FormatTest样本"""
        context = sample.get('context', '')
        question = sample.get('question', '')

        prompt = f"Context:\n{context}\n\nQuestion: {question}"

        if include_answer:
            answer = sample.get('answer', '')
            prompt += f"\n\nAnswer: {answer}"
        else:
            prompt += "\n\nAnswer:"

        return prompt

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt"""
        instruction = (
            "Read the following context carefully and answer the question based on the information provided.\n\n"
        )

        return instruction + self.format_prompt(sample, include_answer=False)

    def parse_response(self, response: str) -> str:
        """Parse响应"""
        return response.strip()

    def check_answer(self, predicted: str, correct: str) -> bool:
        """CheckAnswer"""
        if not predicted:
            return False

        predicted_lower = normalize_text(predicted.lower())
        correct_lower = normalize_text(correct.lower())

        # Check关键词is否存in
        key_phrases = ['sandwich', 'dolores park', 'sunny']
        found_count = sum(1 for phrase in key_phrases if phrase in predicted_lower)

        # 至少包含两关键词认is正确
        return found_count >= 2 or correct_lower in predicted_lower

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """GetCorrect answer"""
        return sample.get('answer', '')

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get类别（onunder文长度 + 深度）"""
        length = sample.get('context_length', 0)
        depth = sample.get('needle_depth', 0)
        return f"len_{length}_depth_{int(depth*100)}"
