"""
WinoGrande Evaluator
WinoGrande DatasetEvaluator

WinoGrande is一大规模代词消解Dataset，TestModel常识推理能力。
每问题包含一带has空格句子，Modelneed选择正确Options来Pad空格。
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_choice_answer


class WinoGrandeEvaluator(BaseEvaluator):
    """
    WinoGrande DatasetEvaluator

    Data格式:
    {
        "sentence": "The trophy doesn't fit into the suitcase because _ is too large.",
        "option1": "trophy",
        "option2": "suitcase",
        "answer": "1"  // 1 or 2
    }
    """

    def __init__(
        self,
        dataset_name: str = "winogrande",
        dataset_path: str = "datasets/winogrande",
        num_shots: int = 5,
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
        """Load WinoGrande Dataset"""
        samples = []

        # use指定子集大小 (xs, s, m, l, xl)
        size = subset if subset in ['xs', 's', 'm', 'l', 'xl'] else 'xl'

        possible_files = [
            os.path.join(self.dataset_path, f"winogrande_{size}.json"),
            os.path.join(self.dataset_path, "validation.json"),
            os.path.join(self.dataset_path, "test.json"),
            os.path.join(self.dataset_path, "winogrande.jsonl"),
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
                sentence = sample.get('sentence', '')
                option1 = sample.get('option1', '')
                option2 = sample.get('option2', '')
                answer = sample.get('answer', '1')

                # ConvertAnsweris 0-indexed
                if isinstance(answer, str):
                    if answer == '1':
                        answer = 0
                    elif answer == '2':
                        answer = 1
                    elif answer in 'AB':
                        answer = ord(answer) - ord('A')
                elif isinstance(answer, int):
                    answer = answer - 1 if answer > 0 else answer

                normalized.append({
                    'sentence': sentence,
                    'choices': [option1, option2],
                    'answer': answer
                })
            except Exception as e:
                continue

        return normalized

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Create示例Data"""
        return [
            {
                "sentence": "The trophy doesn't fit into the suitcase because _ is too large.",
                "choices": ["the trophy", "the suitcase"],
                "answer": 0  # trophy
            },
            {
                "sentence": "The trophy doesn't fit into the suitcase because _ is too small.",
                "choices": ["the trophy", "the suitcase"],
                "answer": 1  # suitcase
            },
            {
                "sentence": "Paul tried to call George on the phone, but _ wasn't available.",
                "choices": ["Paul", "George"],
                "answer": 1  # George
            },
            {
                "sentence": "The dog chased the cat because _ was hungry.",
                "choices": ["the dog", "the cat"],
                "answer": 0  # dog
            },
            {
                "sentence": "The man couldn't lift his son because _ was too heavy.",
                "choices": ["the man", "his son"],
                "answer": 1  # son
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format WinoGrande 样本"""
        sentence = sample.get('sentence', '')
        choices = sample.get('choices', [])

        while len(choices) < 2:
            choices.append("")

        prompt_lines = [
            f"Sentence: {sentence}",
            "",
            "Fill in the blank with the correct option:",
            f"A. {choices[0]}",
            f"B. {choices[1]}",
        ]

        if include_answer:
            answer_idx = sample.get('answer', 0)
            answer_letter = chr(ord('A') + answer_idx) if isinstance(answer_idx, int) else answer_idx
            prompt_lines.append(f"Answer: {answer_letter}")
        else:
            prompt_lines.append("Answer:")

        return "\n".join(prompt_lines)

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt"""
        instruction = (
            "Fill in the blank (_) with the correct option based on the context.\n\n"
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
        return extract_choice_answer(response, ['A', 'B'])

    def check_answer(self, predicted: str, correct: str) -> bool:
        if not predicted:
            return False
        if isinstance(correct, int) or (isinstance(correct, str) and correct.isdigit()):
            correct = chr(ord('A') + int(correct))
        return predicted.upper() == correct.upper()

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        answer = sample.get('answer', 0)
        if isinstance(answer, int):
            return chr(ord('A') + answer)
        return str(answer).upper()
