"""
HellaSwag Evaluator
HellaSwag DatasetEvaluator

HellaSwag is一常识推理Dataset，TestModel对自然语言场景理解。
给定一场景描述，Modelneed选择最合理后续发展。
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_choice_answer


class HellaSwagEvaluator(BaseEvaluator):
    """
    HellaSwag DatasetEvaluator

    Data格式 (HuggingFace):
    {
        "activity_label": "Removing ice from car",
        "ctx": "A man is seen speaking to the camera...",
        "ctx_a": "A man is seen speaking...",
        "ctx_b": "he then begin...",
        "endings": ["ending1", "ending2", "ending3", "ending4"],
        "label": 2  // 0-indexed correct answer
    }
    """

    def __init__(
        self,
        dataset_name: str = "hellaswag",
        dataset_path: str = "datasets/hellaswag",
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
        """Load HellaSwag Dataset"""
        samples = []

        possible_files = [
            os.path.join(self.dataset_path, "hellaswag_val.json"),
            os.path.join(self.dataset_path, "validation.json"),
            os.path.join(self.dataset_path, "test.json"),
            os.path.join(self.dataset_path, "hellaswag.jsonl"),
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
                # Getonunder文
                ctx = sample.get('ctx', '')
                if not ctx:
                    ctx_a = sample.get('ctx_a', '')
                    ctx_b = sample.get('ctx_b', '')
                    ctx = f"{ctx_a} {ctx_b}".strip()

                # GetOptions
                endings = sample.get('endings', [])
                if isinstance(endings, dict):
                    endings = endings.get('text', [])

                while len(endings) < 4:
                    endings.append('')

                # GetLabel
                label = sample.get('label', 0)
                if isinstance(label, str):
                    if label in 'ABCD':
                        label = ord(label) - ord('A')
                    elif label.isdigit():
                        label = int(label)

                normalized.append({
                    'context': ctx,
                    'activity': sample.get('activity_label', ''),
                    'choices': endings[:4],
                    'answer': label
                })
            except Exception as e:
                continue

        return normalized

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Create示例Data"""
        return [
            {
                "context": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She",
                "activity": "Bathing dog",
                "choices": [
                    "rinses the dog off with a hose.",
                    "uses a brush to scrape the dog.",
                    "gets the dog wet with a towel.",
                    "walks away and lets the dog play."
                ],
                "answer": 0
            },
            {
                "context": "A man is sitting on a roof. He starts pulling up shingles. He",
                "activity": "Roof repair",
                "choices": [
                    "takes a nap on the roof.",
                    "removes the old shingles and starts nailing new ones.",
                    "starts to put on a helmet.",
                    "begins to dance on the roof."
                ],
                "answer": 1
            },
            {
                "context": "A young woman is sitting at a table. She picks up a piece of paper and a pencil. She",
                "activity": "Drawing",
                "choices": [
                    "tears the paper into pieces.",
                    "begins to sketch a portrait.",
                    "throws the pencil away.",
                    "starts eating the paper."
                ],
                "answer": 1
            },
            {
                "context": "Two men are in a boxing ring. The referee signals the start of the match. They",
                "activity": "Boxing",
                "choices": [
                    "shake hands and leave the ring.",
                    "start circling each other and throwing punches.",
                    "sit down in their corners.",
                    "begin playing chess."
                ],
                "answer": 1
            },
            {
                "context": "A chef is in a kitchen preparing food. He picks up a knife and a cutting board. He",
                "activity": "Cooking",
                "choices": [
                    "starts juggling the knives.",
                    "begins chopping vegetables.",
                    "throws the cutting board away.",
                    "sits down to rest."
                ],
                "answer": 1
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format HellaSwag 样本"""
        context = sample.get('context', '')
        choices = sample.get('choices', [])

        while len(choices) < 4:
            choices.append("")

        prompt_lines = [
            f"Context: {context}",
            "",
            "What happens next?",
            f"A. {choices[0]}",
            f"B. {choices[1]}",
            f"C. {choices[2]}",
            f"D. {choices[3]}",
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
            "Complete the following scenario by choosing the most plausible continuation.\n\n"
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
        return extract_choice_answer(response, ['A', 'B', 'C', 'D'])

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
