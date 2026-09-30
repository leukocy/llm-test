"""C-Eval test/validation scoring with subject-specific development exemplars.

Official split contract: https://huggingface.co/datasets/ceval/ceval-exam
Public test labels were released in 2025; older unlabeled rows are rejected.
"""

import random
from typing import Any

from . import register_evaluator
from .base_evaluator import BaseEvaluator, DatasetUnavailableError, extract_choice_answer


@register_evaluator("ceval")
class CEvalEvaluator(BaseEvaluator):
    evaluation_split = "test"
    few_shot_split = "dev"

    def __init__(
        self,
        dataset_name="ceval",
        dataset_path="datasets/ceval",
        num_shots=5,
        max_samples=None,
        seed=42,
    ):
        super().__init__(dataset_name, dataset_path, num_shots, max_samples, seed)

    @staticmethod
    def normalize(rows: list[dict[str, Any]], split: str = "val") -> list[dict[str, Any]]:
        samples = []
        seen = set()
        for row in rows:
            subject = row.get("subject")
            answer = row.get("answer")
            question = row.get("question")
            choices = [row.get(letter) for letter in "ABCD"]
            if (
                not isinstance(subject, str)
                or not subject
                or not isinstance(question, str)
                or not question
                or answer not in tuple("ABCD")
                or any(not isinstance(choice, str) or not choice for choice in choices)
            ):
                raise DatasetUnavailableError(
                    "C-Eval scoring/dev rows require subject, question, A-D choices and a labeled answer"
                )
            identifier = f"{split}:{subject}:{row.get('id')}"
            if row.get("id") is None or identifier in seen:
                raise DatasetUnavailableError(
                    "C-Eval subject/sample identifiers must be present and unique"
                )
            seen.add(identifier)
            samples.append({**row, "id": identifier, "choices": choices})
        return samples

    def load_dataset(self, subset=None):
        from core.dataset_manager import get_dataset

        if self.evaluation_split not in {"test", "val"}:
            raise DatasetUnavailableError("C-Eval scoring requires test or val")
        samples = self.normalize(
            get_dataset("ceval", split=self.evaluation_split), self.evaluation_split
        )
        if subset and subset != "all":
            samples = [sample for sample in samples if sample["subject"] == subset]
        if not samples:
            raise DatasetUnavailableError(
                "C-Eval has no labeled scoring samples for this subject; prepare the dataset first"
            )
        self.dataset_source = "ceval_" + self.evaluation_split
        random.Random(self.seed).shuffle(samples)
        self.samples = samples[: self.max_samples] if self.max_samples is not None else samples
        subjects = {sample["subject"] for sample in self.samples}
        self.few_shot_examples = (
            self.normalize(get_dataset("ceval", split="dev"), "dev") if self.num_shots else []
        )
        self.few_shot_examples = [
            sample for sample in self.few_shot_examples if sample["subject"] in subjects
        ]
        if self.num_shots and any(
            sum(ex["subject"] == subject for ex in self.few_shot_examples) < self.num_shots
            for subject in subjects
        ):
            raise DatasetUnavailableError(
                "C-Eval has insufficient same-subject dev exemplars; choose at most the available count (official dev: 5 per subject)"
            )
        return self.samples

    def format_prompt(self, sample, include_answer=False):
        prompt = (
            sample["question"]
            + "\n"
            + "\n".join(f"{letter}. {sample[letter]}" for letter in "ABCD")
            + "\n答案："
        )
        return prompt + (sample["answer"] if include_answer else "")

    def build_full_prompt(self, sample):
        examples = [ex for ex in self.few_shot_examples if ex["subject"] == sample["subject"]][
            : self.num_shots
        ]
        return "\n\n".join(
            [*(self.format_prompt(ex, True) for ex in examples), self.format_prompt(sample)]
        )

    def build_chat_messages(self, sample):
        messages = [
            {
                "role": "system",
                "content": f"以下是关于{sample['subject']}的单项选择题，请只回答正确选项 A、B、C 或 D。",
            }
        ]
        for ex in [ex for ex in self.few_shot_examples if ex["subject"] == sample["subject"]][
            : self.num_shots
        ]:
            messages.extend(
                [
                    {"role": "user", "content": self.format_prompt(ex)},
                    {"role": "assistant", "content": ex["answer"]},
                ]
            )
        messages.append({"role": "user", "content": self.format_prompt(sample)})
        return messages

    def get_correct_answer(self, sample):
        return sample["answer"]

    def parse_response(self, response):
        return extract_choice_answer(response, list("ABCD"))

    def check_answer(self, predicted, correct):
        return predicted == correct
