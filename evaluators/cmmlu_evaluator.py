"""CMMLU labeled test scoring with same-subject dev demonstrations.

Data contract: https://github.com/haonan-li/CMMLU
The official CSV columns are Question, A, B, C, D, Answer.
"""

import random

from . import register_evaluator
from .base_evaluator import DatasetUnavailableError
from .ceval_evaluator import CEvalEvaluator


@register_evaluator("cmmlu")
class CMMLUEvaluator(CEvalEvaluator):
    """Chinese multiple-choice scoring; never borrow test rows as exemplars."""

    def __init__(
        self,
        dataset_name="cmmlu",
        dataset_path="datasets/cmmlu",
        num_shots=5,
        max_samples=None,
        seed=42,
    ):
        super().__init__(dataset_name, dataset_path, num_shots, max_samples, seed)

    @staticmethod
    def normalize(rows, split="test"):
        if any(not str(row.get("id", "")).strip() or row.get("id") is None for row in rows):
            raise DatasetUnavailableError("CMMLU sample IDs must be present and nonempty")
        normalized = [
            {**row, "question": row.get("Question"), "answer": row.get("Answer")} for row in rows
        ]
        try:
            return CEvalEvaluator.normalize(normalized, split)
        except DatasetUnavailableError as exc:
            raise DatasetUnavailableError(
                "Invalid CMMLU subject, ID, question, choices or label"
            ) from exc

    def load_dataset(self, subset=None):
        from core.dataset_manager import get_dataset

        if self.evaluation_split != "test":
            raise DatasetUnavailableError("CMMLU scoring requires the labeled test split")
        samples = self.normalize(get_dataset("cmmlu", split="test"), "test")
        if subset and subset != "all":
            samples = [row for row in samples if row["subject"] == subset]
        if not samples:
            raise DatasetUnavailableError("CMMLU has no scoring samples; prepare the dataset first")
        random.Random(self.seed).shuffle(samples)
        self.samples = samples[: self.max_samples] if self.max_samples is not None else samples
        self.dataset_source = "cmmlu_test"
        subjects = {row["subject"] for row in self.samples}
        self.few_shot_examples = (
            self.normalize(get_dataset("cmmlu", split="dev"), "dev") if self.num_shots else []
        )
        self.few_shot_examples = [
            row for row in self.few_shot_examples if row["subject"] in subjects
        ]
        if self.num_shots and any(
            sum(row["subject"] == subject for row in self.few_shot_examples) < self.num_shots
            for subject in subjects
        ):
            raise DatasetUnavailableError(
                "CMMLU has insufficient same-subject dev exemplars (official dev: 5 per subject)"
            )
        return self.samples
