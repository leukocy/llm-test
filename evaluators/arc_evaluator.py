"""
ARC Evaluator
Evaluator for the AI2 Reasoning Challenge (ARC) dataset.
Contains primary school science questions (Easy and Challenge sets).
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_choice_answer


class ARCEvaluator(BaseEvaluator):
    """
    ARC Dataset Evaluator.
    """

    def __init__(
        self,
        dataset_name: str = "arc",
        dataset_path: str = "datasets/arc",
        num_shots: int = 25,
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

    def load_dataset(self, subset: str | None = "challenge") -> list[dict[str, Any]]:
        """Load ARC dataset (Easy or Challenge)."""
        try:
            from core.dataset_manager import get_dataset
            # split mapped to "challenge" or "easy" in some configs, here we use subset
            samples = get_dataset(
                name=self.dataset_name,
                split=subset or "challenge",
                max_samples=self.max_samples,
                seed=self.seed
            )
            self.samples = samples
            return samples
        except Exception as e:
            print(f"[WARNING] ARC load failed: {e}")
            return []

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        question = sample.get('question', '')
        choices_data = sample.get('choices', {})
        labels = choices_data.get('label', [])
        texts = choices_data.get('text', [])

        prompt = f"Question: {question}\n"
        for label, text in zip(labels, texts):
            prompt += f"{label}. {text}\n"
        
        prompt += "Answer:"
        if include_answer:
            prompt += f" {sample.get('answer', '')}"
        
        return prompt

    def parse_response(self, response: str) -> str:
        return extract_choice_answer(response, ['A', 'B', 'C', 'D', '1', '2', '3', '4'])

    def check_answer(self, predicted: str, correct: str) -> bool:
        if not predicted: return False
        return predicted.upper() == correct.upper()
