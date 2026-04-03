"""
MMLU Evaluator
Evaluator for the MMLU (Massive Multitask Language Understanding) benchmark.
MMLU consists of multiple-choice questions across 57 subjects in STEM, Humanities, Social Sciences, and more.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_choice_answer


class MMLUEvaluator(BaseEvaluator):
    """
    MMLU Dataset Evaluator.
    
    Data Format:
    {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "answer": 1,  # 0-indexed index (1 = B)
        "subject": "geography"
    }
    """

    # MMLU subject classification
    SUBJECTS = {
        "stem": [
            "abstract_algebra", "anatomy", "astronomy", "college_biology",
            "college_chemistry", "college_computer_science", "college_mathematics",
            "college_physics", "computer_security", "conceptual_physics",
            "electrical_engineering", "elementary_mathematics", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_mathematics", "high_school_physics", "high_school_statistics",
            "machine_learning"
        ],
        "humanities": [
            "formal_logic", "high_school_european_history",
            "high_school_us_history", "high_school_world_history", "international_law",
            "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios",
            "philosophy", "prehistory", "professional_law", "world_religions"
        ],
        "social_sciences": [
            "econometrics", "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_microeconomics",
            "high_school_psychology", "human_sexuality", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy"
        ],
        "other": [
            "business_ethics", "clinical_knowledge", "college_medicine",
            "global_facts", "human_aging", "management", "marketing",
            "medical_genetics", "miscellaneous", "nutrition",
            "professional_accounting", "professional_medicine", "virology"
        ]
    }

    def __init__(
        self,
        dataset_name: str = "mmlu",
        dataset_path: str = "datasets/mmlu",
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
        """Load MMLU dataset with subset filtering."""
        try:
            from core.dataset_manager import get_dataset
            
            # Load "test" split via DatasetManager
            all_samples = get_dataset(
                name=self.dataset_name,
                split="test",
                max_samples=None,
                seed=self.seed
            )
            
            if subset and subset != "all":
                subjects_to_keep = []
                if subset in self.SUBJECTS:
                    subjects_to_keep = self.SUBJECTS[subset]
                else:
                    subjects_to_keep = [subset]
                
                if all_samples:
                     if 'subject' in all_samples[0]:
                         samples = [s for s in all_samples if s.get('subject') in subjects_to_keep]
                     else:
                         print("[WARNING] MMLU samples lack 'subject' field, cannot filter.")
                         samples = all_samples
                else:
                    samples = []
            else:
                samples = all_samples

        except Exception as e:
            print(f"[WARNING] DatasetManager failed: {e}")
            samples = []

        # Fallback to sample data if empty
        if not samples:
            samples = self._create_sample_data()

        random.shuffle(samples)

        # Set few-shot examples from dev set
        if self.num_shots > 0:
            dev_samples = self._load_dev_samples()
            if dev_samples:
                self.few_shot_examples = dev_samples[:self.num_shots]
            else:
                # Fallback: steal from test set
                total_needed = self.num_shots + (self.max_samples or 0)
                if len(samples) > total_needed:
                    self.few_shot_examples = samples[:self.num_shots]
                    samples = samples[self.num_shots:]

        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def _load_dev_samples(self) -> list[dict[str, Any]]:
        """Load development set for few-shot examples."""
        try:
            from core.dataset_manager import get_dataset
            return get_dataset(name="mmlu", split="train") 
        except Exception:
            return []

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Mock data for testing."""
        return [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": 1,
                "subject": "geography"
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
                "answer": 1,
                "subject": "astronomy"
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format MMLU question and choices."""
        question = sample.get('question', '')
        choices = sample.get('choices', [])

        while len(choices) < 4:
            choices.append("")

        prompt_lines = [
            f"Question: {question}",
            f"A. {choices[0]}",
            f"B. {choices[1]}",
            f"C. {choices[2]}",
            f"D. {choices[3]}",
        ]

        if include_answer:
            answer_idx = sample.get('answer', 0)
            if isinstance(answer_idx, str) and not answer_idx.isdigit():
                answer_letter = answer_idx.upper()
            else:
                answer_letter = chr(ord('A') + int(answer_idx))
            prompt_lines.append(f"Answer: {answer_letter}")
        else:
            prompt_lines.append("Answer:")

        return "\n".join(prompt_lines)

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build full prompt with subject-specific instructions."""
        subject = sample.get('subject', 'general knowledge')
        subject_display = subject.replace('_', ' ').title()

        instruction = f"The following are multiple choice questions (with answers) about {subject_display}.\n\n"
        
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
        """Extract choice letter from response."""
        return extract_choice_answer(response, ['A', 'B', 'C', 'D'])

    def check_answer(self, predicted: str, correct: str) -> bool:
        """Verify if predicted letter matches correct answer."""
        if not predicted:
            return False

        # Handle numeric index in correct answer
        if isinstance(correct, int) or (isinstance(correct, str) and correct.isdigit()):
            correct = chr(ord('A') + int(correct))

        return predicted.upper() == correct.upper()

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        return sample.get('subject', 'unknown')

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        answer = sample.get('answer', 0)
        if isinstance(answer, int) or (isinstance(answer, str) and answer.isdigit()):
            return chr(ord('A') + int(answer))
        return str(answer)
