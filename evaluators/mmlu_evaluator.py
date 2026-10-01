"""
MMLU Evaluator
Evaluator for the MMLU (Massive Multitask Language Understanding) benchmark.
MMLU consists of multiple-choice questions across 57 subjects in STEM, Humanities, Social Sciences, and more.
"""

import random
from typing import Any

from . import register_evaluator
from .base_evaluator import BaseEvaluator, DatasetUnavailableError, extract_choice_answer


@register_evaluator("mmlu")
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

    evaluation_split = "test"
    few_shot_split: str | None = "dev"
    few_shot_policy = "same_subject_dev"
    answer_protocol = "generated_text_choice"

    # MMLU subject classification
    SUBJECTS = {
        "stem": [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "electrical_engineering",
            "elementary_mathematics",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_statistics",
            "machine_learning",
        ],
        "humanities": [
            "formal_logic",
            "high_school_european_history",
            "high_school_us_history",
            "high_school_world_history",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "moral_disputes",
            "moral_scenarios",
            "philosophy",
            "prehistory",
            "professional_law",
            "world_religions",
        ],
        "social_sciences": [
            "econometrics",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_microeconomics",
            "high_school_psychology",
            "human_sexuality",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
        ],
        "other": [
            "business_ethics",
            "clinical_knowledge",
            "college_medicine",
            "global_facts",
            "human_aging",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "nutrition",
            "professional_accounting",
            "professional_medicine",
            "virology",
        ],
    }

    def __init__(
        self,
        dataset_name: str = "mmlu",
        dataset_path: str = "datasets/mmlu",
        num_shots: int = 5,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_shots=num_shots,
            max_samples=max_samples,
            seed=seed,
        )
        self.few_shot_split = "dev" if num_shots else None

    @staticmethod
    def normalize(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        normalized = []
        seen = set()
        for index, row in enumerate(rows):
            subject, question, choices, answer = (
                row.get(key) for key in ("subject", "question", "choices", "answer")
            )
            if (
                not isinstance(subject, str)
                or not subject.strip()
                or not isinstance(question, str)
                or not question.strip()
                or not isinstance(choices, list)
                or len(choices) != 4
                or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
            ):
                raise DatasetUnavailableError(
                    "MMLU requires subject, question and four nonempty choices"
                )
            if type(answer) is int and 0 <= answer < 4:
                answer = "ABCD"[answer]
            elif isinstance(answer, str) and answer in ("0", "1", "2", "3"):
                answer = "ABCD"[int(answer)]
            if answer not in tuple("ABCD"):
                raise DatasetUnavailableError("MMLU requires a valid labeled answer")
            raw_id = row.get("id", index)
            if type(raw_id) not in (str, int) or not str(raw_id).strip():
                raise DatasetUnavailableError("MMLU requires nonempty sample IDs")
            identifier = f"{split}:{subject}:{raw_id}"
            if identifier in seen:
                raise DatasetUnavailableError("MMLU sample IDs must be unique per split/subject")
            seen.add(identifier)
            normalized.append({**row, "id": identifier, "choices": list(choices), "answer": answer})
        return normalized

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        from core.dataset_manager import get_dataset

        raw = get_dataset(name=self.dataset_name, split="test", max_samples=None, seed=self.seed)
        self.dataset_source = "mmlu_test"
        self.evaluation_split = "test"
        if not raw:
            raw = self._fallback_to_demo_samples(self._create_sample_data)
            self.evaluation_split = "demo"
        samples = self.normalize(raw, self.evaluation_split)
        if subset and subset != "all":
            subjects_to_keep = self.SUBJECTS.get(subset, [subset])
            samples = [sample for sample in samples if sample["subject"] in subjects_to_keep]
        if not samples:
            raise DatasetUnavailableError("MMLU has no scoring samples for the selected subject")
        random.Random(self.seed).shuffle(samples)
        self.samples = samples[: self.max_samples] if self.max_samples is not None else samples
        subjects = {sample["subject"] for sample in self.samples}
        self.few_shot_split = "dev" if self.num_shots else None
        self.few_shot_examples = (
            self.normalize(get_dataset(name=self.dataset_name, split="dev"), "dev")
            if self.num_shots
            else []
        )
        self.few_shot_examples = [
            example for example in self.few_shot_examples if example["subject"] in subjects
        ]
        self.validate_dev(self.samples, self.few_shot_examples, self.num_shots)
        return self.samples

    @staticmethod
    def validate_dev(samples, examples, num_shots):
        for subject in {sample["subject"] for sample in samples}:
            if sum(example["subject"] == subject for example in examples) < num_shots:
                raise DatasetUnavailableError(
                    "MMLU has insufficient same-subject dev examples; never borrow scoring rows"
                )
        scoring_content = {
            (row["subject"], row["question"], tuple(row["choices"])) for row in samples
        }
        if any(
            (row["subject"], row["question"], tuple(row["choices"])) in scoring_content
            for row in examples
        ):
            raise DatasetUnavailableError("MMLU scoring/dev content overlaps")

    def _examples_for(self, sample):
        return [
            example for example in self.few_shot_examples if example["subject"] == sample["subject"]
        ][: self.num_shots]

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Mock data for testing."""
        return [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": 1,
                "subject": "geography",
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
                "answer": 1,
                "subject": "astronomy",
            },
        ]

    def build_chat_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Build structured chat messages with system prompt, few-shot turns, and user question."""
        messages = []

        subject = sample.get("subject", "general knowledge")
        subject_display = subject.replace("_", " ").title()
        system_instruction = (
            f"The following are multiple choice questions (with answers) about {subject_display}."
        )
        messages.append({"role": "system", "content": system_instruction})

        for ex in self._examples_for(sample):
            messages.append(
                {
                    "role": "user",
                    "content": self.format_prompt(ex, include_answer=False),
                }
            )
            messages.append({"role": "assistant", "content": self.get_correct_answer(ex)})

        messages.append(
            {
                "role": "user",
                "content": self.format_prompt(sample, include_answer=False),
            }
        )
        return messages

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format MMLU question and choices."""
        question = sample.get("question", "")
        choices = list(sample.get("choices", []))

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
            answer_idx = sample.get("answer", 0)
            if isinstance(answer_idx, str) and not answer_idx.isdigit():
                answer_letter = answer_idx.upper()
            else:
                answer_letter = chr(ord("A") + int(answer_idx))
            prompt_lines.append(f"Answer: {answer_letter}")
        else:
            prompt_lines.append("Answer:")

        return "\n".join(prompt_lines)

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build full prompt with subject-specific instructions."""
        subject = sample.get("subject", "general knowledge")
        subject_display = subject.replace("_", " ").title()

        instruction = f"The following are multiple choice questions (with answers) about {subject_display}.\n\n"

        examples = []
        for example in self._examples_for(sample):
            examples.append(self.format_prompt(example, include_answer=True))

        question = self.format_prompt(sample, include_answer=False)

        full_prompt = instruction + "\n\n".join(examples)
        if examples:
            full_prompt += "\n\n"
        full_prompt += question

        return full_prompt

    def parse_response(self, response: str) -> str:
        """Extract choice letter from response."""
        return extract_choice_answer(response, ["A", "B", "C", "D"])

    def check_answer(self, predicted: str, correct: str) -> bool:
        """Verify if predicted letter matches correct answer."""
        if not predicted:
            return False

        # Handle numeric index in correct answer
        if isinstance(correct, int) or (isinstance(correct, str) and correct.isdigit()):
            correct = chr(ord("A") + int(correct))

        return predicted.upper() == correct.upper()

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        return str(sample.get("subject", "unknown"))

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        answer = sample.get("answer", 0)
        if isinstance(answer, int) or (isinstance(answer, str) and answer.isdigit()):
            return chr(ord("A") + int(answer))
        return str(answer)
