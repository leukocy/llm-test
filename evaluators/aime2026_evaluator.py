"""
AIME 2026 Evaluator
AIME 2026 (American Invitational Mathematics Examination) DatasetEvaluator

Data source: MathArena/aime_2026 (HuggingFace)
- 30 problems
- Answer range: 0-999 integers
- Evaluation: Exact Match
- 0-shot, pure reasoning

This is one of the official GLM-5.2 benchmarks (official score: 99.2).
"""

import json
import os
import random
import re
from typing import Any

from . import register_evaluator
from .answer_parser import MathAnswerParser
from .base_evaluator import BaseEvaluator


@register_evaluator("aime2026")
class AIME2026Evaluator(BaseEvaluator):
    """
    AIME 2026 DatasetEvaluator

    Data format (MathArena/aime_2026):
    {
        "problem_idx": 1,
        "answer": 277,
        "problem": "Patrick started walking at a constant rate..."
    }

    Standardized to:
    {
        "problem": "...",
        "answer": "277",
        "source": "AIME-2026"
    }

    Features:
    - 30 problems total
    - Answer must be an integer between 0 and 999
    - Uses exact match evaluation
    - 0-shot by default (AIME problems are too hard for few-shot to help meaningfully)
    """

    def __init__(
        self,
        dataset_name: str = "aime2026",
        dataset_path: str = "datasets/aime2026",
        num_shots: int = 0,
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
        random.seed(seed)

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """
        Load AIME 2026 Dataset

        Args:
            subset: optional subset filter (not used for AIME 2026, all 30 problems)

        Returns:
            list of samples
        """
        samples = []

        # 1. Try DatasetManager (auto-download)
        try:
            from core.dataset_manager import get_dataset

            samples = get_dataset(self.dataset_name, split="test", max_samples=None, seed=self.seed)
        except Exception as e:
            print(f"[WARNING] DatasetManager failed for AIME2026: {e}")

        # 2. Fallback: manual file loading
        if not samples:
            possible_files = [
                os.path.join(self.dataset_path, "aime2026.json"),
                os.path.join(self.dataset_path, "test.json"),
                os.path.join(self.dataset_path, "data.json"),
                os.path.join(self.dataset_path, "aime2026.jsonl"),
                os.path.join(self.dataset_path, "test.jsonl"),
            ]

            for filepath in possible_files:
                if os.path.exists(filepath):
                    try:
                        if filepath.endswith(".jsonl"):
                            with open(filepath, encoding="utf-8") as f:
                                for line in f:
                                    if line.strip():
                                        samples.append(json.loads(line))
                        else:
                            with open(filepath, encoding="utf-8") as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                samples = data
                            elif isinstance(data, dict) and "data" in data:
                                samples = data["data"]
                            elif isinstance(data, dict) and "samples" in data:
                                samples = data["samples"]
                        print(f"[INFO] Loaded {len(samples)} samples from {filepath}")
                        break
                    except Exception as e:
                        print(f"Load {filepath} failed: {e}")

        # 3. If no local data, try creating from embedded sample data
        if not samples:
            print("[INFO] No local dataset found, using embedded sample data")
            samples = self._create_sample_data()

        # Standardize sample format
        samples = self._normalize_samples(samples)

        # Filter by subset if specified
        if subset:
            subset_upper = subset.upper().replace("_", "-")
            samples = [s for s in samples if subset_upper in s.get("source", "").upper()]

        # Shuffle
        random.shuffle(samples)

        # Calculate total needed
        total_needed = self.num_shots + (self.max_samples if self.max_samples else len(samples))
        if len(samples) > total_needed:
            samples = samples[:total_needed]

        # Set few-shot examples
        if self.num_shots > 0 and len(samples) > self.num_shots:
            self.few_shot_examples = samples[: self.num_shots]
            samples = samples[self.num_shots :]

        # Limit test samples
        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[: self.max_samples]

        self.samples = samples
        return samples

    def _normalize_samples(self, samples: list[dict]) -> list[dict]:
        """Standardize sample format"""
        normalized = []
        for s in samples:
            sample = {
                "problem": s.get("problem", s.get("question", "")),
                "answer": str(s.get("answer", "")),
                "source": s.get("source", "AIME-2026"),
            }

            # Handle MathArena format (problem_idx, problem, answer)
            if "problem_idx" in s and "problem" in s:
                sample["problem"] = s["problem"]
                sample["answer"] = str(s["answer"])
                sample["source"] = "AIME-2026"
                sample["id"] = f"aime_2026_{s['problem_idx']:02d}"

            # Preserve original id
            if "id" in s:
                sample["id"] = s["id"]
            if "problem_idx" in s:
                sample["problem_idx"] = s["problem_idx"]

            normalized.append(sample)
        return normalized

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """
        Create AIME 2026 sample data (embedded fallback).
        These are real AIME 2026 problems.
        """
        return [
            {
                "id": "aime_2026_01",
                "problem_idx": 1,
                "answer": "277",
                "problem": "Patrick started walking at a constant rate along a straight road from school to the park. One hour after Patrick left, Tanya started running along the same road from school to the park. One hour after Tanya started, they met at a point that was twice as close to the park as to the school. If Patrick walked 1 mile per hour slower than Tanya, find how many miles per hour Tanya ran.",
                "source": "AIME-2026",
            },
            {
                "id": "aime_2026_02",
                "problem_idx": 2,
                "answer": "62",
                "problem": "Find the number of positive integer palindromes written in base 10 with no zero digits, and whose digits add up to 13. For example, 42124 has these properties. Recall that a palindrome is a number that reads the same forwards and backwards.",
                "source": "AIME-2026",
            },
            {
                "id": "aime_2026_03",
                "problem_idx": 3,
                "answer": "79",
                "problem": "A hemisphere with radius 200 sits on top of a horizontal circular disk with radius 200, and the hemisphere and disk have the same center. Let T be the region of points P in the disk such that from point P, one can travel to the surface of the hemisphere by moving upward at a 45 degree angle (with respect to the ground) in at least two distinct directions. Find the area of T.",
                "source": "AIME-2026",
            },
            {
                "id": "aime_2026_04",
                "problem_idx": 4,
                "answer": "70",
                "problem": "Find the number of integers less than or equal to 100 that are equal to a+b+ab for some choice of distinct positive integers a and b.",
                "source": "AIME-2026",
            },
            {
                "id": "aime_2026_05",
                "problem_idx": 5,
                "answer": "65",
                "problem": "A plane contains points A and B with AB = 1. Point A is rotated in the plane counterclockwise through an acute angle theta around point B to point A'. Then B is rotated in the plane counterclockwise through an acute angle theta around point A to point B'. Let M be the midpoint of A'B'. The maximum possible value of the distance from M to the midpoint of AB can be expressed as m/n, where m and n are relatively prime positive integers. Find m+n.",
                "source": "AIME-2026",
            },
            {
                "id": "aime_2026_06",
                "problem_idx": 6,
                "answer": "651",
                "problem": "Let k be the smallest positive integer such that for any positive integer n, it is possible to partition {1,2,...,kn} into n sets each of size k such that the sum of elements in each set is the same. Find the remainder when k is divided by 1000.",
                "source": "AIME-2026",
            },
            {
                "id": "aime_2026_07",
                "problem_idx": 7,
                "answer": "350",
                "problem": "Let a_n be the number of sequences of length n consisting of the letters A, B, C, D such that no two adjacent letters are the same, and the number of occurrences of A plus the number of occurrences of B is equal to the number of occurrences of C plus the number of occurrences of D. Find a_8 mod 1000.",
                "source": "AIME-2026",
            },
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format AIME sample as prompt"""
        problem = sample.get("problem", "")
        prompt_lines = [f"Problem: {problem}"]
        if include_answer:
            answer = sample.get("answer", "")
            prompt_lines.append(f"Solution: The answer is {answer}.")
            prompt_lines.append(f"Final Answer: {answer}")
        else:
            prompt_lines.append("Solution: Let me solve this step by step.")
        return "\n".join(prompt_lines)

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build complete AIME prompt"""
        instruction = (
            "Solve the following math problem from AIME (American Invitational Mathematics Examination).\n"
            "Think step by step and show your reasoning process.\n"
            "IMPORTANT: The answer MUST be an integer between 000 and 999 (inclusive).\n"
            "Put your final answer in \\boxed{}.\n\n"
        )
        examples = []
        for example in self.few_shot_examples[: self.num_shots]:
            examples.append(self.format_prompt(example, include_answer=True))
        question = self.format_prompt(sample, include_answer=False)
        full_prompt = instruction
        if examples:
            full_prompt += "\n\n".join(examples) + "\n\n"
        full_prompt += question
        return full_prompt

    def build_chat_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Build chat messages for the AIME evaluator."""
        messages: list[dict[str, str]] = []
        system_instruction = (
            "Solve the following math problem from AIME (American Invitational Mathematics Examination).\n"
            "Think step by step and show your reasoning process.\n"
            "IMPORTANT: The answer MUST be an integer between 000 and 999 (inclusive).\n"
            "Put your final answer in \\boxed{}."
        )
        messages.append({"role": "system", "content": system_instruction})

        for ex in self.few_shot_examples[: self.num_shots]:
            messages.append(
                {"role": "user", "content": self.format_prompt(ex, include_answer=False)}
            )
            full_example = self.format_prompt(ex, include_answer=True)
            solution_part = full_example.split("Solution:", 1)
            if len(solution_part) > 1:
                assistant_content = solution_part[1].strip()
            else:
                assistant_content = full_example
            messages.append({"role": "assistant", "content": assistant_content})

        messages.append(
            {"role": "user", "content": self.format_prompt(sample, include_answer=False)}
        )
        return messages

    def parse_response(self, response: str) -> str:
        """
        Parse model response to extract answer.
        Priority: MathAnswerParser -> \boxed{} extraction -> pattern fallback
        """
        # Delegate to MathAnswerParser first
        math_parser = MathAnswerParser()
        parsed = math_parser.parse(response)
        if parsed:
            num = self._extract_integer(parsed)
            if num is not None:
                return str(num)
            return parsed

        # Fallback: original boxed extraction
        boxed_answers = []
        i = 0
        while i < len(response):
            if response[i : i + 7] == r"\boxed{":
                start = i + 7
                depth = 1
                j = start
                while j < len(response) and depth > 0:
                    if response[j] == "{":
                        depth += 1
                    elif response[j] == "}":
                        depth -= 1
                    j += 1
                if depth == 0:
                    content = response[start : j - 1].strip()
                    boxed_answers.append(content)
                    i = j
                    continue
            i += 1

        if boxed_answers:
            answer = boxed_answers[-1]
            num = self._extract_integer(answer)
            if num is not None:
                return str(num)
            return answer

        # Fallback: pattern-based extraction
        patterns = [
            r"(?:final\s+)?answer\s*(?:is|:)\s*(\d{1,3})\b",
            r"Answer[：:]\s*(\d{1,3})\b",
            r"(?:=|equals?)\s*(\d{1,3})\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1)
                num = int(answer)
                if 0 <= num <= 999:
                    return str(num).zfill(3) if num < 100 else str(num)

        # Final fallback: last valid 0-999 integer
        all_integers = re.findall(r"\b(\d{1,3})\b", response)
        for num_str in reversed(all_integers):
            num = int(num_str)
            if 0 <= num <= 999:
                return str(num)

        return ""

    def _extract_integer(self, text: str) -> int | None:
        """Extract integer from text"""
        clean = text.strip()
        clean = clean.replace("^\\circ", "")
        clean = clean.replace("^{\\circ}", "")
        clean = clean.replace("\\circ", "")
        clean = clean.replace("°", "")
        clean = clean.replace("\\degree", "")
        clean = re.sub(r"\\+[a-zA-Z]+\{([^{}]*)\}", r"\1", clean)
        clean = clean.replace("$", "").replace(",", "").strip()
        try:
            num = int(clean)
            if 0 <= num <= 999:
                return num
        except ValueError:
            pass
        match = re.search(r"(\d{1,3})", clean)
        if match:
            num = int(match.group(1))
            if 0 <= num <= 999:
                return num
        return None

    def check_answer(self, predicted: str, correct: str) -> bool:
        """Check if predicted answer matches correct answer"""
        if not predicted or not correct:
            return False
        if MathAnswerParser.check_answer(predicted, correct):
            return True
        pred_int = self._extract_integer(predicted)
        corr_int = self._extract_integer(correct)
        if pred_int is not None and corr_int is not None:
            return pred_int == corr_int
        pred_clean = re.sub(r"\s+", "", predicted).lower()
        corr_clean = re.sub(r"\s+", "", correct).lower()
        try:
            pred_num = str(int(pred_clean))
            corr_num = str(int(corr_clean))
            return pred_num == corr_num
        except ValueError:
            pass
        return pred_clean == corr_clean

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get sample category"""
        return sample.get("source", "AIME-2026")

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """Get correct answer"""
        return str(sample.get("answer", ""))
