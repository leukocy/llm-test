"""
MATH-500 Evaluator
Evaluator for the OpenAI MATH-500 dataset.
MATH-500 consists of 500 challenging math problems selected from the MATH benchmark,
introduced in the "Let's Verify Step by Step" paper.
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_numeric_answer


class MATH500Evaluator(BaseEvaluator):
    """
    MATH-500 Dataset Evaluator.
    
    Data Format:
    {
        "problem": "Let $r$ be the positive real solution to $x^3 + \\frac{2}{5} x - 1 = 0.$...",
        "solution": "Let $f(x) = x^3 + \\frac{2}{5} x - 1.$...",
        "answer": "\\frac{1}{4}",  # or "\\boxed{\\frac{1}{4}}"
        "subject": "Intermediate Algebra",
        "level": 5,
        "unique_id": "test/intermediate_algebra/1234"
    }
    """

    # MATH subject categories
    SUBJECTS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ]

    def __init__(
        self,
        dataset_name: str = "math500",
        dataset_path: str = "datasets/math500",
        num_shots: int = 4,
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
        """Load MATH-500 dataset with optional subject filtering."""
        samples = []

        try:
            from core.dataset_manager import get_dataset
            
            # Load "test" split via DatasetManager
            all_samples = get_dataset(
                name=self.dataset_name,
                split="test", 
                max_samples=None,
                seed=self.seed
            )
            
            if subset:
                subset_lower = subset.lower().replace(' ', '_')
                samples = [
                    s for s in all_samples
                    if subset_lower in s.get('subject', '').lower().replace(' ', '_')
                ]
            else:
                samples = all_samples

        except Exception as e:
            print(f"[WARNING] DatasetManager failed: {e}. Falling back to manual load.")
            possible_files = [
                os.path.join(self.dataset_path, "test.json"),
                os.path.join(self.dataset_path, "math500.json"),
                os.path.join(self.dataset_path, "data.json"),
                os.path.join(self.dataset_path, "test.jsonl"),
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
                    except Exception as load_err:
                        print(f"Failed to load {filepath}: {load_err}")
                        
            if samples and subset:
                subset_lower = subset.lower().replace(' ', '_')
                samples = [s for s in samples if subset_lower in s.get('subject', '').lower().replace(' ', '_')]

        if not samples:
            samples = self._create_sample_data()
            if subset:
                 subset_lower = subset.lower().replace(' ', '_')
                 samples = [s for s in samples if subset_lower in s.get('subject', '').lower().replace(' ', '_')]

        random.shuffle(samples)

        # Few-shot management
        # Since MATH-500 is small (500), few-shot usually comes from the set itself
        total_needed = self.num_shots + (self.max_samples or len(samples))
        if len(samples) > total_needed:
            samples = samples[:total_needed]

        if self.num_shots > 0 and len(samples) > self.num_shots:
             self.few_shot_examples = samples[:self.num_shots]
             samples = samples[self.num_shots:]

        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Mock data for testing."""
        return [
            {
                "problem": "Compute $\\dbinom{8}{4}$.",
                "solution": "\\begin{align*}\\dbinom{8}{4} &= \\boxed{70}\\end{align*}",
                "answer": "70",
                "subject": "Counting & Probability",
                "level": 1,
                "unique_id": "test/counting_and_probability/0"
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format sample using CoT (Chain-of-Thought) pattern."""
        problem = sample.get('problem', '')
        prompt_lines = [f"Problem: {problem}"]

        if include_answer:
            solution = sample.get('solution', '')
            prompt_lines.append(f"Solution: {solution}")
        else:
            prompt_lines.append("Solution: Let me solve this step by step.")

        return "\n".join(prompt_lines)

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Construct prompt with instructions to boxed the final answer."""
        instruction = (
            "Solve the following math problem. "
            "Think step by step and put your final answer in \\boxed{}.\n\n"
        )
        examples = []
        for example in self.few_shot_examples[:self.num_shots]:
            examples.append(self.format_prompt(example, include_answer=True))

        current = self.format_prompt(sample, include_answer=False)
        full_prompt = instruction + "\n\n".join(examples)
        if examples:
            full_prompt += "\n\n"
        full_prompt += current

        return full_prompt

    def parse_response(self, response: str) -> str:
        """Extract content from the last \\boxed{...} tag."""
        results = []
        i = 0
        while i < len(response):
            if response[i:i+7] == r'\boxed{':
                start = i + 7
                depth = 1
                j = start
                while j < len(response) and depth > 0:
                    if response[j] == '{': depth += 1
                    elif response[j] == '}': depth -= 1
                    j += 1
                if depth == 0:
                    results.append(response[start:j-1].strip())
                    i = j
                    continue
            i += 1

        if results:
            return results[-1]

        # Fallback: extract last numeric value
        return extract_numeric_answer(response)

    def check_answer(self, predicted: str, correct: str) -> bool:
        """Compare the predicted answer with the correct answer."""
        if not predicted or not correct:
            return False

        # Clean boxed markers
        predicted_clean = self._strip_boxed(predicted)
        correct_clean = self._strip_boxed(correct)

        # Normalize and compare
        norm_pred = self._normalize_answer(predicted_clean)
        norm_corr = self._normalize_answer(correct_clean)

        if norm_pred == norm_corr:
            return True

        # Special case: "x \in [a,b]" vs "[a,b]"
        if (norm_corr.startswith("x\\in") or norm_corr.startswith("xin")) and not (norm_pred.startswith("x\\in") or norm_pred.startswith("xin")):
            norm_corr_val = re.sub(r'^[a-zA-Z](\\in|in)', '', norm_corr)
            if norm_pred == norm_corr_val:
                return True

        # Try evaluating as numeric float
        try:
            pred_val = self._eval_answer(predicted_clean)
            correct_val = self._eval_answer(correct_clean)
            if pred_val is not None and correct_val is not None:
                return abs(pred_val - correct_val) < 1e-6
        except:
            pass

        return False

    def _strip_boxed(self, text: str) -> str:
        """Recursively remove \\boxed tags."""
        while '\\boxed{' in text:
             text = re.sub(r'\\boxed\{([^{}]*)\}', r'\1', text)
             if '\\boxed{' in text: continue
             break
        return text

    def _normalize_answer(self, answer: str) -> str:
        """Standardize LaTeX math format for comparison."""
        if not answer:
            return ""

        answer = answer.strip()
        # Remove common LaTeX markers
        answer = answer.replace('^\\circ', '').replace('^{\\circ}', '').replace('\\degree', '')
        answer = answer.replace('$', '').replace('\\%', '')

        # Handle \text{}, \mathrm{}, etc.
        answer = re.sub(r'\\+text\{([^{}]*)\}', r'\1', answer)
        answer = re.sub(r'\\+mathrm\{([^{}]*)\}', r'\1', answer)
        answer = re.sub(r'\\+mbox\{([^{}]*)\}', r'\1', answer)

        answer = re.sub(r'^[a-zA-Z]\s*=\s*', '', answer) # Remove "x = "
        answer = re.sub(r'\s+', '', answer) # Remove all whitespace

        # Canonicalize LaTeX commands
        for cmd in ['\\frac', '\\dfrac', '\\left', '\\right']:
            answer = answer.replace(cmd, cmd.replace('\\', ''))
            answer = answer.replace('\\' + cmd, cmd.replace('\\', ''))

        # Standardize parentheses for single-letter options
        answer = re.sub(r'^[\(\[\{]([A-Z])[\)\]\}]$', r'\1', answer)

        return answer

    def _eval_answer(self, answer: str) -> float | None:
        """Convert LaTeX math expression to float if possible."""
        try:
            clean_ans = self._normalize_answer(answer)
            # Handle frac{a}{b}
            frac_match = re.match(r'frac\{([\d\.]+)\}\{([\d\.]+)\}', clean_ans)
            if frac_match:
                return float(frac_match.group(1)) / float(frac_match.group(2))
            return float(clean_ans)
        except:
            return None

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        return sample.get('subject', 'unknown')

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        return sample.get('answer', '')
