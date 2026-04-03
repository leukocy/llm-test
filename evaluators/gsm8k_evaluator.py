"""
GSM8K Evaluator
Evaluator for the GSM8K (Grade School Math 8K) dataset.
GSM8K is a dataset of 8.5K high-quality grade school math word problems.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_numeric_answer


class GSM8KEvaluator(BaseEvaluator):
    """
    GSM8K Dataset Evaluator.
    
    Data Format:
    {
        "question": "Janet has 10 apples. She gives 3 to her sister...",
        "answer": ".... #### 7"
    }
    """

    def __init__(
        self,
        dataset_name: str = "gsm8k",
        dataset_path: str = "datasets/gsm8k",
        num_shots: int = 4,
        max_samples: int | None = None,
        seed: int = 42,
        use_llm_judge: bool = False
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_shots=num_shots,
            max_samples=max_samples,
            seed=seed
        )
        self.use_llm_judge = use_llm_judge

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """Load GSM8K dataset using DatasetManager."""
        try:
            from core.dataset_manager import get_dataset
            samples = get_dataset(
                name=self.dataset_name,
                split="test",
                max_samples=self.max_samples,
                seed=self.seed
            )
            self.samples = samples
            return samples
        except Exception as e:
            print(f"[WARNING] GSM8K load failed: {e}")
            return []

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format GSM8K sample into a prompt."""
        question = sample.get('question', '')
        prompt = f"Question: {question}\nAnswer:"
        
        if include_answer:
            answer = sample.get('answer', '')
            prompt += f" {answer}"
            
        return prompt

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build full GSM8K prompt with CoT (Chain of Thought) instruction."""
        instruction = "Solve the following math word problem step by step. End your response with '#### [result]'.\n\n"
        
        # Add few-shot examples
        examples = []
        for example in self.few_shot_examples[:self.num_shots]:
            examples.append(self.format_prompt(example, include_answer=True))
            
        # Add current sample
        current = self.format_prompt(sample, include_answer=False)
        
        full_prompt = instruction + "\n\n".join(examples)
        if examples:
            full_prompt += "\n\n"
        full_prompt += current
        
        return full_prompt

    def parse_response(self, response: str) -> str:
        """Extract the numeric answer from a CoT response."""
        # Try finding the #### marker first
        if "####" in response:
            parts = response.split("####")
            if len(parts) > 1:
                ans = parts[-1].strip().replace(",", "")
                # Extract first number from the remaining string
                match = re.search(r'[-+]?\d*\.?\d+', ans)
                if match:
                    return match.group()
        
        # Fallback to general numeric extractor
        return extract_numeric_answer(response)

    def check_answer(self, predicted: str, correct: str) -> bool:
        """Verify if the mathematical answer is correct."""
        if not predicted:
            return False
            
        # Extract ground truth from ".... #### 7"
        correct_val = ""
        if "####" in correct:
            correct_val = correct.split("####")[-1].strip().replace(",", "")
        else:
            correct_val = correct.strip().replace(",", "")
            
        try:
            # Float comparison to handle 7.0 == 7
            return float(predicted) == float(correct_val)
        except ValueError:
            return predicted.strip() == correct_val.strip()

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """Extract ground truth answer for reporting."""
        ans = sample.get('answer', '')
        if "####" in ans:
            return ans.split("####")[-1].strip()
        return ans
