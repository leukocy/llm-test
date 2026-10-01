"""
HumanEval Evaluator
Evaluator for the OpenAI HumanEval dataset.
HumanEval is a benchmark for code generation containing 164 Python programming problems.
"""

import ast
import json
import keyword
import os
import random
import re
import time
from typing import Any

from . import register_evaluator
from .answer_parser import CodeAnswerParser
from .base_evaluator import BaseEvaluator, DatasetUnavailableError, SampleResult


@register_evaluator("humaneval")
class HumanEvalEvaluator(BaseEvaluator):
    """
    HumanEval Dataset Evaluator.

    Data Format:
    {
        "task_id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    ...",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        ...",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    ...",
        "entry_point": "has_close_elements"
    }
    """

    requires_code_execution = True

    # Execution timeout (seconds)
    EXECUTION_TIMEOUT = 5

    def __init__(
        self,
        dataset_name: str = "humaneval",
        dataset_path: str = "datasets/humaneval",
        num_shots: int = 0,  # HumanEval typically uses 0-shot
        max_samples: int | None = None,
        seed: int = 42,
        timeout: int = 5,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_shots=num_shots,
            max_samples=max_samples,
            seed=seed,
        )
        self.timeout = timeout
        random.seed(seed)

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """Load HumanEval dataset via DatasetManager with fallback."""
        samples = []

        try:
            from core.dataset_manager import get_dataset

            samples = get_dataset(
                name=self.dataset_name, split="test", max_samples=None, seed=self.seed
            )
        except Exception as e:
            print(f"[WARNING] DatasetManager failed: {e}. Falling back to manual load.")
            possible_files = [
                os.path.join(self.dataset_path, "test.json"),
                os.path.join(self.dataset_path, "humaneval.json"),
                os.path.join(self.dataset_path, "HumanEval.jsonl"),
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
                        break
                    except Exception as e:
                        print(f"Failed to load {filepath}: {e}")

        if not samples:
            samples = self._fallback_to_demo_samples(self._create_sample_data)
        identifiers = set()
        for sample in samples:
            identifier, entry, tests = (
                sample.get("task_id"),
                sample.get("entry_point"),
                sample.get("test"),
            )
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise DatasetUnavailableError("HumanEval requires distinct task IDs")
            identifiers.add(identifier)
            if (
                not isinstance(entry, str)
                or not entry.isidentifier()
                or keyword.iskeyword(entry)
                or not isinstance(tests, str)
            ):
                raise DatasetUnavailableError("Invalid HumanEval entry point or tests")
            try:
                tree = ast.parse(tests)
            except SyntaxError as exc:
                raise DatasetUnavailableError("Invalid HumanEval test syntax") from exc
            if not any(
                isinstance(node, ast.FunctionDef) and node.name == "check" for node in tree.body
            ):
                raise DatasetUnavailableError("HumanEval tests must define check(candidate)")

        if self.max_samples and len(samples) > self.max_samples:
            random.shuffle(samples)
            samples = samples[: self.max_samples]

        self.samples = samples
        return samples

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Return a few mock samples for testing."""
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": '''from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if anywhere in the list two numbers are closer than threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    """\n''',
                "canonical_solution": """    for i, x in enumerate(numbers):\n        for j, y in enumerate(numbers):\n            if i != j and abs(x - y) < threshold: return True\n    return False\n""",
                "test": """\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\ncheck(has_close_elements)\n""",
                "entry_point": "has_close_elements",
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format the sample as code prompt."""
        prompt = sample.get("prompt", "")
        if include_answer:
            prompt += sample.get("canonical_solution", "")
        return str(prompt)

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Add system instruction before code prompt."""
        instruction = (
            "Complete the following Python function. "
            "Only provide the function body, not the function signature. "
            "Do not include any explanation or markdown formatting.\n\n"
        )
        return str(instruction + sample.get("prompt", ""))

    def build_chat_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Build chat messages for the HumanEval evaluator."""
        messages: list[dict[str, str]] = []
        system_instruction = (
            "Complete the following Python function. "
            "Only provide the function body, not the function signature."
        )
        messages.append({"role": "system", "content": system_instruction})

        for ex in self.few_shot_examples[: self.num_shots]:
            messages.append({"role": "user", "content": ex.get("prompt", "")})
            messages.append({"role": "assistant", "content": ex.get("canonical_solution", "")})

        messages.append({"role": "user", "content": sample.get("prompt", "")})
        return messages

    def parse_response(self, response: str) -> str:
        """Clean up the generated code (remove markdown blocks, etc.)."""
        # Delegate to CodeAnswerParser first
        code_parser = CodeAnswerParser()
        parsed = code_parser.parse(response)
        if parsed:
            return parsed

        # Fallback: original extraction
        code = response
        code_block_pattern = r"```(?:python)?\s*(.*?)```"
        matches = re.findall(code_block_pattern, code, re.DOTALL)
        if matches:
            code = matches[0]
        return code.strip()

    def check_answer(self, predicted: str, correct: str) -> bool:
        """Verification is handled in evaluate_single via code execution."""
        return False

    async def evaluate_single(
        self, sample: dict[str, Any], get_response_func, sample_index: int = 0
    ) -> SampleResult:
        """Evaluate a code generation sample via execution."""
        task_id = sample.get("task_id", str(sample_index))
        entry_point = sample.get("entry_point", "")
        test_code = sample.get("test", "")
        base_prompt = sample.get("prompt", "")

        input_tokens = output_tokens = 0
        ttft_ms = tps = total_time_ms = 0.0

        try:
            full_prompt = self.build_full_prompt(sample)
            chat_messages = self.build_chat_messages(sample)
            start_time = time.time()
            # Use chat messages if they contain more than a single user message
            if len(chat_messages) > 1 or any(m.get("role") == "system" for m in chat_messages):
                response_data = await get_response_func(messages=chat_messages)
            else:
                response_data = await get_response_func(full_prompt)
            latency_ms = (time.time() - start_time) * 1000

            if isinstance(response_data, dict):
                response = response_data.get("content", "")
                input_tokens = response_data.get("input_tokens", 0)
                output_tokens = response_data.get("output_tokens", 0)
                ttft_ms = response_data.get("ttft_ms", 0.0)
                tps = response_data.get("tps", 0.0)
                total_time_ms = response_data.get("total_time_ms", latency_ms)
                if response_data.get("error"):
                    raise Exception(response_data["error"])
            else:
                response = str(response_data)
                total_time_ms = latency_ms

            generated_code = self.parse_response(response)
            full_code = base_prompt + generated_code

            # Sanity check: code execution
            is_correct, error_msg, _exec_output = self._execute_code(
                full_code, test_code, entry_point
            )

            return SampleResult(
                sample_id=task_id,
                prompt=full_prompt,
                question=base_prompt[:200],
                correct_answer=sample.get("canonical_solution", "")[:100],
                model_response=response,
                predicted_answer=generated_code[:200],
                is_correct=is_correct,
                category="code",
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                tps=tps,
                total_time_ms=total_time_ms,
                execution_error=error_msg if not is_correct else None,
                evaluation_method="code_execution",
            )

        except Exception as e:
            return SampleResult(
                sample_id=task_id,
                prompt=locals().get("full_prompt", str(sample)[:500]),
                question=base_prompt[:200] if "base_prompt" in locals() else "",
                correct_answer="",
                model_response="",
                predicted_answer="",
                is_correct=False,
                category="code",
                error=str(e),
            )

    def _execute_code(
        self, code: str, test_code: str, entry_point: str
    ) -> tuple[bool, str | None, str | None]:
        """Run generated code through the isolated sandbox worker."""
        from core.safe_executor import run_untrusted_code

        if not test_code.strip() or not entry_point.isidentifier():
            return False, "Invalid HumanEval test or entry point", None
        full_code = code + "\n" + test_code + f"\ncheck({entry_point})\n"
        # Code and tests are sent to the container worker; the app never executes them.
        return run_untrusted_code(full_code, timeout_seconds=10.0)

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        return "code"

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        return str(sample.get("canonical_solution", ""))
