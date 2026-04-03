"""
Base Evaluator Module
Defines the core Evaluator base class and common data structures for quality evaluation.
"""

import asyncio
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Try to import enhanced parser
try:
    from core.enhanced_parser import AnswerType as EnhancedAnswerType
    from core.enhanced_parser import EnhancedAnswerParser, get_parser, quick_parse
    from core.enhanced_parser import ParseResult as EnhancedParseResult
    from core.enhanced_parser import compare_answers as enhanced_compare_answers
    ENHANCED_PARSER_AVAILABLE = True
except ImportError:
    ENHANCED_PARSER_AVAILABLE = False
    EnhancedAnswerParser = None


class DatasetType(Enum):
    """Supported dataset types."""
    MMLU = "mmlu"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"
    TRUTHFULQA = "truthfulqa"
    HELLASWAG = "hellaswag"
    ARC = "arc"
    WINOGRANDE = "winogrande"
    CEVAL = "ceval"
    CMMLU = "cmmlu"


@dataclass
class SampleResult:
    """Individual sample evaluation result."""
    sample_id: str
    question: str
    correct_answer: str
    model_response: str
    predicted_answer: str
    is_correct: bool
    category: str = ""
    prompt: str = ""  # Full input prompt
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: str | None = None
    is_judge_corrected: bool = False # Whether it was corrected by an AI judge
    evaluation_method: str = "regex" # Evaluation method: regex, llm_judge, smart_parser

    # Performance Metrics
    input_tokens: int = 0       # Input token count
    output_tokens: int = 0      # Output token count
    ttft_ms: float = 0.0        # Time To First Token (milliseconds)
    tps: float = 0.0            # Tokens Per Second (output)
    total_time_ms: float = 0.0  # Total time (milliseconds)

    # Reasoning-related fields
    reasoning_content: str = ""           # Reasoning process (extracted from thought tag or field)
    reasoning_tokens: int = 0             # Reasoning token count
    ttut_ms: float = 0.0                  # Time To User Text (first non-reasoning token)

    # Reasoning quality assessment
    reasoning_quality_overall: float = 0.0  # Overall reasoning quality (0-10)
    reasoning_coherence: float = 0.0        # Logical coherence (0-10)
    reasoning_completeness: float = 0.0     # Step completeness (0-10)
    reasoning_relevance: float = 0.0        # Relevance to question (0-10)
    reasoning_step_count: int = 0           # Reasoning step count

    # Answer parsing details
    answer_parse_confidence: float = 0.0    # Confidence score (0-1)
    answer_parse_method: str = ""           # Parse method (rule_boxed, rule_hash, llm, etc.)

    # Failure analysis
    failure_category: str = ""              # Category: calculation_error, concept_error, etc.
    failure_analysis: str = ""              # Explanation of failure

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a dataset."""
    dataset_name: str
    model_id: str
    accuracy: float
    total_samples: int
    correct_samples: int
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)  # {category: {accuracy, count}}
    details: list[SampleResult] = field(default_factory=list)
    timestamp: str = ""
    duration_seconds: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    # Aggregated performance metrics
    performance_stats: dict[str, Any] = field(default_factory=dict)

    # Extended metrics (statistics)
    extended_metrics: dict[str, Any] = field(default_factory=dict)
    # May include: stderr, ci_lower, ci_upper, f1, bleu, rouge_l, etc.

    def to_dict(self) -> dict[str, Any]:
        result = {
            "dataset_name": self.dataset_name,
            "model_id": self.model_id,
            "accuracy": self.accuracy,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "by_category": self.by_category,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "config": self.config,
            "performance_stats": self.performance_stats,
            "extended_metrics": self.extended_metrics,
            "details": [d.to_dict() for d in self.details]
        }
        return result

    def compute_performance_stats(self):
        """Aggregate performance metrics from individual samples."""
        if not self.details:
            return

        valid_results = [d for d in self.details if not d.error]
        if not valid_results:
            return

        # Collect metrics
        ttft_values = [d.ttft_ms for d in valid_results if d.ttft_ms > 0]
        tps_values = [d.tps for d in valid_results if d.tps > 0]
        input_tokens = [d.input_tokens for d in valid_results if d.input_tokens > 0]
        output_tokens = [d.output_tokens for d in valid_results if d.output_tokens > 0]
        latencies = [d.latency_ms for d in valid_results if d.latency_ms > 0]

        self.performance_stats = {
            # Token Statistics
            "total_input_tokens": sum(input_tokens) if input_tokens else 0,
            "total_output_tokens": sum(output_tokens) if output_tokens else 0,
            "avg_input_tokens": sum(input_tokens) / len(input_tokens) if input_tokens else 0,
            "avg_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else 0,

            # TTFT Statistics
            "avg_ttft_ms": sum(ttft_values) / len(ttft_values) if ttft_values else 0,
            "min_ttft_ms": min(ttft_values) if ttft_values else 0,
            "max_ttft_ms": max(ttft_values) if ttft_values else 0,

            # TPS Statistics
            "avg_tps": sum(tps_values) / len(tps_values) if tps_values else 0,
            "min_tps": min(tps_values) if tps_values else 0,
            "max_tps": max(tps_values) if tps_values else 0,

            # Latency Statistics
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,

            # Success Rate
            "success_rate": len(valid_results) / len(self.details) if self.details else 0,
            "error_count": len(self.details) - len(valid_results)
        }

        # Compute extended statistical metrics
        self._compute_extended_metrics()

    def _compute_extended_metrics(self):
        """Compute statistical indicators like standard error and confidence intervals."""
        if not self.details:
            return

        try:
            from core.metrics import (
                bootstrap_confidence_interval,
                per_category_accuracy,
                standard_error,
                wilson_score_interval,
            )

            # Collect correct/incorrect scores
            is_correct = [1.0 if d.is_correct else 0.0 for d in self.details if not d.error]

            if not is_correct:
                return

            # Compute standard error
            stderr = standard_error(is_correct)

            # Compute bootstrap confidence interval (95%)
            ci = bootstrap_confidence_interval(is_correct, confidence=0.95)

            # Compute Wilson score interval
            wilson_ci = wilson_score_interval(self.correct_samples, self.total_samples, 0.95)

            self.extended_metrics = {
                "stderr": stderr,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "wilson_ci_lower": wilson_ci[0],
                "wilson_ci_upper": wilson_ci[1],
                "confidence_level": 0.95
            }

            # Per-category breakdown
            if self.by_category:
                self.extended_metrics["per_category"] = self.by_category

        except ImportError:
            # Metrics module not available
            pass
        except Exception:
            # Silent ignore for other errors
            pass

    def save_to_json(self, filepath: str):
        """Save results to a JSON file."""
        if not self.performance_stats:
            self.compute_performance_stats()

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> 'EvaluationResult':
        """Load results from a JSON file."""
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)

        details = [SampleResult(**d) for d in data.get('details', [])]
        result = cls(
            dataset_name=data['dataset_name'],
            model_id=data['model_id'],
            accuracy=data['accuracy'],
            total_samples=data['total_samples'],
            correct_samples=data['correct_samples'],
            by_category=data.get('by_category', {}),
            details=details,
            timestamp=data.get('timestamp', ''),
            duration_seconds=data.get('duration_seconds', 0.0),
            config=data.get('config', {}),
            performance_stats=data.get('performance_stats', {})
        )
        return result


class BaseEvaluator(ABC):
    """
    Abstract base class for all dataset evaluators.
    All specific evaluators should inherit from this class.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        num_shots: int = 0,
        max_samples: int | None = None,
        seed: int = 42,
        prompt_template: Any | None = None,  # PromptTemplate or ChatTemplate
        prompt_format: str = "completion"  # "completion" or "chat"
    ):
        """
        Initialize the evaluator.

        Args:
            dataset_name: Name of the dataset.
            dataset_path: File path to the dataset.
            num_shots: Number of few-shot examples to include.
            max_samples: Limit number of samples (None = use all).
            seed: Random seed for reproducibility.
            prompt_template: Custom prompt template object.
            prompt_format: Format ("completion" or "chat").
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.num_shots = num_shots
        self.max_samples = max_samples
        self.seed = seed
        self.prompt_format = prompt_format

        self.samples: list[dict[str, Any]] = []
        self.few_shot_examples: list[dict[str, Any]] = []

        # Initialize template system
        self._prompt_template = prompt_template
        self._init_template()

    def _init_template(self):
        """Initialize prompt templates from YAML or factory."""
        if self._prompt_template is not None:
            return

        try:
            from core.prompt_template import PromptFormat, TemplateFactory

            # Try to load from task_configs directory
            yaml_path = f"task_configs/{self.dataset_name}.yaml"
            if os.path.exists(yaml_path):
                self._prompt_template = TemplateFactory.from_yaml(
                    yaml_path,
                    format=self.prompt_format
                )
                return

            # Try to get predefined template from factory
            try:
                self._prompt_template = TemplateFactory.get(
                    self.dataset_name,
                    format=self.prompt_format
                )
            except ValueError:
                pass

        except ImportError:
            pass

    def get_template(self):
        """Return the current prompt template."""
        return self._prompt_template

    def set_template(self, template):
        """Set a custom prompt template."""
        self._prompt_template = template

    @abstractmethod
    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """
        Load samples from the dataset.

        Args:
            subset: Optional subset name (e.g., "high_school_math" for MMLU).

        Returns:
            List of sample dictionaries.
        """
        pass

    @abstractmethod
    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """
        Format a single sample into a prompt string.

        Args:
            sample: Sample data dictionary.
            include_answer: Whether to include the answer (for few-shot).

        Returns:
            Formatted prompt string.
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> str:
        """
        Extract the model's answer from its response string.

        Args:
            response: Raw string response from the model.

        Returns:
            Extracted answer as a string.
        """
        pass

    @abstractmethod
    def check_answer(self, predicted: str, correct: str) -> bool:
        """
        Verify if the predicted answer matches the correct answer.

        Args:
            predicted: Answer extracted from the model.
            correct: Ground truth answer.

        Returns:
            True if correct, False otherwise.
        """
        pass

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """
        Build the full prompt including few-shot examples.
        Prioritizes the template system, falls back to legacy methods.

        Args:
            sample: The sample to be evaluated.

        Returns:
            The complete prompt string.
        """
        if self._prompt_template is not None:
            try:
                if hasattr(self._prompt_template, 'render_as_completion'):
                    return self._prompt_template.render_as_completion(
                        sample,
                        self.few_shot_examples[:self.num_shots]
                    )
                elif hasattr(self._prompt_template, 'render_full'):
                    return self._prompt_template.render_full(
                        sample,
                        self.few_shot_examples[:self.num_shots]
                    )
            except Exception:
                pass # Fall back to traditional method

        # Traditional method
        prompt_parts = []
        for example in self.few_shot_examples[:self.num_shots]:
            prompt_parts.append(self.format_prompt(example, include_answer=True))
            prompt_parts.append("") # Separator
        prompt_parts.append(self.format_prompt(sample, include_answer=False))

        return "\n".join(prompt_parts)

    def build_chat_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """
        Build a list of chat messages for chat API.

        Args:
            sample: The sample to be evaluated.

        Returns:
            List of message dictionaries.
        """
        if self._prompt_template is not None and hasattr(self._prompt_template, 'render_messages'):
            try:
                return self._prompt_template.render_messages(
                    sample,
                    self.few_shot_examples[:self.num_shots]
                )
            except Exception:
                pass

        # Fallback: convert completion prompt to simple user message
        prompt = self.build_full_prompt(sample)
        return [{"role": "user", "content": prompt}]

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """
        Return the category of a sample for reporting.
        """
        return sample.get('category', sample.get('subject', ''))

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """
        Return the ground truth answer for a sample.
        """
        return str(sample.get('answer', ''))

    async def evaluate_single(
        self,
        sample: dict[str, Any],
        get_response_func: Callable[[str], Any],
        sample_index: int = 0
    ) -> SampleResult:
        """
        Evaluate a single sample.

        Args:
            sample: Sample dictionary.
            get_response_func: Async function to call the LLM.
            sample_index: Position in the batch.

        Returns:
            A populated SampleResult object.
        """
        sample_id = sample.get('id', str(sample_index))
        category = self.get_sample_category(sample)
        correct_answer = self.get_correct_answer(sample)

        # Initial performance metrics
        input_tokens = 0
        output_tokens = 0
        ttft_ms = 0.0
        tps = 0.0
        total_time_ms = 0.0

        try:
            # Build prompt
            prompt = self.build_full_prompt(sample)

            # Get model response
            start_time = time.time()
            response_data = await get_response_func(prompt)
            latency_ms = (time.time() - start_time) * 1000

            # Process response (handles string or dict return)
            if isinstance(response_data, dict):
                response = response_data.get('content', '')
                input_tokens = response_data.get('input_tokens', 0)
                output_tokens = response_data.get('output_tokens', 0)
                ttft_ms = response_data.get('ttft_ms', 0.0)
                tps = response_data.get('tps', 0.0)
                total_time_ms = response_data.get('total_time_ms', latency_ms)

                if response_data.get('error'):
                    raise Exception(response_data['error'])
            else:
                response = str(response_data)
                total_time_ms = latency_ms

            # Extract reasoning content (think tags or API field)
            reasoning_content = ""
            reasoning_tokens = 0
            
            # 1. Try from API response object
            if isinstance(response_data, dict):
                reasoning_content = response_data.get('reasoning_content', '')
                reasoning_tokens = response_data.get('reasoning_tokens', 0)
            
            # 2. Try parsing <think> tags from content
            if not reasoning_content and response:
                think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
                if think_match:
                    reasoning_content = think_match.group(1).strip()
            
            # Parse and check answer
            predicted = self.parse_response(response)
            is_correct = self.check_answer(predicted, correct_answer)
            error_msg = None
            is_judge_corrected = False
            evaluation_method = "regex"

            judge_enabled = getattr(self, "use_llm_judge", False)

            # --- AI Judge Verification (Only for missed cases with non-empty responses) ---
            if not is_correct and judge_enabled and response.strip():
                print(f"⚡ [Judge Triggered] Sample {sample_id} | Model Answer Preview: {response[:30]}...")
                evaluation_method = "llm_judge_attempted"
                try:
                    judge_q = sample.get('question', '')
                    if judge_q:
                        judge_prompt = (
                            f"You are an impartial judge evaluating mathematical or factual answers.\n\n"
                            f"Question: {judge_q}\n\n"
                            f"Target Answer: {correct_answer}\n\n"
                            f"Model's Full Response: {response}\n\n"
                            f"Task: Determine if the model's final answer is mathematically or semantically equivalent to the Target Answer.\n\n"
                            f"Guidelines:\n"
                            f"1. Look for the final answer in the model's response (e.g., in \\boxed{{...}}, after 'Final Answer:', etc.).\n"
                            f"2. Compare SEMANTIC VALUE. For example: 15400.0 = 15400, 2/4 = 0.5.\n"
                            f"3. Ignore formatting differences or LaTeX inconsistencies.\n\n"
                            f"Reply with ONLY 'YES' if they match, or 'NO' if they don't."
                        )

                        judge_res_data = await get_response_func(judge_prompt)

                        judge_content = ""
                        if isinstance(judge_res_data, dict):
                            judge_content = judge_res_data.get('content', '')
                            input_tokens += judge_res_data.get('input_tokens', 0)
                            output_tokens += judge_res_data.get('output_tokens', 0)
                        else:
                            judge_content = str(judge_res_data)

                        print(f"[AI Judge] Sample {sample_id} | Correct: {correct_answer} | Judge Says: {judge_content}")

                        if "YES" in judge_content.upper():
                            is_correct = True
                            error_msg = "Validated by AI Judge"
                            is_judge_corrected = True
                            evaluation_method = "llm_judge_passed"
                        else:
                            evaluation_method = "llm_judge_rejected"

                except Exception as e:
                    print(f"LLM Judge Error: {e}")

            return SampleResult(
                sample_id=sample_id,
                prompt=prompt,
                question=sample.get('question', prompt[:200]),
                correct_answer=correct_answer,
                model_response=response,
                predicted_answer=predicted,
                is_correct=is_correct,
                is_judge_corrected=is_judge_corrected,
                evaluation_method=evaluation_method,
                category=category,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                tps=tps,
                total_time_ms=total_time_ms,
                error=error_msg,
                reasoning_content=reasoning_content,
                reasoning_tokens=reasoning_tokens
            )

        except Exception as e:
            return SampleResult(
                sample_id=sample_id,
                prompt=locals().get('prompt', str(sample)[:500]),
                question=sample.get('question', ''),
                correct_answer=correct_answer,
                model_response="",
                predicted_answer="",
                is_correct=False,
                category=category,
                error=str(e)
            )

    async def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
        get_response_func: Callable[[str], Any],
        concurrency: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[['SampleResult'], None] | None = None
    ) -> list[SampleResult]:
        """
        Evaluate a list of samples in parallel.

        Args:
            samples: List of samples to evaluate.
            get_response_func: Async LLM caller.
            concurrency: Parallel request limit.
            progress_callback: Progress tracker.
            result_callback: Individual result tracker for real-time reporting.

        Returns:
            List of SampleResult objects.
        """
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0
        results_lock = asyncio.Lock()

        async def evaluate_with_semaphore(sample: dict, index: int) -> SampleResult:
            nonlocal completed
            async with semaphore:
                result = await self.evaluate_single(sample, get_response_func, index)

                async with results_lock:
                    completed += 1
                    if result_callback:
                        try:
                            result_callback(result)
                        except Exception:
                            pass

                    if progress_callback:
                        progress_callback(completed, len(samples))

                return result

        tasks = [
            evaluate_with_semaphore(sample, i)
            for i, sample in enumerate(samples)
        ]

        results = await asyncio.gather(*tasks)
        return list(results)

    def compute_metrics(self, results: list[SampleResult]) -> tuple[float, dict[str, dict[str, float]]]:
        """
        Compute aggregate metrics from evaluation results.

        Returns:
            (Overall accuracy, per-category metrics map)
        """
        if not results:
            return 0.0, {}

        # Filter out empty responses
        valid_results = [r for r in results if r.model_response and r.model_response.strip()]

        if not valid_results:
             return 0.0, {}

        # Overall Accuracy
        correct_count = sum(1 for r in valid_results if r.is_correct)
        accuracy = correct_count / len(valid_results)

        # Per-category stats
        by_category: dict[str, dict[str, Any]] = {}
        for result in valid_results:
            cat = result.category or "unknown"
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0}
            by_category[cat]["total"] += 1
            if result.is_correct:
                by_category[cat]["correct"] += 1

        # Finalize rates
        for cat in by_category:
            total = by_category[cat]["total"]
            correct = by_category[cat]["correct"]
            by_category[cat]["accuracy"] = correct / total if total > 0 else 0.0
            by_category[cat]["count"] = total

        return accuracy, by_category


def extract_choice_answer(
    response: str,
    choices: list[str] = None,
    use_enhanced: bool = True
) -> str:
    """
    Extract multiple-choice answer from model response.
    Supports formats like: "A", "A.", "(A)", "Answer: A", "The answer is A".
    """
    if choices is None:
        choices = ['A', 'B', 'C', 'D']
    if not response:
        return ""

    response = response.strip()

    # Try enhanced parser if available
    if use_enhanced and ENHANCED_PARSER_AVAILABLE:
        try:
            parser = get_parser()
            result = parser.parse(response, EnhancedAnswerType.CHOICE, choices=choices)
            if result.normalized and result.confidence > 0.4:
                return str(result.normalized).upper()
        except Exception:
            pass

    # Fallback to regex logic
    choice_pattern = '|'.join(choices)
    patterns = [
        rf'(?:answer|Answer|The answer is)[:\s]*[（(]?({choice_pattern})[)）]?',
        rf'(?:Selection|Option|Selected)[:\s]*[（(]?({choice_pattern})[)）]?',
        rf'^[（(]?({choice_pattern})[)）.]?\s*$',
        rf'^[（(]?({choice_pattern})[)）.]',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Last resort: find any standalone valid choice
    all_matches = re.findall(rf'\b({choice_pattern})\b', response.upper())
    if all_matches:
        return all_matches[-1] # Usually the last one is the final choice

    return ""


def extract_numeric_answer(
    response: str,
    use_enhanced: bool = True
) -> str:
    """
    Extract numeric answer from model response.
    Supports currency, scientific notation, and commas.
    """
    if not response:
        return ""

    # Try enhanced parser
    if use_enhanced and ENHANCED_PARSER_AVAILABLE:
        try:
            parser = get_parser()
            result = parser.parse(response, EnhancedAnswerType.NUMERIC)
            if result.normalized:
                return str(result.normalized)
        except Exception:
            pass

    # Fallback to simple regex
    # Support scientific notation like 1.23e-4 or simple 12,345.67
    patterns = [
        r'(\\boxed|Final Answer:|####)[:\s]*.*?([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)',
        r'([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            # If multiple matches, find the one that looks most like a standalone number
            val = matches[-1]
            if isinstance(val, tuple): val = val[1]
            return val.replace(',', '')

    return ""

def normalize_text(text: str) -> str:
    """Normalize text forComparison (lowercase, remove punctuation/extra whitespace)."""
    if not text:
        return ""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text
