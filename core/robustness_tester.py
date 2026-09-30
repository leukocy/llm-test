"""
鲁棒性Test系统 (Robustness Testing System)

评估Model对输入微扰敏感性，检测Model鲁棒性。
"""

import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from core.evaluation_control import EvaluationJournal, run_samples
from evaluators.base_evaluator import ProviderRequestError

PERTURBATION_GENERATOR_VERSION = "text-perturber-v2"


class PerturbationType(Enum):
    """扰动类型"""

    SYNONYM_REPLACE = "synonym"  # 同义词替换
    TYPO_INSERT = "typo"  # Insert拼写Error
    WORD_REORDER = "reorder"  # 词序调整
    CASE_CHANGE = "case"  # 大小写变化
    PUNCTUATION = "punctuation"  # 标点变化
    WHITESPACE = "whitespace"  # 空白符变化
    NUMBER_FORMAT = "number_format"  # 数字格式变化
    PARAPHRASE = "paraphrase"  # 同义改写
    CONTEXT_ADD = "context_add"  # Addno关onunder文
    QUESTION_REPHRASE = "rephrase"  # 问题重述


@dataclass
class PerturbedSample:
    """扰动后样本"""

    original_question: str
    perturbed_question: str
    perturbation_type: PerturbationType
    perturbation_details: str = ""


@dataclass
class RobustnessResult:
    """鲁棒性Test Results"""

    sample_id: str
    original_question: str
    correct_answer: str

    # 原始Result
    original_answer: str = ""
    original_correct: bool = False

    # 扰动Result
    perturbed_results: list[dict[str, Any]] = field(default_factory=list)

    # 鲁棒性指标
    robustness_score: float = 0.0  # 扰动后保持正确比例
    consistency_score: float = 0.0  # 扰动后Answer一致比例
    sensitivity_by_type: dict[str, float] = field(default_factory=dict)


@dataclass
class RobustnessReport:
    """鲁棒性Test Report"""

    model_id: str
    total_samples: int
    perturbations_per_sample: int
    perturbation_generator_version: str = PERTURBATION_GENERATOR_VERSION
    metric_contract_version: str = "robustness-accuracy-v2"

    # 总体指标
    original_accuracy: float = 0.0
    perturbed_accuracy: float = 0.0
    accuracy_drop: float = 0.0

    overall_robustness: float = 0.0
    overall_consistency: float = 0.0

    # 按扰动类型敏感性
    sensitivity_by_type: dict[str, float] = field(default_factory=dict)
    most_sensitive_perturbation: str = ""

    # Detailed Results
    results: list[RobustnessResult] = field(default_factory=list)

    # Suggestion
    recommendations: list[str] = field(default_factory=list)


class TextPerturber:
    """
    文本扰动器

    Generate各种类型输入扰动。
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        # 同义词字典
        self.synonyms = {
            "calculate": ["compute", "determine", "find", "work out"],
            "what": ["which", "how much"],
            "is": ["equals", "="],
            "how many": ["what number of", "count of"],
            "total": ["sum", "altogether", "in all"],
            "each": ["every", "per"],
            "if": ["when", "given that", "suppose"],
            "has": ["owns", "possesses", "holds"],
            "买": ["购买", "购入"],
            "卖": ["出售", "售出"],
            "多少": ["几", "什么数量"],
            "Calculate": ["算", "求", "得出"],
        }

    def perturb(self, text: str, perturbation_type: PerturbationType) -> PerturbedSample:
        """
        对文本Apply扰动

        Args:
            text: 原始文本
            perturbation_type: 扰动类型

        Returns:
            PerturbedSample
        """
        if perturbation_type == PerturbationType.SYNONYM_REPLACE:
            return self._synonym_replace(text)
        elif perturbation_type == PerturbationType.TYPO_INSERT:
            return self._insert_typo(text)
        elif perturbation_type == PerturbationType.CASE_CHANGE:
            return self._change_case(text)
        elif perturbation_type == PerturbationType.PUNCTUATION:
            return self._change_punctuation(text)
        elif perturbation_type == PerturbationType.WHITESPACE:
            return self._change_whitespace(text)
        elif perturbation_type == PerturbationType.NUMBER_FORMAT:
            return self._change_number_format(text)
        elif perturbation_type == PerturbationType.CONTEXT_ADD:
            return self._add_context(text)
        elif perturbation_type == PerturbationType.WORD_REORDER:
            return self._reorder_words(text)
        else:
            return PerturbedSample(
                original_question=text,
                perturbed_question=text,
                perturbation_type=perturbation_type,
                perturbation_details="No perturbation applied",
            )

    def _synonym_replace(self, text: str) -> PerturbedSample:
        """同义词替换"""
        perturbed = text
        replaced = []

        for word, synonyms in self.synonyms.items():
            match = re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)
            if match:
                replacement = self.rng.choice(synonyms)
                # 保持原始大小写
                if match[0][0].isupper():
                    replacement = replacement.capitalize()
                perturbed = re.sub(
                    rf"\b{re.escape(word)}\b",
                    replacement,
                    perturbed,
                    flags=re.IGNORECASE,
                    count=1,
                )
                replaced.append(f"{word} → {replacement}")
                break  # 只替换一词

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.SYNONYM_REPLACE,
            perturbation_details=(", ".join(replaced) if replaced else "No synonyms found"),
        )

    def _insert_typo(self, text: str) -> PerturbedSample:
        """Insert拼写Error"""
        words = list(re.finditer(r"\S+", text))
        if len(words) < 3:
            return PerturbedSample(
                original_question=text,
                perturbed_question=text,
                perturbation_type=PerturbationType.TYPO_INSERT,
                perturbation_details="Text too short",
            )

        # 选择一长度>=4单词
        long_words = [match for match in words if len(match[0]) >= 4 and match[0].isalpha()]
        if not long_words:
            return PerturbedSample(
                original_question=text,
                perturbed_question=text,
                perturbation_type=PerturbationType.TYPO_INSERT,
                perturbation_details="No suitable words",
            )

        match = self.rng.choice(long_words)
        word = match[0]

        # 交换两相邻字母
        pos = self.rng.randint(1, len(word) - 2)
        typo_word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]

        perturbed = text[: match.start()] + typo_word + text[match.end() :]

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.TYPO_INSERT,
            perturbation_details=f"{word} → {typo_word}",
        )

    def _change_case(self, text: str) -> PerturbedSample:
        """大小写变化"""
        perturbed = text.lower()  # 全小写

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.CASE_CHANGE,
            perturbation_details="Converted to lowercase",
        )

    def _change_punctuation(self, text: str) -> PerturbedSample:
        """标点变化"""
        # 移除末尾标点
        perturbed = text.rstrip("?.!")

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.PUNCTUATION,
            perturbation_details="Removed ending punctuation",
        )

    def _change_whitespace(self, text: str) -> PerturbedSample:
        """空白符变化"""
        # Add额外空格
        perturbed = re.sub(r" ", "  ", text, count=3)

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.WHITESPACE,
            perturbation_details="Added extra spaces",
        )

    def _change_number_format(self, text: str) -> PerturbedSample:
        """数字格式变化"""

        # 1000 -> 1,000 or反过来
        def swap_format(match):
            num = match.group(0)
            # Consume candidates once, then validate their pieces without regex
            # backtracking or reinterpreting part of an identifier/invalid number.
            before = text[match.start() - 1] if match.start() else ""
            after = text[match.end()] if match.end() < len(text) else ""
            if (before and (before.isalnum() or before in "_.,+-")) or (
                after and (after.isalnum() or after == "_")
            ):
                return num
            if "e" in num.lower():
                return num
            candidate = num.rstrip(".,")
            punctuation = num[len(candidate) :]
            sign = candidate[:1] if candidate.startswith(("-", "+")) else ""
            unsigned = candidate[len(sign) :]
            integer, dot, fraction = unsigned.partition(".")
            if not integer or (dot and (not fraction or not fraction.isdigit())):
                return num
            if "," in integer:
                pieces = integer.split(",")
                if not (
                    1 <= len(pieces[0]) <= 3
                    and pieces[0].isdigit()
                    and all(len(piece) == 3 and piece.isdigit() for piece in pieces[1:])
                ):
                    return num
                integer = integer.replace(",", "")
            elif not integer.isdigit():
                return num
            elif len(integer) >= 4 and not integer.startswith("0"):
                first = len(integer) % 3 or 3
                integer = ",".join(
                    [integer[:first]] + [integer[i : i + 3] for i in range(first, len(integer), 3)]
                )
            return sign + integer + dot + fraction + punctuation

        perturbed = re.sub(
            r"[-+]?[0-9][0-9,.eE+-]*",
            swap_format,
            text,
        )

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.NUMBER_FORMAT,
            perturbation_details="Changed number format",
        )

    def _add_context(self, text: str) -> PerturbedSample:
        """Addno关onunder文"""
        prefixes = [
            "By the way, the weather is nice today. ",
            "Before we start, I should mention this is an interesting problem. ",
            "Let me think about this carefully. ",
        ]

        prefix = self.rng.choice(prefixes)
        perturbed = prefix + text

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.CONTEXT_ADD,
            perturbation_details=f"Added prefix: {prefix[:20]}...",
        )

    def _reorder_words(self, text: str) -> PerturbedSample:
        """词序调整（轻微）"""
        # in问句in移动一副词
        words = text.split()
        if len(words) < 5:
            return PerturbedSample(
                original_question=text,
                perturbed_question=text,
                perturbation_type=PerturbationType.WORD_REORDER,
                perturbation_details="Text too short",
            )

        # 简单实现：交换两相邻非关键词
        perturbed = text  # defaultnot变

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.WORD_REORDER,
            perturbation_details="Minimal reordering",
        )


class RobustnessTester:
    """
    Robustness Tester

    评估Model对各种输入扰动敏感性。

    Usage:
        tester = RobustnessTester()

        result = await tester.test_single(
            sample_id="001",
            question="What is 5 + 3?",
            correct_answer="8",
            get_response_func=my_api_call
        )

        print(result.robustness_score)
    """

    def __init__(self, perturbation_types: list[PerturbationType] | None = None, seed: int = 42):
        self.perturber = TextPerturber(seed)
        self.perturbation_types = perturbation_types or [
            PerturbationType.SYNONYM_REPLACE,
            PerturbationType.CASE_CHANGE,
            PerturbationType.PUNCTUATION,
            PerturbationType.NUMBER_FORMAT,
            PerturbationType.CONTEXT_ADD,
        ]

    async def test_single(
        self,
        sample_id: str,
        question: str,
        correct_answer: str,
        get_response_func: Callable,
        answer_parser: Callable | None = None,
        perturbations: list[PerturbedSample] | None = None,
    ) -> RobustnessResult:
        """
        Test单 samples鲁棒性
        """
        result = RobustnessResult(
            sample_id=sample_id,
            original_question=question,
            correct_answer=correct_answer,
        )

        # 1. Test原始问题
        try:
            original_response = await get_response_func(question)
            result.original_answer = self._extract_answer(original_response, answer_parser)
            result.original_correct = self._check_answer(result.original_answer, correct_answer)
        except Exception as e:
            raise ProviderRequestError(f"Provider request failed: {e}") from e

        # 2. Test各种扰动
        consistent_count = 0
        correct_after_perturbation = 0
        type_results: dict[str, list[bool]] = {t.value: [] for t in self.perturbation_types}

        planned = (
            perturbations
            if perturbations is not None
            else [self.perturber.perturb(question, ptype) for ptype in self.perturbation_types]
        )
        if [p.perturbation_type for p in planned] != self.perturbation_types:
            raise ValueError("Frozen perturbation types differ from the test plan")
        for perturbed in planned:
            ptype = perturbed.perturbation_type

            try:
                response = await get_response_func(perturbed.perturbed_question)
                answer = self._extract_answer(response, answer_parser)
                is_correct = self._check_answer(answer, correct_answer)
                is_consistent = answer == result.original_answer

                result.perturbed_results.append(
                    {
                        "perturbation_type": ptype.value,
                        "perturbed_question": perturbed.perturbed_question[:200],
                        "details": perturbed.perturbation_details,
                        "answer": answer,
                        "is_correct": is_correct,
                        "is_consistent": is_consistent,
                    }
                )

                type_results[ptype.value].append(is_correct)

                if is_consistent:
                    consistent_count += 1
                if is_correct:
                    correct_after_perturbation += 1

            except Exception as e:
                raise ProviderRequestError(f"Provider request failed: {e}") from e

        # 3. Calculated metrics
        if self.perturbation_types:
            result.robustness_score = correct_after_perturbation / len(self.perturbation_types)
            result.consistency_score = consistent_count / len(self.perturbation_types)

        # 按类型敏感性
        for pkey, results in type_results.items():
            if results:
                result.sensitivity_by_type[pkey] = 1 - (sum(results) / len(results))

        return result

    def _extract_answer(self, response, parser=None) -> str:
        """提取Answer"""
        if parser:
            return str(parser(response))

        content = response.get("content", "") if isinstance(response, dict) else str(response)

        # 简单提取最后一数字
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", content)
        return str(numbers[-1]) if numbers else str(content[:50])

    def _check_answer(self, predicted: str, correct: str) -> bool:
        """CheckAnswer"""
        try:
            pred = float(predicted.replace(",", ""))
            corr = float(correct.replace(",", ""))
            return abs(pred - corr) < 0.01
        except Exception:
            return predicted.strip().lower() == correct.strip().lower()

    async def test_batch(
        self,
        samples: list[dict[str, Any]],
        get_response_func: Callable,
        answer_parser: Callable | None = None,
        progress_callback: Callable | None = None,
        *,
        control_checkpoint: Callable[[], Awaitable[None]] | None = None,
        control_poll: Callable[[], bool] | None = None,
        journal: EvaluationJournal | None = None,
    ) -> RobustnessReport:
        """批量鲁棒性Test"""

        report = RobustnessReport(
            model_id="",
            total_samples=len(samples),
            perturbations_per_sample=len(self.perturbation_types),
        )

        scope_key = "robustness"

        def freeze() -> tuple[dict, list[dict]]:
            return {"generator_version": PERTURBATION_GENERATOR_VERSION}, [
                {
                    "sample": sample,
                    "perturbations": [
                        {**asdict(p), "perturbation_type": p.perturbation_type.value}
                        for p in [
                            self.perturber.perturb(sample.get("question", ""), ptype)
                            for ptype in self.perturbation_types
                        ]
                    ],
                }
                for sample in samples
            ]

        metadata, plan = journal.prepare_scope(scope_key, freeze) if journal else freeze()
        if metadata["generator_version"] != PERTURBATION_GENERATOR_VERSION:
            raise ValueError("Frozen perturbation generator version differs")
        restored = (
            {
                index: RobustnessResult(**value)
                for index, value in journal.results(scope_key).items()
            }
            if journal
            else {}
        )

        async def evaluate(index: int) -> RobustnessResult:
            unit = plan[index]
            sample = unit["sample"]
            return await self.test_single(
                sample_id=sample.get("sample_id", str(index)),
                question=sample.get("question", ""),
                correct_answer=sample.get("correct_answer", ""),
                get_response_func=get_response_func,
                answer_parser=answer_parser,
                perturbations=[
                    PerturbedSample(
                        **{**p, "perturbation_type": PerturbationType(p["perturbation_type"])}
                    )
                    for p in unit["perturbations"]
                ],
            )

        report.results = await run_samples(
            evaluate,
            total=len(plan),
            concurrency=1,
            restored=restored,
            checkpoint=control_checkpoint,
            pause_requested=control_poll,
            start=(lambda index: journal.start(scope_key, index)) if journal else None,
            commit=(lambda index, result: journal.commit(scope_key, index, asdict(result)))
            if journal
            else None,
            progress=progress_callback,
        )

        # Calculate汇总指标
        self._calculate_report_metrics(report)

        return report

    def _calculate_report_metrics(self, report: RobustnessReport):
        """Calculate报告指标"""
        if not report.results:
            return

        # 原始Accuracy
        original_correct = sum(1 for r in report.results if r.original_correct)
        report.original_accuracy = original_correct / len(report.results)

        # 扰动后Average指标
        robustness_scores = [r.robustness_score for r in report.results]
        consistency_scores = [r.consistency_score for r in report.results]

        report.overall_robustness = sum(robustness_scores) / len(robustness_scores)
        report.overall_consistency = sum(consistency_scores) / len(consistency_scores)

        # 扰动后Accuracy
        perturbations = [p for result in report.results for p in result.perturbed_results]
        report.perturbed_accuracy = (
            sum(bool(p.get("is_correct")) for p in perturbations) / len(perturbations)
            if perturbations
            else 0.0
        )
        report.accuracy_drop = report.original_accuracy - report.perturbed_accuracy

        # 按类型敏感性
        type_sensitivity: dict[str, list[float]] = {}
        for result in report.results:
            for ptype, sens in result.sensitivity_by_type.items():
                if ptype not in type_sensitivity:
                    type_sensitivity[ptype] = []
                type_sensitivity[ptype].append(sens)

        for ptype, values in type_sensitivity.items():
            report.sensitivity_by_type[ptype] = sum(values) / len(values)

        # 最敏感扰动类型
        if report.sensitivity_by_type:
            report.most_sensitive_perturbation = max(
                report.sensitivity_by_type.items(), key=lambda x: x[1]
            )[0]

        # GenerateSuggestion
        report.recommendations = self._generate_recommendations(report)

    def _generate_recommendations(self, report: RobustnessReport) -> list[str]:
        """Generate改进Suggestion"""
        recommendations = []

        if report.accuracy_drop > 0.1:
            recommendations.append(
                f"Accuracyin扰动后under降 {report.accuracy_drop * 100:.1f}%，Model鲁棒性need改进"
            )

        if report.overall_consistency < 0.7:
            recommendations.append(
                "Answer一致性较低，Suggestion降低 temperature or增加明确输出格式要求"
            )

        if report.most_sensitive_perturbation:
            sens = report.sensitivity_by_type.get(report.most_sensitive_perturbation, 0)
            if sens > 0.3:
                recommendations.append(
                    f"Model对 {report.most_sensitive_perturbation} 类型扰动最敏感 ({sens * 100:.0f}%)"
                )

        if not recommendations:
            recommendations.append("Model鲁棒性表现Good")

        return recommendations
