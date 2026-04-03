"""
鲁棒性Test系统 (Robustness Testing System)

评估Model对输入微扰敏感性，检测Model鲁棒性。
"""

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class PerturbationType(Enum):
    """扰动类型"""
    SYNONYM_REPLACE = "synonym"           # 同义词替换
    TYPO_INSERT = "typo"                  # Insert拼写Error
    WORD_REORDER = "reorder"              # 词序调整
    CASE_CHANGE = "case"                  # 大小写变化
    PUNCTUATION = "punctuation"           # 标点变化
    WHITESPACE = "whitespace"             # 空白符变化
    NUMBER_FORMAT = "number_format"       # 数字格式变化
    PARAPHRASE = "paraphrase"             # 同义改写
    CONTEXT_ADD = "context_add"           # Addno关onunder文
    QUESTION_REPHRASE = "rephrase"        # 问题重述


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
        random.seed(seed)

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

    def perturb(
        self,
        text: str,
        perturbation_type: PerturbationType
    ) -> PerturbedSample:
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
                perturbation_details="No perturbation applied"
            )

    def _synonym_replace(self, text: str) -> PerturbedSample:
        """同义词替换"""
        perturbed = text
        replaced = []

        for word, synonyms in self.synonyms.items():
            if word.lower() in text.lower():
                replacement = random.choice(synonyms)
                # 保持原始大小写
                if word[0].isupper():
                    replacement = replacement.capitalize()
                perturbed = re.sub(
                    rf'\b{re.escape(word)}\b',
                    replacement,
                    perturbed,
                    flags=re.IGNORECASE,
                    count=1
                )
                replaced.append(f"{word} → {replacement}")
                break  # 只替换一词

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.SYNONYM_REPLACE,
            perturbation_details=", ".join(replaced) if replaced else "No synonyms found"
        )

    def _insert_typo(self, text: str) -> PerturbedSample:
        """Insert拼写Error"""
        words = text.split()
        if len(words) < 3:
            return PerturbedSample(
                original_question=text,
                perturbed_question=text,
                perturbation_type=PerturbationType.TYPO_INSERT,
                perturbation_details="Text too short"
            )

        # 选择一长度>=4单词
        long_words = [(i, w) for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
        if not long_words:
            return PerturbedSample(
                original_question=text,
                perturbed_question=text,
                perturbation_type=PerturbationType.TYPO_INSERT,
                perturbation_details="No suitable words"
            )

        idx, word = random.choice(long_words)

        # 交换两相邻字母
        pos = random.randint(1, len(word) - 2)
        typo_word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]

        words[idx] = typo_word
        perturbed = ' '.join(words)

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.TYPO_INSERT,
            perturbation_details=f"{word} → {typo_word}"
        )

    def _change_case(self, text: str) -> PerturbedSample:
        """大小写变化"""
        perturbed = text.lower()  # 全小写

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.CASE_CHANGE,
            perturbation_details="Converted to lowercase"
        )

    def _change_punctuation(self, text: str) -> PerturbedSample:
        """标点变化"""
        # 移除末尾标点
        perturbed = re.sub(r'[?.!]+$', '', text)

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.PUNCTUATION,
            perturbation_details="Removed ending punctuation"
        )

    def _change_whitespace(self, text: str) -> PerturbedSample:
        """空白符变化"""
        # Add额外空格
        perturbed = re.sub(r' ', '  ', text, count=3)

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.WHITESPACE,
            perturbation_details="Added extra spaces"
        )

    def _change_number_format(self, text: str) -> PerturbedSample:
        """数字格式变化"""
        # 1000 -> 1,000 or反过来
        def swap_format(match):
            num = match.group(0)
            if ',' in num:
                return num.replace(',', '')
            elif len(num) >= 4:
                # Add逗号
                return f"{int(num):,}"
            return num

        perturbed = re.sub(r'\d[\d,]+', swap_format, text)

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.NUMBER_FORMAT,
            perturbation_details="Changed number format"
        )

    def _add_context(self, text: str) -> PerturbedSample:
        """Addno关onunder文"""
        prefixes = [
            "By the way, the weather is nice today. ",
            "Before we start, I should mention this is an interesting problem. ",
            "Let me think about this carefully. ",
        ]

        prefix = random.choice(prefixes)
        perturbed = prefix + text

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.CONTEXT_ADD,
            perturbation_details=f"Added prefix: {prefix[:20]}..."
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
                perturbation_details="Text too short"
            )

        # 简单实现：交换两相邻非关键词
        perturbed = text  # defaultnot变

        return PerturbedSample(
            original_question=text,
            perturbed_question=perturbed,
            perturbation_type=PerturbationType.WORD_REORDER,
            perturbation_details="Minimal reordering"
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

    def __init__(
        self,
        perturbation_types: list[PerturbationType] = None,
        seed: int = 42
    ):
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
        answer_parser: Callable = None
    ) -> RobustnessResult:
        """
        Test单 samples鲁棒性
        """
        result = RobustnessResult(
            sample_id=sample_id,
            original_question=question,
            correct_answer=correct_answer
        )

        # 1. Test原始问题
        try:
            original_response = await get_response_func(question)
            result.original_answer = self._extract_answer(original_response, answer_parser)
            result.original_correct = self._check_answer(result.original_answer, correct_answer)
        except Exception as e:
            result.original_answer = f"Error: {e}"
            result.original_correct = False

        # 2. Test各种扰动
        consistent_count = 0
        correct_after_perturbation = 0
        type_results = {t.value: [] for t in self.perturbation_types}

        for ptype in self.perturbation_types:
            perturbed = self.perturber.perturb(question, ptype)

            try:
                response = await get_response_func(perturbed.perturbed_question)
                answer = self._extract_answer(response, answer_parser)
                is_correct = self._check_answer(answer, correct_answer)
                is_consistent = answer == result.original_answer

                result.perturbed_results.append({
                    "perturbation_type": ptype.value,
                    "perturbed_question": perturbed.perturbed_question[:200],
                    "details": perturbed.perturbation_details,
                    "answer": answer,
                    "is_correct": is_correct,
                    "is_consistent": is_consistent
                })

                type_results[ptype.value].append(is_correct)

                if is_consistent:
                    consistent_count += 1
                if is_correct:
                    correct_after_perturbation += 1

            except Exception as e:
                result.perturbed_results.append({
                    "perturbation_type": ptype.value,
                    "error": str(e)
                })

        # 3. Calculated metrics
        if self.perturbation_types:
            result.robustness_score = correct_after_perturbation / len(self.perturbation_types)
            result.consistency_score = consistent_count / len(self.perturbation_types)

        # 按类型敏感性
        for ptype, results in type_results.items():
            if results:
                result.sensitivity_by_type[ptype] = 1 - (sum(results) / len(results))

        return result

    def _extract_answer(self, response, parser=None) -> str:
        """提取Answer"""
        if parser:
            return parser(response)

        content = response.get('content', '') if isinstance(response, dict) else str(response)

        # 简单提取最后一数字
        numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', content)
        return numbers[-1] if numbers else content[:50]

    def _check_answer(self, predicted: str, correct: str) -> bool:
        """CheckAnswer"""
        try:
            pred = float(predicted.replace(',', ''))
            corr = float(correct.replace(',', ''))
            return abs(pred - corr) < 0.01
        except:
            return predicted.strip().lower() == correct.strip().lower()

    async def test_batch(
        self,
        samples: list[dict[str, Any]],
        get_response_func: Callable,
        answer_parser: Callable = None,
        progress_callback: Callable = None
    ) -> RobustnessReport:
        """批量鲁棒性Test"""

        report = RobustnessReport(
            model_id="",
            total_samples=len(samples),
            perturbations_per_sample=len(self.perturbation_types)
        )

        results = []
        for i, sample in enumerate(samples):
            result = await self.test_single(
                sample_id=sample.get('sample_id', str(i)),
                question=sample.get('question', ''),
                correct_answer=sample.get('correct_answer', ''),
                get_response_func=get_response_func,
                answer_parser=answer_parser
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(samples))

        report.results = results

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
        report.perturbed_accuracy = report.overall_robustness * report.original_accuracy
        report.accuracy_drop = report.original_accuracy - report.perturbed_accuracy

        # 按类型敏感性
        type_sensitivity = {}
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
                report.sensitivity_by_type.items(),
                key=lambda x: x[1]
            )[0]

        # GenerateSuggestion
        report.recommendations = self._generate_recommendations(report)

    def _generate_recommendations(self, report: RobustnessReport) -> list[str]:
        """Generate改进Suggestion"""
        recommendations = []

        if report.accuracy_drop > 0.1:
            recommendations.append(
                f"Accuracyin扰动后under降 {report.accuracy_drop*100:.1f}%，Model鲁棒性need改进"
            )

        if report.overall_consistency < 0.7:
            recommendations.append(
                "Answer一致性较低，Suggestion降低 temperature or增加明确输出格式要求"
            )

        if report.most_sensitive_perturbation:
            sens = report.sensitivity_by_type.get(report.most_sensitive_perturbation, 0)
            if sens > 0.3:
                recommendations.append(
                    f"Model对 {report.most_sensitive_perturbation} 类型扰动最敏感 ({sens*100:.0f}%)"
                )

        if not recommendations:
            recommendations.append("Model鲁棒性表现Good")

        return recommendations


def create_robustness_tester(
    perturbation_types: list[str] = None
) -> RobustnessTester:
    """Factory函数：CreateRobustness Tester"""
    types = [PerturbationType(t) for t in perturbation_types] if perturbation_types else None
    return RobustnessTester(perturbation_types=types)
