"""
失败案例分析系统 (Failure Analysis System)

自动分类and分析评估in失败案例，提供详细失败原因and改进Suggestion。
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class FailureCategory(Enum):
    """失败类别"""
    CALCULATION_ERROR = "calculation_error"         # CalculateError
    CONCEPT_MISUNDERSTANDING = "concept_error"      # 概念理解Error
    REASONING_GAP = "reasoning_gap"                 # 推理跳步
    HALLUCINATION = "hallucination"                 # 幻觉（引入Error message）
    FORMAT_MISMATCH = "format_mismatch"             # 格式问题
    KNOWLEDGE_GAP = "knowledge_gap"                 # 知识缺失
    ATTENTION_ERROR = "attention_error"             # 注意力Error（漏读Question信息）
    MAGNITUDE_ERROR = "magnitude_error"             # 数量级Error
    UNIT_CONVERSION = "unit_error"                  # 单位换算Error
    LOGIC_ERROR = "logic_error"                     # 逻辑Error
    NO_RESPONSE = "no_response"                     # no响应
    TIMEOUT = "timeout"                             # 超时
    API_ERROR = "api_error"                         # API Error
    UNKNOWN = "unknown"                             # 未知Error


@dataclass
class FailureCase:
    """失败案例"""
    sample_id: str
    question: str
    correct_answer: str
    predicted_answer: str
    model_response: str
    reasoning_content: str = ""

    # 分析Result
    category: FailureCategory = FailureCategory.UNKNOWN
    confidence: float = 0.0
    analysis: str = ""
    root_cause: str = ""
    suggestions: list[str] = field(default_factory=list)

    # 辅助信息
    question_type: str = ""  # math, choice, reasoning, etc.
    difficulty: str = ""     # easy, medium, hard


@dataclass
class FailureAnalysisReport:
    """失败分析报告"""
    total_samples: int
    failed_samples: int
    failure_rate: float

    # 按类别Statistics
    category_distribution: dict[str, int] = field(default_factory=dict)
    category_percentage: dict[str, float] = field(default_factory=dict)

    # 详细案例
    cases: list[FailureCase] = field(default_factory=list)

    # 汇总Suggestion
    top_issues: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)


class FailureAnalyzer:
    """
    失败案例分析器

    自动分类失败原因并提供改进Suggestion。

    Usage:
        analyzer = FailureAnalyzer()

        # 分析单失败案例
        case = analyzer.analyze_single(
            sample_id="001",
            question="What is 15 + 27?",
            correct_answer="42",
            predicted_answer="41",
            model_response="15 + 27 = 41",
            reasoning_content="..."
        )

        # 批量分析
        report = analyzer.analyze_batch(failed_samples)

        print(report.category_distribution)
        print(report.improvement_suggestions)
    """

    def __init__(self):
        # 关键词模式
        self.calculation_indicators = [
            r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+',
            r'Calculate|compute|calculate',
        ]

        self.knowledge_gaps = [
            'i don\'t know', 'i\'m not sure', 'unclear',
            'not确定', 'not知道', 'no法确定'
        ]

        self.hallucination_indicators = [
            'assume', 'suppose', 'let\'s say',
            '假设', '假如'
        ]

    def analyze_single(
        self,
        sample_id: str,
        question: str,
        correct_answer: str,
        predicted_answer: str,
        model_response: str,
        reasoning_content: str = "",
        error: str | None = None
    ) -> FailureCase:
        """
        分析单失败案例

        Args:
            sample_id: 样本 ID
            question: 问题
            correct_answer: Correct answer
            predicted_answer: 预测Answer
            model_response: Model响应
            reasoning_content: 推理内容
            error: Error message（ifhas）

        Returns:
            FailureCase
        """
        case = FailureCase(
            sample_id=sample_id,
            question=question,
            correct_answer=correct_answer,
            predicted_answer=predicted_answer,
            model_response=model_response,
            reasoning_content=reasoning_content
        )

        # 1. Checkis否is系统Error
        if error:
            if 'timeout' in error.lower():
                case.category = FailureCategory.TIMEOUT
                case.analysis = "Request timeout"
                case.confidence = 1.0
                case.suggestions = ["增加超时时间", "Check网络Connect"]
                return case
            elif any(code in error for code in ['429', '500', '502', '503']):
                case.category = FailureCategory.API_ERROR
                case.analysis = f"API Error: {error}"
                case.confidence = 1.0
                case.suggestions = ["Add重试机制", "Check API 配额"]
                return case

        # 2. Checkno响应
        if not model_response or not model_response.strip():
            case.category = FailureCategory.NO_RESPONSE
            case.analysis = "Model未Returnhas效响应"
            case.confidence = 1.0
            case.suggestions = ["Check prompt 格式", "调整 max_tokens"]
            return case

        # 3. 分析失败原因
        category, confidence, analysis, root_cause = self._classify_failure(
            question, correct_answer, predicted_answer, model_response, reasoning_content
        )

        case.category = category
        case.confidence = confidence
        case.analysis = analysis
        case.root_cause = root_cause
        case.suggestions = self._generate_suggestions(category, analysis)
        case.question_type = self._detect_question_type(question)

        return case

    def _classify_failure(
        self,
        question: str,
        correct: str,
        predicted: str,
        response: str,
        reasoning: str
    ) -> tuple[FailureCategory, float, str, str]:
        """
        分类失败原因

        Returns:
            (类别, 置信度, 分析说明, 根因)
        """
        full_text = (response + " " + reasoning).lower()

        # 1. Check数量级Error
        try:
            correct_num = float(correct.replace(',', ''))
            predicted_num = float(predicted.replace(',', ''))

            if correct_num != 0:
                ratio = predicted_num / correct_num
                if ratio in [10, 100, 1000, 0.1, 0.01, 0.001]:
                    return (
                        FailureCategory.MAGNITUDE_ERROR,
                        0.9,
                        f"Answer数量级Error（预测值is正确值 {ratio} 倍）",
                        "可能is单位换算or小数点位置Error"
                    )

                # 接近butnot完全正确
                if 0.9 <= ratio <= 1.1:
                    return (
                        FailureCategory.CALCULATION_ERROR,
                        0.8,
                        "Answer接近正确值，可能isCalculate误差",
                        "Calculate过程in精度问题or舍入Error"
                    )
        except (ValueError, TypeError):
            pass

        # 2. CheckCalculateError
        calc_matches = []
        for pattern in self.calculation_indicators:
            calc_matches.extend(re.findall(pattern, full_text, re.IGNORECASE))

        if calc_matches:
            # Checkis否存inErrorCalculate
            for match in calc_matches:
                if self._is_calculation_wrong(match):
                    return (
                        FailureCategory.CALCULATION_ERROR,
                        0.85,
                        f"检测到CalculateError: {match}",
                        "数学运算过程in出现Error"
                    )

        # 3. Check格式问题
        if predicted and correct:
            # Check内容is否etc.价but格式not同
            pred_clean = re.sub(r'[^\w]', '', predicted.lower())
            correct_clean = re.sub(r'[^\w]', '', correct.lower())

            if pred_clean == correct_clean:
                return (
                    FailureCategory.FORMAT_MISMATCH,
                    0.95,
                    "Answer内容etc.价but格式not匹配",
                    "AnswerParse未能正确识别etc.价格式"
                )

        # 4. Check知识缺失
        if any(gap in full_text for gap in self.knowledge_gaps):
            return (
                FailureCategory.KNOWLEDGE_GAP,
                0.8,
                "Model表示not确定ornot知道",
                "Model缺乏解决该问题所需知识"
            )

        # 5. Check注意力Error
        question_numbers = set(re.findall(r'\d+(?:\.\d+)?', question))
        response_numbers = set(re.findall(r'\d+(?:\.\d+)?', response))

        missing_numbers = question_numbers - response_numbers
        if len(missing_numbers) > len(question_numbers) * 0.5:
            return (
                FailureCategory.ATTENTION_ERROR,
                0.7,
                f"Model可能遗漏QuestioninData: {missing_numbers}",
                "未能完整捕获Question信息"
            )

        # 6. Check推理跳步
        if reasoning:
            step_count = len(re.findall(r'(?:step|\s*\d|首先|然后|因此)', reasoning, re.IGNORECASE))
            if step_count <= 1 and len(question) > 100:
                return (
                    FailureCategory.REASONING_GAP,
                    0.65,
                    "Reasoning process过于简略",
                    "可能跳过关键Reasoning Steps"
                )

        # 7. Check幻觉
        if any(ind in full_text for ind in self.hallucination_indicators):
            # Checkis否引入Questioninnot存in数字
            extra_numbers = response_numbers - question_numbers
            if len(extra_numbers) > 3:
                return (
                    FailureCategory.HALLUCINATION,
                    0.6,
                    f"Model可能引入额外信息: {extra_numbers}",
                    "Reasoning processin引入Question未提and假设"
                )

        # 8. default：概念理解Error
        return (
            FailureCategory.CONCEPT_MISUNDERSTANDING,
            0.5,
            "no法确定具体原因，可能is概念理解问题",
            "Model对问题理解可能存in偏差"
        )

    def _is_calculation_wrong(self, expr: str) -> bool:
        """CheckCalculate表达式is否Error"""
        # 提取 a op b = c 格式
        match = re.match(r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', expr)
        if not match:
            return False

        try:
            a, op, b, c = match.groups()
            a, b, c = float(a), float(b), float(c)

            if op == '+':
                return abs((a + b) - c) > 0.01
            elif op == '-':
                return abs((a - b) - c) > 0.01
            elif op == '*':
                return abs((a * b) - c) > 0.01
            elif op == '/':
                return abs((a / b) - c) > 0.01
        except:
            pass

        return False

    def _generate_suggestions(self, category: FailureCategory, analysis: str) -> list[str]:
        """Generate改进Suggestion"""
        suggestions = {
            FailureCategory.CALCULATION_ERROR: [
                "增加 few-shot 示例展示正确Calculate过程",
                "in prompt in强调仔细CheckCalculate",
                "考虑use更强数学推理Model"
            ],
            FailureCategory.CONCEPT_MISUNDERSTANDING: [
                "提供更清晰问题描述",
                "增加相关概念背景知识",
                "use chain-of-thought Tip"
            ],
            FailureCategory.REASONING_GAP: [
                "要求Model展示详细Reasoning Steps",
                "use step-by-step Tip词",
                "增加推理 token 预算"
            ],
            FailureCategory.HALLUCINATION: [
                "in prompt in强调只useQuestion提供信息",
                "要求Model标注use哪些Data",
                "降低 temperature 参数"
            ],
            FailureCategory.FORMAT_MISMATCH: [
                "改进AnswerParse器格式兼容性",
                "in prompt in明确Answer格式要求",
                "use LLM AnswerParse兜底"
            ],
            FailureCategory.KNOWLEDGE_GAP: [
                "提供必要背景知识",
                "考虑use知识增强 (RAG)",
                "选择Update训练Model"
            ],
            FailureCategory.ATTENTION_ERROR: [
                "in prompt in高亮关键Data",
                "要求Model先列出所has已知条件",
                "缩短Question长度or分段呈现"
            ],
            FailureCategory.MAGNITUDE_ERROR: [
                "in prompt in明确单位要求",
                "要求ModelCheckAnswer合理性",
                "增加单位Convert示例"
            ],
            FailureCategory.NO_RESPONSE: [
                "Check prompt 格式is否Trigger安全Filter",
                "增加 max_tokens 限制",
                "简化问题表述"
            ],
            FailureCategory.TIMEOUT: [
                "增加超时时间",
                "减少 prompt 长度",
                "CheckModel负载情况"
            ],
            FailureCategory.API_ERROR: [
                "Add重试机制",
                "Check API 配额",
                "切换备用 API 端点"
            ]
        }

        return suggestions.get(category, ["need人工分析具体原因"])

    def _detect_question_type(self, question: str) -> str:
        """检测问题类型"""
        q_lower = question.lower()

        if any(word in q_lower for word in ['calculate', 'compute', 'solve', 'Calculate', '求']):
            return "math"
        elif any(word in q_lower for word in ['which', 'choose', 'select', 'option', '选择', 'Options']):
            return "choice"
        elif any(word in q_lower for word in ['why', 'explain', 'how', 'is什么', '解释', '如何']):
            return "reasoning"
        elif any(word in q_lower for word in ['true', 'false', 'yes', 'no', 'is否', '对错']):
            return "boolean"
        else:
            return "general"

    def analyze_batch(
        self,
        failed_samples: list[dict[str, Any]],
        total_samples: int = None
    ) -> FailureAnalysisReport:
        """
        批量分析失败案例

        Args:
            failed_samples: 失败样本列表，每 samples应包含：
                - sample_id, question, correct_answer, predicted_answer,
                - model_response, reasoning_content, error
            total_samples: 总Sample count（用于Calculate失败率）

        Returns:
            FailureAnalysisReport
        """
        if total_samples is None:
            total_samples = len(failed_samples)

        # 分析每案例
        cases = []
        for sample in failed_samples:
            case = self.analyze_single(
                sample_id=sample.get('sample_id', ''),
                question=sample.get('question', ''),
                correct_answer=sample.get('correct_answer', ''),
                predicted_answer=sample.get('predicted_answer', ''),
                model_response=sample.get('model_response', ''),
                reasoning_content=sample.get('reasoning_content', ''),
                error=sample.get('error')
            )
            cases.append(case)

        # Statistics分布
        category_counts = Counter(case.category.value for case in cases)
        category_distribution = dict(category_counts)
        category_percentage = {
            k: v / len(cases) * 100 if cases else 0
            for k, v in category_distribution.items()
        }

        # Generate顶级问题
        top_issues = [
            f"{cat}: {count} 例 ({count/len(cases)*100:.1f}%)"
            for cat, count in category_counts.most_common(3)
        ]

        # 汇总Suggestion
        all_suggestions = []
        for case in cases:
            all_suggestions.extend(case.suggestions)

        suggestion_counts = Counter(all_suggestions)
        improvement_suggestions = [
            suggestion for suggestion, _ in suggestion_counts.most_common(5)
        ]

        return FailureAnalysisReport(
            total_samples=total_samples,
            failed_samples=len(cases),
            failure_rate=len(cases) / total_samples * 100 if total_samples else 0,
            category_distribution=category_distribution,
            category_percentage=category_percentage,
            cases=cases,
            top_issues=top_issues,
            improvement_suggestions=improvement_suggestions
        )


def analyze_failures(failed_samples: list[dict], total: int = None) -> FailureAnalysisReport:
    """便捷函数：批量分析失败案例"""
    analyzer = FailureAnalyzer()
    return analyzer.analyze_batch(failed_samples, total)
