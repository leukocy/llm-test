"""
Reasoning processQuality Evaluator (Reasoning Quality Evaluator)

not仅评估最终Answer，还评估ModelReasoning process质量。
including:逻辑连贯性、步骤完整性、推理相关性etc.维度。
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ReasoningQualityDimension(Enum):
    """推理Quality Assessment维度"""
    COHERENCE = "coherence"           # 逻辑连贯性
    COMPLETENESS = "completeness"     # 步骤完整性
    RELEVANCE = "relevance"           # 与问题相关性
    CORRECTNESS = "correctness"       # Reasoning Steps正确性
    EFFICIENCY = "efficiency"         # 推理效率（is否has冗余）


@dataclass
class ReasoningStep:
    """Reasoning Steps"""
    step_number: int
    content: str
    step_type: str = "reasoning"  # reasoning, calculation, conclusion, observation
    is_valid: bool | None = None
    validation_note: str = ""


@dataclass
class ReasoningQualityScore:
    """推理质量Score"""
    coherence: float = 0.0           # 连贯性 (0-10)
    completeness: float = 0.0        # 完整性 (0-10)
    relevance: float = 0.0           # 相关性 (0-10)
    correctness: float = 0.0         # 正确性 (0-10)
    efficiency: float = 0.0          # 效率 (0-10)

    overall: float = 0.0             # 综合分 (0-10)

    # 详细信息
    step_count: int = 0              # Reasoning Steps数
    valid_step_count: int = 0        # has效步骤数
    redundant_step_count: int = 0    # 冗余步骤数

    # 定性评价
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def calculate_overall(self):
        """Calculate综合分"""
        weights = {
            'coherence': 0.25,
            'completeness': 0.20,
            'relevance': 0.20,
            'correctness': 0.25,
            'efficiency': 0.10
        }
        self.overall = (
            self.coherence * weights['coherence'] +
            self.completeness * weights['completeness'] +
            self.relevance * weights['relevance'] +
            self.correctness * weights['correctness'] +
            self.efficiency * weights['efficiency']
        )


@dataclass
class ReasoningEvaluationResult:
    """推理评估完整Result"""
    # Answer评估
    final_answer_correct: bool = False
    answer_confidence: float = 0.0

    # 推理Quality Assessment
    quality_score: ReasoningQualityScore = field(default_factory=ReasoningQualityScore)

    # Reasoning Steps分析
    steps: list[ReasoningStep] = field(default_factory=list)

    # 失败分析（ifAnswerError）
    failure_category: str = ""  # calculation_error, concept_error, reasoning_gap, etc.
    failure_analysis: str = ""

    # 元信息
    evaluation_method: str = "rule"  # rule, llm, hybrid
    evaluation_confidence: float = 0.0


class ReasoningQualityEvaluator:
    """
    Reasoning processQuality Evaluator

    Usage:
        evaluator = ReasoningQualityEvaluator()

        # 基于规则快速评估
        result = evaluator.evaluate(
            question="What is 2 + 3?",
            reasoning="Let me think... 2 plus 3 equals 5.",
            final_answer="5",
            correct_answer="5"
        )

        # use LLM 进行深度评估
        result = await evaluator.evaluate_with_llm(
            question="...",
            reasoning="...",
            final_answer="...",
            correct_answer="...",
            llm_func=my_llm_call
        )
    """

    def __init__(self):
        # 推理指示词模式
        self.step_indicators = [
            r'(?:step\s*\d+|\s*\d+\s*步)',
            r'(?:first|second|third|fourth|fifth|首先|其次|然后|接着|最后)',
            r'(?:therefore|thus|hence|so|因此|therefore|由此)',
            r'(?:because|since|as|due to|because)',
            r'(?:let\'s|we need to|we can|我们need|让我们)',
        ]

        # Calculate模式
        self.calculation_patterns = [
            r'=\s*[\d.]+',
            r'[\d.]+\s*[+\-*/]\s*[\d.]+',
            r'\d+\s*(?:times|multiplied by|divided by|plus|minus)',
        ]

        # 结论模式
        self.conclusion_patterns = [
            r'(?:therefore|thus|hence|so|in conclusion|finally)',
            r'(?:the answer is|Answeris|Resultis|最终)',
            r'(?:####|\\boxed)',
        ]

    def evaluate(
        self,
        question: str,
        reasoning: str,
        final_answer: str,
        correct_answer: str,
        is_answer_correct: bool | None = None
    ) -> ReasoningEvaluationResult:
        """
        基于规则推理Quality Assessment

        Args:
            question: 原始问题
            reasoning: Reasoning process（canis空字符串）
            final_answer: Model最终Answer
            correct_answer: Correct answer
            is_answer_correct: Answeris否正确（if已知）

        Returns:
            ReasoningEvaluationResult
        """
        result = ReasoningEvaluationResult()

        # 1. 判断Answer正确性
        if is_answer_correct is not None:
            result.final_answer_correct = is_answer_correct
        else:
            result.final_answer_correct = self._check_answer_equivalence(
                final_answer, correct_answer
            )

        # 2. ParseReasoning Steps
        result.steps = self._parse_reasoning_steps(reasoning)
        result.quality_score.step_count = len(result.steps)

        # 3. 评估各维度
        if reasoning and reasoning.strip():
            result.quality_score.coherence = self._evaluate_coherence(reasoning, result.steps)
            result.quality_score.completeness = self._evaluate_completeness(reasoning, question, final_answer)
            result.quality_score.relevance = self._evaluate_relevance(reasoning, question)
            result.quality_score.correctness = self._evaluate_correctness(result.steps, result.final_answer_correct)
            result.quality_score.efficiency = self._evaluate_efficiency(reasoning, result.steps)
        else:
            # 没hasReasoning process
            result.quality_score.coherence = 0.0
            result.quality_score.completeness = 0.0
            result.quality_score.relevance = 0.0
            result.quality_score.correctness = 5.0 if result.final_answer_correct else 0.0
            result.quality_score.efficiency = 10.0 if result.final_answer_correct else 0.0
            result.quality_score.weaknesses.append("没has提供Reasoning process")

        # 4. Calculate综合分
        result.quality_score.calculate_overall()

        # 5. ifAnswerError，分析失败原因
        if not result.final_answer_correct:
            result.failure_category, result.failure_analysis = self._analyze_failure(
                reasoning, final_answer, correct_answer, result.steps
            )

        # 6. Generate评价
        self._generate_feedback(result)

        result.evaluation_method = "rule"
        result.evaluation_confidence = 0.7

        return result

    def _parse_reasoning_steps(self, reasoning: str) -> list[ReasoningStep]:
        """ParseReasoning Steps"""
        if not reasoning:
            return []

        steps = []

        # 按段落or换行分割
        paragraphs = re.split(r'\n\n+', reasoning.strip())

        step_num = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 进一步按句子分割长段落
            sentences = re.split(r'(?<=[.。!?！？])\s*', para) if len(para) > 200 else [para]

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 10:
                    continue

                step_num += 1
                step_type = self._classify_step_type(sentence)

                steps.append(ReasoningStep(
                    step_number=step_num,
                    content=sentence[:500],  # 限制长度
                    step_type=step_type
                ))

        return steps

    def _classify_step_type(self, text: str) -> str:
        """分类Reasoning Steps类型"""
        text_lower = text.lower()

        # Checkis否is结论
        for pattern in self.conclusion_patterns:
            if re.search(pattern, text_lower):
                return "conclusion"

        # Checkis否isCalculate
        for pattern in self.calculation_patterns:
            if re.search(pattern, text):
                return "calculation"

        # Checkis否is观察/引用
        if any(word in text_lower for word in ['given', 'according to', 'based on', 'Questionin']):
            return "observation"

        return "reasoning"

    def _evaluate_coherence(self, reasoning: str, steps: list[ReasoningStep]) -> float:
        """评估逻辑连贯性"""
        score = 5.0  # 基础分

        if not steps:
            return 0.0

        # 1. Checkis否has明确步骤指示
        step_indicator_count = 0
        for pattern in self.step_indicators:
            step_indicator_count += len(re.findall(pattern, reasoning, re.IGNORECASE))

        if step_indicator_count >= len(steps) * 0.5:
            score += 2.0  # hasGood步骤指示

        # 2. Checkis否has逻辑Connect词
        connectors = ['therefore', 'thus', 'hence', 'so', 'because', 'since',
                      '因此', 'therefore', '由此', 'because', 'due to']
        connector_count = sum(1 for c in connectors if c in reasoning.lower())
        if connector_count >= 2:
            score += 1.5

        # 3. Checkis否has明确结论
        has_conclusion = any(s.step_type == "conclusion" for s in steps)
        if has_conclusion:
            score += 1.5

        return min(10.0, score)

    def _evaluate_completeness(self, reasoning: str, question: str, final_answer: str) -> float:
        """评估步骤完整性"""
        score = 5.0

        # 1. 推理长度与问题复杂度匹配
        question_words = len(question.split())
        reasoning_words = len(reasoning.split())

        # 简单问题：推理notneed太长
        if question_words < 20:
            if reasoning_words >= 30:
                score += 2.0
        else:
            # 复杂问题：推理应该更详细
            if reasoning_words >= 100:
                score += 2.0
            elif reasoning_words >= 50:
                score += 1.0

        # 2. Checkis否hasin间Result
        intermediate_results = re.findall(r'=\s*[\d.]+', reasoning)
        if len(intermediate_results) >= 2:
            score += 2.0
        elif len(intermediate_results) >= 1:
            score += 1.0

        # 3. Check最终Answeris否in推理in被推Export
        if final_answer in reasoning:
            score += 1.0

        return min(10.0, score)

    def _evaluate_relevance(self, reasoning: str, question: str) -> float:
        """评估与问题相关性"""
        score = 5.0

        # 提取问题in关键词
        question_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        reasoning_words = set(re.findall(r'\b\w{4,}\b', reasoning.lower()))

        # Calculate关键词重叠度
        if question_words:
            overlap = len(question_words & reasoning_words) / len(question_words)
            score += overlap * 3.0

        # Checkis否引用问题in数字
        question_numbers = set(re.findall(r'\d+(?:\.\d+)?', question))
        reasoning_numbers = set(re.findall(r'\d+(?:\.\d+)?', reasoning))

        if question_numbers:
            number_overlap = len(question_numbers & reasoning_numbers) / len(question_numbers)
            score += number_overlap * 2.0

        return min(10.0, score)

    def _evaluate_correctness(self, steps: list[ReasoningStep], final_correct: bool) -> float:
        """评估Reasoning Steps正确性"""
        # 这里use启发式方法，真正正确性Validateneed LLM

        if final_correct:
            # Answer正确，假设大部分推理也is正确
            return 8.0
        else:
            # AnswerError，降低推理正确性Score
            # butnotis 0，because可能只is最后一步出错
            return 4.0

    def _evaluate_efficiency(self, reasoning: str, steps: list[ReasoningStep]) -> float:
        """评估推理效率"""
        score = 7.0  # 基础分

        if not steps:
            return 5.0

        # 1. Checkis否has重复内容
        step_contents = [s.content.lower() for s in steps]
        unique_ratio = len(set(step_contents)) / len(step_contents)

        if unique_ratio < 0.7:
            score -= 2.0  # has较多重复
        elif unique_ratio >= 0.9:
            score += 1.0  # 几乎没has重复

        # 2. Check推理长度is否过长
        reasoning_len = len(reasoning)
        if reasoning_len > 2000:
            score -= 1.5  # 可能过于冗长
        elif reasoning_len < 500:
            score += 0.5  # 简洁

        # 3. Checkis否hasno关发散
        # （viaCheckis否has与数学/逻辑no关内容）
        distracting_patterns = [
            r'(?:however|but this depends|alternatively)',
            r'(?:not过|butis这取决于|or者)',
        ]
        distraction_count = 0
        for pattern in distracting_patterns:
            distraction_count += len(re.findall(pattern, reasoning, re.IGNORECASE))

        if distraction_count > 2:
            score -= 1.0

        return max(0.0, min(10.0, score))

    def _check_answer_equivalence(self, predicted: str, correct: str) -> bool:
        """简单Answeretc.价性Check"""
        if not predicted or not correct:
            return False

        # 规范化
        pred_clean = predicted.strip().lower().replace(',', '')
        correct_clean = correct.strip().lower().replace(',', '')

        # 直接比较
        if pred_clean == correct_clean:
            return True

        # 尝试数值比较
        try:
            pred_num = float(pred_clean)
            correct_num = float(correct_clean)
            return abs(pred_num - correct_num) < 0.001
        except:
            pass

        return False

    def _analyze_failure(
        self,
        reasoning: str,
        final_answer: str,
        correct_answer: str,
        steps: list[ReasoningStep]
    ) -> tuple[str, str]:
        """分析失败原因"""
        # Calculate步骤数
        calc_steps = [s for s in steps if s.step_type == "calculation"]

        # 1. Checkis否isCalculateError
        if calc_steps:
            return "calculation_error", "Reasoning processin可能存inCalculateError"

        # 2. Checkis否缺少Reasoning Steps
        if len(steps) <= 2:
            return "reasoning_gap", "Reasoning Stepsnot完整，可能跳过关键步骤"

        # 3. Checkis否Answer格式问题
        try:
            final_num = float(final_answer.replace(',', ''))
            correct_num = float(correct_answer.replace(',', ''))
            ratio = final_num / correct_num if correct_num != 0 else 0

            if 0.9 <= ratio <= 1.1:
                return "rounding_error", "Answer接近正确，可能is精度or舍入问题"
            elif ratio in [10, 100, 0.1, 0.01]:
                return "magnitude_error", "Answer数量级Error（可能is单位换算问题）"
        except:
            pass

        # 4. default：概念理解Error
        return "concept_error", "可能存in概念理解偏差or推理逻辑Error"

    def _generate_feedback(self, result: ReasoningEvaluationResult):
        """Generate评价反馈"""
        qs = result.quality_score

        # 优点
        if qs.coherence >= 7:
            qs.strengths.append("推理逻辑连贯清晰")
        if qs.completeness >= 7:
            qs.strengths.append("Reasoning Steps完整")
        if qs.relevance >= 7:
            qs.strengths.append("推理与问题紧密相关")
        if qs.efficiency >= 7:
            qs.strengths.append("推理简洁高效")

        # 缺点
        if qs.coherence < 5:
            qs.weaknesses.append("推理逻辑not够连贯")
        if qs.completeness < 5:
            qs.weaknesses.append("Reasoning Steps可能not完整")
        if qs.relevance < 5:
            qs.weaknesses.append("推理内容与问题相关性not足")
        if qs.efficiency < 5:
            qs.weaknesses.append("Reasoning process可能存in冗余")

        # Suggestion
        if qs.coherence < 7:
            qs.suggestions.append("Suggestionuse更多逻辑Connect词来增强推理连贯性")
        if qs.completeness < 7:
            qs.suggestions.append("Suggestion展示更多in间步骤andCalculate过程")
        if not result.final_answer_correct:
            qs.suggestions.append(f"AnswerError，失败原因可能is：{result.failure_analysis}")

    async def evaluate_with_llm(
        self,
        question: str,
        reasoning: str,
        final_answer: str,
        correct_answer: str,
        llm_func,
        is_answer_correct: bool | None = None
    ) -> ReasoningEvaluationResult:
        """
        use LLM 进行深度推理Quality Assessment

        Args:
            question: 问题
            reasoning: Reasoning process
            final_answer: 最终Answer
            correct_answer: Correct answer
            llm_func: LLM 调用函数 async (prompt) -> str
            is_answer_correct: Answeris否正确

        Returns:
            ReasoningEvaluationResult
        """
        # 先进行规则评估
        result = self.evaluate(question, reasoning, final_answer, correct_answer, is_answer_correct)

        if not reasoning or len(reasoning.strip()) < 50:
            return result

        # use LLM 深度评估
        try:
            llm_eval = await self._llm_evaluate_reasoning(
                question, reasoning, final_answer, correct_answer, llm_func
            )

            # 融合规则评估and LLM 评估
            result.quality_score.coherence = (result.quality_score.coherence + llm_eval.get('coherence', 5)) / 2
            result.quality_score.completeness = (result.quality_score.completeness + llm_eval.get('completeness', 5)) / 2
            result.quality_score.relevance = (result.quality_score.relevance + llm_eval.get('relevance', 5)) / 2
            result.quality_score.correctness = (result.quality_score.correctness + llm_eval.get('correctness', 5)) / 2
            result.quality_score.efficiency = (result.quality_score.efficiency + llm_eval.get('efficiency', 5)) / 2

            result.quality_score.calculate_overall()

            # Add LLM 反馈
            if llm_eval.get('feedback'):
                result.quality_score.suggestions.append(f"[AI评价] {llm_eval['feedback']}")

            result.evaluation_method = "hybrid"
            result.evaluation_confidence = 0.85

        except Exception as e:
            result.quality_score.suggestions.append(f"LLM 评估失败: {str(e)}")

        return result

    async def _llm_evaluate_reasoning(
        self,
        question: str,
        reasoning: str,
        final_answer: str,
        correct_answer: str,
        llm_func
    ) -> dict[str, Any]:
        """use LLM 评估推理质量"""
        prompt = f"""You are an expert evaluator assessing the quality of mathematical/logical reasoning.

Question: {question[:500]}

Reasoning Process:
\"\"\"
{reasoning[:1500]}
\"\"\"

Model's Answer: {final_answer}
Correct Answer: {correct_answer}

Please evaluate the reasoning quality on these dimensions (0-10 scale):
1. Coherence: Is the logic flow clear and connected?
2. Completeness: Are all necessary steps shown?
3. Relevance: Is the reasoning focused on the question?
4. Correctness: Are the reasoning steps mathematically/logically correct?
5. Efficiency: Is the reasoning concise without redundancy?

Respond in JSON format:
{{
    "coherence": <0-10>,
    "completeness": <0-10>,
    "relevance": <0-10>,
    "correctness": <0-10>,
    "efficiency": <0-10>,
    "feedback": "<one sentence summary of main issue or strength>"
}}"""

        response = await llm_func(prompt)

        # 尝试Parse JSON
        import json
        try:
            # 提取 JSON 部分
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass

        return {}


# 便捷函数
def quick_evaluate_reasoning(
    question: str,
    reasoning: str,
    final_answer: str,
    correct_answer: str
) -> ReasoningEvaluationResult:
    """快速评估推理质量（仅规则，no LLM）"""
    evaluator = ReasoningQualityEvaluator()
    return evaluator.evaluate(question, reasoning, final_answer, correct_answer)
