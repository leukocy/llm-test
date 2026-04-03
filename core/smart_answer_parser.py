"""
智能AnswerParse器 (Smart Answer Parser)

解决传统正则Parse脆弱性问题，引入 LLM 兜底机制。
支持多种Answer类型：数值、选择题、文本etc.。
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple


class AnswerType(Enum):
    """Answer类型"""
    NUMBER = "number"           # 数值Answer
    CHOICE = "choice"           # 选择题 (A/B/C/D)
    TEXT = "text"               # 文本Answer
    BOOLEAN = "boolean"         # is/否
    CODE = "code"               # 代码
    MATH_EXPRESSION = "math"    # 数学表达式


@dataclass
class ParseResult:
    """Parse result"""
    extracted_answer: str           # 提取Answer
    confidence: float               # 置信度 (0-1)
    method: str                     # Parse方法 (rule/llm/hybrid)
    normalized_value: Any           # 规范化后值（数值类型）
    raw_match: str                  # 原始匹配字符串
    error: str | None = None     # Error信息


class SmartAnswerParser:
    """
    智能AnswerParse器

    use分层策略：
    1. 规则Parse（快速、低成本）
    2. 启发式规则（inetc.复杂度）
    3. LLM Parse兜底（高精度、高成本）

    Usage:
        parser = SmartAnswerParser()

        # Parse数值Answer
        result = parser.parse("The answer is 42", AnswerType.NUMBER)
        print(result.extracted_answer)  # "42"
        print(result.confidence)        # 0.95

        # use LLM 兜底（need提供Callback）
        async def llm_call(prompt):
            # 调用你 LLM API
            return response

        result = await parser.parse_with_llm_fallback(
            response="Answer大约is三点一四",
            answer_type=AnswerType.NUMBER,
            llm_func=llm_call
        )
    """

    def __init__(self, llm_fallback_threshold: float = 0.6):
        """
        InitializeParse器

        Args:
            llm_fallback_threshold: LLM 兜底阈值，低于此置信度时use LLM
        """
        self.llm_fallback_threshold = llm_fallback_threshold

        # 选择题Options
        self.choice_options = ['A', 'B', 'C', 'D', 'E', 'F']

    def parse(self, response: str, answer_type: AnswerType,
              expected_answer: str | None = None) -> ParseResult:
        """
        ParseModel响应，提取Answer（仅规则Parse）

        Args:
            response: Model响应
            answer_type: 期望Answer类型
            expected_answer: 期望Correct answer（用于提升置信度判断）

        Returns:
            ParseResult
        """
        if answer_type == AnswerType.NUMBER:
            return self._parse_number(response)
        elif answer_type == AnswerType.CHOICE:
            return self._parse_choice(response)
        elif answer_type == AnswerType.BOOLEAN:
            return self._parse_boolean(response)
        elif answer_type == AnswerType.MATH_EXPRESSION:
            return self._parse_math_expression(response)
        else:
            return self._parse_text(response)

    def _parse_number(self, response: str) -> ParseResult:
        """Parse数值Answer"""
        response = response.strip()

        # 策略1: 匹配 \boxed{...}（MATH 标准格式）
        boxed_matches = re.findall(r'\\boxed\{([^{}]+)\}', response)
        if boxed_matches:
            value_str = boxed_matches[-1].strip()
            normalized = self._normalize_number(value_str)
            if normalized is not None:
                return ParseResult(
                    extracted_answer=value_str,
                    confidence=0.95,
                    method="rule_boxed",
                    normalized_value=normalized,
                    raw_match=f"\\boxed{{{value_str}}}"
                )

        # 策略2: 匹配 #### 后数值（GSM8K 标准格式）
        hash_match = re.search(r'####\s*([-+]?[\d,]+(?:\.\d+)?)', response)
        if hash_match:
            value_str = hash_match.group(1)
            normalized = self._normalize_number(value_str)
            if normalized is not None:
                return ParseResult(
                    extracted_answer=value_str,
                    confidence=0.95,
                    method="rule_hash",
                    normalized_value=normalized,
                    raw_match=hash_match.group(0)
                )

        # 策略3: 匹配 "answer is X" 格式
        answer_patterns = [
            r'(?:answer|result|Answer|Result)[:\s]*(?:is\s*)?[$]?\s*([-+]?[\d,]+(?:\.\d+)?)',
            r'(?:=|etc.于|得)\s*([-+]?[\d,]+(?:\.\d+)?)\s*(?:$|[。.\s])',
            r'(?:最终|final)\s*[:：]?\s*([-+]?[\d,]+(?:\.\d+)?)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                normalized = self._normalize_number(value_str)
                if normalized is not None:
                    return ParseResult(
                        extracted_answer=value_str,
                        confidence=0.85,
                        method="rule_pattern",
                        normalized_value=normalized,
                        raw_match=match.group(0)
                    )

        # 策略4: 匹配末尾etc.式Result
        eq_match = re.search(r'=\s*([-+]?[\d,]+(?:\.\d+)?)[^=\n]*$', response.strip())
        if eq_match:
            value_str = eq_match.group(1)
            normalized = self._normalize_number(value_str)
            if normalized is not None:
                return ParseResult(
                    extracted_answer=value_str,
                    confidence=0.7,
                    method="rule_equation",
                    normalized_value=normalized,
                    raw_match=eq_match.group(0)
                )

        # 策略5: 取最后一出现数字（低置信度）
        all_numbers = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', response)
        if all_numbers:
            value_str = all_numbers[-1]
            normalized = self._normalize_number(value_str)
            if normalized is not None:
                return ParseResult(
                    extracted_answer=value_str,
                    confidence=0.4,
                    method="rule_last_number",
                    normalized_value=normalized,
                    raw_match=value_str
                )

        return ParseResult(
            extracted_answer="",
            confidence=0.0,
            method="rule_failed",
            normalized_value=None,
            raw_match="",
            error="No number found in response"
        )

    def _parse_choice(self, response: str) -> ParseResult:
        """Parse选择题Answer"""
        response_clean = response.strip()
        choice_pattern = '|'.join(self.choice_options)

        # 策略1: 明确Answer声明
        patterns = [
            rf'(?:answer|Answer)[:\s]*[（(]?({choice_pattern})[)）]?',
            rf'(?:选择|选|choose)[:\s]*[（(]?({choice_pattern})[)）]?',
            rf'^[（(]?({choice_pattern})[)）.]?\s*$',  # 单独Options
            rf'\b({choice_pattern})\b(?:\s*[.。:：])',  # Options后跟标点
        ]

        for pattern in patterns:
            match = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if match:
                choice = match.group(1).upper()
                return ParseResult(
                    extracted_answer=choice,
                    confidence=0.9,
                    method="rule_explicit",
                    normalized_value=choice,
                    raw_match=match.group(0)
                )

        # 策略2: 响应开头Options
        first_char_match = re.match(rf'^[（(]?({choice_pattern})[)）.]?', response_clean, re.IGNORECASE)
        if first_char_match:
            choice = first_char_match.group(1).upper()
            return ParseResult(
                extracted_answer=choice,
                confidence=0.8,
                method="rule_first_char",
                normalized_value=choice,
                raw_match=first_char_match.group(0)
            )

        # 策略3: 任意位置一Options（低置信度）
        for char in response_clean[:100]:
            if char.upper() in self.choice_options:
                return ParseResult(
                    extracted_answer=char.upper(),
                    confidence=0.5,
                    method="rule_any_choice",
                    normalized_value=char.upper(),
                    raw_match=char
                )

        return ParseResult(
            extracted_answer="",
            confidence=0.0,
            method="rule_failed",
            normalized_value=None,
            raw_match="",
            error="No choice found in response"
        )

    def _parse_boolean(self, response: str) -> ParseResult:
        """Parseis/否Answer"""
        response_lower = response.lower().strip()

        yes_indicators = ['yes', 'true', 'is', '对', '正确', '确实']
        no_indicators = ['no', 'false', '否', 'not', 'Error', 'not对']

        for indicator in yes_indicators:
            if indicator in response_lower:
                return ParseResult(
                    extracted_answer="Yes",
                    confidence=0.85,
                    method="rule_keyword",
                    normalized_value=True,
                    raw_match=indicator
                )

        for indicator in no_indicators:
            if indicator in response_lower:
                return ParseResult(
                    extracted_answer="No",
                    confidence=0.85,
                    method="rule_keyword",
                    normalized_value=False,
                    raw_match=indicator
                )

        return ParseResult(
            extracted_answer="",
            confidence=0.0,
            method="rule_failed",
            normalized_value=None,
            raw_match="",
            error="No boolean indicator found"
        )

    def _parse_math_expression(self, response: str) -> ParseResult:
        """Parse数学表达式"""
        # 匹配 \boxed{} 内表达式
        boxed_matches = re.findall(r'\\boxed\{([^{}]+)\}', response)
        if boxed_matches:
            expr = boxed_matches[-1].strip()
            # 尝试求值
            try:
                normalized = self._evaluate_expression(expr)
            except:
                normalized = expr

            return ParseResult(
                extracted_answer=expr,
                confidence=0.9,
                method="rule_boxed",
                normalized_value=normalized,
                raw_match=f"\\boxed{{{expr}}}"
            )

        # 回退到数值Parse
        return self._parse_number(response)

    def _parse_text(self, response: str) -> ParseResult:
        """Parse文本Answer（通用）"""
        # 尝试提取最后一句作isAnswer
        sentences = re.split(r'[.。!！?？\n]', response.strip())
        last_sentence = sentences[-1].strip() if sentences else response.strip()

        return ParseResult(
            extracted_answer=last_sentence[:200],  # 限制长度
            confidence=0.5,
            method="rule_last_sentence",
            normalized_value=last_sentence[:200],
            raw_match=last_sentence[:200]
        )

    def _normalize_number(self, value_str: str) -> float | None:
        """规范化数值字符串"""
        try:
            # 移除逗号、货币符号、空格
            clean = value_str.replace(',', '').replace('$', '').replace(' ', '').strip()
            return float(clean)
        except:
            return None

    def _evaluate_expression(self, expr: str) -> float | None:
        """安全地求值简单数学表达式"""
        try:
            # 只允许数字and基本运算符
            safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expr)
            return float(eval(safe_expr))
        except:
            return None

    async def parse_with_llm_fallback(
        self,
        response: str,
        answer_type: AnswerType,
        llm_func,
        expected_answer: str | None = None
    ) -> ParseResult:
        """
        带 LLM 兜底Parse

        Args:
            response: Model响应
            answer_type: Answer类型
            llm_func: LLM 调用函数 async (prompt) -> str
            expected_answer: 期望Answer（optional）

        Returns:
            ParseResult
        """
        # 先尝试规则Parse
        rule_result = self.parse(response, answer_type, expected_answer)

        # if置信度足够高，直接Return
        if rule_result.confidence >= self.llm_fallback_threshold:
            return rule_result

        # use LLM 兜底
        try:
            llm_result = await self._llm_parse(response, answer_type, llm_func)

            # if LLM Parsesucceeded且置信度更高
            if llm_result.confidence > rule_result.confidence:
                return llm_result

            return rule_result

        except Exception as e:
            # LLM 失败，Return规则Parse result
            rule_result.error = f"LLM fallback failed: {e}"
            return rule_result

    async def _llm_parse(
        self,
        response: str,
        answer_type: AnswerType,
        llm_func
    ) -> ParseResult:
        """use LLM ParseAnswer"""
        type_instruction = {
            AnswerType.NUMBER: "Extract the final numerical answer. Return ONLY the number, nothing else. If the answer is a fraction, convert it to decimal.",
            AnswerType.CHOICE: "Extract the selected option (A, B, C, D, or E). Return ONLY the letter, nothing else.",
            AnswerType.BOOLEAN: "Determine if the answer is Yes or No. Return ONLY 'Yes' or 'No', nothing else.",
            AnswerType.TEXT: "Extract the final answer. Return ONLY the answer text, nothing else.",
            AnswerType.MATH_EXPRESSION: "Extract the final mathematical answer. If it's an expression, evaluate it to a number. Return ONLY the number.",
        }

        prompt = f"""You are an answer extraction assistant. Your task is to extract the final answer from the following response.

{type_instruction.get(answer_type, "Extract the final answer.")}

Response to analyze:
\"\"\"
{response[:2000]}
\"\"\"

Extracted answer:"""

        llm_response = await llm_func(prompt)
        extracted = llm_response.strip()

        # Validate提取Result
        if answer_type == AnswerType.NUMBER:
            normalized = self._normalize_number(extracted)
            if normalized is not None:
                return ParseResult(
                    extracted_answer=extracted,
                    confidence=0.85,
                    method="llm",
                    normalized_value=normalized,
                    raw_match=extracted
                )
        elif answer_type == AnswerType.CHOICE:
            if extracted.upper() in self.choice_options:
                return ParseResult(
                    extracted_answer=extracted.upper(),
                    confidence=0.85,
                    method="llm",
                    normalized_value=extracted.upper(),
                    raw_match=extracted
                )
        else:
            return ParseResult(
                extracted_answer=extracted,
                confidence=0.8,
                method="llm",
                normalized_value=extracted,
                raw_match=extracted
            )

        return ParseResult(
            extracted_answer=extracted,
            confidence=0.5,
            method="llm_uncertain",
            normalized_value=extracted,
            raw_match=extracted
        )


def compare_answers(
    predicted: Any,
    expected: Any,
    answer_type: AnswerType,
    tolerance: float = 0.001
) -> tuple[bool, float]:
    """
    比较两Answeris否etc.价

    Args:
        predicted: 预测Answer（已规范化）
        expected: 期望Answer
        answer_type: Answer类型
        tolerance: 数值比较容差

    Returns:
        (is否正确, 相似度Score)
    """
    if predicted is None or expected is None:
        return False, 0.0

    if answer_type == AnswerType.NUMBER:
        try:
            pred_val = float(predicted) if not isinstance(predicted, (int, float)) else predicted
            exp_val = float(str(expected).replace(',', ''))

            # 完全相etc.
            if pred_val == exp_val:
                return True, 1.0

            # in容差范围内
            if abs(pred_val - exp_val) <= tolerance:
                return True, 0.95

            # 百分比容差（Process浮点精度问题）
            if exp_val != 0 and abs((pred_val - exp_val) / exp_val) < 0.001:
                return True, 0.9

            return False, 0.0

        except (ValueError, TypeError):
            return False, 0.0

    elif answer_type == AnswerType.CHOICE:
        pred_str = str(predicted).upper().strip()
        exp_str = str(expected).upper().strip()
        return pred_str == exp_str, 1.0 if pred_str == exp_str else 0.0

    elif answer_type == AnswerType.BOOLEAN:
        pred_bool = predicted if isinstance(predicted, bool) else str(predicted).lower() in ['true', 'yes', 'is']
        exp_bool = expected if isinstance(expected, bool) else str(expected).lower() in ['true', 'yes', 'is']
        return pred_bool == exp_bool, 1.0 if pred_bool == exp_bool else 0.0

    else:
        # 文本比较：规范化后比较
        pred_norm = str(predicted).lower().strip()
        exp_norm = str(expected).lower().strip()

        if pred_norm == exp_norm:
            return True, 1.0

        # Calculate相似度（简单包含Check）
        if exp_norm in pred_norm or pred_norm in exp_norm:
            return True, 0.8

        return False, 0.0
