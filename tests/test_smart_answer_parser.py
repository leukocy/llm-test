"""
SmartAnswerParser 模块单元Test

Test智能AnswerParse器核心功能：
- 数值AnswerParse（多种格式）
- 选择题AnswerParse
- 布尔AnswerParse
- 数学表达式Parse
- Answer比较功能
"""

import pytest

from core.smart_answer_parser import (
    SmartAnswerParser,
    AnswerType,
    ParseResult,
    compare_answers
)


class TestSmartAnswerParser:
    """Test SmartAnswerParser 类"""

    @pytest.fixture
    def parser(self):
        """CreateParse器实例"""
        return SmartAnswerParser()

    # ==================== 数值ParseTest ====================

    def test_parse_number_boxed_format(self, parser):
        """Test \boxed{...} 格式（MATH 标准）"""
        result = parser.parse("The answer is \\boxed{42}", AnswerType.NUMBER)
        assert result.extracted_answer == "42"
        assert result.confidence == 0.95
        assert result.method == "rule_boxed"
        assert result.normalized_value == 42

    def test_parse_number_boxed_with_decimal(self, parser):
        """Test带小数 boxed 格式"""
        result = parser.parse("Result: \\boxed{3.14}", AnswerType.NUMBER)
        assert result.extracted_answer == "3.14"
        assert result.normalized_value == 3.14

    def test_parse_number_hash_format(self, parser):
        """Test #### 数值格式（GSM8K 标准）"""
        result = parser.parse("Calculating...\n#### 1,234", AnswerType.NUMBER)
        assert result.extracted_answer == "1,234"
        assert result.normalized_value == 1234

    def test_parse_number_with_commas(self, parser):
        """Test带逗号数字"""
        result = parser.parse("The total is 1,234,567", AnswerType.NUMBER)
        assert result.extracted_answer == "1,234,567"
        assert result.normalized_value == 1234567

    def test_parse_number_negative(self, parser):
        """Test负数"""
        result = parser.parse("Answer: -42", AnswerType.NUMBER)
        assert result.extracted_answer == "-42"
        assert result.normalized_value == -42

    def test_parse_number_fraction(self, parser):
        """Test分数 - etc.式模式匹配（Not supported分数）"""
        result = parser.parse("Result = 3/4", AnswerType.NUMBER)
        # etc.式模式匹配 "= 3/4"，提取一数字 3
        assert result.extracted_answer == "3"
        assert result.method == "rule_equation"

    def test_parse_number_explicit_answer(self, parser):
        """Test明确Answer声明"""
        result = parser.parse("The answer is 42", AnswerType.NUMBER)
        assert result.extracted_answer == "42"
        assert result.method == "rule_pattern"

    def test_parse_number_chinese_format(self, parser):
        """Testin文Answer格式"""
        result = parser.parse("Answeris 100", AnswerType.NUMBER)
        assert result.extracted_answer == "100"

    def test_parse_number_equation_format(self, parser):
        """Testetc.式格式"""
        result = parser.parse("2 + 2 = 4", AnswerType.NUMBER)
        assert result.extracted_answer == "4"
        # etc.号模式willin方程模式之前匹配
        assert result.method == "rule_pattern"

    def test_parse_number_last_fallback(self, parser):
        """Test最后一数字兜底策略"""
        result = parser.parse("After calculation we get approximately 42 units", AnswerType.NUMBER)
        assert result.extracted_answer == "42"
        assert result.method == "rule_last_number"
        assert result.confidence == 0.4

    def test_parse_number_empty_response(self, parser):
        """Test空响应"""
        result = parser.parse("", AnswerType.NUMBER)
        assert result.extracted_answer == ""
        assert result.confidence == 0.0
        assert result.error is not None

    def test_parse_number_malformed_response(self, parser):
        """TestFormat error响应"""
        result = parser.parse("No numbers here!", AnswerType.NUMBER)
        assert result.extracted_answer == ""
        assert result.confidence == 0.0

    # ==================== 选择题ParseTest ====================

    def test_parse_choice_explicit(self, parser):
        """Test明确选择题Answer"""
        result = parser.parse("Answer: (A)", AnswerType.CHOICE)
        assert result.extracted_answer == "A"
        assert result.confidence >= 0.9

    def test_parse_choice_implicit(self, parser):
        """Test隐式选择题Answer"""
        result = parser.parse("B", AnswerType.CHOICE)
        assert result.extracted_answer == "B"

    def test_parse_choice_lowercase(self, parser):
        """Test小写Options"""
        result = parser.parse("answer: c", AnswerType.CHOICE)
        assert result.extracted_answer == "C"

    def test_parse_choice_chinese(self, parser):
        """Testin文选择题格式"""
        result = parser.parse("选 A", AnswerType.CHOICE)
        assert result.extracted_answer == "A"

    def test_parse_choice_with_punctuation(self, parser):
        """Test带标点Options"""
        result = parser.parse("The answer is B.", AnswerType.CHOICE)
        assert result.extracted_answer == "B"

    def test_parse_choice_empty_response(self, parser):
        """Test空响应"""
        result = parser.parse("", AnswerType.CHOICE)
        assert result.extracted_answer == ""
        assert result.error is not None

    # ==================== 布尔AnswerParseTest ====================

    def test_parse_boolean_yes_no(self, parser):
        """Test yes/no 格式"""
        result = parser.parse("yes", AnswerType.BOOLEAN)
        assert result.extracted_answer == "Yes"
        assert result.normalized_value is True

    def test_parse_boolean_chinese(self, parser):
        """Testin文is/否"""
        result = parser.parse("is", AnswerType.BOOLEAN)
        assert result.extracted_answer == "Yes"
        assert result.normalized_value is True

    def test_parse_boolean_true_false(self, parser):
        """Test true/false"""
        result = parser.parse("false", AnswerType.BOOLEAN)
        assert result.extracted_answer == "No"
        assert result.normalized_value is False

    # ==================== Answer比较Test ====================

    def test_compare_answers_numeric(self):
        """Test数值Answer比较"""
        is_correct, score = compare_answers("42", "42", AnswerType.NUMBER)
        assert is_correct == True
        assert score == 1.0

    def test_compare_answers_numeric_wrong(self):
        """Test数值Answer比较（Error）"""
        is_correct, score = compare_answers("42", "43", AnswerType.NUMBER)
        assert is_correct == False
        assert score == 0.0

    def test_compare_answers_numeric_with_tolerance(self):
        """Test数值容差比较"""
        is_correct, score = compare_answers("3.14159", "3.141", AnswerType.NUMBER)
        # indefault容差 0.001 范围内
        assert is_correct == True

    def test_compare_answers_choice(self):
        """Test选择题Answer比较"""
        is_correct, score = compare_answers("A", "a", AnswerType.CHOICE)
        assert is_correct == True
        assert score == 1.0

    def test_compare_answers_text(self):
        """Test文本Answer比较"""
        is_correct, score = compare_answers("hello", "hello", AnswerType.TEXT)
        assert is_correct == True

    def test_compare_answers_boolean(self):
        """Test布尔Answer比较"""
        is_correct, score = compare_answers("yes", "true", AnswerType.BOOLEAN)
        assert is_correct == True

    # ==================== 边界条件Test ====================

    def test_parse_with_llm_fallback_threshold(self):
        """Test LLM 兜底阈值Set"""
        parser = SmartAnswerParser(llm_fallback_threshold=0.8)
        assert parser.llm_fallback_threshold == 0.8

    def test_parse_response_with_whitespace(self, parser):
        """Test带空白字符响应"""
        result = parser.parse("  The answer is 42  ", AnswerType.NUMBER)
        assert result.extracted_answer == "42"

    def test_parse_multiline_response(self, parser):
        """Test多行响应"""
        response = """
        Let me solve this step by step.
        First, we calculate...
        Then, the answer is 100.
        """
        result = parser.parse(response, AnswerType.NUMBER)
        assert result.extracted_answer == "100"


class TestParseResult:
    """Test ParseResult Data类"""

    def test_parse_result_creation(self):
        """Test ParseResult Create"""
        result = ParseResult(
            extracted_answer="42",
            confidence=0.95,
            method="test",
            normalized_value=42,
            raw_match="42"
        )
        assert result.extracted_answer == "42"
        assert result.confidence == 0.95
        assert result.error is None

    def test_parse_result_with_error(self):
        """Test带Error ParseResult"""
        result = ParseResult(
            extracted_answer="",
            confidence=0.0,
            method="failed",
            normalized_value=None,
            raw_match="",
            error="No answer found"
        )
        assert result.error == "No answer found"
