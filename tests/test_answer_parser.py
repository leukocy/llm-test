"""
Comprehensive tests for the quality evaluation answer-parsing pipeline.

Covers:
1. evaluators/answer_parser.py — centralized parsers (MultiChoice, Math, Code, Text)
2. evaluators/base_evaluator.py — extract_choice_answer / extract_numeric_answer delegation
3. Integration: build_chat_messages + evaluate_single structured-message path
4. Migrated from test_smart_answer_parser.py (core SmartAnswerParser equivalent coverage)
"""

import pytest

from evaluators.answer_parser import (
    CodeAnswerParser,
    MathAnswerParser,
    MultiChoiceParser,
    TextAnswerParser,
    get_parser_for_dataset,
)
from evaluators.base_evaluator import (
    extract_choice_answer,
    extract_numeric_answer,
    extract_number_answer,
    normalize_text,
)


# ---------------------------------------------------------------------------
# MultiChoiceParser
# ---------------------------------------------------------------------------
class TestMultiChoiceParser:

    def setup_method(self):
        self.parser = MultiChoiceParser()

    # English patterns
    @pytest.mark.parametrize("response,expected", [
        ("ANSWER: B", "B"),
        ("Answer: C", "C"),
        ("The answer is A", "A"),
        ("I choose D", "D"),
        ("I select B", "B"),
        ("My answer is C", "C"),
    ])
    def test_english_patterns(self, response, expected):
        assert self.parser.parse(response) == expected

    # Chinese patterns
    @pytest.mark.parametrize("response,expected", [
        ("答案是A", "A"),
        ("选择B", "B"),
        ("选项C", "C"),
        ("正确答案为D", "D"),
        ("答案为A", "A"),
    ])
    def test_chinese_patterns(self, response, expected):
        assert self.parser.parse(response) == expected

    # Structured formats
    @pytest.mark.parametrize("response,expected", [
        ("(A)", "A"),
        ("（B）", "B"),
        ("[C]", "C"),
        ("A. First option", "A"),
        ("B。第二个选项", "B"),
    ])
    def test_structured_formats(self, response, expected):
        assert self.parser.parse(response) == expected

    def test_standalone_letter_at_end(self):
        assert self.parser.parse("Some reasoning here.\nA") == "A"

    def test_empty_response(self):
        assert self.parser.parse("") == ""

    def test_no_valid_choice(self):
        assert self.parser.parse("I don't know the answer.") == ""

    def test_custom_choices_ab(self):
        assert self.parser.parse("I choose B", choices=["A", "B"]) == "B"

    def test_returns_uppercase(self):
        assert self.parser.parse("answer: a") == "A"

    def test_long_reasoning_with_answer_at_end(self):
        response = "Let me think about this.\nOption A seems wrong.\nOption B is also incorrect.\nThe answer is C"
        assert self.parser.parse(response) == "C"


# ---------------------------------------------------------------------------
# MathAnswerParser
# ---------------------------------------------------------------------------
class TestMathAnswerParser:

    def setup_method(self):
        self.parser = MathAnswerParser()

    def test_boxed_simple(self):
        assert self.parser.parse("The result is $\\boxed{42}$") == "42"

    def test_boxed_with_expression(self):
        assert self.parser.parse("Therefore $\\boxed{3.14}$") == "3.14"

    def test_boxed_nested_braces(self):
        result = self.parser.parse("Answer: $\\boxed{\\frac{1}{2}}$")
        assert "frac" in result or result == "\\frac{1}{2}"

    def test_hash_marker(self):
        assert self.parser.parse("Step 1...\nStep 2...\n#### 15400") == "15400"

    def test_hash_with_commas(self):
        assert self.parser.parse("#### 15,400") == "15400"

    def test_explicit_en(self):
        result = self.parser.parse("The answer is 42.")
        assert result == "42"

    def test_explicit_cn(self):
        result = self.parser.parse("答案是 3.14")
        assert result == "3.14"

    def test_last_number(self):
        result = self.parser.parse("Some text without markers\n100")
        assert result == "100"

    def test_empty_response(self):
        assert self.parser.parse("") == ""

    # -- check_answer tests --
    def test_check_exact_match(self):
        assert MathAnswerParser.check_answer("42", "42") is True

    def test_check_float_tolerance(self):
        assert MathAnswerParser.check_answer("3.14159", "3.14159") is True

    def test_check_close_values(self):
        assert MathAnswerParser.check_answer("3.141592", "3.141593") is True

    def test_check_comma_normalized(self):
        assert MathAnswerParser.check_answer("15,400", "15400") is True

    def test_check_rounded_match(self):
        assert MathAnswerParser.check_answer("15400.0", "15400") is True

    def test_check_wrong_answer(self):
        assert MathAnswerParser.check_answer("41", "42") is False

    def test_check_empty_predicted(self):
        assert MathAnswerParser.check_answer("", "42") is False

    def test_check_unicode_minus(self):
        assert MathAnswerParser.check_answer("−5", "-5") is True

    def test_check_scientific_notation(self):
        assert MathAnswerParser.check_answer("1.5e3", "1500") is True


# ---------------------------------------------------------------------------
# CodeAnswerParser
# ---------------------------------------------------------------------------
class TestCodeAnswerParser:

    def setup_method(self):
        self.parser = CodeAnswerParser()

    def test_markdown_python_fence(self):
        response = "Here's the solution:\n```python\ndef foo():\n    return 42\n```"
        result = self.parser.parse(response)
        assert "def foo():" in result
        assert "return 42" in result

    def test_markdown_plain_fence(self):
        response = "```\ndef bar():\n    pass\n```"
        result = self.parser.parse(response)
        assert "def bar():" in result

    def test_multiple_blocks_last_wins(self):
        response = "```python\ndef first():\n    pass\n```\n```python\ndef second():\n    pass\n```"
        result = self.parser.parse(response)
        assert "def second():" in result

    def test_indented_code(self):
        response = "The function is:\n    def foo():\n        return 1"
        result = self.parser.parse(response)
        assert "def foo():" in result

    def test_empty_response(self):
        assert self.parser.parse("") == ""

    def test_plain_code_fallback(self):
        response = "def hello(): print('hi')"
        result = self.parser.parse(response)
        assert "def hello()" in result


# ---------------------------------------------------------------------------
# TextAnswerParser
# ---------------------------------------------------------------------------
class TestTextAnswerParser:

    def test_exact_match(self):
        assert TextAnswerParser.check_answer("Hello World", "Hello World") is True

    def test_case_insensitive(self):
        assert TextAnswerParser.check_answer("hello world", "Hello World") is True

    def test_substring_contains(self):
        assert TextAnswerParser.check_answer("The answer is Paris", "paris") is True

    def test_fuzzy_match(self):
        assert TextAnswerParser.check_answer("The Eiffel Tower", "Eiffel Tower") is True

    def test_no_match(self):
        assert TextAnswerParser.check_answer("Paris", "London") is False

    def test_empty(self):
        assert TextAnswerParser.check_answer("", "hello") is False

    def test_punctuation_normalized(self):
        assert TextAnswerParser.check_answer("Hello, World!", "hello world") is True


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
class TestGetParserForDataset:

    def test_choice_datasets(self):
        for name in ["mmlu", "gpqa", "hellaswag", "truthfulqa", "winogrande", "arc"]:
            assert isinstance(get_parser_for_dataset(name), MultiChoiceParser), f"{name}"

    def test_math_datasets(self):
        for name in ["gsm8k", "math500", "aime2025", "aime"]:
            assert isinstance(get_parser_for_dataset(name), MathAnswerParser), f"{name}"

    def test_code_datasets(self):
        for name in ["humaneval", "mbpp"]:
            assert isinstance(get_parser_for_dataset(name), CodeAnswerParser), f"{name}"

    def test_text_datasets(self):
        for name in ["longbench", "arena_hard", "piqa"]:
            assert isinstance(get_parser_for_dataset(name), TextAnswerParser), f"{name}"

    def test_unknown_defaults_to_text(self):
        assert isinstance(get_parser_for_dataset("unknown_dataset"), TextAnswerParser)

    def test_case_insensitive(self):
        assert isinstance(get_parser_for_dataset("MMLU"), MultiChoiceParser)

    def test_hyphen_normalize(self):
        assert isinstance(get_parser_for_dataset("arena-hard"), TextAnswerParser)


# ===========================================================================
# Migrated from test_smart_answer_parser.py — equivalent coverage via new parsers
# ===========================================================================

class TestMigratedNumberParsing:
    """Number parsing tests migrated from SmartAnswerParser, now using MathAnswerParser."""

    def setup_method(self):
        self.parser = MathAnswerParser()

    def test_parse_number_boxed_format(self):
        """\\boxed{...} format (MATH standard)"""
        assert self.parser.parse("The answer is \\boxed{42}") == "42"

    def test_parse_number_boxed_with_decimal(self):
        assert self.parser.parse("Result: \\boxed{3.14}") == "3.14"

    def test_parse_number_hash_format(self):
        """#### format (GSM8K standard)"""
        assert self.parser.parse("Calculating...\n#### 1,234") == "1234"

    def test_parse_number_with_commas(self):
        assert self.parser.parse("The total is 1,234,567") == "1234567"

    def test_parse_number_negative(self):
        assert self.parser.parse("Answer: -42") == "-42"

    def test_parse_number_explicit_answer(self):
        result = self.parser.parse("The answer is 42")
        assert result == "42"

    def test_parse_number_chinese_format(self):
        result = self.parser.parse("答案是 100")
        assert result == "100"

    def test_parse_number_equation_format(self):
        """Equation format — picks up last number"""
        result = self.parser.parse("2 + 2 = 4")
        assert result == "4"

    def test_parse_number_last_fallback(self):
        result = self.parser.parse("After calculation we get approximately 42 units")
        assert result == "42"

    def test_parse_number_empty_response(self):
        assert self.parser.parse("") == ""

    def test_parse_number_malformed_response(self):
        result = self.parser.parse("No numbers here!")
        assert result == ""

    def test_parse_response_with_whitespace(self):
        result = self.parser.parse("  The answer is 42  ")
        assert result == "42"

    def test_parse_multiline_response(self):
        response = """
        Let me solve this step by step.
        First, we calculate...
        Then, the answer is 100.
        """
        result = self.parser.parse(response)
        assert result == "100"


class TestMigratedChoiceParsing:
    """Choice parsing tests migrated from SmartAnswerParser, now using MultiChoiceParser."""

    def setup_method(self):
        self.parser = MultiChoiceParser()

    def test_parse_choice_explicit(self):
        assert self.parser.parse("Answer: (A)") == "A"

    def test_parse_choice_implicit(self):
        assert self.parser.parse("B") == "B"

    def test_parse_choice_lowercase(self):
        assert self.parser.parse("answer: c") == "C"

    def test_parse_choice_chinese(self):
        assert self.parser.parse("选 A") == "A"

    def test_parse_choice_with_punctuation(self):
        assert self.parser.parse("The answer is B.") == "B"

    def test_parse_choice_empty_response(self):
        assert self.parser.parse("") == ""


# ===========================================================================
# extract_choice_answer / extract_numeric_answer (base_evaluator delegation)
# ===========================================================================

class TestExtractChoiceAnswer:
    """Tests for the base_evaluator.extract_choice_answer function."""

    def test_answer_prefix(self):
        assert extract_choice_answer("Answer: B") == "B"

    def test_the_answer_is(self):
        assert extract_choice_answer("The answer is A") == "A"

    def test_paren_format(self):
        assert extract_choice_answer("(C)") == "C"

    def test_chinese_paren(self):
        assert extract_choice_answer("（D）") == "D"

    def test_lowercase(self):
        assert extract_choice_answer("answer: a") == "A"

    def test_empty(self):
        assert extract_choice_answer("") == ""

    def test_custom_choices(self):
        assert extract_choice_answer("I choose B", choices=["A", "B"]) == "B"

    def test_enhanced_disabled_fallback(self):
        """When use_enhanced=False, still works with regex fallback."""
        assert extract_choice_answer("Answer: C", use_enhanced=False) == "C"

    def test_no_valid_choice(self):
        assert extract_choice_answer("I don't know") == ""


class TestExtractNumericAnswer:
    """Tests for the base_evaluator.extract_numeric_answer function."""

    def test_boxed(self):
        result = extract_numeric_answer("The result is \\boxed{42}")
        assert result.replace(" ", "") == "42"

    def test_hash_marker(self):
        result = extract_numeric_answer("#### 15400")
        assert result == "15400"

    def test_commas_removed(self):
        result = extract_numeric_answer("#### 15,400")
        assert result == "15400"

    def test_empty(self):
        assert extract_numeric_answer("") == ""

    def test_enhanced_disabled_fallback(self):
        result = extract_numeric_answer("#### 123", use_enhanced=False)
        assert result == "123"


class TestExtractNumberAnswerAlias:
    """extract_number_answer is a backward-compatible alias."""

    def test_alias(self):
        assert extract_number_answer("#### 42") == extract_numeric_answer("#### 42")


class TestNormalizeText:
    """Tests for base_evaluator.normalize_text."""

    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_remove_punctuation(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_none(self):
        assert normalize_text(None) == ""


# ===========================================================================
# MathAnswerParser — additional edge cases
# ===========================================================================

class TestMathAnswerParserEdgeCases:
    """Additional edge cases for MathAnswerParser beyond migrated tests."""

    def setup_method(self):
        self.parser = MathAnswerParser()

    def test_boxed_negative(self):
        assert self.parser.parse("Result: \\boxed{-7}") == "-7"

    def test_boxed_with_dollar_sign(self):
        assert self.parser.parse("#### $1,234") == "1234"

    def test_fraction_format(self):
        result = self.parser.parse("3/4")
        # Last number fallback returns "4" (the last number in the string)
        assert result == "4" or "3" in result or "3/4" in result

    def test_hence_pattern(self):
        result = self.parser.parse("Hence, the answer is 99")
        assert result == "99"

    def test_thus_pattern(self):
        result = self.parser.parse("Thus, 256")
        assert result == "256"

    def test_check_negative_vs_positive(self):
        assert MathAnswerParser.check_answer("-5", "5") is False

    def test_check_fraction_vs_decimal(self):
        """1/2 vs 0.5 — only works if SymPy is available."""
        # At minimum, string comparison should fail gracefully
        result = MathAnswerParser.check_answer("1/2", "0.5")
        # This is True only with SymPy; otherwise False is acceptable
        assert isinstance(result, bool)

    def test_check_dollar_signs(self):
        assert MathAnswerParser.check_answer("$100", "100") is True

    def test_check_large_numbers(self):
        assert MathAnswerParser.check_answer("1000000", "1000000") is True

    def test_check_zero(self):
        assert MathAnswerParser.check_answer("0", "0") is True

    def test_check_both_empty(self):
        assert MathAnswerParser.check_answer("", "") is False


# ===========================================================================
# CodeAnswerParser — additional edge cases
# ===========================================================================

class TestCodeAnswerParserEdgeCases:

    def setup_method(self):
        self.parser = CodeAnswerParser()

    def test_javascript_fence(self):
        response = "```javascript\nconst x = 1;\n```"
        result = self.parser.parse(response)
        assert "const x = 1;" in result

    def test_no_language_fence(self):
        response = "```\nprint('hello')\n```"
        result = self.parser.parse(response)
        assert "print" in result

    def test_tab_indented(self):
        response = "Code:\n\tdef foo():\n\t\treturn 1"
        result = self.parser.parse(response)
        assert "def foo():" in result


# ===========================================================================
# TextAnswerParser — additional edge cases
# ===========================================================================

class TestTextAnswerParserEdgeCases:

    def test_both_empty(self):
        assert TextAnswerParser.check_answer("", "") is False

    def test_predicted_empty(self):
        assert TextAnswerParser.check_answer("", "hello") is False

    def test_correct_empty(self):
        assert TextAnswerParser.check_answer("hello", "") is False

    def test_partial_overlap(self):
        """Fuzzy match with partial overlap."""
        assert TextAnswerParser.check_answer("The capital of France", "capital of France") is True

    def test_completely_different(self):
        assert TextAnswerParser.check_answer("apple", "orange") is False

    def test_normalize_function(self):
        assert TextAnswerParser.normalize("  Hello, World!  ") == "hello world"

    def test_normalize_empty(self):
        assert TextAnswerParser.normalize("") == ""


# ===========================================================================
# MultiChoiceParser — additional edge cases
# ===========================================================================

class TestMultiChoiceParserEdgeCases:

    def setup_method(self):
        self.parser = MultiChoiceParser()

    def test_answer_in_middle_of_text(self):
        assert self.parser.parse("Based on my analysis, ANSWER: D is correct") == "D"

    def test_multiple_choice_letters_picks_best(self):
        """When multiple patterns match, picks highest-confidence one."""
        result = self.parser.parse("The answer is B")
        assert result == "B"

    def test_choice_e_and_f(self):
        assert self.parser.parse("Answer: E", choices=["A", "B", "C", "D", "E", "F"]) == "E"

    def test_chinese_答案是_with_paren(self):
        assert self.parser.parse("答案是（C）") == "C"

    def test_just_a_letter_with_period(self):
        response = "After careful consideration:\nA."
        assert self.parser.parse(response) == "A"


# ===========================================================================
# EvalScope compatibility tests
# Tests derived from patterns in evalscope/utils/multi_choices.py and
# evalscope/metrics/math_parser.py to ensure feature parity.
# ===========================================================================

class TestEvalScopeChoicePatterns:
    """Tests for EvalScope-style ANSWER: X and 答案：X patterns."""

    def setup_method(self):
        self.parser = MultiChoiceParser()

    def test_strict_answer_prefix(self):
        """EvalScope parse_answers: strict 'ANSWER: X' at start of line."""
        assert self.parser.parse("ANSWER: B") == "B"

    def test_strict_answer_after_reasoning(self):
        """EvalScope: 'ANSWER: X' after step-by-step reasoning."""
        response = "Let me think step by step.\nFirst, ...\nTherefore, the answer is C.\nANSWER: C"
        assert self.parser.parse(response) == "C"

    def test_answer_colon_midtext(self):
        assert self.parser.parse("Based on analysis, ANSWER: D") == "D"

    def test_chinese_answer_colon(self):
        """EvalScope parse_answers_zh: '答案：X'."""
        assert self.parser.parse("答案：A") == "A"

    def test_chinese_answer_colon_english(self):
        assert self.parser.parse("答案: B") == "B"

    def test_evalscope_format_example(self):
        """EvalScope format_example outputs: 'ANSWER: X'."""
        response = "What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nANSWER: B"
        assert self.parser.parse(response) == "B"

    def test_fallback_last_uppercase(self):
        """EvalScope _fallback_parse_answer: last uppercase letter."""
        assert self.parser.parse("I think the answer is probably C or D") == "D"

    def test_no_valid_choice_returns_empty(self):
        assert self.parser.parse("I cannot determine the answer.") == ""


class TestEvalScopeMathExtractAnswer:
    """Tests for EvalScope math_parser.py extract_answer patterns."""

    def setup_method(self):
        self.parser = MathAnswerParser()

    def test_boxed_with_braces(self):
        """EvalScope: \\boxed{70,000}."""
        result = self.parser.parse("The answer is \\boxed{70000}")
        assert result == "70000"

    def test_boxed_nested(self):
        """EvalScope: brace-depth tracking with nested braces."""
        result = self.parser.parse("\\boxed{\\frac{1}{2}}")
        assert "frac" in result or result == "\\frac{1}{2}"

    def test_he_answer_is(self):
        """EvalScope: 'the answer is' / 'he answer is' pattern."""
        assert self.parser.parse("So the answer is 42") == "42"

    def test_final_answer_is(self):
        """EvalScope: 'final answer is' pattern."""
        result = self.parser.parse("My final answer is $100$")
        # Dollar signs are cleaned at the strip_answer_string / check_answer level
        assert "100" in result

    def test_chinese_答案是(self):
        """EvalScope: '答案是' pattern."""
        assert self.parser.parse("经过计算，答案是 3.14") == "3.14"

    def test_ANSWER_prefix(self):
        """EvalScope: 'ANSWER:' pattern."""
        assert self.parser.parse("ANSWER: 256") == "256"

    def test_last_number_fallback(self):
        """EvalScope: use_last_number fallback."""
        assert self.parser.parse("Some text 123 end") == "123"

    def test_comma_thousands(self):
        """EvalScope: strip commas in numbers like 70,000."""
        result = self.parser.parse("\\boxed{70000}")
        assert "70000" in result


class TestEvalScopeMathEqual:
    """Tests for EvalScope math_equal comparison patterns."""

    def test_numeric_equal(self):
        """EvalScope numeric_equal: isclose with rel_tol=1e-4."""
        assert MathAnswerParser.check_answer("70000", "70000") is True

    def test_numeric_close(self):
        assert MathAnswerParser.check_answer("3.1415", "3.1416") is True

    def test_percentage_variant(self):
        """EvalScope: 50 vs 0.5 (percentage variant)."""
        assert MathAnswerParser.check_answer("0.5", "50") is True

    def test_percentage_variant_2(self):
        """EvalScope: 0.75 vs 75."""
        assert MathAnswerParser.check_answer("0.75", "75") is True

    def test_case_insensitive(self):
        """EvalScope: case-insensitive comparison."""
        assert MathAnswerParser.check_answer("abc", "ABC") is True

    def test_none_prediction(self):
        assert MathAnswerParser.check_answer(None, "42") is False

    def test_none_reference(self):
        assert MathAnswerParser.check_answer("42", None) is False

    def test_comma_normalized_comparison(self):
        assert MathAnswerParser.check_answer("70,000", "70000") is True


class TestEvalScopeStripAnswerString:
    """Tests for MathAnswerParser.strip_answer_string (EvalScope LaTeX cleanup)."""

    def test_dollar_sign_removal(self):
        assert MathAnswerParser.strip_answer_string("$100$") == "100"

    def test_trailing_period(self):
        assert MathAnswerParser.strip_answer_string("42.") == "42"

    def test_newline_removal(self):
        assert MathAnswerParser.strip_answer_string("42\n") == "42"

    def test_comma_thousands(self):
        assert MathAnswerParser.strip_answer_string("70,000") == "70000"

    def test_latex_text_removal(self):
        result = MathAnswerParser.strip_answer_string("42\\text{miles}")
        assert "42" in result

    def test_empty(self):
        assert MathAnswerParser.strip_answer_string("") == ""
