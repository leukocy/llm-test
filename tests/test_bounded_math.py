"""Safe equivalence must reject executable or unbounded expressions."""

import pytest

from core.bounded_math import bounded_math_equal, parse_bounded_math


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("1/2", "0.5"),
        (r"\frac{1}{2}", "50%"),
        (r"\sqrt{4}", "2"),
        ("(x+1)^2", "x*x+2*x+1"),
        ("-1,000.5", "-1000.5"),
        ("2^-2", "0.25"),
        ("1e-5", "0.00001"),
    ],
)
def test_bounded_equivalence(left, right):
    assert bounded_math_equal(left, right) is True


@pytest.mark.parametrize(
    ("left", "right"), [("0.5", "50"), ("0.4", "0.49"), ("x+1", "x+2"), ("1,2", "12")]
)
def test_different_values_do_not_use_implicit_percent_rounding_or_invalid_commas(left, right):
    assert bounded_math_equal(left, right) is not True


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "Symbol('x')",
        "sin(2)",
        "x.__class__",
        "(lambda: 1)()",
        "[1][0]",
        "True",
        "1/0",
        "1e309",
        "10**10000000",
        "((((10**8)**8)**8)**8)**8",
        "(x+y+z+a)^8",
        "x*y*z*a*b",
        "x^9",
        "1" + "+1" * 40,
        "1" * 257,
        "sqrt(x)",
        "sqrt(-1)",
        "x^-1",
    ],
)
def test_unsupported_or_excessive_work_is_undetermined(expression):
    with pytest.raises((ValueError, SyntaxError, TypeError, OverflowError, RecursionError)):
        parse_bounded_math(expression)
    assert bounded_math_equal(expression, "1") is None


def test_no_sympy_source_evaluation(monkeypatch):
    import sympy

    def reject_source(*args, **kwargs):
        raise AssertionError("Source text must never reach sympify")

    monkeypatch.setattr(sympy, "sympify", reject_source)
    assert bounded_math_equal("(x+1)^2", "x*x+2*x+1") is True
    assert bounded_math_equal("__import__('os').system('true')", "1") is None


def test_shared_smart_parser_uses_bounded_math_without_source_execution():
    from core.smart_answer_parser import AnswerType, SmartAnswerParser

    parser = SmartAnswerParser()
    result = parser.parse(r"\boxed{2+3*4}", AnswerType.MATH_EXPRESSION)
    assert result.normalized_value == 14
    assert parser._evaluate_expression("10**10000000") is None
    assert parser._evaluate_expression("__import__('os').getcwd()") is None
    assert parser._evaluate_expression("x+1") is None
    assert parser._evaluate_expression("abc2+3") is None
    result = parser.parse("value =," + "0" * 10000 + "!", AnswerType.NUMBER)
    assert result.normalized_value == 0.0
