"""Construct bounded arithmetic/polynomial expressions without evaluating source code."""

from __future__ import annotations

import ast
import math
import re
from typing import Any


def parse_bounded_math(text: str) -> Any:
    """Accept small polynomials and numeric radicals; reject unsupported syntax."""
    import sympy

    if not text or len(text) > 256:
        raise ValueError("表达式为空或超过 256 字符")
    source = text.strip().strip("$").replace("−", "-")
    source = source.replace(r"\left", "").replace(r"\right", "")
    for _ in range(4):
        source = re.sub(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", source)
        source = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", source)
    source = source.replace(r"\times", "*").replace(r"\cdot", "*").replace("×", "*")
    source = source.replace(r"\div", "/").replace("÷", "/").replace("^", "**")
    source = source.replace("{", "(").replace("}", ")")
    source = re.sub(
        r"(?<![\w.])[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?(?![0-9,.])",
        lambda m: m[0].replace(",", ""),
        source,
    )
    if source.endswith("%"):
        source = "(" + source[:-1] + ")/100"
    tree = ast.parse(source, mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("表达式节点过多")
    symbols: set[str] = set()

    def build(node: ast.AST) -> tuple[Any, int, int]:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            if not math.isfinite(node.value) or abs(node.value) > 10**12:
                raise ValueError("数值超出范围")
            return sympy.Rational(str(node.value)), 1, 0
        if isinstance(node, ast.Name) and re.fullmatch("[A-Za-z]", node.id):
            symbols.add(node.id)
            if len(symbols) > 4:
                raise ValueError("最多四个变量")
            return sympy.Symbol(node.id), 1, 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value, terms, degree = visit(node.operand)
            return (-value if isinstance(node.op, ast.USub) else value), terms, degree
        if isinstance(node, ast.BinOp):
            left, left_terms, left_degree = visit(node.left)
            right, right_terms, right_degree = visit(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                terms, degree = left_terms + right_terms, max(left_degree, right_degree)
                if terms > 128:
                    raise ValueError("多项式展开超过上限")
                return left + (right if isinstance(node.op, ast.Add) else -right), terms, degree
            if isinstance(node.op, ast.Mult):
                terms, degree = left_terms * right_terms, left_degree + right_degree
                if terms > 128 or degree > 8:
                    raise ValueError("多项式展开超过上限")
                return left * right, terms, degree
            if isinstance(node.op, ast.Div) and right.is_number and right != 0:
                return left / right, left_terms, left_degree
            if isinstance(node.op, ast.Pow) and right.is_Integer and -8 <= right <= 8:
                if right < 0 and (not left.is_number or left == 0):
                    raise ValueError("只支持非零常数的负整数次幂")
                exponent = int(right)
                terms, degree = left_terms ** abs(exponent), left_degree * abs(exponent)
                coefficient_bits = max(
                    (
                        abs(int(value.p)).bit_length() + int(value.q).bit_length()
                        for value in left.atoms(sympy.Rational)
                    ),
                    default=1,
                )
                if terms > 128 or degree > 8 or coefficient_bits * abs(exponent) > 4096:
                    raise ValueError("幂运算超过上限")
                return left**exponent, terms, degree
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            value, _, _ = visit(node.args[0])
            if value.is_number and value.is_nonnegative:
                return sympy.sqrt(value), 1, 0
        raise ValueError("不支持的数学语法")

    def visit(node: ast.AST) -> tuple[Any, int, int]:
        value, terms, degree = build(node)
        if any(
            abs(int(item.p)).bit_length() + int(item.q).bit_length() > 4096
            for item in value.atoms(sympy.Rational)
        ):
            raise ValueError("数学系数超过上限")
        return value, terms, degree

    result, _, _ = visit(tree.body)
    expanded = result.expand()
    if expanded.is_number and not expanded.is_finite:
        raise ValueError("非有限数学结果")
    return expanded


def bounded_math_equal(left: str, right: str) -> bool | None:
    """None means equivalence cannot be safely decided by this bounded rule."""
    try:
        actual, expected = parse_bounded_math(left), parse_bounded_math(right)
        if actual == expected:
            return True
        if actual.is_number and expected.is_number:
            a, b = float(actual), float(expected)
            if math.isfinite(a) and math.isfinite(b):
                return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
            return None
        if actual.is_polynomial() and expected.is_polynomial():
            return bool((actual - expected).expand() == 0)
    except (ValueError, SyntaxError, TypeError, OverflowError, ZeroDivisionError, RecursionError):
        return None
    return None
