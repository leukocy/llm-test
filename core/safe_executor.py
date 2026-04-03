"""
Safe code execution utilities for evaluating code and expressions.
Uses AST validation to prevent arbitrary code execution.
"""

import ast
import math
import re
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Optional, Tuple


class SafeExecutionError(Exception):
    """Raised when safe execution fails."""
    pass


def validate_math_expression(expr: str) -> bool:
    """
    Validate that a math expression contains only safe operations.

    Args:
        expr: The expression to validate

    Returns:
        True if safe, False otherwise
    """
    if not expr or not isinstance(expr, str):
        return False

    expr = expr.strip()

    # Check for suspicious patterns
    dangerous_patterns = [
        r'__.*__',  # Double underscores (magic methods/dunder)
        r'import\s',  # Import statements
        r'from\s',     # From imports
        r'exec\s*\(',  # exec function
        r'eval\s*\(',  # eval function
        r'open\s*\(',  # open function
        r'compile\s*\(',  # compile function
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, expr, re.IGNORECASE):
            return False

    # Only allow safe characters: digits, operators, parens, dots, spaces, word chars
    safe_chars = re.compile(r'^[0-9+\-*/().\s\w]+$')
    if not safe_chars.match(expr):
        return False

    # Try to parse as AST to ensure it's a valid expression
    try:
        tree = ast.parse(expr, mode='eval')
        # Walk the AST to check for dangerous constructs
        for node in ast.walk(tree):
            # Check for attribute access (could be used to access dangerous modules)
            if isinstance(node, ast.Attribute):
                # Only allow math module attributes
                if isinstance(node.value, ast.Name) and node.value.id == 'math':
                    continue
                # Other attribute access is not allowed
                if not (isinstance(node.value, ast.Name) and node.value.id == 'math'):
                    return False
            # Check for function calls
            elif isinstance(node, ast.Call):
                # Only allow math function calls
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'math':
                        continue
                return False
            # Check for imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                return False

    except (SyntaxError, ValueError):
        return False

    return True


def safe_eval_math(expr: str) -> Optional[float]:
    """
    Safely evaluate a mathematical expression.

    Args:
        expr: Mathematical expression (e.g., "2 + 2", "math.sqrt(16)")

    Returns:
        Result as float, or None if evaluation fails

    Raises:
        SafeExecutionError: If expression is unsafe
    """
    if not expr or not isinstance(expr, str):
        return None

    expr = expr.strip()

    # Validate the expression first
    if not validate_math_expression(expr):
        raise SafeExecutionError(f"Unsafe mathematical expression: {expr}")

    # Create a restricted namespace with only math functions
    safe_namespace = {
        "__builtins__": {},
        "math": math,
        # Add specific math functions for convenience
        "sqrt": math.sqrt,
        "pow": math.pow,
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        result = eval(expr, safe_namespace, {})
        return float(result)
    except (NameError, SyntaxError, TypeError, ValueError) as e:
        raise SafeExecutionError(f"Failed to evaluate expression: {e}")
    except Exception as e:
        raise SafeExecutionError(f"Unexpected error evaluating expression: {e}")


def safe_exec_code(
    code: str,
    test_code: Optional[str] = None,
    timeout_seconds: int = 5
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Safely execute Python code in a restricted environment.

    Args:
        code: The code to execute
        test_code: Optional test code to append and run
        timeout_seconds: Maximum execution time (not enforced in this implementation)

    Returns:
        Tuple of (success: bool, error_message: Optional[str], output: Optional[str])

    Raises:
        SafeExecutionError: If code is deemed unsafe
    """
    if not code or not isinstance(code, str):
        return False, "No code provided", None

    # Basic validation - check for obviously dangerous operations
    dangerous_keywords = [
        'os', 'subprocess', 'sys', 'importlib', 'builtins',
        'eval', 'exec', 'compile', 'open', '__import__',
        'globals', 'locals', 'vars', 'dir'
    ]

    code_lower = code.lower()

    # Check for import statements
    for dangerous in dangerous_keywords:
        dangerous_lower = dangerous.lower()
        # Check for "import os" or "from os import"
        if f'import {dangerous_lower}' in code_lower or f'from {dangerous_lower}' in code_lower:
            return False, f"Import of '{dangerous}' is not allowed", None

    # Check for direct access to dangerous modules through builtins
    if '__builtins__' in code_lower or '__import__' in code_lower:
        return False, "Access to __builtins__ or __import__ is not allowed", None

    # Check for dangerous attribute access patterns
    dangerous_patterns = [
        r'__class__',
        r'__base__',
        r'__subclasses__',
        r'__mro__',
        r'__globals__',
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return False, f"Dangerous pattern detected: {pattern}", None

    # Prepare the full code
    full_code = code
    if test_code:
        full_code = code + "\n" + test_code

    # Create restricted execution environment
    exec_globals = {
        "__builtins__": {
            # Only allow specific, safe builtins
            'print': print,
            'range': range,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'any': any,
            'all': all,
            'isinstance': isinstance,
            'type': type,
            'reversed': reversed,
        },
        # Add typing support (commonly needed in HumanEval)
        'List': list,
        'Dict': dict,
        'Tuple': tuple,
        'Optional': type(None),  # Simplified
        'Any': object,
        'Union': lambda *args: object,  # Simplified
    }

    # Capture output
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(full_code, exec_globals)

        output = stdout_capture.getvalue()
        return True, None, output

    except AssertionError as e:
        return False, f"AssertionError: {str(e)}", None
    except SyntaxError as e:
        return False, f"SyntaxError: {str(e)}", None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}", None


# For backward compatibility, provide the old interface but with warnings
def safe_exec_legacy(code: str, globals_dict: dict, locals_dict: dict = None):
    """
    Legacy compatibility wrapper for safe_exec_code.
    Issues a deprecation warning.
    """
    import warnings
    warnings.warn(
        "safe_exec_legacy is deprecated. Use safe_exec_code instead.",
        DeprecationWarning,
        stacklevel=2
    )
    success, error, output = safe_exec_code(code)
    if not success:
        raise Exception(error)
    return output
