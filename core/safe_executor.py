"""Math expression validation and client for the isolated code execution worker.

Generated code is never executed in the application process or a child that shares
its filesystem and credentials. A separately deployed worker runs each submission
inside a restricted container. An unavailable worker is an infrastructure error.
"""

import ast
import math
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlparse

import httpx

from core.sandbox_identity import validate_identity

_expected_sandbox: ContextVar[str | None] = ContextVar("expected_sandbox_contract", default=None)


@contextmanager
def sandbox_session(identity: dict | None):
    expected = validate_identity(identity)["sha256"] if identity is not None else None
    token = _expected_sandbox.set(expected)
    try:
        yield
    finally:
        _expected_sandbox.reset(token)


DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MEM_LIMIT_MB = 1024
_MAX_OUTPUT_BYTES = 1_000_000
_SAFE_MATH_ATTRIBUTES = {"sqrt", "sin", "cos", "tan", "log", "log10", "exp", "pi", "e"}


class SafeExecutionError(Exception):
    """Raised when safe execution fails."""

    pass


class SandboxUnavailableError(SafeExecutionError):
    """The isolated code execution service is missing or unhealthy."""

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
    if len(expr) > 256:
        return False

    # Check for suspicious patterns
    dangerous_patterns = [
        r"__.*__",  # Double underscores (magic methods/dunder)
        r"import\s",  # Import statements
        r"from\s",  # From imports
        r"exec\s*\(",  # exec function
        r"eval\s*\(",  # eval function
        r"open\s*\(",  # open function
        r"compile\s*\(",  # compile function
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, expr, re.IGNORECASE):
            return False

    # Only allow safe characters: digits, operators, parens, dots, spaces, word chars
    safe_chars = re.compile(r"^[0-9+\-*/().\s\w]+$")
    if not safe_chars.match(expr):
        return False

    # Try to parse as AST to ensure it's a valid expression
    try:
        tree = ast.parse(expr, mode="eval")
        nodes = list(ast.walk(tree))
        if len(nodes) > 128:
            return False
        power_count = 0
        # Walk the AST to check for dangerous constructs
        for node in nodes:
            if isinstance(node, ast.Constant):
                if (
                    isinstance(node.value, bool)
                    or not isinstance(node.value, (int, float))
                    or abs(node.value) > 10**18
                ):
                    return False
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                power_count += 1
                if (
                    power_count > 2
                    or not isinstance(node.right, ast.Constant)
                    or not isinstance(node.right.value, int)
                    or not 0 <= node.right.value <= 64
                ):
                    return False
            # Check for attribute access (could be used to access dangerous modules)
            if isinstance(node, ast.Attribute):
                # Only allow math module attributes
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "math"
                    and node.attr in _SAFE_MATH_ATTRIBUTES
                ):
                    continue
                return False
            # Check for function calls
            elif isinstance(node, ast.Call):
                if not (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in _SAFE_MATH_ATTRIBUTES - {"pi", "e"}
                    and len(node.args) == 1
                    and not node.keywords
                ):
                    return False
            # Check for imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                return False

    except (SyntaxError, ValueError):
        return False

    return True


def safe_eval_math(expr: str) -> float | None:
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
        result = eval(  # noqa: S307 - 表达式经 AST 白名单校验且只暴露 math
            expr, {"__builtins__": {}, **safe_namespace}, {}
        )
        return float(result)
    except (NameError, SyntaxError, TypeError, ValueError) as e:
        raise SafeExecutionError(f"Failed to evaluate expression: {e}") from e
    except Exception as e:
        raise SafeExecutionError(f"Unexpected error evaluating expression: {e}") from e


def _worker_settings() -> tuple[str, str]:
    url = os.environ.get("LLM_TEST_SANDBOX_URL", "").rstrip("/")
    token = os.environ.get("LLM_TEST_SANDBOX_TOKEN", "")
    if not url or not token:
        raise SandboxUnavailableError(
            "Sandbox unavailable: set LLM_TEST_SANDBOX_URL and LLM_TEST_SANDBOX_TOKEN "
            "before evaluating generated code."
        )
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise SandboxUnavailableError("Sandbox unavailable: worker URL must use HTTP(S).")
    if len(token) < 32:
        raise SandboxUnavailableError("Sandbox unavailable: worker token is too short.")
    return url, token


def _worker_request(method: str, url: str, token: str, timeout: float, **kwargs):
    # Worker traffic must stay on the private network even if HTTP_PROXY is set.
    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        return client.request(method, url, headers={"Authorization": f"Bearer {token}"}, **kwargs)


def require_sandbox_available() -> dict:
    """Check the worker before model requests are sent for code benchmarks."""
    url, token = _worker_settings()
    try:
        response = _worker_request("GET", f"{url}/health", token, 12.0)
        payload = response.json() if response.status_code == 200 else None
        if (
            response.status_code != 200
            or not isinstance(payload, dict)
            or payload.get("status") != "ok"
        ):
            raise SandboxUnavailableError(
                f"Sandbox unavailable: worker health check returned HTTP {response.status_code}."
            )
        return validate_identity(payload.get("sandbox_identity"))
    except (httpx.HTTPError, ValueError) as exc:
        raise SandboxUnavailableError(
            f"Sandbox unavailable: worker health check failed: {exc}"
        ) from exc


def run_untrusted_code(
    full_code: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_mb: int = DEFAULT_MEM_LIMIT_MB,
) -> tuple[bool, str | None, str | None]:
    """Execute untrusted code only through the isolated worker."""
    url, token = _worker_settings()
    if not full_code or not isinstance(full_code, str):
        return False, "No code provided", None
    if len(full_code.encode("utf-8")) > 256_000:
        return False, "Submission exceeds the 256 KB code limit", None
    try:
        response = _worker_request(
            "POST",
            f"{url}/execute",
            token,
            max(1.0, timeout_seconds) + 15.0,
            json={
                "code": full_code,
                "timeout_seconds": timeout_seconds,
                "mem_limit_mb": mem_limit_mb,
                **(
                    {"sandbox_contract": _expected_sandbox.get()} if _expected_sandbox.get() else {}
                ),
            },
        )
        if response.status_code != 200:
            raise SandboxUnavailableError(
                f"Sandbox unavailable: worker returned HTTP {response.status_code}."
            )
        result = response.json()
        if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
            raise ValueError("missing success status")
        if _expected_sandbox.get():
            identity = validate_identity(result.get("sandbox_identity"))
            if identity["sha256"] != _expected_sandbox.get():
                raise SandboxUnavailableError("Sandbox unavailable: grading environment changed")
        error = result.get("error")
        output = result.get("output")
        if error is not None and not isinstance(error, str):
            raise ValueError("invalid error field")
        if output is not None and not isinstance(output, str):
            raise ValueError("invalid output field")
        return (
            result["success"],
            error[:2000] if error else None,
            output[:_MAX_OUTPUT_BYTES] if output else None,
        )
    except (httpx.HTTPError, ValueError) as exc:
        raise SandboxUnavailableError(f"Sandbox unavailable: worker request failed: {exc}") from exc


def safe_exec_code(
    code: str, test_code: str | None = None, timeout_seconds: int = 5
) -> tuple[bool, str | None, str | None]:
    """
    Execute code in the isolated worker with a real timeout.

    Args:
        code: The code to execute (LLM-generated submission)
        test_code: Optional test code to append and run
        timeout_seconds: Maximum wall-clock time before the child is killed

    Returns:
        Tuple of (success: bool, error_message: Optional[str], output: Optional[str])

    Raises:
        SafeExecutionError: If code is deemed unsafe
    """
    if not code or not isinstance(code, str):
        return False, "No code provided", None

    full_code = code if not test_code else code + "\n" + test_code
    return run_untrusted_code(full_code, timeout_seconds=float(timeout_seconds))
