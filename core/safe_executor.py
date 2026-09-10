"""
Safe code execution utilities for evaluating code and expressions.

Security model (进程隔离, CWE-250/400 防护):
- LLM 生成的被测代码在**独立子进程**中执行, 与主进程(持有 API key、
  数据库、docker socket 访问)完全隔离 —— 解释器内逃逸只能打到子进程自身。
- 真实超时: 主进程 kill 子进程, while True / fork 类失控代码不再挂死评测线程。
- 资源限制: Linux 下通过 preexec_fn 施加 CPU 秒数与地址空间上限,
  抑制内存炸弹与 fork 炸弹。
- 执行结果仅经 stdout/stderr/exit code 回传, 无对象引用泄漏。

历史版本曾用「AST/正则黑名单 + exec」在同一解释器内执行, 已被证实可
通过运行时字符串拼接绕过 dunder 黑名单(docs/security_audit_2026-08.md #1)。
"""

import ast
import math
import os
import re
import subprocess
import sys

# 子进程单次执行的默认资源上限
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MEM_LIMIT_MB = 1024
# 子进程输出上限, 防止 print 炸弹撑爆管道
_MAX_OUTPUT_BYTES = 1_000_000

_CHILD_TEMPLATE = """\
import sys

# 允许被测代码定义函数/类所需的最小 builtins 集(仅作用于子进程;
# 真正的隔离是进程边界本身, 不依赖这份名单的完备性)
_safe = ("print range len str int float bool list dict tuple set sum min max abs "
         "round sorted enumerate zip map filter any all isinstance type reversed "
         "object super staticmethod classmethod property ValueError TypeError "
         "ZeroDivisionError IndexError KeyError AssertionError StopIteration "
         "Exception RuntimeError ArithmeticError OverflowError").split()
_real = {}
for _name in _safe:
    try:
        import builtins as _b
        _real[_name] = getattr(_b, _name)
    except AttributeError:
        pass

_globals = {"__builtins__": _real, "__name__": "__main__"}

try:
    exec(compile(sys.stdin.read(), "<submission>", "exec"), _globals)
except BaseException as e:
    sys.stderr.write(f"{type(e).__name__}: {e}\\n")
    sys.exit(1)
sys.exit(0)
"""


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
        # Walk the AST to check for dangerous constructs
        for node in ast.walk(tree):
            # Check for attribute access (could be used to access dangerous modules)
            if isinstance(node, ast.Attribute):
                # Only allow math module attributes
                if isinstance(node.value, ast.Name) and node.value.id == "math":
                    continue
                # Other attribute access is not allowed
                if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                    return False
            # Check for function calls
            elif isinstance(node, ast.Call):
                # Only allow math function calls
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                        continue
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


def _child_preexec(timeout_seconds: float, mem_limit_mb: int):  # noqa: ANN202
    """子进程资源限制(Linux): CPU 秒数 + 地址空间上限。失败则中止启动。"""

    def apply() -> None:  # pragma: no cover - 在子进程中运行
        import resource

        cpu = max(1, int(timeout_seconds)) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        mem_bytes = max(64, mem_limit_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        # 收窄 fd 上限(硬上限不可低于当前 soft, 否则 EPERM)
        cur_soft, cur_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        cap = min(cur_soft, 256)
        resource.setrlimit(resource.RLIMIT_NOFILE, (cap, max(cur_hard, cap)))
        os.umask(0o077)

    return apply


def run_untrusted_code(
    full_code: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_mb: int = DEFAULT_MEM_LIMIT_MB,
) -> tuple[bool, str | None, str | None]:
    """
    在隔离子进程中执行不可信代码。

    Args:
        full_code: 待执行代码(如 HumanEval 提交 + 测试断言)
        timeout_seconds: 墙钟超时, 到点强杀子进程(真实生效)
        mem_limit_mb: 子进程地址空间上限(MB)

    Returns:
        Tuple of (success: bool, error_message: Optional[str], output: Optional[str])
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _CHILD_TEMPLATE],
            input=full_code,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            preexec_fn=(
                _child_preexec(timeout_seconds, mem_limit_mb) if os.name == "posix" else None
            ),
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            f"TimeoutError: execution exceeded {timeout_seconds}s",
            None,
        )
    except OSError as e:
        return False, f"ExecutionError: failed to spawn sandbox process: {e}", None

    output = (proc.stdout or "")[:_MAX_OUTPUT_BYTES] or None
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        # 压缩 traceback 噪音, 只保留最后一行错误摘要
        tail = err.splitlines()[-1] if err else f"exit code {proc.returncode}"
        return False, tail[:2000], output
    return True, None, output


def safe_exec_code(
    code: str, test_code: str | None = None, timeout_seconds: int = 5
) -> tuple[bool, str | None, str | None]:
    """
    Safely execute code in an isolated child process with a real timeout.

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


