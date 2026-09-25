"""Source of the Python program executed inside a restricted container."""

CHILD_TEMPLATE = """\
import sys

# Coding benchmarks need ordinary imports and builtins. The container boundary,
# rather than Python-level filtering, protects the application and host.
_globals = {"__name__": "__main__"}

try:
    exec(compile(sys.stdin.read(), "<submission>", "exec"), _globals)
except BaseException as e:
    sys.stderr.write(f"{type(e).__name__}: {e}\\n")
    sys.exit(1)
sys.exit(0)
"""
