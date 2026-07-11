"""Run Ruff only on Python files changed from a chosen Git base."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_changed_python_files(output: str) -> list[str]:
    """Return normalized Python paths from ``git diff --name-only`` output."""

    return [line.strip() for line in output.splitlines() if line.strip().endswith(".py")]


def changed_python_files(base: str) -> list[str]:
    """Collect added, copied, modified, or renamed Python files since ``base``."""

    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base}...HEAD",
            "--",
            "*.py",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Unable to collect changed Python files")
    return parse_changed_python_files(result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="master", help="Git base ref (default: master)")
    args = parser.parse_args(argv)

    paths = changed_python_files(args.base)
    if not paths:
        print(f"No changed Python files relative to {args.base}.")
        return 0

    print(f"Running Ruff on {len(paths)} changed Python files relative to {args.base}.")
    return subprocess.run(["ruff", "check", *paths], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
