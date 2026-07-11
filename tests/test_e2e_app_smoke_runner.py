"""Run the real Streamlit AppTest smoke test outside the global test mock."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_real_app_smoke_script_passes_strict_copy_checks():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tests/e2e_app_smoke.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "PASS: Default page copy and primary action verified" in output
    assert "WARN:" not in output
    assert "SKIP:" not in output
