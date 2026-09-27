"""The worker persists the whole structured log even after the memory window rolls."""

from __future__ import annotations

import json

from server.runner_adapter import _LogTee
from utils.logger import BenchmarkLogger


def test_log_tee_keeps_entries_after_memory_window_rolls(tmp_path):
    logger = BenchmarkLogger(max_entries=3)
    path = tmp_path / "logs.jsonl"
    tee = _LogTee(path)
    for index in range(8):
        logger.info(f"sample-{index}", metrics={"ttft": index / 10})
        tee(logger)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["message"] for row in rows] == [f"sample-{index}" for index in range(8)]
    assert rows[0]["level"] == "INFO"
    assert rows[-1]["metrics"] == {"ttft": 0.7}
