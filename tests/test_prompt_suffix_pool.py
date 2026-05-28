import json

import core.benchmark_runner as benchmark_runner


def test_suffix_prompt_pool_prefers_hard_long_output_datasets(tmp_path, monkeypatch):
    datasets_dir = tmp_path / "datasets"
    (datasets_dir / "mmlu").mkdir(parents=True)
    (datasets_dir / "aime2024").mkdir(parents=True)
    (datasets_dir / "aime2025").mkdir(parents=True)
    (datasets_dir / "aime2026").mkdir(parents=True)
    (datasets_dir / "swebench_lite").mkdir(parents=True)
    (datasets_dir / "longbench").mkdir(parents=True)

    (datasets_dir / "mmlu" / "test.json").write_text(
        json.dumps(
            [
                {
                    "question": "Easy lookup question?",
                    "choices": ["A", "B", "C", "D"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (datasets_dir / "aime2024" / "aime2024.json").write_text(
        json.dumps(
            [
                {
                    "problem": "Find the sum of all integer bases b > 9 with a divisibility property.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (datasets_dir / "aime2025" / "aime2025.json").write_text(
        json.dumps(
            [
                {
                    "problem": "Find all integer bases b > 9 with a divisibility property.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (datasets_dir / "aime2026" / "aime2026.json").write_text(
        json.dumps(
            [
                {
                    "problem": "Find the number of integers less than 100 equal to a+b+ab.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (datasets_dir / "swebench_lite" / "swe-bench-lite.jsonl").write_text(
        json.dumps(
            {
                "problem_statement": (
                    "Nested model composition produces an incorrect separability matrix; "
                    "diagnose the bug and propose the minimal regression test."
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (datasets_dir / "longbench" / "repobench-p.jsonl").write_text(
        json.dumps(
            {
                "input": "class Solver:\n    def repair_nested_state(self):",
                "context": "Several modules mutate shared graph state during symbolic analysis.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark_runner, "_SUFFIX_PROMPT_POOL", None)

    pool = benchmark_runner._load_suffix_prompt_pool()

    assert any("AIME 2024" in prompt for prompt in pool)
    assert any("AIME 2025" in prompt for prompt in pool)
    assert any("AIME 2026" in prompt for prompt in pool)
    assert any("Find all integer bases" in prompt for prompt in pool)
    assert any("SWE-Bench Lite" in prompt for prompt in pool)
    assert any("LongBench RepoBench" in prompt for prompt in pool)
    assert all("IMPORTANT: You MUST write a very long" in prompt for prompt in pool)
    assert not any("Easy lookup question" in prompt for prompt in pool)
