"""Versioned local reasoning diagnostics; never an independent reasoning proof."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

from core.reasoning_evaluator import ReasoningQualityEvaluator

VERSION = "heuristic-reasoning-v1"
DIMENSIONS = ("coherence", "completeness", "relevance", "correctness", "efficiency")
WEIGHTS = dict(zip(DIMENSIONS, (0.25, 0.20, 0.20, 0.25, 0.10), strict=True))


def assess_reasoning(
    question: str, reasoning: str, final_answer: str, correct_answer: str, verdict: bool | None
) -> dict[str, Any]:
    inputs = [question, reasoning, final_answer, correct_answer, verdict]
    fingerprint = hashlib.sha256(json.dumps(inputs, ensure_ascii=False).encode()).hexdigest()
    assessment: dict[str, Any] = {
        "version": VERSION,
        "source": "local_rule_heuristics",
        "correctness_basis": "standard_answer_verdict" if verdict is not None else "unavailable",
        "input_sha256": fingerprint,
        "weights": dict(WEIGHTS),
        "dimensions": dict.fromkeys(DIMENSIONS),
        "overall": None,
        "status": "unavailable",
    }
    if not reasoning.strip() or len(reasoning) > 100000:
        assessment["reason"] = (
            "missing_reasoning" if not reasoning.strip() else "reasoning_too_long"
        )
        return assessment
    result = ReasoningQualityEvaluator().evaluate(
        question, reasoning, final_answer, correct_answer, is_answer_correct=verdict or False
    )
    scores = asdict(result.quality_score)
    assessment["dimensions"] = {dimension: scores[dimension] for dimension in DIMENSIONS}
    assessment["status"] = "recorded"
    if verdict is None:
        assessment["dimensions"]["correctness"] = None
    else:
        assessment["overall"] = sum(
            scores[dimension] * weight for dimension, weight in WEIGHTS.items()
        )
    return assessment
