"""Offline analysis routes; the caller supplies the shared authentication dependency."""

import dataclasses
from typing import Any

from fastapi import APIRouter
from pydantic import Field, model_validator

from server.advanced import (
    PERTURBATIONS,
    ConsistencyBody,
    ParseBody,
    PerturbBody,
    analyze_consistency,
    parse_answer,
    perturb_text,
    snapshot,
)
from server.specs import StrictSpec


class ReasoningBody(StrictSpec):
    question: str = Field(min_length=1, max_length=20000)
    reasoning: str = Field(default="", max_length=100000)
    final_answer: str = Field(default="", max_length=20000)
    correct_answer: str = Field(default="", max_length=20000)

    @model_validator(mode="after")
    def validate_question(self) -> "ReasoningBody":
        if not self.question.strip():
            raise ValueError("问题不能为空白")
        return self


def advanced_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/advanced")

    @router.get("/catalog")
    def catalog():
        return {
            "datasets": [
                "auto",
                "mmlu",
                "gsm8k",
                "math500",
                "humaneval",
                "gpqa",
                "truthfulqa",
                "longbench",
            ],
            "perturbations": [
                {"id": key, "label": values[0], "supported": values[1], "description": values[2]}
                for key, values in PERTURBATIONS.items()
            ],
        }

    @router.post("/parse")
    def parse(body: ParseBody):
        return parse_answer(body)

    @router.post("/consistency")
    def consistency(body: ConsistencyBody):
        return analyze_consistency(body)

    @router.post("/perturb")
    def perturb(body: PerturbBody):
        return perturb_text(body)

    @router.post("/reasoning")
    def reasoning(body: ReasoningBody):
        from core.reasoning_evaluator import ReasoningQualityEvaluator

        match = None
        if body.final_answer.strip() and body.correct_answer.strip():
            parsed = parse_answer(
                ParseBody(
                    response=body.final_answer,
                    dataset_type="auto",
                    expected_answer=body.correct_answer,
                )
            )
            match = parsed.get("is_correct")
        result = ReasoningQualityEvaluator().evaluate(
            question=body.question,
            reasoning=body.reasoning,
            final_answer=body.final_answer,
            correct_answer=body.correct_answer,
            is_answer_correct=match if match is not None else False,
        )
        out: dict[str, Any] = dataclasses.asdict(result)
        warnings = ["规则评分是启发式指标，不是经校准的能力评分；不调用 Judge 模型。"]
        if match is None:
            # Correctness-dependent scores cannot be inferred without a usable
            # reference comparison. Do not expose the evaluator's false default.
            out["final_answer_correct"] = None
            out["answer_confidence"] = None
            out["quality_score"]["correctness"] = None
            out["quality_score"]["overall"] = None
            out["failure_category"] = "not_assessed"
            out["failure_analysis"] = ""
            warnings.append("未提供可判定的最终答案/参考答案，正确性和综合分不予计算。")
        warnings.append("参考判定为文本或有界数学规则匹配；代码不执行，匹配不证明推理步骤正确。")
        return {
            **out,
            **snapshot(body),
            "warnings": warnings,
        }

    return router
