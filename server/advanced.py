"""Reproducible offline parsing, agreement analysis and text perturbation tools."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from core.bounded_math import bounded_math_equal, parse_bounded_math
from core.robustness_tester import PERTURBATION_GENERATOR_VERSION, PerturbationType, TextPerturber
from core.smart_answer_parser import AnswerType, SmartAnswerParser
from evaluators.answer_parser import (
    CodeAnswerParser,
    MathAnswerParser,
    MultiChoiceParser,
    TextAnswerParser,
    get_parser_for_dataset,
)
from server.specs import StrictSpec

TOOL_VERSION = "offline-tools-v1"
DatasetType = Literal[
    "auto", "mmlu", "gsm8k", "math500", "humaneval", "gpqa", "truthfulqa", "longbench"
]
AnswerKind = Literal["number", "choice", "text", "boolean", "code", "math"]
PerturbationKind = Literal[
    "synonym",
    "typo",
    "reorder",
    "case",
    "punctuation",
    "whitespace",
    "number_format",
    "paraphrase",
    "context_add",
    "rephrase",
]
PERTURBATIONS = {
    "synonym": ("同义词替换", True, "按小型词表替换；不保证语义等价，请人工复核。"),
    "typo": ("拼写扰动", True, "交换相邻字母；可能改变词义。"),
    "reorder": ("词序调整", False, "原版仅有占位规则，尚未实现。"),
    "case": ("大小写变化", True, "转换为小写；大小写可能影响专名和代码。"),
    "punctuation": ("标点变化", True, "移除结尾英文标点。"),
    "whitespace": ("空白符变化", True, "增加前三处空格。"),
    "number_format": ("数字格式变化", True, "切换整数部分的千位分隔，不改小数、符号及科学计数法。"),
    "paraphrase": ("同义改写", False, "原版未实现；不调用模型生成改写。"),
    "context_add": ("添加上下文", True, "添加固定词表中的前缀；可能影响答案。"),
    "rephrase": ("问题重述", False, "原版未实现；不调用模型重述问题。"),
}


class ParseBody(StrictSpec):
    response: str = Field(min_length=1, max_length=50000)
    answer_type: AnswerKind | None = None
    dataset_type: DatasetType | None = None
    expected_answer: str | None = Field(default=None, max_length=10000)

    @model_validator(mode="after")
    def validate_mode(self) -> ParseBody:
        if not self.response.strip():
            raise ValueError("响应不能为空白")
        if self.answer_type and self.dataset_type:
            raise ValueError("请选择数据集解析或答案类型中的一种")
        return self


class ConsistencyBody(StrictSpec):
    question: str = Field(min_length=1, max_length=10000)
    runs: int = Field(default=5, ge=2, le=10)
    threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    dataset_type: DatasetType = "auto"
    expected_answer: str | None = Field(default=None, max_length=10000)
    responses: list[Annotated[str, Field(max_length=10000)]] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def validate_count(self) -> ConsistencyBody:
        if not self.question.strip():
            raise ValueError("问题不能为空白")
        if len(self.responses) != self.runs:
            raise ValueError("回答条数必须与设置的重复次数一致；空回答可用空字符串记录")
        if sum(map(len, self.responses)) > 50000:
            raise ValueError("回答总长度最多 50,000 字符")
        return self


class PerturbBody(StrictSpec):
    text: str = Field(min_length=1, max_length=10000)
    perturbation_type: PerturbationKind
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def validate_text(self) -> PerturbBody:
        if not self.text.strip():
            raise ValueError("原文不能为空白")
        return self


def snapshot(body: StrictSpec) -> dict[str, Any]:
    payload = body.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return {
        "tool_version": TOOL_VERSION,
        "input": payload,
        "input_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _select_parser(response: str, dataset: str) -> tuple[Any, str, list[str]]:
    if dataset != "auto":
        parser = get_parser_for_dataset(dataset)
        kind = (
            "choice"
            if isinstance(parser, MultiChoiceParser)
            else "math"
            if isinstance(parser, MathAnswerParser)
            else "code"
            if isinstance(parser, CodeAnswerParser)
            else "text"
        )
        return parser, kind, ["数据集抽取规则可能使用兜底匹配，建议复核提取结果。"]
    if "```" in response:
        return CodeAnswerParser(), "code", []
    if re.search(
        r"\\boxed\s*\{|####|(?:answer|答案|结果)\s*(?:is|是|为|:|：|=)\s*[-+]?\d", response, re.I
    ):
        return MathAnswerParser(), "math", []
    if re.search(
        r"(?:answer|choose|select|答案|选择|选项)\s*(?:is|是|为|:|：)?\s*[A-D]\b", response, re.I
    ) or re.fullmatch(r"[A-D]", response.strip(), re.I):
        return MultiChoiceParser(), "choice", []
    try:
        parse_bounded_math(response)
        return TextAnswerParser(), "math", []
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
        return (
            TextAnswerParser(),
            "text",
            ["自动解析未识别明确的答案标记，保留完整文本；可改选数据集或答案类型。"],
        )


def _compare(actual: str, expected: str, kind: str) -> tuple[bool | None, str]:
    if not actual or not expected.strip():
        return None, "not_available"
    if kind in {"math", "number"}:
        return bounded_math_equal(actual, expected), "bounded_math"
    if kind == "boolean":
        boolean = {
            "yes": True,
            "true": True,
            "是": True,
            "正确": True,
            "no": False,
            "false": False,
            "否": False,
            "错误": False,
        }
        a, b = boolean.get(_text_key(actual)), boolean.get(_text_key(expected))
        return (a == b if a is not None and b is not None else None), "explicit_boolean"
    if kind == "code":
        return actual.strip() == expected.strip(), "code_text_exact"
    return _text_key(actual) == _text_key(expected), "normalized_text_exact"


def _extract(parser: Any, response: str, kind: str) -> str:
    if kind == "math":
        # Preserve a bare expression, including unsupported syntax. A last-number
        # fallback must not silently turn e.g. sin(2) or 1/0 into the answer 2/0.
        source = response.strip()
        try:
            parse_bounded_math(source)
            return source
        except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
            if (
                "\n" not in source
                and re.fullmatch(r"[A-Za-z0-9_\\{}+\-*/^%.,()\s$]+", source)
                and (re.search(r"[+*/^%()]", source) or source.startswith(r"\frac"))
            ):
                return source
    return str(parser.parse(response))


def parse_answer(body: ParseBody) -> dict[str, Any]:
    warnings: list[str] = []
    out: dict[str, Any]
    kind: str = body.answer_type or "text"
    if body.dataset_type:
        parser, kind, warnings = _select_parser(body.response, body.dataset_type)
        extracted = _extract(parser, body.response, kind)
        out = {
            "extracted_answer": extracted,
            "confidence": None,
            "method": type(parser).__name__,
            "raw_match": extracted,
            "error": None if extracted else "未提取到答案",
        }
    elif kind in {"math", "code", "text"}:
        parser_types: dict[
            str, type[MathAnswerParser] | type[CodeAnswerParser] | type[TextAnswerParser]
        ] = {
            "math": MathAnswerParser,
            "code": CodeAnswerParser,
            "text": TextAnswerParser,
        }
        parser = parser_types[kind]()
        extracted = _extract(parser, body.response, kind)
        out = {
            "extracted_answer": extracted,
            "confidence": None,
            "method": type(parser).__name__,
            "raw_match": extracted,
            "error": None if extracted else "未提取到答案",
        }
    else:
        out = dataclasses.asdict(SmartAnswerParser().parse(body.response, AnswerType(kind)))
    normalized: Any = out["extracted_answer"]
    if kind in {"math", "number"}:
        try:
            normalized = str(parse_bounded_math(normalized))
        except (
            ValueError,
            SyntaxError,
            TypeError,
            OverflowError,
            ZeroDivisionError,
            RecursionError,
        ):
            normalized = None
            warnings.append("数学表达式不在有界算术/多项式规则内，未计算其值或证明等价。")
    out.update(normalized_value=normalized, answer_type=kind, dataset_type=body.dataset_type)
    if body.expected_answer and body.expected_answer.strip():
        correct, method = _compare(out["extracted_answer"], body.expected_answer, kind)
        out.update(
            is_correct=correct,
            score=None if correct is None else float(correct),
            comparison_method=method,
        )
    warnings.extend(
        [
            "规则匹配强度不是模型正确概率。",
            "代码仅抽取和文本比较，不执行代码；文本匹配不代表语义或事实正确性。",
        ]
    )
    return {**out, **snapshot(body), "warnings": warnings}


def analyze_consistency(body: ConsistencyBody) -> dict[str, Any]:
    results = []
    keys: list[str] = []
    for index, response in enumerate(body.responses, 1):
        if response.strip():
            parsed = parse_answer(
                ParseBody(
                    response=response,
                    dataset_type=body.dataset_type,
                    expected_answer=body.expected_answer,
                )
            )
        else:
            parsed = {
                "extracted_answer": "",
                "normalized_value": None,
                "error": "空回答",
                "is_correct": None,
                "answer_type": None,
            }
        key = None
        if parsed["extracted_answer"] and not parsed["error"]:
            if parsed["answer_type"] in {"math", "number"}:
                key = parsed["normalized_value"]
            elif parsed["answer_type"] == "code":
                key = parsed["extracted_answer"].strip()
            else:
                key = _text_key(parsed["extracted_answer"])
        if key is not None:
            key = str(parsed["answer_type"]) + ":" + key
            keys.append(key)
        results.append(
            {
                "index": index,
                "response": response,
                "extracted_answer": parsed["extracted_answer"],
                "normalized_value": parsed["normalized_value"],
                "error": parsed["error"] or ("表达式无法规范化分组" if key is None else None),
                "is_correct": parsed.get("is_correct"),
                "group_key": key,
            }
        )
    counts = Counter(keys)
    maximum = max(counts.values(), default=0)
    leaders = [key for key, count in counts.items() if count == maximum]
    rate = maximum / body.runs if maximum else None
    groups = [
        {
            "answer": next(
                item["extracted_answer"] for item in results if item["group_key"] == key
            ),
            "count": count,
            "normalized_key": key,
        }
        for key, count in counts.most_common()
    ]
    correct = sum(item["is_correct"] is True for item in results)
    has_reference = bool(body.expected_answer and body.expected_answer.strip())
    return {
        **snapshot(body),
        "total": body.runs,
        "parsed": len(keys),
        "unparsed": body.runs - len(keys),
        "groups": groups,
        "results": results,
        "consistency_rate": rate,
        "tied_majority": len(leaders) > 1,
        "stable": None if rate is None else rate >= body.threshold and len(leaders) == 1,
        "reference_match_rate": correct / body.runs if has_reference else None,
        "reference_matched": correct if has_reference else None,
        "reference_undetermined": sum(item["is_correct"] is None for item in results)
        if has_reference
        else None,
        "warnings": [
            "一致率 = 唯一最多的规范化答案次数 / 全部回答次数；空回答和解析失败仍在分母中。",
            "并列最多不标为稳定；全部无法解析时不判定一致性。",
            "数字按有界表达式规范化，文本仅归一空白与大小写；一致不代表正确。",
            "仅分析用户提供的同一问题回答；来源、采样独立性及生成配置未经平台核验。",
            "2～10 次重复仅作单问题描述性分析，不代表总体能力或统计显著性。",
        ],
    }


def perturb_text(body: PerturbBody) -> dict[str, Any]:
    kind = body.perturbation_type
    label, supported, description = PERTURBATIONS[kind]
    result = TextPerturber(body.seed).perturb(body.text, PerturbationType(kind))
    changed = result.original_question != result.perturbed_question
    return {
        **snapshot(body),
        "label": label,
        "perturbation_generator_version": PERTURBATION_GENERATOR_VERSION,
        "original_text": result.original_question,
        "perturbed_text": result.perturbed_question,
        "details": result.perturbation_details,
        "supported": supported,
        "changed": changed,
        "status": "unsupported" if not supported else "applied" if changed else "unchanged",
        "warnings": [
            description,
            "此工具只生成文本，不调用模型；文本变化不能直接解释为模型鲁棒性。",
            "生成器使用独立种子，复现需同时保留工具版本、原文、类型与种子。",
        ],
    }
