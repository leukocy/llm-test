"""Pair observed grades only after validating identity, coverage and scoring source."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from core.model_comparator import mcnemar_test

CONDITIONS = (
    "num_shots",
    "temperature",
    "max_tokens",
    "concurrency",
    "model_type",
    "thinking_enabled",
    "thinking_budget",
    "reasoning_effort",
    "use_cache",
    "use_llm_judge",
)


class PairingConflict(ValueError):
    """A sample ID must not conceal conflicting input or duplicate observations."""


def _rows(dataset: dict, side: str) -> tuple[dict[str, dict], dict[str, int]]:
    if not isinstance(dataset, dict) or not isinstance(dataset.get("details"), list):
        raise PairingConflict(f"{side} has no complete sample detail list")
    selected = {}
    counts = {"missing_id": 0, "invalid_grade": 0, "request_error": 0}
    seen = set()
    for row in dataset.get("details", []):
        if not isinstance(row, dict):
            counts["invalid_grade"] += 1
            continue
        identifier = row.get("sample_id")
        if not isinstance(identifier, str) or not identifier.strip():
            counts["missing_id"] += 1
            continue
        if identifier in seen:
            raise PairingConflict(f"{side} contains duplicate sample_id: {identifier[:80]}")
        seen.add(identifier)
        if (
            type(row.get("is_correct")) is not bool
            or type(row.get("is_judge_corrected", False)) is not bool
            or (row.get("is_judge_corrected") and not row["is_correct"])
        ):
            counts["invalid_grade"] += 1
            continue
        if row.get("error") and not (
            row.get("is_judge_corrected") and row["error"] == "Validated by AI Judge"
        ):
            counts["request_error"] += 1
            continue
        selected[identifier] = row
    return selected, counts


def compare_dataset(
    a: dict, b: dict, basis: str = "standard", dataset_name: str = ""
) -> dict[str, Any]:
    if basis not in {"standard", "final"}:
        raise ValueError("Score basis must be standard or final")
    rows_a, excluded_a = _rows(a, "A")
    rows_b, excluded_b = _rows(b, "B")
    shared = sorted(rows_a.keys() & rows_b.keys())
    if not shared:
        raise PairingConflict("No valid paired grades")
    warnings = []
    digest = hashlib.sha256()
    for identifier in shared:
        left, right = rows_a[identifier], rows_b[identifier]
        for field in ("question", "correct_answer"):
            if (
                not isinstance(left.get(field), str)
                or not left[field]
                or not isinstance(right.get(field), str)
                or not right[field]
            ):
                raise PairingConflict(
                    f"Sample {identifier[:80]} is missing {field} identity evidence"
                )
            if left[field] != right[field]:
                raise PairingConflict(f"Sample {identifier[:80]} has conflicting {field}")
        if not left.get("prompt") or left.get("prompt") != right.get("prompt"):
            warnings.append("配对提示词缺失或不一致，无法核验共同工作负载。")
        encoded = json.dumps(
            [identifier, left["question"], left["correct_answer"], left.get("prompt")],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    config_a, config_b = a.get("config") or {}, b.get("config") or {}
    if not isinstance(config_a, dict) or not isinstance(config_b, dict):
        raise PairingConflict("Scoring configuration must be an object")
    for config in (config_a, config_b):
        if not isinstance(config.get("dataset_provenance") or {}, dict) or not isinstance(
            config.get("dataset_overrides") or {}, dict
        ):
            raise PairingConflict("Scoring provenance and overrides must be objects")
        if any(
            not isinstance(value, dict)
            for value in (config.get("dataset_overrides") or {}).values()
        ):
            raise PairingConflict("Dataset override must be an object")
    for config, rows in ((config_a, rows_a), (config_b, rows_b)):
        corrected = [row for row in rows.values() if row.get("is_judge_corrected")]
        if corrected and config.get("use_llm_judge") is False:
            warnings.append("Judge 改判标记与关闭复核的配置冲突。")
        if any(row.get("judge_verdict") != "YES" for row in corrected):
            warnings.append("Judge 改判缺少完整 YES 判定来源。")
    for field in (*CONDITIONS, *(("ceval_split",) if dataset_name.startswith("ceval") else ())):
        if field not in config_a or field not in config_b:
            warnings.append(f"生成/评分条件未记录：{field}。")
        elif (
            (config_a.get("dataset_overrides") or {})
            .get(dataset_name, {})
            .get(field, config_a[field])
        ) != (
            (config_b.get("dataset_overrides") or {})
            .get(dataset_name, {})
            .get(field, config_b[field])
        ):
            warnings.append(f"生成/评分条件不同：{field}。")
    for field in ("sample_sha256", "few_shot_sha256"):
        pa = (config_a.get("dataset_provenance") or {}).get(field)
        pb = (config_b.get("dataset_provenance") or {}).get(field)
        if not isinstance(pa, str) or not re.fullmatch(r"[0-9a-f]{64}", pa) or pa != pb:
            warnings.append(f"有序样本/few-shot 来源未通过核验：{field}。")
    ca, cb = config_a.get("scoring_contract"), config_b.get("scoring_contract")
    if not isinstance(ca, str) or not re.fullmatch(r"[0-9a-f]{64}", ca) or ca != cb:
        warnings.append("评分代码来源缺失或不同，不能给出模型差异的显著性结论。")
    if config_a.get("requires_code_execution") or config_b.get("requires_code_execution"):
        environment_a, environment_b = (
            config_a.get("sandbox_contract"),
            config_b.get("sandbox_contract"),
        )
        if (
            not isinstance(environment_a, str)
            or not re.fullmatch(r"[0-9a-f]{64}", environment_a)
            or environment_a != environment_b
        ):
            warnings.append("代码评分的隔离执行环境缺少共同身份凭证；只描述配对，不检验模型差异。")
    for field in ("evaluation_split", "few_shot_split", "selection_seed"):
        pa = (config_a.get("dataset_provenance") or {}).get(field)
        pb = (config_b.get("dataset_provenance") or {}).get(field)
        if pa != pb:
            warnings.append(f"数据准备条件不同：{field}。")
    if (
        rows_a.keys() != rows_b.keys()
        or any(excluded_a.values())
        or any(excluded_b.values())
        or type(a.get("total_samples")) is not int
        or a["total_samples"] != len(rows_a)
        or type(b.get("total_samples")) is not int
        or b["total_samples"] != len(rows_b)
    ):
        warnings.append(
            "样本覆盖不完整或存在请求错误/无效成绩；统计只描述有效配对，不能代表完整计划。"
        )

    def grade(row: dict) -> bool:
        return row["is_correct"] and (basis == "final" or not row.get("is_judge_corrected", False))

    correct_a, correct_b = [grade(rows_a[s]) for s in shared], [grade(rows_b[s]) for s in shared]
    test = mcnemar_test(correct_a, correct_b)
    verified = not warnings
    if not verified:
        test.update(
            method="descriptive-pairs-v1",
            statistic=None,
            p_value=None,
            log_p_value=None,
            p_value_label="未检验",
            significant=None,
            interpretation="仅描述相同题目与参考答案的有效配对；来源、条件或覆盖未核验，不给出显著性结论。",
        )
    return {
        **test,
        "samples": len(shared),
        "accuracy_a": sum(correct_a) / len(shared),
        "accuracy_b": sum(correct_b) / len(shared),
        "accuracy_difference": (sum(correct_a) - sum(correct_b)) / len(shared),
        "score_basis": basis,
        "verified": verified,
        "paired_sha256": digest.hexdigest(),
        "excluded_a": excluded_a,
        "excluded_b": excluded_b,
        "unpaired_a": len(rows_a.keys() - rows_b.keys()),
        "unpaired_b": len(rows_b.keys() - rows_a.keys()),
        "warnings": list(dict.fromkeys(warnings)),
    }


def adjust_family(datasets: dict[str, dict]) -> None:
    """Holm correction over the explicitly tested dataset family, in log space."""
    eligible = sorted(
        (entry for entry in datasets.values() if entry["verified"]),
        key=lambda entry: entry["log_p_value"],
    )
    previous = -math.inf
    for index, entry in enumerate(eligible):
        adjusted_log = min(
            0.0, max(previous, entry["log_p_value"] + math.log(len(eligible) - index))
        )
        previous = adjusted_log
        adjusted = math.exp(adjusted_log)
        entry.update(
            adjusted_p_value=adjusted or float.fromhex("0x0.0000000000001p-1022"),
            adjusted_p_value_label=f"{adjusted:.6g}" if adjusted else "<5e-324",
            adjusted_significant=adjusted_log < math.log(0.05),
        )
    for entry in datasets.values():
        entry["test_family_size"] = len(eligible)
        if not entry["verified"]:
            entry.update(
                adjusted_p_value=None, adjusted_p_value_label="未检验", adjusted_significant=None
            )
