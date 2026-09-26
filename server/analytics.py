"""Statistics based on persisted per-request observations."""

from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from typing import Any

from core.benchmark.metrics import METRIC_CONTRACT_VERSION

NUMERIC_FIELDS = ("ttft", "tpot", "tps", "total_time", "prefill_speed")
GROUP_FIELDS = {
    "concurrency": ("concurrency_level",),
    "prefill": ("input_tokens_target",),
    "segmented_prefill": ("context_length_target",),
    "long_context": ("context_length_target",),
    "throughput_matrix": ("concurrency_level", "context_length_target"),
    "stability": ("concurrency_level",),
    "custom_text": ("concurrency_level",),
}
GROUP_AXIS = {
    "concurrency_level": "并发",
    "input_tokens_target": "输入长度",
    "context_length_target": "上下文长度",
}

LEGACY_CONTRACT_VERSION = "legacy-unversioned"


class MetricContractConflict(ValueError):
    """A report cannot aggregate observations with conflicting metric definitions."""


def _contract_version(raw: str | None, source: str) -> str:
    if not raw:
        return LEGACY_CONTRACT_VERSION
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise MetricContractConflict(f"Invalid metric provenance in {source}") from exc
    if not isinstance(payload, dict):
        raise MetricContractConflict(f"Invalid metric provenance in {source}")
    version = payload.get("metric_contract_version")
    if version is None or version == "":
        return LEGACY_CONTRACT_VERSION
    if not isinstance(version, str) or not version.strip():
        raise MetricContractConflict(f"Invalid metric contract version in {source}")
    return version.strip()


def _run_contract_version(config_json: str | None, rows: list[dict[str, Any]]) -> str:
    run_version = _contract_version(config_json, "run configuration")
    row_versions = {_contract_version(row["extra_metrics"], f"request {row['id']}") for row in rows}
    if len(row_versions) > 1 or (row_versions and row_versions != {run_version}):
        raise MetricContractConflict("Run contains mixed metric contract versions")
    return run_version


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float] | None:
    if trials == 0:
        return None
    fraction = successes / trials
    denominator = 1 + z * z / trials
    midpoint = (fraction + z * z / (2 * trials)) / denominator
    width = (
        z
        * math.sqrt(fraction * (1 - fraction) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return [max(0.0, midpoint - width), min(1.0, midpoint + width)]


def _describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row["error"]]
    metrics: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        values = [
            float(row[field])
            for row in successful
            if row[field] is not None and math.isfinite(float(row[field])) and float(row[field]) > 0
        ]
        metrics[field] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    n = len(rows)
    ok = len(successful)
    return {
        "requests": n,
        "successes": ok,
        "failures": n - ok,
        "success_rate": ok / n if n else None,
        "success_rate_ci95": wilson_interval(ok, n),
        "metrics": metrics,
    }


def run_summary(db_path: str, run_id: int, *, job: dict[str, Any] | None = None) -> dict[str, Any]:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        run = conn.execute(
            """SELECT id, test_id, test_type, status, model_id, provider, created_at,
                      started_at, completed_at, config_json, system_info_json
               FROM test_runs WHERE id = ?""",
            (run_id,),
        ).fetchone()
        if run is None:
            raise LookupError(run_id)
        rows = [
            dict(row)
            for row in conn.execute(
                """SELECT id, concurrency_level, input_tokens_target, context_length_target,
                          ttft, tpot, tps, total_time,
                          prefill_speed, prefill_tokens, decode_tokens, error, error_type,
                          token_source, token_calc_method, cache_hit_source, extra_metrics
                   FROM test_results WHERE run_id = ? ORDER BY id""",
                (run_id,),
            )
        ]
    finally:
        conn.close()
    version = _run_contract_version(run["config_json"], rows)
    expected = int(job["progress_total"]) if job and job["progress_total"] > 0 else None
    integrity_reasons = []
    if job is None:
        integrity_reasons.append("缺少控制任务，无法核验执行状态和计划请求数。")
    else:
        if job["result_run_id"] != run_id or job["job_id"] != run["test_id"]:
            integrity_reasons.append("任务与测量记录未正确关联。")
        if job["status"] != "completed":
            integrity_reasons.append("任务尚未成功完成。")
    if run["status"] != "completed":
        integrity_reasons.append("测量记录尚未成功完成。")
    if expected is None:
        integrity_reasons.append("缺少计划请求数，无法核验结果完整性。")
    elif len(rows) != expected:
        integrity_reasons.append(f"计划 {expected} 次请求，数据库实际记录 {len(rows)} 次。")
    if not rows:
        integrity_reasons.append("没有逐请求观测值。")
    if version != METRIC_CONTRACT_VERSION:
        integrity_reasons.append(f"指标口径 {version} 尚未通过当前版本验收。")
    fields = GROUP_FIELDS.get(run["test_type"], ("concurrency_level",))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)

    def group_label(key: tuple[Any, ...]) -> str:
        parts = []
        for field, value in zip(fields, key, strict=True):
            text = f"{value:,}" if isinstance(value, int) else "未记录"
            suffix = "并发" if field == "concurrency_level" else "tokens"
            parts.append(f"{text} {suffix}" if value is not None else text)
        return " / ".join(parts)

    sliced = [
        {"label": group_label(key), **_describe(group)}
        for key, group in sorted(
            groups.items(),
            key=lambda pair: tuple(
                (item is None, item if item is not None else 0) for item in pair[0]
            ),
        )
    ]
    return {
        "metric_contract_version": version,
        "integrity": {
            "verified": not integrity_reasons,
            "expected_requests": expected,
            "recorded_requests": len(rows),
            "reasons": integrity_reasons,
        },
        "run": {
            key: value
            for key, value in dict(run).items()
            if key not in {"config_json", "system_info_json"}
        },
        "overall": _describe(rows),
        "group_axis": " × ".join(GROUP_AXIS.get(field, field) for field in fields),
        "groups": sliced,
        "provenance": {
            "config_json": run["config_json"],
            "system_info_json": run["system_info_json"],
            "token_sources": sorted({row["token_source"] for row in rows if row["token_source"]}),
            "token_methods": sorted(
                {row["token_calc_method"] for row in rows if row["token_calc_method"]}
            ),
        },
        "notes": [
            "延迟和吞吐统计只使用成功且数值有限、大于零的请求；零表示未采集。",
            "分位数使用相邻观测值线性插值。",
            "成功率区间采用双侧 95% Wilson score。",
            "本报告仅描述观测结果，不推断模型间差异的统计显著性。",
            "跨运行对比仍须核对硬件、模型配置、工作负载和 token 来源。",
        ],
    }


def run_results(
    db_path: str, run_id: int, *, limit: int, offset: int, since_id: int = 0
) -> dict[str, Any]:
    """逐请求结果分页；since_id > 0 时只返回 id 更大的增量行（运行中轮询用）。"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM test_results WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        rows = conn.execute(
            """SELECT id, session_id, request_index, round, concurrency_level,
                      input_tokens_target, context_length_target, ttft, tpot, tps,
                      total_time, prefill_tokens, decode_tokens, token_source,
                      token_calc_method, cache_hit_source, error, error_type
               FROM test_results WHERE run_id = ? AND id > ?
               ORDER BY id LIMIT ? OFFSET ?""",
            (run_id, since_id, limit, offset),
        ).fetchall()
    finally:
        conn.close()
    return {
        "total": count,
        "items": [dict(row) for row in rows],
        "limit": limit,
        "offset": offset,
        "since_id": since_id,
    }


def run_results_csv(db_path: str, run_id: int) -> str:
    """Export numeric observations without prompt or response text."""
    columns = (
        "id",
        "session_id",
        "concurrency_level",
        "input_tokens_target",
        "context_length_target",
        "ttft",
        "tpot",
        "tps",
        "total_time",
        "prefill_tokens",
        "decode_tokens",
        "token_source",
        "token_calc_method",
        "error_type",
        "metric_contract_version",
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow((*columns, "success"))
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        stored_columns = tuple(field for field in columns if field != "metric_contract_version")
        rows = conn.execute(
            f"SELECT {', '.join(stored_columns)}, extra_metrics, error "
            "FROM test_results WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        for row in rows:
            values = []
            for field in columns:
                value = (
                    _contract_version(row["extra_metrics"], f"request {row['id']}")
                    if field == "metric_contract_version"
                    else row[field]
                )
                if isinstance(value, str) and value.startswith(
                    ("=", "+", "-", "@", "\t", "\r", "\n")
                ):
                    value = "'" + value
                values.append(value)
            writer.writerow((*values, int(not bool(row["error"]))))
    finally:
        conn.close()
    return buffer.getvalue()
