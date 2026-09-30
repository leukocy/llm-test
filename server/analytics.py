"""Statistics based on persisted per-request observations."""

from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from config.tokenizer_paths import describe_tokenizer_installation
from core.benchmark.metrics import METRIC_CONTRACT_VERSION
from server.extended_analytics import extended_observations
from server.observations import describe_values, percentile
from server.scenario_reports import scenario_analysis
from server.specs import describe_report_environment
from server.time_series import resolve_stability_row, stability_time_series, timing_observation

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


def _prompt_sha256(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    value = payload.get("prompt_sha256") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and len(value) == 64 else None


def _run_contract_version(config_json: str | None, rows: list[dict[str, Any]]) -> str:
    run_version = _contract_version(config_json, "run configuration")
    row_versions = {_contract_version(row["extra_metrics"], f"request {row['id']}") for row in rows}
    if len(row_versions) > 1 or (row_versions and row_versions != {run_version}):
        raise MetricContractConflict("Run contains mixed metric contract versions")
    return run_version


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


def describe_observations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row["error"]]
    metrics: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        values = [
            float(row[field])
            for row in successful
            if row[field] is not None and math.isfinite(float(row[field])) and float(row[field]) > 0
        ]
        metrics[field] = describe_values(values)
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


def run_summary(
    db_path: str,
    run_id: int,
    *,
    job: dict[str, Any] | None = None,
    warmup_path: Path | None = None,
) -> dict[str, Any]:
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
                """SELECT id, request_index, batch_id, concurrency_level, input_tokens_target, context_length_target,
                          ttft, tpot, tps, total_time,
                          prefill_speed, prefill_tokens, decode_tokens, error, error_type,
                          token_source, token_calc_method, cache_hit_source, cache_hit_tokens,
                          api_prefill, effective_prefill_tokens, extra_metrics
                   FROM test_results WHERE run_id = ? ORDER BY id""",
                (run_id,),
            )
        ]
        batch = conn.execute(
            "SELECT max_parallel, stop_on_error FROM control_batches WHERE batch_id = ?",
            (job.get("parent_job_id") if job else None,),
        ).fetchone()
    finally:
        conn.close()
    version = _run_contract_version(run["config_json"], rows)
    config = json.loads(run["config_json"] or "{}")
    if run["test_type"] == "stability":
        rows = [resolve_stability_row(row, config.get("stability_window")) for row in rows]
    try:
        report_environment = describe_report_environment(config.get("report_environment"))
    except ValidationError as exc:
        raise MetricContractConflict("Invalid user-reported report environment") from exc
    try:
        tokenizer_installation = describe_tokenizer_installation(
            config.get("tokenizer_installation")
        )
    except ValueError as exc:
        raise MetricContractConflict("Invalid tokenizer installation provenance") from exc
    stored_control = config.get("execution_control")
    control = dict(stored_control) if isinstance(stored_control, dict) else {}
    if job:
        control.update(
            pause_count=job.get("pause_count", 0),
            paused_seconds=job.get("paused_seconds", 0.0)
            + (
                max(0.0, time.time() - job["pause_started_at"])
                if job.get("pause_started_at")
                else 0.0
            ),
            pause_policy=config.get("pause_policy"),
            batch_id=job.get("parent_job_id"),
            max_parallel=1,
            stop_on_error=False,
        )
    if batch:
        control.update(
            max_parallel=batch["max_parallel"], stop_on_error=bool(batch["stop_on_error"])
        )
    protocol = config.get("measurement_protocol") if isinstance(config, dict) else None
    if not isinstance(protocol, dict):
        protocol = None
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
    if protocol is not None:
        if protocol.get("measured_requests") != len(rows):
            integrity_reasons.append("固定工作负载的正式请求数与逐请求记录不一致。")
        if protocol.get("warmup_recorded") != protocol.get("warmup_requests"):
            integrity_reasons.append("预热请求未全部记录，预热状态不可核验。")
        if protocol.get("warmup_failures", 0):
            integrity_reasons.append("预热请求出现失败。")
        if protocol.get("warmup_requests", 0) and warmup_path is not None:
            try:
                with warmup_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames or "condition" not in reader.fieldnames:
                        raise ValueError("Invalid warmup header")
                    artifact_count = sum(1 for _ in reader)
            except (OSError, ValueError, csv.Error):
                artifact_count = None
            if artifact_count != protocol.get("warmup_recorded"):
                integrity_reasons.append("预热观测文件缺失或记录数与运行配置不一致。")
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

    planned_by_label = {
        cell["label"]: cell
        for cell in (protocol or {}).get("cells", [])
        if isinstance(cell, dict) and isinstance(cell.get("label"), str)
    }
    plan_fields = {
        "concurrency_level": "concurrency",
        "input_tokens_target": "input_tokens_target",
        "context_length_target": "input_tokens_target",
    }
    planned_by_key = {
        tuple(cell.get(plan_fields[field]) for field in fields): cell
        for cell in planned_by_label.values()
        if all(isinstance(cell.get(plan_fields[field]), int) for field in fields)
    }
    observed_plan_labels: set[str] = set()
    quality_warnings: list[str] = []
    if control.get("pause_count", 0):
        quality_warnings.append(
            "运行曾在请求组之间暂停；单请求计时不含暂停，缓存、温度与资源监控条件可能变化。"
        )
    if control.get("max_parallel", 1) > 1:
        quality_warnings.append(
            "批次允许任务并行，共享受测服务容量；与串行结果比较前需核对重叠负载。"
        )
    sliced = []
    for key, group in sorted(
        groups.items(),
        key=lambda pair: tuple((item is None, item if item is not None else 0) for item in pair[0]),
    ):
        label = group_label(key)
        planned = planned_by_key.get(key) or planned_by_label.get(label)
        if planned:
            observed_plan_labels.add(planned["label"])
        target = planned.get("input_tokens_target") if planned else None
        actual_tokens = [
            float(row["prefill_tokens"])
            for row in group
            if not row["error"] and row["prefill_tokens"] is not None and row["prefill_tokens"] > 0
        ]
        token_median = percentile(actual_tokens, 0.5)
        drift_pct = (
            (token_median - target) / target * 100
            if token_median is not None and isinstance(target, int) and target > 0
            else None
        )
        slice_data = {
            "label": label,
            "dimensions": dict(zip(fields, key, strict=True)),
            **describe_observations(group),
            "planned_requests": planned.get("measured_requests") if planned else None,
            "input_tokens": {
                "target": target,
                "count": len(actual_tokens),
                "median": token_median,
                "target_deviation_pct": drift_pct,
            },
        }
        extended = extended_observations(group)
        slice_data["metrics"].update(extended["metrics"])
        slice_data["extended_observations"] = {
            key: value for key, value in extended.items() if key != "metrics"
        }
        sliced.append(slice_data)
        if planned and len(group) != planned["measured_requests"]:
            integrity_reasons.append(
                f"{label} 计划 {planned['measured_requests']} 次，实际 {len(group)} 次。"
            )
        if drift_pct is not None and abs(drift_pct) > 10:
            quality_warnings.append(f"{label} 的实际输入 token 中位数偏离目标 {drift_pct:+.1f}%。")
        if len(group) < 20:
            quality_warnings.append(f"{label} 仅 {len(group)} 个样本，尾部分位数分辨率有限。")
    for label, planned in planned_by_label.items():
        if label not in observed_plan_labels:
            integrity_reasons.append(f"{label} 计划 {planned['measured_requests']} 次，实际 0 次。")
    token_sources = sorted({row["token_source"] for row in rows if row["token_source"]})
    token_methods = sorted({row["token_calc_method"] for row in rows if row["token_calc_method"]})
    if len(token_sources) > 1 or len(token_methods) > 1:
        quality_warnings.append("本次运行混用了不同的 token 来源或算法；比较前需核对口径。")
    summary: dict[str, Any] = {
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
        "overall": describe_observations(rows),
        "group_axis": " × ".join(GROUP_AXIS.get(field, field) for field in fields),
        "groups": sliced,
        "measurement_protocol": protocol,
        "execution_control": control,
        "report_environment": report_environment,
        "tokenizer_installation": tokenizer_installation,
        "data_quality": {"warnings": quality_warnings},
        "provenance": {
            "config_json": run["config_json"],
            "system_info_json": run["system_info_json"],
            "token_sources": token_sources,
            "token_methods": token_methods,
        },
        "notes": [
            "逐请求延迟和速度统计只使用成功且数值有限、大于零的请求；这些指标的零表示未采集。系统批次速率与明确的缓存零值另按来源统计。",
            "分位数使用相邻观测值线性插值。",
            "成功率区间采用双侧 95% Wilson score。",
            "本报告仅描述观测结果，不推断模型间差异的统计显著性。",
            "跨运行对比仍须核对硬件、模型配置、工作负载和 token 来源。",
        ],
    }
    extended = extended_observations(rows)
    summary["overall"]["metrics"].update(extended["metrics"])
    summary["extended_observations"] = {
        key: value for key, value in extended.items() if key != "metrics"
    }
    summary["scenario_analysis"] = scenario_analysis(summary)
    if run["test_type"] == "stability":
        summary["time_series"] = stability_time_series(rows, config.get("stability_window"))
        if summary["time_series"]["live"] and (
            run["status"] != "running"
            or (
                job is not None
                and job["status"] not in {"running", "pausing", "paused", "cancelling"}
            )
        ):
            summary["time_series"]["live"] = False
            summary["time_series"]["window_state"] = "interrupted"
            summary["time_series"]["notes"].append(
                "执行已结束，但时间窗口未最终关闭；曲线只展示最后一次成功保存的窗口，不能视为完整运行。"
            )
    return summary


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
                      token_calc_method, cache_hit_source, error, error_type, extra_metrics
               FROM test_results WHERE run_id = ? AND id > ?
               ORDER BY id LIMIT ? OFFSET ?""",
            (run_id, since_id, limit, offset),
        ).fetchall()
    finally:
        conn.close()
    return {
        "total": count,
        "items": [
            {
                **{key: value for key, value in dict(row).items() if key != "extra_metrics"},
                "prompt_sha256": _prompt_sha256(row["extra_metrics"]),
            }
            for row in rows
        ],
        "limit": limit,
        "offset": offset,
        "since_id": since_id,
    }


def latest_output(db_path: str, run_id: int, *, max_chars: int = 4000) -> dict[str, Any]:
    """Return a bounded preview of the latest recorded model response on demand."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        row = conn.execute(
            """SELECT id, substr(output_text, 1, ?) AS output,
                      length(output_text) > ? AS truncated
               FROM test_results
               WHERE run_id = ? AND output_text IS NOT NULL AND output_text != ''
               ORDER BY id DESC LIMIT 1""",
            (max_chars, max_chars, run_id),
        ).fetchone()
    finally:
        conn.close()
    return (
        {"result_id": row[0], "output": row[1], "truncated": bool(row[2])}
        if row
        else {"result_id": None, "output": None, "truncated": False}
    )


def run_results_csv(db_path: str, run_id: int) -> str:
    """Export numeric observations without prompt or response text."""
    columns = (
        "id",
        "session_id",
        "batch_id",
        "request_index",
        "concurrency_level",
        "input_tokens_target",
        "context_length_target",
        "ttft",
        "tpot",
        "tps",
        "total_time",
        "prefill_tokens",
        "decode_tokens",
        "cache_hit_tokens",
        "cache_hit_source",
        "api_prefill",
        "effective_prefill_tokens",
        "token_source",
        "token_calc_method",
        "error_type",
        "metric_contract_version",
        "prompt_sha256",
        "system_measurement_json",
        "timing_observation_json",
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow((*columns, "success"))
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        stored_columns = tuple(
            field
            for field in columns
            if field
            not in {
                "metric_contract_version",
                "prompt_sha256",
                "system_measurement_json",
                "timing_observation_json",
            }
        )
        conn.execute("BEGIN")
        stored_run = conn.execute(
            "SELECT config_json FROM test_runs WHERE id = ?", (run_id,)
        ).fetchone()
        try:
            stored_config = json.loads(stored_run[0] or "{}") if stored_run else {}
        except (ValueError, TypeError):
            stored_config = {}
        window = stored_config.get("stability_window") if isinstance(stored_config, dict) else None
        rows = conn.execute(
            f"SELECT {', '.join(stored_columns)}, extra_metrics, error "
            "FROM test_results WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        for row in rows:
            row = resolve_stability_row(dict(row), window)
            values = []
            for field in columns:
                value: Any
                if field == "metric_contract_version":
                    value = _contract_version(row["extra_metrics"], f"request {row['id']}")
                elif field == "prompt_sha256":
                    value = _prompt_sha256(row["extra_metrics"])
                elif field == "timing_observation_json":
                    timing = timing_observation(row["extra_metrics"])
                    value = json.dumps(timing, ensure_ascii=False) if timing is not None else None
                elif field == "system_measurement_json":
                    payload = json.loads(row["extra_metrics"] or "{}")
                    observation = (
                        payload.get("system_measurement") if isinstance(payload, dict) else None
                    )
                    value = (
                        json.dumps(
                            {
                                key: observation.get(key)
                                for key in [
                                    "version",
                                    "id",
                                    "elapsed_seconds",
                                    "expected_requests",
                                    "recorded_requests",
                                ]
                            },
                            ensure_ascii=False,
                        )
                        if isinstance(observation, dict)
                        else None
                    )
                else:
                    value = row[field]
                if isinstance(value, str) and value.startswith(
                    ("=", "+", "-", "@", "\t", "\r", "\n")
                ):
                    value = "'" + value
                values.append(value)
            writer.writerow((*values, int(not bool(row["error"]))))
    finally:
        conn.close()
    return buffer.getvalue()
