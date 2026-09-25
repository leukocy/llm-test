"""Descriptive statistics shared by the live view and exported reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.result_metrics import success_mask_from_error


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str
    unit: str
    tail: float

    @property
    def tail_label(self) -> str:
        return "P95" if self.tail == 0.95 else "P10"


METRICS = (
    MetricSpec("ttft", "TTFT", "s", 0.95),
    MetricSpec("tpot", "TPOT", "ms", 0.95),
    MetricSpec("tps", "Decode TPS", "tokens/s", 0.10),
    MetricSpec("prefill_speed", "Prefill speed", "tokens/s", 0.10),
    MetricSpec("system_output_throughput", "Output throughput", "tokens/s", 0.10),
)

GROUP_COLUMNS = {
    "concurrency": ("concurrency",),
    "prefill": ("input_tokens_target",),
    "long_context": ("context_length_target",),
    "matrix": ("context_length_target", "concurrency"),
    "segmented": ("context_length_target", "cumulative_mode"),
    "stability": ("concurrency",),
    "custom_text": ("concurrency",),
    "all": ("test_type",),
}

GROUP_LABELS = {
    "concurrency": "Concurrency",
    "input_tokens_target": "Input tokens",
    "context_length_target": "Context tokens",
    "cumulative_mode": "Cumulative",
    "test_type": "Test",
}


@dataclass(frozen=True)
class AnalysisReport:
    test_type: str
    rows: int
    success_count: int | None
    success_ci: tuple[float, float] | None
    groups: pd.DataFrame
    notes: tuple[str, ...]
    metric_contract_version: str
    findings: tuple[str, ...] = ()
    token_sources: tuple[str, ...] = ()

    @property
    def success_rate(self) -> float | None:
        if self.success_count is None or self.rows == 0:
            return None
        return self.success_count / self.rows * 100


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float] | None:
    """Two-sided 95% Wilson interval for a binomial proportion, on a 0–1 scale."""
    if total <= 0 or successes < 0 or successes > total:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _metric_contract_version(df: pd.DataFrame) -> str:
    if "metric_contract_version" not in df.columns:
        return "legacy-unversioned"
    values = df["metric_contract_version"].fillna("").astype(str).str.strip()
    versions = set(values.replace("", "legacy-unversioned"))
    if len(versions) != 1:
        raise ValueError("Report cannot combine different metric contract versions.")
    return str(versions.pop())


def _group_label(columns: tuple[str, ...], key: Any) -> str:
    if not columns:
        return "All requests"
    values = key if isinstance(key, tuple) else (key,)
    parts = []
    for column, value in zip(columns, values, strict=True):
        if pd.isna(value):
            display = "Missing"
        elif isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
            display = f"{int(value):,}"
        else:
            display = str(value)
        parts.append(f"{GROUP_LABELS.get(column, column)} {display}")
    return " · ".join(parts)


def _positive_success_values(group: pd.DataFrame, column: str, ok: pd.Series) -> pd.Series:
    values = pd.to_numeric(group[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.loc[ok & values.gt(0)].dropna()


def _descriptive_findings(groups: pd.DataFrame) -> tuple[str, ...]:
    """Point out supported patterns without claiming significance or causality."""
    findings = []
    median = "TTFT (s) P50"
    tail = "TTFT (s) P95"
    valid = "TTFT (s) valid n"
    if all(column in groups for column in (median, tail, valid)):
        eligible = groups.loc[groups[valid].ge(20) & groups[median].gt(0)].copy()
        if not eligible.empty:
            eligible["tail_ratio"] = eligible[tail] / eligible[median]
            worst = eligible.loc[eligible["tail_ratio"].idxmax()]
            if worst["tail_ratio"] >= 2:
                findings.append(
                    f"Tail latency: {worst['Configuration']} has TTFT P95/P50 "
                    f"of {worst['tail_ratio']:.1f}× across {int(worst[valid])} valid measurements."
                )

    throughput = "Decode TPS (tokens/s) P50"
    throughput_valid = "Decode TPS (tokens/s) valid n"
    if all(column in groups for column in (throughput, throughput_valid)):
        eligible = groups.loc[groups[throughput_valid].ge(20) & groups[throughput].gt(0)]
        if len(eligible) >= 2:
            highest = eligible.loc[eligible[throughput].idxmax()]
            lowest = eligible.loc[eligible[throughput].idxmin()]
            if highest[throughput] >= lowest[throughput] * 1.25:
                findings.append(
                    "Observed median decode throughput ranges from "
                    f"{lowest[throughput]:,.1f} tokens/s ({lowest['Configuration']}) to "
                    f"{highest[throughput]:,.1f} tokens/s ({highest['Configuration']}); "
                    "these are descriptive estimates, not a significance test."
                )
    return tuple(findings)


def build_scientific_summary(df: pd.DataFrame, test_type: str) -> AnalysisReport:
    """Summarize recorded requests without turning missing metrics into zeroes."""
    if df is None or df.empty:
        raise ValueError("No request rows are available for analysis.")

    work = df.reset_index(drop=True).copy()
    contract = _metric_contract_version(work)
    group_columns = tuple(c for c in GROUP_COLUMNS.get(test_type, ()) if c in work.columns)
    has_status = "error" in work.columns
    if has_status:
        work["_report_success"] = success_mask_from_error(work["error"]).astype(bool)
    else:
        work["_report_success"] = True

    total = len(work)
    success_count = int(work["_report_success"].sum()) if has_status else None
    overall_ci = wilson_interval(success_count, total) if success_count is not None else None
    available = tuple(spec for spec in METRICS if spec.column in work.columns)

    grouped = (
        work.groupby(list(group_columns), dropna=False, sort=True)
        if group_columns
        else [((), work)]
    )
    records: list[dict[str, Any]] = []
    small_groups = 0
    missing_metrics: set[str] = set()
    for key, group in grouped:
        n = len(group)
        ok = group["_report_success"].astype(bool)
        succeeded = int(ok.sum()) if has_status else None
        interval = wilson_interval(succeeded, n) if succeeded is not None else None
        record: dict[str, Any] = {"Configuration": _group_label(group_columns, key), "Requests": n}
        for column in group_columns:
            record[column] = group.iloc[0][column]
        record["Succeeded"] = succeeded
        record["Success rate (%)"] = succeeded / n * 100 if succeeded is not None else np.nan
        record["Success CI low (%)"] = interval[0] * 100 if interval else np.nan
        record["Success CI high (%)"] = interval[1] * 100 if interval else np.nan
        if n < 20:
            small_groups += 1
        for spec in available:
            values = _positive_success_values(group, spec.column, ok)
            factor = 1000 if spec.column == "tpot" else 1
            prefix = f"{spec.label} ({spec.unit})"
            record[f"{prefix} valid n"] = len(values)
            record[f"{prefix} P50"] = float(values.median() * factor) if len(values) else np.nan
            record[f"{prefix} {spec.tail_label}"] = (
                float(values.quantile(spec.tail) * factor) if len(values) else np.nan
            )
            if len(values) < int(ok.sum()):
                missing_metrics.add(spec.label)
        records.append(record)

    groups = pd.DataFrame.from_records(records)
    notes: list[str] = []
    if not has_status:
        notes.append("Request status is missing; success rate and its interval are unavailable.")
    elif success_count is not None and success_count < total:
        notes.append(
            f"{total - success_count} failed request(s) remain in the success-rate denominator and are excluded from timing summaries."
        )
    if success_count == 0 and available:
        notes.append("No successful requests; timing and throughput summaries are unavailable.")
    if small_groups:
        notes.append(
            f"{small_groups} configuration(s) have fewer than 20 recorded requests; percentiles are descriptive and may be unstable."
        )
    if missing_metrics:
        notes.append(
            "Missing or nonpositive values were excluded from: "
            + ", ".join(sorted(missing_metrics))
            + ". Check each metric's valid n."
        )
    token_sources: tuple[str, ...] = ()
    if "token_calc_method" in work.columns:
        token_sources = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in work["token_calc_method"].dropna()
                    if str(value).strip()
                }
            )
        )
    if len(token_sources) > 1:
        notes.append(
            "Multiple token counting methods occur in this run; compare token-based rates with care."
        )
    notes.append(
        "P50/P95 latency and P10/P50 throughput are empirical request quantiles. The 95% Wilson interval applies only to the request success rate."
    )
    notes.append(
        "Statistics describe recorded rows; compare against planned request counts before treating a run as complete."
    )
    if overall_ci is not None:
        notes.append(
            "The Wilson interval assumes approximately independent request outcomes; correlated failures can make it optimistic."
        )

    findings = _descriptive_findings(groups)
    return AnalysisReport(
        test_type,
        total,
        success_count,
        overall_ci,
        groups,
        tuple(notes),
        contract,
        findings,
        token_sources,
    )


def compact_summary_table(analysis: AnalysisReport) -> pd.DataFrame:
    """Select the core comparisons while preserving the complete table separately."""
    groups = analysis.groups
    columns = [
        "Configuration",
        "Requests",
        "Success rate (%)",
        "TTFT (s) valid n",
        "TTFT (s) P50",
        "TTFT (s) P95",
        "Decode TPS (tokens/s) valid n",
        "Decode TPS (tokens/s) P50",
        "Decode TPS (tokens/s) P10",
    ]
    compact = groups[[column for column in columns if column in groups]].copy()
    if {"Success CI low (%)", "Success CI high (%)"}.issubset(groups.columns):
        compact.insert(
            min(3, len(compact.columns)),
            "Success 95% CI (%)",
            [
                f"{low:.1f}–{high:.1f}" if pd.notna(low) and pd.notna(high) else "—"
                for low, high in zip(
                    groups["Success CI low (%)"], groups["Success CI high (%)"], strict=True
                )
            ],
        )
    return compact


def analysis_to_markdown(analysis: AnalysisReport) -> str:
    """A compact, plain Markdown block for downloaded reports."""
    rate = "Unavailable" if analysis.success_rate is None else f"{analysis.success_rate:.1f}%"
    ci = (
        "Unavailable"
        if analysis.success_ci is None
        else f"{analysis.success_ci[0] * 100:.1f}%–{analysis.success_ci[1] * 100:.1f}%"
    )
    columns = [
        "Configuration",
        "Requests",
        "Succeeded",
        "Success rate (%)",
        "Success CI low (%)",
        "Success CI high (%)",
    ]
    columns += [
        c
        for c in analysis.groups
        if c.endswith(" valid n") or c.endswith(" P50") or c.endswith(" P95") or c.endswith(" P10")
    ]
    columns = [c for c in columns if c in analysis.groups.columns]
    table = analysis.groups[columns].copy()
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: (
                    "—"
                    if pd.isna(value)
                    else (
                        f"{value:,.0f}"
                        if column in ("Requests", "Succeeded") or column.endswith(" valid n")
                        else f"{value:,.3f}" if "TTFT" in column else f"{value:,.2f}"
                    )
                )
            )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(
            str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ") for value in row
        )
        + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    notes = "\n".join(f"- {note}" for note in analysis.notes)
    findings = "\n".join(f"- {finding}" for finding in analysis.findings)
    return (
        "## Statistical summary\n\n"
        f"Recorded requests: **{analysis.rows:,}** · Success rate: **{rate}** "
        f"(95% Wilson interval: {ci}) · Metric contract: `{analysis.metric_contract_version}`\n\n"
        + (
            "Token count source(s): " + ", ".join(analysis.token_sources) + "\n\n"
            if analysis.token_sources
            else ""
        )
        + "\n".join([header, separator, *body])
        + ("\n\n### Observed patterns\n\n" + findings if findings else "")
        + "\n\n### Interpretation notes\n\n"
        + notes
        + "\n\n"
    )
