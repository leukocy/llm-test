"""Self-contained, printable report exports with escaped provenance."""

from __future__ import annotations

from html import escape
from typing import Any


def render_html(job: dict[str, Any], summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    metric = overall["metrics"]
    integrity = summary["integrity"]

    def fmt(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:,.{digits}f}"

    def cell(value: Any) -> str:
        return escape(str(value))

    def rate(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.1f}%"

    rows = "".join(
        f"<tr><th scope='row'>{cell(group['label'])}</th>"
        f"<td>{group['requests']}</td><td>{rate(group['success_rate'])}</td>"
        f"<td>{fmt(group['metrics']['ttft']['median'])}</td>"
        f"<td>{fmt(group['metrics']['ttft']['p95'])}</td>"
        f"<td>{fmt(group['metrics']['tps']['median'])}</td></tr>"
        for group in summary["groups"]
    )
    notes = "".join(f"<li>{cell(note)}</li>" for note in summary["notes"])
    integrity_reasons = "".join(f"<li>{cell(reason)}</li>" for reason in integrity["reasons"])
    integrity_label = (
        "单次运行完整性核验通过" if integrity["verified"] else "仅供诊断 · 未通过完整性核验"
    )
    expected = integrity["expected_requests"]
    integrity_detail = (
        f"数据库已记录 {integrity['recorded_requests']} / 计划 {expected} 次请求"
        if expected is not None
        else f"数据库已记录 {integrity['recorded_requests']} 次请求；计划请求数未知"
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Test 报告 · {cell(job["job_id"])}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, 'Noto Sans SC', system-ui, sans-serif; }}
body {{ margin:0; background:#f5f7fb; color:#17243b; }}
main {{ max-width:1040px; margin:40px auto; padding:48px; background:white; box-shadow:0 12px 45px #13244512; }}
h1 {{ font-size:34px; letter-spacing:-.04em; margin:8px 0; }}
.eyebrow {{ color:#315eaa; font-weight:700; letter-spacing:.16em; font-size:12px; }}
.muted {{ color:#65758e; }}
.grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:32px 0; }}
.card {{ padding:22px; border:1px solid #dce5f2; border-radius:14px; background:#f9fbff; }}
.card b {{ display:block; font-size:27px; margin-top:12px; }}
.integrity {{ padding:18px 22px; border:1px solid; border-radius:12px; margin:26px 0; }}
.integrity strong {{ display:block; font-size:17px; margin-bottom:4px; }}
.integrity p {{ margin:0; }} .integrity ul {{ margin:10px 0 0; padding-left:22px; }}
.integrity.ready {{ background:#edf8f5; border-color:#b7e3d6; color:#286a60; }}
.integrity.limited {{ background:#fff8ea; border-color:#efd6a4; color:#845b16; }}
table {{ border-collapse:collapse; width:100%; margin:24px 0; }}
th,td {{ text-align:left; border-bottom:1px solid #dce5f2; padding:13px 10px; }}
th {{ color:#536780; font-size:13px; }}
code {{ overflow-wrap:anywhere; }}
@media print {{ body {{ background:white; }} main {{ margin:0; box-shadow:none; padding:0; }} }}
</style></head><body><main>
<div class="eyebrow">LLM TEST / MEASUREMENT REPORT</div>
<h1>{cell(job["model_id"])}</h1>
<p class="muted">{cell(job["test_type"])} · {cell(job["status"])} · {cell(job["job_id"])}</p>
<section class="integrity {"ready" if integrity["verified"] else "limited"}">
<strong>{integrity_label}</strong><p>{cell(integrity_detail)}</p>
{"<ul>" + integrity_reasons + "</ul>" if integrity_reasons else ""}</section>
<div class="grid">
<div class="card">请求数<b>{overall["requests"]}</b></div>
<div class="card">成功率<b>{rate(overall["success_rate"])}</b></div>
<div class="card">TTFT p50 · s<b>{fmt(metric["ttft"]["median"])}</b></div>
<div class="card">TTFT p95 · s<b>{fmt(metric["ttft"]["p95"])}</b></div>
</div>
<h2>条件切片</h2><table><thead><tr><th>{cell(summary["group_axis"])}</th><th>请求数</th><th>成功率</th>
<th>TTFT p50 · s</th><th>TTFT p95 · s</th><th>TPS p50</th></tr></thead><tbody>{rows}</tbody></table>
<h2>方法与限制</h2><ul>{notes}</ul>
<p>成功率区间：{cell(overall["success_rate_ci95"])}；指标契约：{cell(summary["metric_contract_version"])}。</p>
<h2>可追溯信息</h2><p>Run ID: <code>{cell(summary["run"]["test_id"])}</code></p>
<p>Token 来源：<code>{cell(summary["provenance"]["token_sources"])}</code></p>
<p>Token 算法：<code>{cell(summary["provenance"]["token_methods"])}</code></p>
<p class="muted">由逐请求记录计算。空值表示样本不足或该指标未采集。</p>
</main></body></html>"""


def render_quality_html(job: dict[str, Any], report: dict[str, Any]) -> str:
    """Printable quality report with sample counts and source provenance."""

    def safe(value: Any) -> str:
        return escape(str(value))

    cards = []
    for name, result in report.get("datasets", {}).items():
        metrics = result.get("extended_metrics") or {}
        provenance = (result.get("config") or {}).get("dataset_provenance") or {}
        accuracy = float(result.get("accuracy") or 0)
        lower = metrics.get("wilson_ci_lower")
        upper = metrics.get("wilson_ci_upper")
        interval = (
            f"{float(lower) * 100:.1f}%–{float(upper) * 100:.1f}%"
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float))
            else "未计算"
        )
        cards.append(
            f"<section class='card'><span class='eyebrow'>DATASET</span><h2>{safe(name)}</h2>"
            f"<strong>{accuracy * 100:.1f}%</strong>"
            f"<p>{safe(result.get('correct_samples', 0))} / {safe(result.get('total_samples', 0))} 正确"
            f" · 95% Wilson 区间 {safe(interval)}</p>"
            f"<dl><dt>数据来源</dt><dd>{safe(provenance.get('source', '未记录'))}</dd>"
            f"<dt>样本 SHA-256</dt><dd><code>{safe(provenance.get('sample_sha256', '未记录'))}</code></dd>"
            f"<dt>Few-shot SHA-256</dt><dd><code>{safe(provenance.get('few_shot_sha256', '未记录'))}</code></dd>"
            f"</dl></section>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Test 质量报告 · {safe(job["job_id"])}</title><style>
:root {{ color-scheme:light; font-family:Inter,'Noto Sans SC',system-ui,sans-serif; }}
body {{ background:#f5f7fb;color:#17243b;margin:0; }}
main {{ max-width:960px;background:white;margin:40px auto;padding:48px;box-shadow:0 12px 45px #13244512; }}
.eyebrow {{ color:#168f79;font-size:11px;font-weight:800;letter-spacing:.16em; }}
h1 {{ font-size:34px;letter-spacing:-.04em;margin:10px 0; }}
.muted {{ color:#75879b; }}
.card {{ border:1px solid #dce7ee;border-radius:14px;padding:27px;margin:20px 0;break-inside:avoid; }}
.card h2 {{ margin:10px 0; }} .card strong {{ display:block;font-size:36px;color:#159478;margin:16px 0; }}
.card p {{ color:#5c7286; }} dl {{ display:grid;grid-template-columns:145px 1fr;gap:12px;border-top:1px solid #e3eaf0;padding-top:17px; }}
dt {{ color:#7c8da0; }} dd {{ margin:0;overflow-wrap:anywhere; }} code {{ overflow-wrap:anywhere; }}
.note {{ background:#edf8f4;padding:18px;border-left:3px solid #2bb69b;line-height:1.6; }}
@media print {{ body {{ background:white; }}main {{ margin:0;padding:0;box-shadow:none; }} }}
</style></head><body><main><span class="eyebrow">LLM TEST / QUALITY REPORT</span>
<h1>{safe(job["model_id"])}</h1><p class="muted">{safe(job["job_id"])} · {safe(job["status"])}</p>
{"".join(cards) if cards else "<p>无可用数据集结果。</p>"}
<p class="note">准确率只反映所列样本与评分口径。比较模型前请核对数据来源、样本指纹、few-shot 设置和样本数量。</p>
</main></body></html>"""
