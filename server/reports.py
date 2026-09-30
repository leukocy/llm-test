"""Self-contained, printable report exports with escaped provenance."""

from __future__ import annotations

import json
from html import escape
from typing import Any

from server.scenario_reports import METRICS, profile_svg, sampling_unit, scenario_analysis
from server.specs import REPORT_ENVIRONMENT_FIELDS
from server.time_series import REQUEST_METRICS, timeline_svg

REPORT_ENVIRONMENT_SCOPES = {
    "model_server": "受测模型服务器",
    "test_client": "测试客户端",
    "unspecified": "未明确对象",
}


def report_environment_html(environment: dict[str, Any] | None) -> str:
    if not environment or not environment.get("fields"):
        return ""
    scope = REPORT_ENVIRONMENT_SCOPES.get(str(environment.get("scope")), "未明确对象")
    rows = "".join(
        f"<tr><th scope='row'>{label}</th><td style='overflow-wrap:anywhere'>"
        f"{escape(str(environment['fields'][key]))}</td></tr>"
        for key, label in REPORT_ENVIRONMENT_FIELDS.items()
        if environment["fields"].get(key)
    )
    return (
        f"<section><h2>报告环境信息</h2><p>对象：{scope} · 来源：用户填写，未经自动核验。"
        "自动硬件快照来自执行端，与本表分别记录。</p>"
        f"<table><tbody>{rows}</tbody></table></section>"
    )


def report_environment_markdown(environment: dict[str, Any] | None) -> str:
    if not environment or not environment.get("fields"):
        return ""
    scope = REPORT_ENVIRONMENT_SCOPES.get(str(environment.get("scope")), "未明确对象")
    lines = [
        "",
        "## 报告环境信息",
        "",
        f"对象：{scope} · 来源：用户填写，未经自动核验。自动硬件快照来自执行端，与本表分别记录。",
        "",
        "| 项目 | 用户填写的值 |",
        "|---|---|",
    ]
    for key, label in REPORT_ENVIRONMENT_FIELDS.items():
        value = environment["fields"].get(key)
        if value:
            safe = _md(value)
            for char in "`*_[]":
                safe = safe.replace(char, f"\\{char}")
            lines.append(f"| {label} | {safe} |")
    return "\n".join(lines) + "\n"


def _md(value: Any) -> str:
    return escape(
        (
            str(value if value is not None else "—")
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\n", " ")
        ),
        quote=False,
    )


def tokenizer_installation_html(installation: dict[str, Any] | None) -> str:
    if not installation:
        return ""
    rows = "".join(
        f"<tr><td>{escape(item['name'])}</td><td>{item['size']:,}</td>"
        f"<td style='overflow-wrap:anywhere'><code>{item['sha256']}</code></td></tr>"
        for item in installation["files"]
    )
    return (
        "<section><h2>Tokenizer 安装来源</h2><p>公开 Hugging Face 仓库："
        f"{escape(installation['repo_id'])} · 名称：{escape(installation['name'])}</p>"
        f"<p>固定版本：<code>{installation['revision']}</code> · 离线编码校验通过，禁止执行远程代码。</p>"
        "<p>该记录描述输入校准和本地计数器；最终 Token 指标仍以逐请求记录的来源为准。</p>"
        f"<table><thead><tr><th>文件</th><th>字节</th><th>SHA-256</th></tr></thead><tbody>{rows}</tbody></table></section>"
    )


def tokenizer_installation_markdown(installation: dict[str, Any] | None) -> str:
    if not installation:
        return ""
    lines = [
        "",
        "## Tokenizer 安装来源",
        "",
        f"- 仓库：{installation['repo_id']}；名称：{installation['name']}",
        f"- 固定版本：`{installation['revision']}`；离线编码校验通过，禁止执行远程代码。",
        "- 该记录描述输入校准和本地计数器；最终 Token 指标仍以逐请求记录的来源为准。",
        "",
        "| 文件 | 字节 | SHA-256 |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| {item['name']} | {item['size']} | `{item['sha256']}` |"
        for item in installation["files"]
    )
    return "\n".join(lines) + "\n"


def render_markdown(job: dict[str, Any], summary: dict[str, Any]) -> str:
    """Portable performance report with the same integrity and metric contract as HTML."""
    overall = summary["overall"]
    metrics = overall["metrics"]
    integrity = summary["integrity"]
    protocol = summary.get("measurement_protocol") or {}
    control = summary.get("execution_control") or {}
    historical = summary.get("origin", {}).get("kind") == "saved_csv"

    def number(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:,.{digits}f}"

    def rate(value: Any) -> str:
        return "—" if value is None else f"{value * 100:.1f}%"

    def token_list(values: list[str]) -> str:
        safe = _md(", ".join(values))
        for char in "`*_[]":
            safe = safe.replace(char, f"\\{char}")
        return safe

    lines = [
        "# LLM Test 历史 CSV 报告"
        if historical
        else f"# LLM Test 测量报告 · {_md(job['model_id'])}",
        "",
        f"- 作业 ID：`{_md(job['job_id'])}`",
        f"- 测试类型：{_md(job['test_type'])}",
        f"- 状态：{_md(job['status'])}",
        f"- 指标契约：{_md(summary['metric_contract_version'])}",
        "",
        "## 完整性",
        "",
        f"**{'通过' if integrity['verified'] else '仅供诊断，未通过'}**。"
        f"记录 {integrity['recorded_requests']} / "
        f"{_md(integrity['expected_requests'])} 次计划请求。",
    ]
    lines.extend(f"- {_md(reason)}" for reason in integrity["reasons"])
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            "| 请求 | 成功率 | TTFT p50 (s) | TTFT p95 (s) | TPS p50 (token/s) |",
            "|---:|---:|---:|---:|---:|",
            f"| {overall['requests']} | {rate(overall['success_rate'])} | "
            f"{number(metrics['ttft']['median'])} | {number(metrics['ttft']['p95'])} | "
            f"{number(metrics['tps']['median'], 1)} |",
            "",
            "## 条件切片",
            "",
            f"| {_md(summary['group_axis'])} | 正式 / 计划 | 失败 | 成功率 | 输入 token 中位数 | TTFT p50 (s) | TTFT p95 (s) | TPS p50 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["groups"]:
        lines.append(
            f"| {_md(group['label'])} | {group['requests']} / {_md(group.get('planned_requests'))} "
            f"| {group['failures']} | {rate(group['success_rate'])} "
            f"| {number((group.get('input_tokens') or {}).get('median'), 1)} "
            f"| {number(group['metrics']['ttft']['median'])} "
            f"| {number(group['metrics']['ttft']['p95'])} "
            f"| {number(group['metrics']['tps']['median'], 1)} |"
        )
    lines.extend(
        [
            "",
            "## 测量方法与来源",
            "",
            f"- 协议版本：{_md(protocol.get('protocol_version'))}",
            f"- 负载模型：{_md(protocol.get('workload_model'))}",
            f"- 正式请求：{_md(protocol.get('measured_requests'))}；预热请求：{_md(protocol.get('warmup_requests'))}",
            f"- Token 来源：{token_list(summary['provenance']['token_sources'])}",
            f"- Token 算法：{token_list(summary['provenance']['token_methods'])}",
            "- 原运行的暂停、批次并行与失败策略未经核验。"
            if historical
            else f"- 暂停：{control.get('pause_count', 0)} 次，共 {number(control.get('paused_seconds', 0))} 秒；仅在请求组之间暂停",
            "- 此处不提供新测量的执行完整性证明。"
            if historical
            else f"- 批次并行上限：{control.get('max_parallel', 1)}；失败即停：{'开启' if control.get('stop_on_error') else '关闭'}",
        ]
    )
    lines.extend(f"- {_md(note)}" for note in summary["notes"])
    lines.extend(
        f"- 数据质量：{_md(note)}" for note in summary.get("data_quality", {}).get("warnings", [])
    )
    analysis = scenario_analysis(summary)
    lines.extend(["", "## 场景分析：" + analysis["title"], ""])
    lines.extend("- " + _md(item["text"]) for item in analysis["observations"])
    lines.extend("- " + note for note in analysis["notes"])
    lines.extend(
        [
            "",
            "### 完整指标切片",
            "",
            "| 条件 | 指标 | 有效 n | 采样单位 | 均值 | p50 | p95 | p99 | 最小 | 最大 | 单位 |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for group in summary["groups"]:
        for key, (_, abbreviation, unit) in METRICS.items():
            if key not in group["metrics"]:
                continue
            value = group["metrics"][key]
            stats = " | ".join(
                number(value[stat]) for stat in ["mean", "median", "p95", "p99", "min", "max"]
            )
            lines.append(
                f"| {_md(group['label'])} | {abbreviation} | {value['count']} | {sampling_unit(key)} | {stats} | {unit} |"
            )
    timeline = summary.get("time_series")
    if timeline:
        lines.extend(["", "## 稳定性完成时间序列", "", _timeline_description(timeline), ""])
        lines.extend("- " + _md(note) for note in timeline["notes"])
        lines.extend(
            [
                "",
                "| 时间窗 (s) | 完成 | 失败 | 指标 | 有效 n | 均值 | p50 | p95 | p99 | 最小 | 最大 |",
                "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        lines.extend(
            "| " + " | ".join(_md(v) for v in row) + " |" for row in _timeline_rows(timeline)
        )
    return (
        "\n".join(lines)
        + "\n"
        + history_origin_markdown(summary)
        + checkpoint_markdown(summary)
        + report_environment_markdown(summary.get("report_environment"))
        + tokenizer_installation_markdown(summary.get("tokenizer_installation"))
    )


def checkpoint_markdown(report: dict[str, Any]) -> str:
    checkpoint = report.get("checkpoint")
    if not checkpoint:
        return ""
    return (
        f"\n## {_md(checkpoint.get('unit_label', '样本'))}检查点与执行中断\n\n"
        f"- 已提交 {checkpoint['committed_units']} / {checkpoint['planned_units']} 个{_md(checkpoint.get('unit_label', '样本'))}；"
        f"恢复 {checkpoint['recoveries']} 次；重复发起 {checkpoint['repeated_unit_attempts']} 次。\n"
        + "\n".join("- " + _md(note) for note in checkpoint.get("notes", []))
        + "\n"
        + f"- 暂停 {report.get('execution_control', {}).get('pause_count', 0)} 次；"
        f"累计暂停 {report.get('execution_control', {}).get('paused_seconds', 0):.3f} 秒。\n"
    )


def checkpoint_html(report: dict[str, Any]) -> str:
    checkpoint = report.get("checkpoint")
    if not checkpoint:
        return ""
    return (
        f"<section><h2>{escape(str(checkpoint.get('unit_label', '样本')))}检查点与执行中断</h2>"
        f"<p>已提交 {int(checkpoint['committed_units'])} / {int(checkpoint['planned_units'])} 个{escape(str(checkpoint.get('unit_label', '样本')))}；"
        f"恢复 {int(checkpoint['recoveries'])} 次；重复发起 {int(checkpoint['repeated_unit_attempts'])} 次。</p>"
        + "<ul>"
        + "".join("<li>" + escape(str(note)) + "</li>" for note in checkpoint.get("notes", []))
        + "</ul>"
        + f"<p>暂停 {int(report.get('execution_control', {}).get('pause_count', 0))} 次；"
        f"累计暂停 {report.get('execution_control', {}).get('paused_seconds', 0):.3f} 秒。</p></section>"
    )


def render_quality_markdown(job: dict[str, Any], report: dict[str, Any]) -> str:
    lines = [
        f"# LLM Test 质量报告 · {_md(job['model_id'])}",
        "",
        f"- 作业 ID：`{_md(job['job_id'])}`",
        f"- 状态：{_md(job['status'])}",
        "",
        "| 数据集 | 正确 / 总数 | 准确率 | 样本 SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for name, result in report.get("datasets", {}).items():
        provenance = (result.get("config") or {}).get("dataset_provenance") or {}
        lines.append(
            f"| {_md(name)} | {result.get('correct_samples', 0)} / "
            f"{result.get('total_samples', 0)} | {float(result.get('accuracy') or 0) * 100:.1f}% "
            f"| {_md(provenance.get('sample_sha256'))} |"
        )
    return (
        "\n".join(lines)
        + "\n"
        + checkpoint_markdown(report)
        + report_environment_markdown(report.get("report_environment"))
    )


def render_robustness_markdown(job: dict[str, Any], report: dict[str, Any]) -> str:
    robustness = report["robustness"]
    lines = [
        f"# LLM Test 鲁棒性报告 · {_md(job['model_id'])}",
        "",
        f"- 作业 ID：`{_md(job['job_id'])}`",
        f"- 原始准确率：{float(robustness.get('original_accuracy') or 0) * 100:.1f}%",
        f"- 扰动后准确率：{float(robustness.get('perturbed_accuracy') or 0) * 100:.1f}%",
        f"- 指标契约：{_md(robustness.get('metric_contract_version') or '未记录，可能使用旧公式；对比前请重新评测')}",
        f"- 总体鲁棒性：{float(robustness.get('overall_robustness') or 0):.3f}",
        "",
        "| 扰动类型 | 错误率（越高越敏感） |",
        "|---|---:|",
    ]
    for name, score in (robustness.get("sensitivity_by_type") or {}).items():
        lines.append(f"| {_md(name)} | {float(score):.3f} |")
    return (
        "\n".join(lines)
        + "\n"
        + checkpoint_markdown(report)
        + report_environment_markdown(report.get("report_environment"))
    )


def render_html(job: dict[str, Any], summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    metric = overall["metrics"]
    integrity = summary["integrity"]
    protocol = summary.get("measurement_protocol")

    def fmt(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{value:,.{digits}f}"

    def cell(value: Any) -> str:
        return escape(str(value))

    def rate(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.1f}%"

    rows = "".join(
        f"<tr><th scope='row'>{cell(group['label'])}</th>"
        f"<td>{group['requests']} / {cell(group.get('planned_requests') or '—')}</td>"
        f"<td>{rate(group['success_rate'])}</td>"
        f"<td>{fmt(group.get('input_tokens', {}).get('median'), 1)}</td>"
        f"<td>{fmt(group['metrics']['ttft']['median'])}</td>"
        f"<td>{fmt(group['metrics']['ttft']['p95'])}</td>"
        f"<td>{fmt(group['metrics']['tps']['median'])}</td></tr>"
        for group in summary["groups"]
    )
    notes = "".join(f"<li>{cell(note)}</li>" for note in summary["notes"])
    integrity_reasons = "".join(f"<li>{cell(reason)}</li>" for reason in integrity["reasons"])
    quality_warnings = "".join(
        f"<li>{cell(reason)}</li>" for reason in summary.get("data_quality", {}).get("warnings", [])
    )
    protocol_html = (
        f"<h2>测量协议</h2><p>版本：{cell(protocol.get('protocol_version'))}；"
        f"负载模型：{cell(protocol.get('workload_model'))}；"
        f"正式请求：{cell(protocol.get('measured_requests'))}；"
        f"预热：{cell(protocol.get('warmup_recorded'))} / "
        f"{cell(protocol.get('warmup_requests'))} 次。</p>"
        if protocol
        else "<h2>测量协议</h2><p>历史运行未记录固定工作负载协议。</p>"
    )
    control = summary.get("execution_control") or {}
    control_html = checkpoint_html(summary) + (
        f"<h2>执行条件</h2><p>暂停 {cell(control.get('pause_count', 0))} 次，"
        f"共 {fmt(control.get('paused_seconds', 0))} 秒；"
        f"批次并行上限 {cell(control.get('max_parallel', 1))}；"
        f"失败即停{'开启' if control.get('stop_on_error') else '关闭'}。</p>"
    )
    if summary.get("origin", {}).get("kind") == "saved_csv":
        control_html = "<h2>执行条件</h2><p>原运行的暂停、批次并行与失败策略未经核验。</p>"
    integrity_label = (
        "请求记录完整性核验通过" if integrity["verified"] else "仅供诊断 · 未通过完整性核验"
    )
    expected = integrity["expected_requests"]
    record_source = "CSV" if summary.get("origin", {}).get("kind") == "saved_csv" else "数据库"
    integrity_detail = (
        f"{record_source}已记录 {integrity['recorded_requests']} / 计划 {expected} 次请求"
        if expected is not None
        else f"{record_source}已记录 {integrity['recorded_requests']} 次请求；计划请求数未知"
    )
    analysis = scenario_analysis(summary)
    scenario_html = (
        f"<h2>场景分析：{cell(analysis['title'])}</h2><ul>"
        + "".join(f"<li>{cell(item['text'])}</li>" for item in analysis["observations"])
        + "".join(f"<li>{cell(note)}</li>" for note in analysis["notes"])
        + "</ul><h2>指标图形 · p50</h2>"
        + "".join(profile_svg(summary, key) for key in METRICS)
    )
    timeline = summary.get("time_series")
    if timeline:
        timeline_rows = "".join(
            "<tr>" + "".join(f"<td>{cell(value)}</td>" for value in row) + "</tr>"
            for row in _timeline_rows(timeline)
        )
        scenario_html += (
            "<section><h2>稳定性完成时间序列</h2><p>"
            + cell(_timeline_description(timeline))
            + "</p><ul>"
            + "".join(f"<li>{cell(note)}</li>" for note in timeline["notes"])
            + "</ul>"
            + "".join(
                timeline_svg(timeline, key, METRICS[key][1], METRICS[key][2])
                for key in REQUEST_METRICS
            )
            + "<div style='overflow-x:auto'><table><thead><tr>"
            + "".join(
                f"<th>{label}</th>"
                for label in [
                    "时间窗 (s)",
                    "完成",
                    "失败",
                    "指标",
                    "有效 n",
                    "均值",
                    "p50",
                    "p95",
                    "p99",
                    "最小",
                    "最大",
                ]
            )
            + "</tr></thead><tbody>"
            + timeline_rows
            + "</tbody></table></div></section>"
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
figure {{ margin:24px 0; break-inside:avoid; }} figure svg {{ width:100%; height:auto; }}
figcaption {{ font-size:13px; color:#536780; }}
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
<th>输入 token 中位数</th><th>TTFT p50 · s</th><th>TTFT p95 · s</th><th>TPS p50</th></tr></thead><tbody>{rows}</tbody></table>
{protocol_html}
{control_html}
{history_origin_html(summary)}
{report_environment_html(summary.get("report_environment"))}
{tokenizer_installation_html(summary.get("tokenizer_installation"))}
{scenario_html}
{"<h2>数据质量提示</h2><ul>" + quality_warnings + "</ul>" if quality_warnings else ""}
<h2>方法与限制</h2><ul>{notes}</ul>
<p>成功率区间：{cell(overall["success_rate_ci95"])}；指标契约：{cell(summary["metric_contract_version"])}。</p>
<h2>可追溯信息</h2><p>Run ID: <code>{cell(summary["run"]["test_id"])}</code></p>
<p>Token 来源：<code>{cell(summary["provenance"]["token_sources"])}</code></p>
<p>Token 算法：<code>{cell(summary["provenance"]["token_methods"])}</code></p>
<p class="muted">由逐请求记录计算。空值表示样本不足或该指标未采集。</p>
</main></body></html>"""


def _timeline_description(timeline: dict[str, Any]) -> str:
    window = timeline["window_seconds"]
    window_text = f"{window:.3f}" if window is not None else "未知"
    return (
        f"stability-clock-v1 · 窗口状态：{timeline.get('window_state') or '历史记录'}；"
        f"计时记录完整：{'是' if timeline['complete'] else '否'}；"
        f"有效计时 {timeline['timed_requests']}，缺失 {timeline['missing_requests']}，无效 {timeline['invalid_requests']}；"
        f"计划发起 {timeline['planned_seconds'] if timeline['planned_seconds'] is not None else '未知'} 秒，"
        f"调度至排空 {window_text} 秒；每窗 {timeline['bin_seconds'] if timeline['bin_seconds'] is not None else '未知'} 秒。"
    )


def _timeline_rows(timeline: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for b in timeline["bins"]:
        for key in REQUEST_METRICS:
            stats = b["metrics"][key]
            rows.append(
                [
                    f"{b['start_seconds']:.3f}–{b['end_seconds']:.3f}",
                    b["requests"],
                    b["failures"],
                    f"{METRICS[key][1]} ({METRICS[key][2]})",
                    stats["count"],
                    *(
                        "—" if stats[s] is None else f"{stats[s]:.4g}"
                        for s in ["mean", "median", "p95", "p99", "min", "max"]
                    ),
                ]
            )
    return rows


def history_origin_html(summary: dict[str, Any]) -> str:
    origin = summary.get("origin")
    if not origin or origin.get("kind") != "saved_csv":
        return ""
    context = escape(json.dumps({"run": summary["run"], **origin}, ensure_ascii=False, indent=2))
    unknown = summary["overall"].get("unknown_outcomes", 0)
    return (
        "<section><h2>历史文件来源与配置</h2><p>由保存的 CSV 重新计算，未重新运行测量。"
        f"未知成功状态：{unknown} 次；成功率的分母仅含已知状态的请求。"
        "配置与硬件说明来自配套元数据，未经核验；哈希仅用于识别本次读取的文件内容。</p>"
        f"<pre style='white-space:pre-wrap;overflow-wrap:anywhere'>{context}</pre></section>"
    )


def history_origin_markdown(summary: dict[str, Any]) -> str:
    origin = summary.get("origin")
    if not origin or origin.get("kind") != "saved_csv":
        return ""
    context = json.dumps({"run": summary["run"], **origin}, ensure_ascii=False, indent=2)
    return (
        "\n## 历史文件来源与配置\n\n由保存的 CSV 重新计算，未重新运行测量。"
        f"未知成功状态：{summary['overall'].get('unknown_outcomes', 0)} 次；成功率的分母仅含已知状态的请求。"
        "配置与硬件说明来自配套元数据，未经核验；哈希仅用于识别本次读取的文件内容。\n\n"
        f"```json\n{context}\n```\n"
    )


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
{checkpoint_html(report)}
{report_environment_html(report.get("report_environment"))}
<p class="note">准确率只反映所列样本与评分口径。比较模型前请核对数据来源、样本指纹、few-shot 设置和样本数量。</p>
</main></body></html>"""
