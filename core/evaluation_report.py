"""
Phase 5: 增强Evaluation报告 (Enhanced Evaluation Report)

提供详细Evaluation报告结构and多种Export格式：
- JSON 格式（机器可读）
- Markdown 格式（人类可读）
- HTML 格式（Web 展示）
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import ThinkingMetricsResult
from .response_parser import ParsedResponse


@dataclass
class ModelConfig:
    """Model Configuration"""
    platform: str = ""
    model_id: str = ""
    api_base_url: str = ""
    thinking_enabled: bool = False
    thinking_budget: int | None = None
    reasoning_effort: str = "medium"
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class LatencyMetrics:
    """Latency指标"""
    ttft_ms: float | None = None
    ttr_ms: float | None = None
    ttut_ms: float | None = None
    total_ms: float | None = None
    reasoning_phase_ms: float | None = None


@dataclass
class TokenMetrics:
    """Token 指标"""
    prompt_tokens: int = 0
    reasoning_tokens: int = 0
    content_tokens: int = 0
    total_tokens: int = 0
    reasoning_ratio: float = 0.0


@dataclass
class QualityMetrics:
    """质量指标"""
    accuracy_score: float | None = None
    reasoning_coherence: float | None = None
    response_completeness: float | None = None
    overall_score: float | None = None


@dataclass
class CostMetrics:
    """成本指标"""
    estimated_cost_usd: float | None = None
    quality_per_dollar: float | None = None
    cost_per_1k_tokens: float | None = None


@dataclass
class RawData:
    """Raw data"""
    prompt: str = ""
    system_prompt: str = ""
    reasoning_content: str = ""
    final_content: str = ""
    stream_snapshots: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class EvaluationReport:
    """Evaluation报告"""
    # 元信息
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"

    # Config
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # 指标
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)

    # Raw data
    raw_data: RawData = field(default_factory=RawData)

    # 额外信息
    notes: str = ""
    tags: list[str] = field(default_factory=list)


class ReportBuilder:
    """
    Evaluation报告Build器

    Usage:
        builder = ReportBuilder()
        builder.set_model_config(platform="mimo", model_id="mimo-v2-flash", ...)
        builder.set_prompt("Test问题")
        builder.set_response(parsed_response)
        builder.set_metrics(thinking_metrics_result)
        builder.set_quality_scores(accuracy=8.5, coherence=9.0)

        report = builder.build()

        # Export
        builder.export_json("report.json")
        builder.export_markdown("report.md")
    """

    def __init__(self):
        self._report = EvaluationReport()

    def set_model_config(
        self,
        platform: str = "",
        model_id: str = "",
        api_base_url: str = "",
        thinking_enabled: bool = False,
        thinking_budget: int | None = None,
        reasoning_effort: str = "medium",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> "ReportBuilder":
        """SetModel Configuration"""
        self._report.model_config = ModelConfig(
            platform=platform,
            model_id=model_id,
            api_base_url=api_base_url,
            thinking_enabled=thinking_enabled,
            thinking_budget=thinking_budget,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return self

    def set_prompt(self, prompt: str, system_prompt: str = "") -> "ReportBuilder":
        """SetTip词"""
        self._report.raw_data.prompt = prompt
        self._report.raw_data.system_prompt = system_prompt
        return self

    def set_response(self, parsed: ParsedResponse) -> "ReportBuilder":
        """SetParse后响应"""
        self._report.raw_data.reasoning_content = parsed.full_reasoning
        self._report.raw_data.final_content = parsed.full_content
        self._report.raw_data.stream_snapshots = parsed.raw_snapshots
        return self

    def set_metrics(self, metrics: ThinkingMetricsResult) -> "ReportBuilder":
        """SetMetric calculationResult"""
        # Latency指标
        self._report.latency = LatencyMetrics(
            ttft_ms=metrics.ttft_ms,
            ttr_ms=metrics.ttr_ms,
            ttut_ms=metrics.ttut_ms,
            total_ms=metrics.total_time_ms,
            reasoning_phase_ms=metrics.reasoning_time_ms
        )

        # Token 指标
        self._report.tokens = TokenMetrics(
            reasoning_tokens=metrics.reasoning_tokens,
            content_tokens=metrics.content_tokens,
            total_tokens=metrics.total_tokens,
            reasoning_ratio=metrics.reasoning_ratio
        )

        # 成本指标
        self._report.cost = CostMetrics(
            estimated_cost_usd=metrics.estimated_cost_usd,
            quality_per_dollar=metrics.quality_per_dollar
        )

        return self

    def set_usage(self, usage: dict[str, Any]) -> "ReportBuilder":
        """Set Token use信息"""
        self._report.raw_data.usage = usage
        if usage:
            self._report.tokens.prompt_tokens = usage.get("prompt_tokens", 0)
            if not self._report.tokens.total_tokens:
                self._report.tokens.total_tokens = usage.get("total_tokens", 0)
        return self

    def set_quality_scores(
        self,
        accuracy: float | None = None,
        coherence: float | None = None,
        completeness: float | None = None,
        overall: float | None = None
    ) -> "ReportBuilder":
        """Set质量分数"""
        self._report.quality = QualityMetrics(
            accuracy_score=accuracy,
            reasoning_coherence=coherence,
            response_completeness=completeness,
            overall_score=overall or (
                (accuracy or 0 + coherence or 0 + completeness or 0) / 3
                if any([accuracy, coherence, completeness]) else None
            )
        )
        return self

    def set_error(self, error: str) -> "ReportBuilder":
        """SetError message"""
        self._report.raw_data.error = error
        return self

    def add_note(self, note: str) -> "ReportBuilder":
        """Add备注"""
        self._report.notes = note
        return self

    def add_tags(self, *tags: str) -> "ReportBuilder":
        """AddLabel"""
        self._report.tags.extend(tags)
        return self

    def build(self) -> EvaluationReport:
        """Build报告"""
        return self._report

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        return asdict(self._report)

    def export_json(self, filepath: str, indent: int = 2) -> str:
        """
        Export JSON 格式

        Args:
            filepath: File path
            indent: 缩进空格数

        Returns:
            ExportFile path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)

        return str(path)

    def export_markdown(self, filepath: str) -> str:
        """
        Export Markdown 格式

        Args:
            filepath: File path

        Returns:
            ExportFile path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = self._report
        config = report.model_config
        latency = report.latency
        tokens = report.tokens
        quality = report.quality
        cost = report.cost
        raw = report.raw_data

        # Build Markdown
        lines = [
            "# ModelEvaluation报告",
            "",
            f"**Test ID**: `{report.test_id}`",
            f"**时间**: {report.timestamp}",
            f"**Version**: {report.version}",
            "",
            "---",
            "",
            "## 1. Model Configuration",
            "",
            "|  items目 | 值 |",
            "|------|-----|",
            f"| 平台 | {config.platform} |",
            f"| Model | {config.model_id} |",
            f"| Thinking mode | {'启用' if config.thinking_enabled else '禁用'} |",
            f"| Thinking budget | {config.thinking_budget or 'N/A'} |",
            f"| 推理强度 | {config.reasoning_effort} |",
            f"| 温度 | {config.temperature} |",
            f"| 最大Token | {config.max_tokens} |",
            "",
            "---",
            "",
            "## 2. Latency指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| TTFT (首Token) | {f'{latency.ttft_ms:.0f}ms' if latency.ttft_ms else 'N/A'} |",
            f"| TTR (首推理) | {f'{latency.ttr_ms:.0f}ms' if latency.ttr_ms else 'N/A'} |",
            f"| TTUT (首正文) | {f'{latency.ttut_ms:.0f}ms' if latency.ttut_ms else 'N/A'} |",
            f"| 总耗时 | {f'{latency.total_ms:.0f}ms' if latency.total_ms else 'N/A'} |",
            f"| 推理阶段 | {f'{latency.reasoning_phase_ms:.0f}ms' if latency.reasoning_phase_ms else 'N/A'} |",
            "",
            "---",
            "",
            "## 3. Token Statistics",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 输入 Token | {tokens.prompt_tokens} |",
            f"| 推理 Token | {tokens.reasoning_tokens} |",
            f"| 正文 Token | {tokens.content_tokens} |",
            f"| 总 Token | {tokens.total_tokens} |",
            f"| 推理占比 | {f'{tokens.reasoning_ratio:.1%}' if tokens.reasoning_ratio else 'N/A'} |",
            "",
            "---",
            "",
            "## 4. 质量Score",
            "",
            "| 指标 | 分数 |",
            "|------|------|",
            f"| 准确性 | {f'{quality.accuracy_score:.1f}/10' if quality.accuracy_score else 'N/A'} |",
            f"| 推理连贯性 | {f'{quality.reasoning_coherence:.1f}/10' if quality.reasoning_coherence else 'N/A'} |",
            f"| 完整性 | {f'{quality.response_completeness:.1f}/10' if quality.response_completeness else 'N/A'} |",
            f"| **综合分** | {f'{quality.overall_score:.1f}/10' if quality.overall_score else 'N/A'} |",
            "",
            "---",
            "",
            "## 5. 成本分析",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 预估成本 | {f'${cost.estimated_cost_usd:.6f}' if cost.estimated_cost_usd else 'N/A'} |",
            f"| 质量/美元 | {f'{cost.quality_per_dollar:.2f}' if cost.quality_per_dollar else 'N/A'} |",
            "",
            "---",
            "",
            "## 6. 原始内容",
            "",
            "### 6.1 Tip词",
            "",
            "```",
            raw.prompt[:500] + ("..." if len(raw.prompt) > 500 else ""),
            "```",
            "",
            "### 6.2 Reasoning process",
            "",
            "```",
            raw.reasoning_content[:1000] + ("..." if len(raw.reasoning_content) > 1000 else "") if raw.reasoning_content else "(no)",
            "```",
            "",
            "### 6.3 最终输出",
            "",
            "```",
            raw.final_content[:1000] + ("..." if len(raw.final_content) > 1000 else "") if raw.final_content else "(no)",
            "```",
            "",
        ]

        if raw.error:
            lines.extend([
                "### 6.4 Error message",
                "",
                "```",
                raw.error,
                "```",
                "",
            ])

        if report.notes:
            lines.extend([
                "---",
                "",
                "## 7. 备注",
                "",
                report.notes,
                "",
            ])

        if report.tags:
            lines.extend([
                "---",
                "",
                f"**Label**: {', '.join(report.tags)}",
                "",
            ])

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return str(path)

    def export_html(self, filepath: str) -> str:
        """
        Export HTML 格式

        Args:
            filepath: File path

        Returns:
            ExportFile path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = self._report
        config = report.model_config
        latency = report.latency
        tokens = report.tokens
        quality = report.quality
        cost = report.cost

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModelEvaluation报告 - {config.model_id}</title>
    <style>
        :root {{
            --primary: #6366f1;
            --bg: #0f172a;
            --card: #1e293b;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --border: #334155;
            --success: #22c55e;
            --warning: #eab308;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        .meta {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 2rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
        .card {{
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
        }}
        .card h2 {{ font-size: 1rem; color: var(--muted); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .metric {{ display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid var(--border); }}
        .metric:last-child {{ border-bottom: none; }}
        .metric .label {{ color: var(--muted); }}
        .metric .value {{ font-weight: 600; }}
        .score {{ font-size: 2rem; font-weight: 700; color: var(--success); }}
        .content-block {{ background: var(--card); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid var(--border); }}
        .content-block h3 {{ margin-bottom: 1rem; }}
        pre {{ background: var(--bg); padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.875rem; white-space: pre-wrap; }}
        .tags {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
        .tag {{ background: var(--primary); color: white; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ModelEvaluation报告</h1>
        <p class="meta">
            <strong>{config.platform}</strong> / {config.model_id}<br>
            Test时间: {report.timestamp}
        </p>

        <div class="grid">
            <div class="card">
                <h2>Latency指标</h2>
                <div class="metric"><span class="label">TTFT</span><span class="value">{f'{latency.ttft_ms:.0f}ms' if latency.ttft_ms else 'N/A'}</span></div>
                <div class="metric"><span class="label">TTR</span><span class="value">{f'{latency.ttr_ms:.0f}ms' if latency.ttr_ms else 'N/A'}</span></div>
                <div class="metric"><span class="label">TTUT</span><span class="value">{f'{latency.ttut_ms:.0f}ms' if latency.ttut_ms else 'N/A'}</span></div>
                <div class="metric"><span class="label">总耗时</span><span class="value">{f'{latency.total_ms:.0f}ms' if latency.total_ms else 'N/A'}</span></div>
            </div>

            <div class="card">
                <h2>Token Statistics</h2>
                <div class="metric"><span class="label">输入</span><span class="value">{tokens.prompt_tokens}</span></div>
                <div class="metric"><span class="label">推理</span><span class="value">{tokens.reasoning_tokens}</span></div>
                <div class="metric"><span class="label">正文</span><span class="value">{tokens.content_tokens}</span></div>
                <div class="metric"><span class="label">推理占比</span><span class="value">{f'{tokens.reasoning_ratio:.1%}' if tokens.reasoning_ratio else 'N/A'}</span></div>
            </div>

            <div class="card">
                <h2>质量Score</h2>
                <div class="score">{f'{quality.overall_score:.1f}' if quality.overall_score else 'N/A'}</div>
                <div class="metric"><span class="label">准确性</span><span class="value">{f'{quality.accuracy_score:.1f}' if quality.accuracy_score else 'N/A'}</span></div>
                <div class="metric"><span class="label">连贯性</span><span class="value">{f'{quality.reasoning_coherence:.1f}' if quality.reasoning_coherence else 'N/A'}</span></div>
            </div>

            <div class="card">
                <h2>成本分析</h2>
                <div class="metric"><span class="label">预估成本</span><span class="value">{f'${cost.estimated_cost_usd:.6f}' if cost.estimated_cost_usd else 'N/A'}</span></div>
                <div class="metric"><span class="label">质量/美元</span><span class="value">{f'{cost.quality_per_dollar:.2f}' if cost.quality_per_dollar else 'N/A'}</span></div>
            </div>
        </div>

        {'<div class="tags">' + ''.join([f'<span class="tag">{t}</span>' for t in report.tags]) + '</div>' if report.tags else ''}
    </div>
</body>
</html>"""

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)

        return str(path)


def create_report_from_test(
    platform: str,
    model_id: str,
    prompt: str,
    parsed_response: ParsedResponse,
    metrics: ThinkingMetricsResult,
    quality_scores: dict[str, float] | None = None,
    **config_kwargs
) -> EvaluationReport:
    """
    便捷函数：从Test ResultsCreate报告

    Args:
        platform: 平台标识
        model_id: ModelID
        prompt: Tip词
        parsed_response: Parse后响应
        metrics: Metric calculationResult
        quality_scores: 质量分数字典
        **config_kwargs: otherModel Configuration

    Returns:
        EvaluationReport: Evaluation报告
    """
    builder = ReportBuilder()
    builder.set_model_config(platform=platform, model_id=model_id, **config_kwargs)
    builder.set_prompt(prompt)
    builder.set_response(parsed_response)
    builder.set_metrics(metrics)

    if quality_scores:
        builder.set_quality_scores(**quality_scores)

    return builder.build()
