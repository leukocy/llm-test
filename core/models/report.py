"""
报告Model
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ReportType(Enum):
    """报告类型"""
    STANDARD = "standard"
    EVALUATION = "evaluation"
    FAILURE_ANALYSIS = "failure_analysis"
    PERFORMANCE = "performance"


@dataclass
class Report:
    """报告Model"""

    # 主键
    id: int | None = None

    # 唯一标识
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # 关联Test运行
    run_id: int | None = None

    # 基本信息
    report_type: str = ReportType.STANDARD.value
    version: str = "1.0"

    # Model信息
    model_id: str = ""
    model_type: str | None = None
    provider: str | None = None

    # 报告内容
    model_info: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)
    failure_analysis: dict[str, Any] = field(default_factory=dict)

    # 质量Evaluation特has
    latency_metrics: dict[str, Any] = field(default_factory=dict)
    token_metrics: dict[str, Any] = field(default_factory=dict)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    cost_metrics: dict[str, Any] = field(default_factory=dict)

    # ExportFile path
    json_path: str | None = None
    html_path: str | None = None
    markdown_path: str | None = None
    excel_path: str | None = None

    # 元Data
    tags: list[str] = field(default_factory=list)
    notes: str | None = None

    # 时间
    created_at: datetime | None = None

    @classmethod
    def create(
        cls,
        model_id: str,
        report_type: str = ReportType.STANDARD.value,
        run_id: int = None,
        **kwargs
    ) -> "Report":
        """Create报告"""
        return cls(
            run_id=run_id,
            report_type=report_type,
            model_id=model_id,
            created_at=datetime.now(),
            **kwargs
        )

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        return {
            'id': self.id,
            'report_id': self.report_id,
            'run_id': self.run_id,
            'report_type': self.report_type,
            'version': self.version,
            'model_id': self.model_id,
            'model_type': self.model_type,
            'provider': self.provider,
            'model_info_json': json.dumps(self.model_info, ensure_ascii=False) if self.model_info else None,
            'environment_json': json.dumps(self.environment, ensure_ascii=False) if self.environment else None,
            'config_json': json.dumps(self.config, ensure_ascii=False) if self.config else None,
            'results_json': json.dumps(self.results, ensure_ascii=False) if self.results else None,
            'aggregate_json': json.dumps(self.aggregate, ensure_ascii=False) if self.aggregate else None,
            'failure_analysis_json': json.dumps(self.failure_analysis, ensure_ascii=False) if self.failure_analysis else None,
            'latency_metrics_json': json.dumps(self.latency_metrics, ensure_ascii=False) if self.latency_metrics else None,
            'token_metrics_json': json.dumps(self.token_metrics, ensure_ascii=False) if self.token_metrics else None,
            'quality_metrics_json': json.dumps(self.quality_metrics, ensure_ascii=False) if self.quality_metrics else None,
            'cost_metrics_json': json.dumps(self.cost_metrics, ensure_ascii=False) if self.cost_metrics else None,
            'json_path': self.json_path,
            'html_path': self.html_path,
            'markdown_path': self.markdown_path,
            'excel_path': self.excel_path,
            'tags': ",".join(self.tags) if self.tags else None,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Report":
        """从Database行Create"""
        return cls(
            id=row.get('id'),
            report_id=row.get('report_id', ''),
            run_id=row.get('run_id'),
            report_type=row.get('report_type', ReportType.STANDARD.value),
            version=row.get('version', '1.0'),
            model_id=row.get('model_id', ''),
            model_type=row.get('model_type'),
            provider=row.get('provider'),
            model_info=json.loads(row.get('model_info_json', '{}') or '{}'),
            environment=json.loads(row.get('environment_json', '{}') or '{}'),
            config=json.loads(row.get('config_json', '{}') or '{}'),
            results=json.loads(row.get('results_json', '{}') or '{}'),
            aggregate=json.loads(row.get('aggregate_json', '{}') or '{}'),
            failure_analysis=json.loads(row.get('failure_analysis_json', '{}') or '{}'),
            latency_metrics=json.loads(row.get('latency_metrics_json', '{}') or '{}'),
            token_metrics=json.loads(row.get('token_metrics_json', '{}') or '{}'),
            quality_metrics=json.loads(row.get('quality_metrics_json', '{}') or '{}'),
            cost_metrics=json.loads(row.get('cost_metrics_json', '{}') or '{}'),
            json_path=row.get('json_path'),
            html_path=row.get('html_path'),
            markdown_path=row.get('markdown_path'),
            excel_path=row.get('excel_path'),
            tags=cls._parse_tags(row.get('tags')),
            notes=row.get('notes'),
            created_at=cls._parse_datetime(row.get('created_at')),
        )

    @staticmethod
    def _parse_datetime(value) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_tags(value) -> list[str]:
        if not value:
            return []
        return [t.strip() for t in str(value).split(',') if t.strip()]

    def to_json_dict(self) -> dict[str, Any]:
        """Export as JSON 格式（用于Export文件）"""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'model': self.model_info,
            'environment': self.environment,
            'config': self.config,
            'results': self.results,
            'aggregate': self.aggregate,
            'failure_analysis': self.failure_analysis,
            'latency_metrics': self.latency_metrics,
            'token_metrics': self.token_metrics,
            'quality_metrics': self.quality_metrics,
            'cost_metrics': self.cost_metrics,
        }
