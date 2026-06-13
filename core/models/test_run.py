"""
Test运行Model
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TestRunStatus(Enum):
    """Test运行Status"""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TestRun:
    """Test运行Model"""

    __test__ = False  # 防止 pytest 把本 dataclass 误当作测试类收集

    # 主键
    id: int | None = None

    # 唯一标识
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Test类型
    test_type: str = ""  # concurrency/prefill/segmented_prefill/long_context/matrix/custom/dataset/stability

    # Status
    status: str = TestRunStatus.RUNNING.value
    progress_percent: float = 0.0

    # ModelConfigure
    model_id: str = ""
    provider: str | None = None
    api_base_url: str | None = None

    # Test Parameters
    concurrency: int = 1
    max_tokens: int = 512
    temperature: float = 0.0
    thinking_enabled: bool = False
    thinking_budget: int | None = None
    reasoning_effort: str = "medium"

    # 完整Configure快照
    config: dict[str, Any] = field(default_factory=dict)

    # 时间信息
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Statistics
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    success_rate: float | None = None

    # Aggregate指标
    avg_ttft: float | None = None
    avg_tps: float | None = None
    avg_tpot: float | None = None
    p50_ttft: float | None = None
    p95_ttft: float | None = None
    p99_ttft: float | None = None
    total_tokens: int | None = None

    # 环境信息
    system_info: dict[str, Any] = field(default_factory=dict)
    python_version: str | None = None
    git_hash: str | None = None

    # 元Data
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    csv_path: str | None = None

    # ===== 1.2.0 数据仓库扩展字段（手册：报告是切片，仓库是全集）=====
    # 一等列（筛选/分组/对外口径）
    machine_id: str | None = None
    tester: str | None = None
    external_level: str = "internal"  # internal / review / publishable
    bottleneck: str | None = None
    next_action: str | None = None
    supersedes_test_id: str | None = None
    comparison_group: str | None = None
    mtp_enabled: bool | None = None

    # 资源监控 / 等效带宽 头条指标
    effective_bandwidth_gbps: float | None = None
    bandwidth_utilization_pct: float | None = None
    gpu_vram_peak_gb: float | None = None
    system_memory_peak_gb: float | None = None

    # 变长 JSON 字段（dataclass 里以 dict 表示，to_dict 序列化为 *_json）
    model_spec: dict = field(default_factory=dict)
    serving_config: dict = field(default_factory=dict)
    resource_monitor: dict = field(default_factory=dict)
    status_detail: str | None = None

    # ===== 1.3.0 推理引擎运行时（/metrics 轮询 + KV 实况）=====
    engine_metrics: dict = field(default_factory=dict)
    gpu_kv_cache_usage_peak_pct: float | None = None
    num_preemption_total: int | None = None
    engine_running_requests_peak: int | None = None
    kv_cache_capacity_tokens: int | None = None

    @classmethod
    def create(
        cls,
        test_type: str,
        model_id: str,
        provider: str = None,
        **kwargs
    ) -> "TestRun":
        """Factory方法：Create新Test运行"""
        return cls(
            test_type=test_type,
            model_id=model_id,
            provider=provider,
            status=TestRunStatus.RUNNING.value,
            created_at=datetime.now(),
            started_at=datetime.now(),
            **kwargs
        )

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        data = asdict(self)
        # Process特殊字段
        data['config_json'] = json.dumps(self.config, ensure_ascii=False) if self.config else None
        data['system_info_json'] = json.dumps(self.system_info, ensure_ascii=False) if self.system_info else None
        data['tags'] = ",".join(self.tags) if self.tags else None
        data['thinking_enabled'] = int(self.thinking_enabled)

        # 1.2.0 JSON 字段序列化
        data['model_spec_json'] = (
            json.dumps(self.model_spec, ensure_ascii=False) if self.model_spec else None
        )
        data['serving_config_json'] = (
            json.dumps(self.serving_config, ensure_ascii=False) if self.serving_config else None
        )
        data['resource_monitor_json'] = (
            json.dumps(self.resource_monitor, ensure_ascii=False) if self.resource_monitor else None
        )
        data['engine_metrics_json'] = (
            json.dumps(self.engine_metrics, ensure_ascii=False) if self.engine_metrics else None
        )
        if self.mtp_enabled is not None:
            data['mtp_enabled'] = int(self.mtp_enabled)
        # external_level 默认 internal（None → internal）
        if data.get('external_level') is None:
            data['external_level'] = "internal"

        # Process时间
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None

        # 移除原始字段
        data.pop('config', None)
        data.pop('system_info', None)
        data.pop('model_spec', None)
        data.pop('serving_config', None)
        data.pop('resource_monitor', None)
        data.pop('engine_metrics', None)

        return data

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TestRun":
        """从Database行Create"""
        return cls(
            id=row.get('id'),
            test_id=row.get('test_id', ''),
            test_type=row.get('test_type', ''),
            status=row.get('status', TestRunStatus.RUNNING.value),
            progress_percent=row.get('progress_percent', 0.0),
            model_id=row.get('model_id', ''),
            provider=row.get('provider'),
            api_base_url=row.get('api_base_url'),
            concurrency=row.get('concurrency', 1),
            max_tokens=row.get('max_tokens', 512),
            temperature=row.get('temperature', 0.0),
            thinking_enabled=bool(row.get('thinking_enabled', 0)),
            thinking_budget=row.get('thinking_budget'),
            reasoning_effort=row.get('reasoning_effort', 'medium'),
            config=json.loads(row.get('config_json', '{}') or '{}'),
            created_at=cls._parse_datetime(row.get('created_at')),
            started_at=cls._parse_datetime(row.get('started_at')),
            completed_at=cls._parse_datetime(row.get('completed_at')),
            duration_seconds=row.get('duration_seconds'),
            total_requests=row.get('total_requests', 0),
            completed_requests=row.get('completed_requests', 0),
            failed_requests=row.get('failed_requests', 0),
            success_rate=row.get('success_rate'),
            avg_ttft=row.get('avg_ttft'),
            avg_tps=row.get('avg_tps'),
            avg_tpot=row.get('avg_tpot'),
            p50_ttft=row.get('p50_ttft'),
            p95_ttft=row.get('p95_ttft'),
            p99_ttft=row.get('p99_ttft'),
            total_tokens=row.get('total_tokens'),
            system_info=json.loads(row.get('system_info_json', '{}') or '{}'),
            python_version=row.get('python_version'),
            git_hash=row.get('git_hash'),
            tags=cls._parse_tags(row.get('tags')),
            notes=row.get('notes'),
            csv_path=row.get('csv_path'),
            # 1.2.0 数据仓库扩展字段
            machine_id=row.get('machine_id'),
            tester=row.get('tester'),
            external_level=row.get('external_level') or "internal",
            bottleneck=row.get('bottleneck'),
            next_action=row.get('next_action'),
            supersedes_test_id=row.get('supersedes_test_id'),
            comparison_group=row.get('comparison_group'),
            mtp_enabled=cls._parse_bool(row.get('mtp_enabled')),
            effective_bandwidth_gbps=row.get('effective_bandwidth_gbps'),
            bandwidth_utilization_pct=row.get('bandwidth_utilization_pct'),
            gpu_vram_peak_gb=row.get('gpu_vram_peak_gb'),
            system_memory_peak_gb=row.get('system_memory_peak_gb'),
            model_spec=json.loads(row.get('model_spec_json', '{}') or '{}'),
            serving_config=json.loads(row.get('serving_config_json', '{}') or '{}'),
            resource_monitor=json.loads(row.get('resource_monitor_json', '{}') or '{}'),
            status_detail=row.get('status_detail'),
            engine_metrics=json.loads(row.get('engine_metrics_json', '{}') or '{}'),
            gpu_kv_cache_usage_peak_pct=row.get('gpu_kv_cache_usage_peak_pct'),
            num_preemption_total=row.get('num_preemption_total'),
            engine_running_requests_peak=row.get('engine_running_requests_peak'),
            kv_cache_capacity_tokens=row.get('kv_cache_capacity_tokens'),
        )

    @staticmethod
    def _parse_datetime(value) -> datetime | None:
        """Parse日期时间"""
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
        """ParseLabel"""
        if not value:
            return []
        return [t.strip() for t in str(value).split(',') if t.strip()]

    @staticmethod
    def _parse_bool(value) -> bool | None:
        """Parse布尔（DB 存 0/1/None）"""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return None

    def complete(self, success: bool = True):
        """标记Test completed"""
        self.status = TestRunStatus.COMPLETED.value if success else TestRunStatus.FAILED.value
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def cancel(self):
        """CancelTest"""
        self.status = TestRunStatus.CANCELLED.value
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def pause(self):
        """Pause Test"""
        self.status = TestRunStatus.PAUSED.value

    def resume(self):
        """Resume Test"""
        self.status = TestRunStatus.RUNNING.value

    def update_progress(self, completed: int, total: int):
        """Update进度"""
        self.completed_requests = completed
        self.total_requests = total
        if total > 0:
            self.progress_percent = (completed / total) * 100
