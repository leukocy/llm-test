"""
Test运行Model
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import json
import uuid


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

    # 主键
    id: Optional[int] = None

    # 唯一标识
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Test类型
    test_type: str = ""  # concurrency/prefill/segmented_prefill/long_context/matrix/custom/dataset/stability

    # Status
    status: str = TestRunStatus.RUNNING.value
    progress_percent: float = 0.0

    # ModelConfigure
    model_id: str = ""
    provider: Optional[str] = None
    api_base_url: Optional[str] = None

    # Test Parameters
    concurrency: int = 1
    max_tokens: int = 512
    temperature: float = 0.0
    thinking_enabled: bool = False
    thinking_budget: Optional[int] = None
    reasoning_effort: str = "medium"

    # 完整Configure快照
    config: Dict[str, Any] = field(default_factory=dict)

    # 时间信息
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Statistics
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    success_rate: Optional[float] = None

    # Aggregate指标
    avg_ttft: Optional[float] = None
    avg_tps: Optional[float] = None
    avg_tpot: Optional[float] = None
    p50_ttft: Optional[float] = None
    p95_ttft: Optional[float] = None
    p99_ttft: Optional[float] = None
    total_tokens: Optional[int] = None

    # 环境信息
    system_info: Dict[str, Any] = field(default_factory=dict)
    python_version: Optional[str] = None
    git_hash: Optional[str] = None

    # 元Data
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    csv_path: Optional[str] = None

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

    def to_dict(self) -> Dict[str, Any]:
        """Convertis字典"""
        data = asdict(self)
        # Process特殊字段
        data['config_json'] = json.dumps(self.config, ensure_ascii=False) if self.config else None
        data['system_info_json'] = json.dumps(self.system_info, ensure_ascii=False) if self.system_info else None
        data['tags'] = ",".join(self.tags) if self.tags else None
        data['thinking_enabled'] = int(self.thinking_enabled)

        # Process时间
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None

        # 移除原始字段
        data.pop('config', None)
        data.pop('system_info', None)

        return data

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "TestRun":
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
        )

    @staticmethod
    def _parse_datetime(value) -> Optional[datetime]:
        """Parse日期时间"""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except:
            return None

    @staticmethod
    def _parse_tags(value) -> List[str]:
        """ParseLabel"""
        if not value:
            return []
        return [t.strip() for t in str(value).split(',') if t.strip()]

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
