"""
API LogModel
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import json
import uuid


class ApiLogStatus(Enum):
    """API LogStatus"""
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ApiLog:
    """API Request LogsModel"""

    # 主键
    id: Optional[int] = None

    # 唯一标识
    log_id: str = field(default_factory=lambda: f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}")

    # 关联Test运行
    run_id: Optional[int] = None

    # 基本信息
    session_id: Optional[str] = None
    test_type: Optional[str] = None
    status: str = ApiLogStatus.SUCCESS.value

    # ModelConfigure
    provider: Optional[str] = None
    model_id: Optional[str] = None
    api_base_url: Optional[str] = None

    # Performance Metrics
    ttft: Optional[float] = None
    total_time: Optional[float] = None

    # 详细信息
    request: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Error
    error: Optional[str] = None

    # 时间
    created_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        session_id: str,
        test_type: str,
        provider: str,
        model_id: str,
        request: Dict[str, Any],
        run_id: int = None,
    ) -> "ApiLog":
        """CreateLog条目"""
        return cls(
            run_id=run_id,
            session_id=str(session_id),
            test_type=test_type,
            provider=provider,
            model_id=model_id,
            request=request,
            created_at=datetime.now(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convertis字典"""
        return {
            'id': self.id,
            'log_id': self.log_id,
            'run_id': self.run_id,
            'session_id': self.session_id,
            'test_type': self.test_type,
            'status': self.status,
            'provider': self.provider,
            'model_id': self.model_id,
            'api_base_url': self.api_base_url,
            'ttft': self.ttft,
            'total_time': self.total_time,
            'request_json': json.dumps(self.request, ensure_ascii=False) if self.request else None,
            'response_json': json.dumps(self.response, ensure_ascii=False) if self.response else None,
            'metrics_json': json.dumps(self.metrics, ensure_ascii=False) if self.metrics else None,
            'error': self.error,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ApiLog":
        """从Database行Create"""
        return cls(
            id=row.get('id'),
            log_id=row.get('log_id', ''),
            run_id=row.get('run_id'),
            session_id=row.get('session_id'),
            test_type=row.get('test_type'),
            status=row.get('status', ApiLogStatus.SUCCESS.value),
            provider=row.get('provider'),
            model_id=row.get('model_id'),
            api_base_url=row.get('api_base_url'),
            ttft=row.get('ttft'),
            total_time=row.get('total_time'),
            request=json.loads(row.get('request_json', '{}') or '{}'),
            response=json.loads(row.get('response_json', '{}') or '{}'),
            metrics=json.loads(row.get('metrics_json', '{}') or '{}'),
            error=row.get('error'),
            created_at=cls._parse_datetime(row.get('created_at')),
        )

    @staticmethod
    def _parse_datetime(value) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except:
            return None

    def mark_success(self, response: Dict[str, Any], ttft: float = None, total_time: float = None):
        """标记succeeded"""
        self.status = ApiLogStatus.SUCCESS.value
        self.response = response
        self.ttft = ttft
        self.total_time = total_time

    def mark_error(self, error: str):
        """标记Error"""
        self.status = ApiLogStatus.ERROR.value
        self.error = error
