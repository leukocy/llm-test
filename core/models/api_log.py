"""
API LogModel
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ApiLogStatus(Enum):
    """API LogStatus"""

    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ApiLog:
    """API Request LogsModel"""

    # 主键
    id: int | None = None

    # 唯一标识
    log_id: str = field(
        default_factory=lambda: f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"
    )

    # 关联Test运行
    run_id: int | None = None

    # 基本信息
    session_id: str | None = None
    test_type: str | None = None
    status: str = ApiLogStatus.SUCCESS.value

    # ModelConfigure
    provider: str | None = None
    model_id: str | None = None
    api_base_url: str | None = None

    # Performance Metrics
    ttft: float | None = None
    total_time: float | None = None

    # 详细信息
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Error
    error: str | None = None

    # 时间
    created_at: datetime | None = None

    @classmethod
    def create(
        cls,
        session_id: str,
        test_type: str,
        provider: str,
        model_id: str,
        request: dict[str, Any],
        run_id: int | None = None,
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

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        return {
            "id": self.id,
            "log_id": self.log_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "test_type": self.test_type,
            "status": self.status,
            "provider": self.provider,
            "model_id": self.model_id,
            "api_base_url": self.api_base_url,
            "ttft": self.ttft,
            "total_time": self.total_time,
            "request_json": (
                json.dumps(self.request, ensure_ascii=False) if self.request else None
            ),
            "response_json": (
                json.dumps(self.response, ensure_ascii=False) if self.response else None
            ),
            "metrics_json": (
                json.dumps(self.metrics, ensure_ascii=False) if self.metrics else None
            ),
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ApiLog":
        """从Database行Create"""
        return cls(
            id=row.get("id"),
            log_id=row.get("log_id", ""),
            run_id=row.get("run_id"),
            session_id=row.get("session_id"),
            test_type=row.get("test_type"),
            status=row.get("status", ApiLogStatus.SUCCESS.value),
            provider=row.get("provider"),
            model_id=row.get("model_id"),
            api_base_url=row.get("api_base_url"),
            ttft=row.get("ttft"),
            total_time=row.get("total_time"),
            request=json.loads(row.get("request_json", "{}") or "{}"),
            response=json.loads(row.get("response_json", "{}") or "{}"),
            metrics=json.loads(row.get("metrics_json", "{}") or "{}"),
            error=row.get("error"),
            created_at=cls._parse_datetime(row.get("created_at")),
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

    def mark_success(
        self,
        response: dict[str, Any],
        ttft: float | None = None,
        total_time: float | None = None,
    ):
        """标记succeeded"""
        self.status = ApiLogStatus.SUCCESS.value
        self.response = response
        self.ttft = ttft
        self.total_time = total_time

    def mark_error(self, error: str):
        """标记Error"""
        self.status = ApiLogStatus.ERROR.value
        self.error = error
