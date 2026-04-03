"""
执行LogModel
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import json


class LogLevel(Enum):
    """Log Level"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ExecLog:
    """执行LogModel"""

    # 主键
    id: Optional[int] = None

    # 关联Test运行
    run_id: Optional[int] = None

    # Log内容
    level: str = LogLevel.INFO.value
    message: str = ""
    session_id: Optional[str] = None

    # 指标
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Error
    error: Optional[str] = None

    # 时间
    timestamp: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        message: str,
        level: str = LogLevel.INFO.value,
        run_id: int = None,
        session_id: str = None,
        metrics: Dict[str, Any] = None,
        error: str = None,
    ) -> "ExecLog":
        """CreateLog条目"""
        return cls(
            run_id=run_id,
            level=level,
            message=message,
            session_id=str(session_id) if session_id else None,
            metrics=metrics or {},
            error=error,
            timestamp=datetime.now(),
        )

    @classmethod
    def debug(cls, message: str, **kwargs) -> "ExecLog":
        return cls.create(message, LogLevel.DEBUG.value, **kwargs)

    @classmethod
    def info(cls, message: str, **kwargs) -> "ExecLog":
        return cls.create(message, LogLevel.INFO.value, **kwargs)

    @classmethod
    def success(cls, message: str, **kwargs) -> "ExecLog":
        return cls.create(message, LogLevel.SUCCESS.value, **kwargs)

    @classmethod
    def warning(cls, message: str, **kwargs) -> "ExecLog":
        return cls.create(message, LogLevel.WARNING.value, **kwargs)

    @classmethod
    def error(cls, message: str, **kwargs) -> "ExecLog":
        return cls.create(message, LogLevel.ERROR.value, **kwargs)

    @classmethod
    def critical(cls, message: str, **kwargs) -> "ExecLog":
        return cls.create(message, LogLevel.CRITICAL.value, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convertis字典"""
        return {
            'id': self.id,
            'run_id': self.run_id,
            'level': self.level,
            'message': self.message,
            'session_id': self.session_id,
            'metrics_json': json.dumps(self.metrics, ensure_ascii=False) if self.metrics else None,
            'error': self.error,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ExecLog":
        """从Database行Create"""
        return cls(
            id=row.get('id'),
            run_id=row.get('run_id'),
            level=row.get('level', LogLevel.INFO.value),
            message=row.get('message', ''),
            session_id=row.get('session_id'),
            metrics=json.loads(row.get('metrics_json', '{}') or '{}'),
            error=row.get('error'),
            timestamp=cls._parse_datetime(row.get('timestamp')),
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

    def to_text(self) -> str:
        """Convertis可读文本"""
        level_emoji = {
            LogLevel.DEBUG.value: "🔍",
            LogLevel.INFO.value: "ℹ️",
            LogLevel.SUCCESS.value: "✅",
            LogLevel.WARNING.value: "⚠️",
            LogLevel.ERROR.value: "❌",
            LogLevel.CRITICAL.value: "🔥",
        }

        time_str = self.timestamp.strftime("%H:%M:%S") if self.timestamp else ""
        emoji = level_emoji.get(self.level, "📝")
        parts = [f"[{time_str}] {emoji}"]

        if self.session_id:
            parts.append(f"[{self.session_id}]")

        parts.append(self.message)

        if self.metrics:
            metrics_str = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in self.metrics.items())
            parts.append(f"({metrics_str})")

        if self.error:
            parts.append(f"Error: {self.error}")

        return " ".join(parts)
