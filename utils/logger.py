"""
Structured logging system for benchmark tests.

Provides professional logging with levels, filtering, analytics, and export.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """Log Level"""
    DEBUG = 0
    INFO = 1
    SUCCESS = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


@dataclass
class LogEntry:
    """结构化Log条目"""
    timestamp: datetime
    level: LogLevel
    message: str
    session_id: str | None = None
    test_type: str | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Convertis字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'message': self.message,
            'session_id': self.session_id,
            'test_type': self.test_type,
            'metrics': self.metrics,
            'error': self.error
        }

    def to_json(self) -> str:
        """ConvertisJSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_text(self, include_metrics=True) -> str:
        """Convertis可读文本"""
        time_str = self.timestamp.strftime("%H:%M:%S")
        level_emoji = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.SUCCESS: "✅",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.CRITICAL: "🔥"
        }

        emoji = level_emoji.get(self.level, "📝")
        parts = [f"[{time_str}] {emoji}"]

        if self.session_id:
            parts.append(f"[{self.session_id}]")

        parts.append(self.message)

        if include_metrics and self.metrics:
            metrics_items = []
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    if 'ttft' in k.lower() or 'time' in k.lower():
                        metrics_items.append(f"{k}={v:.3f}s")
                    else:
                        metrics_items.append(f"{k}={v:.2f}")
                else:
                    metrics_items.append(f"{k}={v}")

            if metrics_items:
                parts.append(f"({', '.join(metrics_items)})")

        if self.error:
            parts.append(f"⚠ Error: {self.error}")

        return " ".join(parts)


class BenchmarkLogger:
    """基准TestLog管理器"""

    def __init__(self, max_entries=500):
        self.entries: list[LogEntry] = []
        self.max_entries = max_entries
        self.stats = {
            'total': 0,
            'by_level': {level.name: 0 for level in LogLevel},
            'errors': 0,
            'warnings': 0,
            'success': 0
        }

    def log(self, level: LogLevel, message: str, **kwargs) -> LogEntry:
        """记录Log"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            **kwargs
        )

        self.entries.append(entry)

        # 限制条目数量
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        # UpdateStatistics
        self.stats['total'] += 1
        self.stats['by_level'][level.name] += 1

        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            self.stats['errors'] += 1
        elif level == LogLevel.WARNING:
            self.stats['warnings'] += 1
        elif level == LogLevel.SUCCESS:
            self.stats['success'] += 1

        return entry

    def debug(self, message: str, **kwargs) -> LogEntry:
        """记录DEBUG级别Log"""
        return self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> LogEntry:
        """记录INFO级别Log"""
        return self.log(LogLevel.INFO, message, **kwargs)

    def success(self, message: str, **kwargs) -> LogEntry:
        """记录SUCCESS级别Log"""
        return self.log(LogLevel.SUCCESS, message, **kwargs)

    def warning(self, message: str, **kwargs) -> LogEntry:
        """记录WARNING级别Log"""
        return self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> LogEntry:
        """记录ERROR级别Log"""
        return self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> LogEntry:
        """记录CRITICAL级别Log"""
        return self.log(LogLevel.CRITICAL, message, **kwargs)

    def filter(self,
               level: LogLevel | None = None,
               levels: list[LogLevel] | None = None,
               session_id: str | None = None,
               test_type: str | None = None,
               search_text: str | None = None) -> list[LogEntry]:
        """FilterLog"""
        filtered = self.entries

        if level:
            filtered = [e for e in filtered if e.level == level]
        elif levels:
            filtered = [e for e in filtered if e.level in levels]

        if session_id:
            filtered = [e for e in filtered if e.session_id == session_id]

        if test_type:
            filtered = [e for e in filtered if e.test_type == test_type]

        if search_text:
            search_lower = search_text.lower()
            filtered = [e for e in filtered
                       if search_lower in e.message.lower()]

        return filtered

    def get_recent(self, n=50) -> list[LogEntry]:
        """Get最近N条Log"""
        return self.entries[-n:] if self.entries else []

    def get_stats(self) -> dict:
        """GetStatistics信息"""
        return self.stats.copy()

    def export_json(self) -> str:
        """ExportisJSON"""
        return json.dumps([e.to_dict() for e in self.entries],
                         ensure_ascii=False, indent=2)

    def export_text(self, include_metrics=True) -> str:
        """Exportis文本"""
        return "\n".join([e.to_text(include_metrics) for e in self.entries])

    def export_csv(self) -> str:
        """ExportisCSV"""
        import csv
        from io import StringIO

        output = StringIO()
        fieldnames = ['timestamp', 'level', 'message', 'session_id',
                     'test_type', 'metrics', 'error']
        writer = csv.DictWriter(output, fieldnames=fieldnames)

        writer.writeheader()
        for entry in self.entries:
            row = entry.to_dict()
            row['metrics'] = json.dumps(row['metrics']) if row['metrics'] else ''
            writer.writerow(row)

        return output.getvalue()

    def clear(self):
        """Clear Logs"""
        self.entries = []
        self.stats = {
            'total': 0,
            'by_level': {level.name: 0 for level in LogLevel},
            'errors': 0,
            'warnings': 0,
            'success': 0
        }
