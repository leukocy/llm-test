"""
Repository 模块

提供Data访问层，封装所hasDatabase操作。
"""

from .base import BaseRepository
from .test_run import TestRunRepository
from .test_result import TestResultRepository
from .api_log import ApiLogRepository
from .exec_log import ExecLogRepository
from .report import ReportRepository

__all__ = [
    'BaseRepository',
    'TestRunRepository',
    'TestResultRepository',
    'ApiLogRepository',
    'ExecLogRepository',
    'ReportRepository',
]
