"""
DataModel模块

定义所hasDatabase表对应 Python Data类。
"""

from .test_run import TestRun, TestRunStatus
from .test_result import TestResult
from .api_log import ApiLog, ApiLogStatus
from .exec_log import ExecLog, LogLevel
from .report import Report, ReportType

__all__ = [
    'TestRun',
    'TestRunStatus',
    'TestResult',
    'ApiLog',
    'ApiLogStatus',
    'ExecLog',
    'LogLevel',
    'Report',
    'ReportType',
]
