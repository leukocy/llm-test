"""
pytest Configure文件

提供Test所需 fixtures andConfigure
"""

import os
import pathlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# The separate engine/ package is still planned. The implemented server/ API
# must participate in the default test command as well as targeted security tests.
collect_ignore = [
    str(pathlib.Path(__file__).parent / "engine"),
]

# Add items目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# === 全局 Mock Streamlit ===
class DictLikeSessionState:
    """模拟 Streamlit  session_state"""

    def __init__(self):
        self._data = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return self._data.get(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __contains__(self, key):
        return key in self._data

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


@pytest.fixture
def sample_predictions():
    """示例预测列表"""
    return ["A", "B", "C", "A", "B"]


@pytest.fixture
def sample_references():
    """示例参考Answer列表"""
    return ["A", "B", "D", "A", "C"]


@pytest.fixture
def sample_text_pairs():
    """示例文本对"""
    return [
        ("the cat is on the mat", "the cat is on the mat"),
        ("hello world", "hello there"),
        ("test case", "test example"),
    ]


@pytest.fixture
def mock_api_response():
    """模拟 API 响应"""
    return {
        "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def mock_stream_chunks():
    """模拟流式响应块"""
    return [
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
