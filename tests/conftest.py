"""
pytest Configure文件

提供Test所需 fixtures andConfigure
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add items目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# === 全局 Mock Streamlit ===
# mustinImport任何use streamlit 模块之前Set
class DictLikeSessionState:
    """模拟 Streamlit  session_state"""

    def __init__(self):
        self._data = {}

    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._data.get(name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __contains__(self, key):
        return key in self._data

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


# Mock streamlit 模块and其子模块
def setup_streamlit_mock():
    """Set streamlit 全局 mock"""

    # 主 streamlit 模块
    st_mock = MagicMock()
    st_mock.session_state = DictLikeSessionState()

    # streamlit.runtime.scriptrunner
    runtime_mock = MagicMock()
    runtime_mock.scriptrunner.get_script_run_ctx = MagicMock(return_value=None)
    runtime_mock.scriptrunner.add_script_run_ctx = MagicMock()
    runtime_mock.scriptrunner.ScriptRunContext = MagicMock()

    # streamlit.runtime.exists
    runtime_mock.exists = MagicMock(return_value=False)

    # 组装模块结构
    st_mock.runtime = runtime_mock

    # Register到 sys.modules
    sys.modules['streamlit'] = st_mock
    sys.modules['streamlit.runtime'] = runtime_mock
    sys.modules['streamlit.runtime.scriptrunner'] = runtime_mock.scriptrunner

    return st_mock


# inImport时Set mock
_streamlit_mock = setup_streamlit_mock()


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
        "choices": [{
            "delta": {"content": "Hello"},
            "finish_reason": None
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }


@pytest.fixture
def mock_stream_chunks():
    """模拟流式响应块"""
    return [
        {"choices": [{"delta": {"content": "Hello "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
