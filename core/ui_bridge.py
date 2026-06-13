"""
UI 状态桥——core 通过它读写跨重跑状态（resume / results_df / current_test_id 等），
不再直接访问 st.session_state（模式 E 解耦）。

UI 层注入一个 session_state 支持的桥（见 ui/test_runner.SessionStateBridge）；
无 UI（headless / 单测）时用 NullStateBridge（内存 dict，不跨重跑，仅供离线/测试）。

协议极简：get(key, default) / set(key, value)，鸭子类型，不强约束 ABC。
"""

from __future__ import annotations

from typing import Any


class NullStateBridge:
    """内存 dict 桥（默认；进程内，不跨 Streamlit 重跑，仅供 headless 与单测）。"""

    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._d[key] = value

    def clear(self) -> None:
        self._d.clear()
