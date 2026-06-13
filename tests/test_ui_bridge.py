"""core.ui_bridge 单元测试。"""

from __future__ import annotations

from core.ui_bridge import NullStateBridge


def test_get_set_roundtrip():
    b = NullStateBridge()
    assert b.get("x") is None
    assert b.get("x", "default") == "default"
    b.set("x", 42)
    assert b.get("x") == 42
    assert b.get("x", "default") == 42


def test_set_overwrites():
    b = NullStateBridge()
    b.set("k", 1)
    b.set("k", 2)
    assert b.get("k") == 2


def test_clear():
    b = NullStateBridge()
    b.set("a", 1)
    b.set("b", 2)
    b.clear()
    assert b.get("a") is None
    assert b.get("b") is None


def test_holds_complex_values():
    b = NullStateBridge()
    payload = {"results": [1, 2, 3], "nested": {"k": "v"}}
    b.set("resume_data", payload)
    assert b.get("resume_data") == payload
