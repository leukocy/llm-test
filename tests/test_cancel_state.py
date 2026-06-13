"""core.cancel_state 单元测试。"""

from __future__ import annotations

import threading

from core import cancel_state
from core.cancel_state import CancellationToken


def test_token_stop_lifecycle():
    t = CancellationToken()
    assert not t.stop_requested
    t.request_stop()
    assert t.stop_requested
    t.clear_stop()
    assert not t.stop_requested


def test_token_pause_lifecycle():
    t = CancellationToken()
    assert not t.pause_requested
    t.request_pause()
    assert t.pause_requested
    t.clear_pause()
    assert not t.pause_requested


def test_token_batch_stop_lifecycle():
    t = CancellationToken()
    assert not t.batch_stop_requested
    t.request_batch_stop()
    assert t.batch_stop_requested
    t.clear_batch_stop()
    assert not t.batch_stop_requested


def test_reset_clears_all():
    t = CancellationToken()
    t.request_stop()
    t.request_pause()
    t.request_batch_stop()
    t.reset()
    assert not t.stop_requested
    assert not t.pause_requested
    assert not t.batch_stop_requested


def test_module_functions_operate_on_default():
    cancel_state.reset_all()
    assert not cancel_state.is_stop_requested()
    cancel_state.request_stop()
    assert cancel_state.is_stop_requested()
    cancel_state.clear_stop()
    assert not cancel_state.is_stop_requested()


def test_default_token_is_process_singleton():
    a = cancel_state.default_token()
    b = cancel_state.default_token()
    assert a is b


def test_cross_thread_visibility():
    """取消信号必须能跨线程可见（核心语义：UI 线程 set，测试线程见）。"""
    t = CancellationToken()
    seen = []

    def worker():
        # 自旋等待 stop 信号（带短 sleep 避免死循环）
        for _ in range(200):
            if t.stop_requested:
                seen.append(True)
                return
            threading.Event().wait(0.01)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    threading.Event().wait(0.05)  # 让 worker 跑起来
    t.request_stop()  # 主线程触发停止
    th.join(timeout=3.0)
    assert seen == [True]
