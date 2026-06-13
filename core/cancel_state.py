"""
进程级取消 / 暂停信号——取代 core 直接读 st.session_state 的取消机制。

为什么需要进程级：取消信号必须跨 Streamlit 重跑边界。Stop/Pause 按钮在新的 script run
里触发，而测试跑在后台线程。threading.Event 是进程级单例，模块在进程内持久存在，
故 UI 线程 set 与测试线程 is_set 能跨重跑互通。

边界：core 只读本模块，不再 import streamlit。UI 层（config/session_state.py 的桥接
函数）可同时写 session_state（供界面显示）与 cancel_state（供 core 读取），二者解耦。
"""

from __future__ import annotations

import threading


class CancellationToken:
    """单次测试运行的可取消句柄（threading.Event 封装）。可注入，也可用模块级默认实例。"""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._batch_stop = threading.Event()

    # ---- 停止 ----
    def request_stop(self) -> None:
        self._stop.set()

    def clear_stop(self) -> None:
        self._stop.clear()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    # ---- 暂停 ----
    def request_pause(self) -> None:
        self._pause.set()

    def clear_pause(self) -> None:
        self._pause.clear()

    @property
    def pause_requested(self) -> bool:
        return self._pause.is_set()

    # ---- 批量测试停止（独立信号）----
    def request_batch_stop(self) -> None:
        self._batch_stop.set()

    def clear_batch_stop(self) -> None:
        self._batch_stop.clear()

    @property
    def batch_stop_requested(self) -> bool:
        return self._batch_stop.is_set()

    def reset(self) -> None:
        """清空所有信号（新一轮测试前调用）。"""
        self._stop.clear()
        self._pause.clear()
        self._batch_stop.clear()


# 进程级默认实例——供旧式模块函数（request_stop/is_stop_requested 等）使用，
# 兼容既有 UI 调用 providers.openai.set_stop_requested 的路径。
_default = CancellationToken()


# ---- 模块级便捷函数（操作默认实例）----
def request_stop() -> None:
    _default.request_stop()


def clear_stop() -> None:
    _default.clear_stop()


def is_stop_requested() -> bool:
    return _default.stop_requested


def request_pause() -> None:
    _default.request_pause()


def clear_pause() -> None:
    _default.clear_pause()


def is_pause_requested() -> bool:
    return _default.pause_requested


def request_batch_stop() -> None:
    _default.request_batch_stop()


def clear_batch_stop() -> None:
    _default.clear_batch_stop()


def is_batch_stop_requested() -> bool:
    return _default.batch_stop_requested


def reset_all() -> None:
    _default.reset()


def default_token() -> CancellationToken:
    """暴露默认实例（供需要在 set/is 之外做更复杂控制的调用方）。"""
    return _default
