"""Tests for moving hardware discovery off the first-request critical path."""

import importlib
import threading


def test_nonblocking_system_info_read_starts_prefetch(monkeypatch):
    from core import system_info

    module = importlib.reload(system_info)
    release_capture = threading.Event()

    def slow_capture(sudo_password=None):
        release_capture.wait(timeout=1)
        return {"machine_id": "prefetched"}

    monkeypatch.setattr(module, "capture_system_info", slow_capture)

    assert module.get_cached_system_info(wait=False) == {}

    release_capture.set()
    assert module.get_cached_system_info() == {"machine_id": "prefetched"}

    # Restore process-level cache state for tests that run later in the suite.
    importlib.reload(module)
