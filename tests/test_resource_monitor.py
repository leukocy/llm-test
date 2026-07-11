"""core.resource_monitor 单元测试。

避免时序抖动：汇总逻辑直接构造 _samples 后断言；生命周期用短 interval 实跑。
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from core.resource_monitor import ResourceMonitor

# ---------- 汇总逻辑 ----------


def _sample(t, cpu, mem, util=40.0, vram=10.0, power=200.0, temp=50.0):
    return {
        "t": t,
        "cpu_percent": cpu,
        "system_memory_gb": mem,
        "gpu_util_percent": util,
        "gpu_vram_gb": vram,
        "gpu_power_w": power,
        "gpu_temp_c": temp,
    }


def test_summarize_peaks_and_means():
    mon = ResourceMonitor(interval=0.01)
    mon._samples = [_sample(0.0, 10, 5), _sample(0.01, 50, 8), _sample(0.02, 30, 12)]
    summary = mon._summarize()
    assert summary["sample_count"] == 3
    assert summary["peaks"]["cpu_percent"] == 50
    assert summary["peaks"]["system_memory_gb"] == 12
    assert summary["means"]["cpu_percent"] == pytest.approx(30, abs=0.1)
    assert summary["gpu_monitored"] is True
    assert len(summary["timeline"]) == 3


def test_summarize_handles_none_gpu_when_nvml_absent():
    mon = ResourceMonitor(interval=0.01)
    mon._samples = [
        {
            "t": 0.0,
            "cpu_percent": 20,
            "system_memory_gb": 4,
            "gpu_util_percent": None,
            "gpu_vram_gb": None,
            "gpu_power_w": None,
            "gpu_temp_c": None,
        },
    ]
    summary = mon._summarize()
    assert summary["peaks"]["cpu_percent"] == 20
    assert summary["peaks"]["gpu_vram_gb"] is None
    assert summary["gpu_monitored"] is False


def test_empty_summary():
    mon = ResourceMonitor()
    summary = mon.stop()  # 未 start
    assert summary["sample_count"] == 0
    assert summary["timeline"] == []
    assert summary["peaks"]["cpu_percent"] is None


def test_downsample_caps_timeline():
    mon = ResourceMonitor(interval=0.01)
    mon._samples = [_sample(i * 0.01, i % 100, i % 50) for i in range(10000)]
    summary = mon._summarize()
    assert len(summary["timeline"]) <= 3600
    assert summary["sample_count"] == 10000


# ---------- 生命周期 + NVML 降级 ----------


def test_start_stop_lifecycle_samples_cpu_mem():
    mon = ResourceMonitor(interval=0.05)

    counter = {"i": 0}

    def fake_cpu_percent(interval=None):
        counter["i"] += 1
        return float(counter["i"] * 10)

    class FakeMem:
        used = 8 * 1024**3

    # 强制 NVML 不可用 → 只采 CPU/内存
    with (
        patch("psutil.cpu_percent", side_effect=fake_cpu_percent),
        patch("psutil.virtual_memory", return_value=FakeMem()),
    ):
        mon._init_nvml = lambda: setattr(mon, "_nvml_ok", False)  # type: ignore[method-assign]
        mon.start()
        time.sleep(0.3)
        summary = mon.stop()

    assert summary["sample_count"] >= 2
    assert summary["peaks"]["cpu_percent"] is not None and summary["peaks"]["cpu_percent"] > 0
    assert summary["peaks"]["system_memory_gb"] == pytest.approx(8.0, abs=0.1)
    assert summary["peaks"]["gpu_vram_gb"] is None  # NVML 关闭
    assert summary["duration_seconds"] >= 0.2
    assert mon._thread is None  # 线程已 join


def test_stop_returns_promptly_via_event():
    """停止信号应让 stop() 在 ~interval 内返回，而非阻塞。"""
    mon = ResourceMonitor(interval=1.0)
    with (
        patch("psutil.cpu_percent", return_value=5.0),
        patch("psutil.virtual_memory") as vm,
    ):
        vm.return_value.used = 1 * 1024**3
        mon._init_nvml = lambda: setattr(mon, "_nvml_ok", False)  # type: ignore[method-assign]
        mon.start()
        t0 = time.monotonic()
        mon.stop()
        elapsed = time.monotonic() - t0
    assert elapsed < 1.5  # 远小于 interval*2 的 join 超时
    assert mon._thread is None


# ---------- 多卡 NVML 聚合（注入假 pynvml） ----------


def test_gpu_aggregation_across_multiple_gpus():
    """假 pynvml 模拟 2 张卡，验证显存/功耗求和、利用率取均、温度取最大。"""

    class FakeUtil:
        def __init__(self, gpu):
            self.gpu = gpu

    class FakeMemInfo:
        def __init__(self, used):
            self.used = used

    class FakePynvml:
        NVML_TEMPERATURE_GPU = 0

        @staticmethod
        def nvmlInit():
            pass

        @staticmethod
        def nvmlShutdown():
            pass

        @staticmethod
        def nvmlDeviceGetCount():
            return 2

        @staticmethod
        def nvmlDeviceGetHandleByIndex(i):
            return i  # handle = index

        @staticmethod
        def nvmlDeviceGetUtilizationRates(h):
            return FakeUtil(60 if h == 0 else 40)

        @staticmethod
        def nvmlDeviceGetMemoryInfo(h):
            return FakeMemInfo(4 * 1024**3 if h == 0 else 6 * 1024**3)

        @staticmethod
        def nvmlDeviceGetPowerUsage(h):
            return 100000 if h == 0 else 150000  # mW

        @staticmethod
        def nvmlDeviceGetTemperature(h, sensor):
            return 60 if h == 0 else 75

    import sys

    mon = ResourceMonitor(interval=0.05)
    with (
        patch.dict(sys.modules, {"pynvml": FakePynvml}),
        patch("psutil.cpu_percent", return_value=10.0),
        patch("psutil.virtual_memory") as vm,
    ):
        vm.return_value.used = 1 * 1024**3
        mon.start()
        time.sleep(0.2)
        summary = mon.stop()

    assert summary["gpu_monitored"] is True
    assert summary["peaks"]["gpu_vram_gb"] == pytest.approx(10.0, abs=0.2)  # 4+6
    assert summary["peaks"]["gpu_util_percent"] == pytest.approx(50.0, abs=0.2)  # (60+40)/2
    assert summary["peaks"]["gpu_power_w"] == pytest.approx(250.0, abs=0.5)  # 100+150
    assert summary["peaks"]["gpu_temp_c"] == 75  # max


def test_thread_is_daemon():
    mon = ResourceMonitor(interval=0.5)
    mon._init_nvml = lambda: setattr(mon, "_nvml_ok", False)  # type: ignore[method-assign]
    with (
        patch("psutil.cpu_percent", return_value=1.0),
        patch("psutil.virtual_memory") as vm,
    ):
        vm.return_value.used = 1
        mon.start()
        try:
            assert mon._thread is not None
            assert mon._thread.daemon is True
        finally:
            mon.stop()
