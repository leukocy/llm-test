"""
测试中后台资源监控

在每次 benchmark 测试期间，用独立 daemon 线程按固定间隔采样 CPU / 内存 / GPU
(利用率、显存、温度、功耗)，停止时返回汇总(peaks + means + timeline)。

为什么用线程而非 asyncio：benchmark runner 是 async + asyncio.gather 驱动的，
psutil / NVML 都是阻塞调用，放进事件循环会卡住并发请求。独立线程不会阻塞事件循环。
数据库是 thread-local 的（core/database/connection.py），所以监控线程**不碰 DB**，
只在内存累积，由主线程在 _complete_db_run 里持久化。

多卡场景：对每张 GPU 采样后做聚合（显存/功耗求和，利用率/温度取聚合），
因为对 LLM 推理盒子而言“整机资源画像”比单卡更有意义。timeline 上限 ~3600 点。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_MAX_TIMELINE_POINTS = 3600


class ResourceMonitor:
    """后台资源采样器。start()/stop() 成对使用，stop() 返回汇总 dict。"""

    def __init__(self, interval: float = 1.0, gpu_indices: list[int] | None = None):
        self.interval = max(0.2, float(interval))
        self.gpu_indices = gpu_indices  # None = 全部可见 GPU
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._start_ts: float | None = None
        self._end_ts: float | None = None
        self._nvml_ok = False
        self._nvml_handles: list[Any] = []

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动采样线程。重复 start 安全（忽略）。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._samples = []
        self._start_ts = time.monotonic()

        # 预热 CPU% 指标（psutil 首次调用返回 0.0，先喂一次基准）
        try:
            import psutil
            psutil.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            pass

        self._init_nvml()
        self._thread = threading.Thread(target=self._run, name="ResourceMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """停止采样并返回汇总。未 start 过则返回空汇总。"""
        if self._thread is None:
            return self._empty_summary()
        self._stop_event.set()
        self._thread.join(timeout=self.interval * 2 + 1.0)
        self._thread = None
        self._end_ts = time.monotonic()
        self._shutdown_nvml()
        return self._summarize()

    # ------------------------------------------------------------------
    # 采样循环
    # ------------------------------------------------------------------
    def _run(self) -> None:
        # 首个采样点对齐 start_ts≈0
        while not self._stop_event.is_set():
            sample = self._sample_once()
            self._samples.append(sample)
            # 等待 interval 或被停止信号唤醒（保证 stop() 及时返回）
            self._stop_event.wait(timeout=self.interval)

    def _sample_once(self) -> dict[str, Any]:
        t = time.monotonic() - (self._start_ts or time.monotonic())
        sample: dict[str, Any] = {"t": round(t, 3)}

        # CPU / 内存
        try:
            import psutil
            sample["cpu_percent"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            sample["system_memory_gb"] = round(mem.used / 1024 ** 3, 3)
        except Exception:  # noqa: BLE001
            sample["cpu_percent"] = None
            sample["system_memory_gb"] = None

        # GPU（聚合）
        gpu = self._sample_gpu()
        sample.update(gpu)
        return sample

    def _sample_gpu(self) -> dict[str, Any]:
        result = {
            "gpu_util_percent": None,
            "gpu_vram_gb": None,
            "gpu_power_w": None,
            "gpu_temp_c": None,
        }
        if not self._nvml_ok or not self._nvml_handles:
            return result
        try:
            import pynvml  # type: ignore
            utils = []
            vram_used = 0.0
            power_w = 0.0
            temp_c = 0.0
            for handle in self._nvml_handles:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utils.append(util.gpu)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    vram_used += pynvml.nvmlDeviceGetMemoryInfo(handle).used
                except Exception:  # noqa: BLE001
                    pass
                try:
                    power_w += pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                except Exception:  # noqa: BLE001
                    pass
                try:
                    temp_c = max(temp_c, pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:  # noqa: BLE001
                    pass
            if utils:
                result["gpu_util_percent"] = round(sum(utils) / len(utils), 1)
            result["gpu_vram_gb"] = round(vram_used / 1024 ** 3, 3) if vram_used else None
            result["gpu_power_w"] = round(power_w, 1) if power_w else None
            result["gpu_temp_c"] = temp_c or None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"GPU 采样失败: {e}")
        return result

    # ------------------------------------------------------------------
    # NVML 初始化 / 关闭
    # ------------------------------------------------------------------
    def _init_nvml(self) -> None:
        try:
            try:
                import pynvml  # type: ignore
            except ImportError:
                from nvidia_ml_py import pynvml  # type: ignore  # noqa: F401
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            indices = self.gpu_indices if self.gpu_indices is not None else range(count)
            self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices if 0 <= i < count]
            self._nvml_ok = bool(self._nvml_handles)
        except Exception as e:  # noqa: BLE001
            # 无 GPU / 无驱动 / 无 nvidia-ml-py：降级为仅 CPU/内存
            logger.debug(f"NVML 不可用，仅采样 CPU/内存: {e}")
            self._nvml_ok = False
            self._nvml_handles = []

    def _shutdown_nvml(self) -> None:
        if not self._nvml_ok:
            return
        try:
            import pynvml  # type: ignore
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    METRIC_KEYS = ("cpu_percent", "system_memory_gb", "gpu_util_percent",
                   "gpu_vram_gb", "gpu_power_w", "gpu_temp_c")

    def _summarize(self) -> dict[str, Any]:
        samples = self._samples
        duration = (self._end_ts or time.monotonic()) - (self._start_ts or time.monotonic())

        if not samples:
            return self._empty_summary(duration)

        peaks: dict[str, float | None] = {}
        means: dict[str, float | None] = {}
        for key in self.METRIC_KEYS:
            vals = [s[key] for s in samples if s.get(key) is not None]
            if vals:
                # 内存/显存/功耗/温度取峰值；CPU/GPU 利用率峰值更有意义但也给均值
                peaks[key] = round(max(vals), 3)
                means[key] = round(sum(vals) / len(vals), 3)
            else:
                peaks[key] = None
                means[key] = None

        timeline = self._downsample(samples)
        gpu_available = any(s.get("gpu_util_percent") is not None for s in samples)

        return {
            "duration_seconds": round(duration, 3),
            "interval": self.interval,
            "gpu_monitored": gpu_available,
            "sample_count": len(samples),
            "peaks": peaks,
            "means": means,
            "timeline": timeline,
        }

    def _downsample(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(samples) <= _MAX_TIMELINE_POINTS:
            return samples
        stride = len(samples) / _MAX_TIMELINE_POINTS
        idxs = sorted({int(i * stride) for i in range(_MAX_TIMELINE_POINTS)})
        return [samples[i] for i in idxs if i < len(samples)]

    def _empty_summary(self, duration: float = 0.0) -> dict[str, Any]:
        peaks = dict.fromkeys(self.METRIC_KEYS)
        return {
            "duration_seconds": round(duration, 3),
            "interval": self.interval,
            "gpu_monitored": False,
            "sample_count": 0,
            "peaks": peaks,
            "means": dict(peaks),
            "timeline": [],
        }
