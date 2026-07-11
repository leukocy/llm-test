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

# Clocks throttle reason 位掩码 → 人话(NVML 文档值)
_THROTTLE_BITS: dict[int, str] = {
    0: "idle",
    1: "app_clocks",
    2: "sw_power_cap",
    4: "hw_slowdown",
    8: "sync_boost",
    16: "sw_thermal",
    32: "hw_thermal",
    64: "hw_power_brake",
}


def _decode_throttle(bitmask: int) -> str:
    """把 NVML throttle reason 位掩码解码成逗号分隔的原因名;0/None='none'。"""
    if not bitmask:
        return "none"
    names = [name for bit, name in _THROTTLE_BITS.items() if bit and bitmask & bit]
    return ",".join(names) if names else f"0x{bitmask:x}"


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
            sample["system_memory_gb"] = round(mem.used / 1024**3, 3)
        except Exception:  # noqa: BLE001
            sample["cpu_percent"] = None
            sample["system_memory_gb"] = None

        # GPU（聚合）
        gpu = self._sample_gpu()
        sample.update(gpu)
        return sample

    def _sample_gpu(self) -> dict[str, Any]:
        """采样 GPU:聚合字段(向后兼容)+ per_gpu 分卡明细(nvtop 级)。"""
        result: dict[str, Any] = {
            "gpu_util_percent": None,
            "gpu_vram_gb": None,
            "gpu_power_w": None,
            "gpu_temp_c": None,
            "per_gpu": [],
        }
        if not self._nvml_ok or not self._nvml_handles:
            return result
        try:
            import pynvml

            utils = []
            vram_used = 0.0
            power_w = 0.0
            temp_c = 0.0
            per_gpu: list[dict[str, Any]] = []
            for idx, handle in enumerate(self._nvml_handles):
                g: dict[str, Any] = {"index": idx}
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utils.append(util.gpu)
                    g["util"] = int(util.gpu)
                    g["mem_util"] = int(util.memory)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    mi = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_used += mi.used
                    g["vram_used_gb"] = round(mi.used / 1024**3, 2)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    pw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                    power_w += pw
                    g["power_w"] = round(pw, 1)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    tc = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temp_c = max(temp_c, tc)
                    g["temp_c"] = int(tc)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    g["sm_clock_mhz"] = int(
                        pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    g["mem_clock_mhz"] = int(
                        pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    g["fan_pct"] = int(pynvml.nvmlDeviceGetFanSpeed(handle))
                except Exception:  # noqa: BLE001
                    pass
                try:
                    g["pcie_rx_mbs"] = round(
                        pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                        / 1024.0,
                        2,
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    g["pcie_tx_mbs"] = round(
                        pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                        / 1024.0,
                        2,
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    g["throttle"] = _decode_throttle(
                        pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                    )
                except Exception:  # noqa: BLE001
                    pass
                per_gpu.append(g)
            if utils:
                result["gpu_util_percent"] = round(sum(utils) / len(utils), 1)
            result["gpu_vram_gb"] = round(vram_used / 1024**3, 3) if vram_used else None
            result["gpu_power_w"] = round(power_w, 1) if power_w else None
            result["gpu_temp_c"] = temp_c or None
            result["per_gpu"] = per_gpu
        except Exception as e:  # noqa: BLE001
            logger.debug(f"GPU 采样失败: {e}")
        return result

    # ------------------------------------------------------------------
    # NVML 初始化 / 关闭
    # ------------------------------------------------------------------
    def _init_nvml(self) -> None:
        try:
            try:
                import pynvml
            except ImportError:
                from nvidia_ml_py import pynvml  # noqa: F401
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            indices = self.gpu_indices if self.gpu_indices is not None else range(count)
            self._nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices if 0 <= i < count
            ]
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
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    METRIC_KEYS = (
        "cpu_percent",
        "system_memory_gb",
        "gpu_util_percent",
        "gpu_vram_gb",
        "gpu_power_w",
        "gpu_temp_c",
    )

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

        # per-GPU 峰值(分卡归因:哪张卡最热/最耗电/PCIe 最忙/是否降频)
        per_gpu_peaks = self._per_gpu_peaks(samples)
        # 全局观测到的 throttle 原因(去重)
        throttle_seen = sorted(
            {
                t
                for s in samples
                for g in (s.get("per_gpu") or [])
                for t in ([g["throttle"]] if g.get("throttle") and g["throttle"] != "none" else [])
                if t
            }
        )

        timeline = self._downsample(samples)
        gpu_available = any(s.get("gpu_util_percent") is not None for s in samples)

        return {
            "duration_seconds": round(duration, 3),
            "interval": self.interval,
            "gpu_monitored": gpu_available,
            "sample_count": len(samples),
            "peaks": peaks,
            "means": means,
            "per_gpu_peaks": per_gpu_peaks,
            "throttle_reasons_seen": throttle_seen,
            "timeline": timeline,
        }

    def _per_gpu_peaks(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """每张卡的峰值:最高温/最大功耗/最大利用率/PCIe 峰值/观测到的降频。"""
        by_idx: dict[int, dict[str, Any]] = {}
        peak_keys = (
            "util",
            "power_w",
            "temp_c",
            "sm_clock_mhz",
            "mem_clock_mhz",
            "pcie_rx_mbs",
            "pcie_tx_mbs",
        )
        for s in samples:
            for g in s.get("per_gpu") or []:
                idx = g.get("index")
                if idx is None:
                    continue
                slot = by_idx.setdefault(idx, {"index": idx, "throttle": set()})
                for k in peak_keys:
                    v = g.get(k)
                    if v is not None:
                        if k not in slot or v > slot[k]:
                            slot[k] = v
                th = g.get("throttle")
                if th and th != "none":
                    for t in th.split(","):
                        slot["throttle"].add(t)
        # throttle set → 排序列表(可 JSON 序列化)
        for slot in by_idx.values():
            slot["throttle"] = sorted(slot["throttle"])
        return [by_idx[i] for i in sorted(by_idx)]

    def _downsample(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # timeline 只保留聚合字段(去掉 per_gpu 明细,控体积);per-GPU 信息在 per_gpu_peaks
        light = [{k: v for k, v in s.items() if k != "per_gpu"} for s in samples]
        if len(light) <= _MAX_TIMELINE_POINTS:
            return light
        stride = len(light) / _MAX_TIMELINE_POINTS
        idxs = sorted({int(i * stride) for i in range(_MAX_TIMELINE_POINTS)})
        return [light[i] for i in idxs if i < len(light)]

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
