"""
推理引擎运行时指标轮询器（EngineMetricsPoller）。

测试期间用独立线程按固定间隔轮询 vLLM/SGLang 的 Prometheus `/metrics` 端点，
捕获引擎自身的运行视图——这是“记录推理引擎运行”的核心：
- GPU KV cache 占用率（gpu_cache_usage_perc）——手册 F 维：KV 实况
- 调度器队列：运行中 / 等待请求数
- 抢救数（num_preemption，KV 驱逐）——稳定性信号
- 引擎侧 TTFT / TPOT（直方图均值）——对照客户端测量值
- cache_config_info：block_size / num_gpu_blocks → KV 容量

为什么用线程：与 ResourceMonitor 同理，httpx 同步请求是阻塞的，不能进 async 事件循环。
端点不可达时优雅降级为 no-op（不影响测试）。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from core.prometheus_parser import (
    detect_engine_family,
    extract_sglang_runtime,
    extract_vllm_runtime,
    parse_prometheus,
)

logger = logging.getLogger(__name__)

_MAX_TIMELINE_POINTS = 3600


def default_metrics_url(api_base_url: str | None) -> str | None:
    """从 api_base_url 推导默认 /metrics 端点（同 host:port，路径换 /metrics）。"""
    if not api_base_url:
        return None
    url = api_base_url.strip()
    if "://" not in url:
        url = "http://" + url
    # 取 scheme://host[:port]
    try:
        scheme, rest = url.split("://", 1)
        host_port = rest.split("/", 1)[0]
        return f"{scheme}://{host_port}/metrics"
    except ValueError:
        return None


class EngineMetricsPoller:
    """轮询引擎 /metrics 的后台采样器。start()/stop() 成对，stop() 返回汇总 dict。"""

    def __init__(self, metrics_url: str | None, interval: float = 1.0, timeout: float = 2.0):
        self.metrics_url = metrics_url
        self.interval = max(0.2, float(interval))
        self.timeout = timeout
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._start_ts: float | None = None
        self._end_ts: float | None = None
        self._cache_config: dict[str, Any] = {}
        self._engine_family: str = "unknown"
        self._preemption_first: float | None = None
        self._client = None  # httpx.Client，惰性创建

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self.metrics_url:
            logger.debug("未配置引擎 metrics 端点，跳过引擎运行时采集")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import httpx
            self._client = httpx.Client(timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"httpx 不可用，引擎指标采集跳过: {e}")
            return
        self._stop_event.clear()
        self._samples = []
        self._start_ts = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="EngineMetricsPoller", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread is None:
            return self._empty_summary()
        self._stop_event.set()
        self._thread.join(timeout=self.interval * 2 + 1.0)
        self._thread = None
        self._end_ts = time.monotonic()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        return self._summarize()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_once()
            if sample is not None:
                self._samples.append(sample)
            self._stop_event.wait(timeout=self.interval)

    def _sample_once(self) -> dict[str, Any] | None:
        if self._client is None:
            return None
        t = time.monotonic() - (self._start_ts or time.monotonic())
        try:
            resp = self._client.get(self.metrics_url)
            if resp.status_code != 200:
                return None
            parsed = parse_prometheus(resp.text)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"引擎 metrics 轮询失败: {e}")
            return None

        if self._engine_family == "unknown":
            self._engine_family = detect_engine_family(parsed)

        rt = self._extract(parsed)
        # 首次成功时冻结 cache_config（运行期不变）
        if not self._cache_config and rt.get("cache_config"):
            self._cache_config = rt["cache_config"]
        if self._preemption_first is None and rt.get("num_preemption") is not None:
            self._preemption_first = rt["num_preemption"]

        return {
            "t": round(t, 3),
            "gpu_cache_usage_perc": rt.get("gpu_cache_usage_perc"),
            "num_requests_running": rt.get("num_requests_running"),
            "num_requests_waiting": rt.get("num_requests_waiting"),
            "num_preemption": rt.get("num_preemption"),
            "ttft_mean_s": rt.get("ttft_mean_s"),
            "tpot_mean_s": rt.get("tpot_mean_s"),
        }

    def _extract(self, parsed: dict[str, Any]) -> dict[str, Any]:
        if self._engine_family == "sglang" or (
            self._engine_family == "unknown" and detect_engine_family(parsed) == "sglang"
        ):
            return extract_sglang_runtime(parsed)
        return extract_vllm_runtime(parsed)

    # ------------------------------------------------------------------
    def _summarize(self) -> dict[str, Any]:
        duration = (self._end_ts or time.monotonic()) - (self._start_ts or time.monotonic())
        if not self._samples:
            return self._empty_summary(duration)

        def _peak(key):
            vals = [s[key] for s in self._samples if s.get(key) is not None]
            return max(vals) if vals else None

        def _last(key):
            # 末采样点值（vLLM/SGLang 直方图是累积的，末点均值 = 引擎整段整体值）
            for s in reversed(self._samples):
                if s.get(key) is not None:
                    return s[key]
            return None

        # num_preemption 是累加 counter：取窗口内增量（需 ≥2 个样本才能算）
        preemption_vals = [s["num_preemption"] for s in self._samples if s.get("num_preemption") is not None]
        preemption_total = None
        if len(preemption_vals) >= 2:
            preemption_total = preemption_vals[-1] - preemption_vals[0]

        cc = self._cache_config or {}
        kv_capacity_tokens = None
        if cc.get("num_gpu_blocks") and cc.get("block_size"):
            kv_capacity_tokens = cc["num_gpu_blocks"] * cc["block_size"]

        return {
            "engine_family": self._engine_family,
            "metrics_url": self.metrics_url,
            "duration_seconds": round(duration, 3),
            "interval": self.interval,
            "sample_count": len(self._samples),
            "peaks": {
                "gpu_cache_usage_perc": _peak("gpu_cache_usage_perc"),
                "num_requests_running": _peak("num_requests_running"),
                "num_requests_waiting": _peak("num_requests_waiting"),
                "ttft_mean_s": _peak("ttft_mean_s"),
                "tpot_mean_s": _peak("tpot_mean_s"),
            },
            "engine_means": {
                # 末采样点的直方图均值（直方图累积，末点 = 引擎整段整体 TTFT/TPOT）
                "ttft_s": _last("ttft_mean_s"),
                "tpot_s": _last("tpot_mean_s"),
            },
            "preemption_total": preemption_total,
            "cache_config": {
                "block_size": cc.get("block_size"),
                "num_gpu_blocks": cc.get("num_gpu_blocks"),
                "num_cpu_blocks": cc.get("num_cpu_blocks"),
                "kv_capacity_tokens": kv_capacity_tokens,
            },
            "timeline": self._downsample(),
        }

    def _downsample(self) -> list[dict[str, Any]]:
        if len(self._samples) <= _MAX_TIMELINE_POINTS:
            return self._samples
        stride = len(self._samples) / _MAX_TIMELINE_POINTS
        idxs = sorted({int(i * stride) for i in range(_MAX_TIMELINE_POINTS)})
        return [self._samples[i] for i in idxs if i < len(self._samples)]

    def _empty_summary(self, duration: float = 0.0) -> dict[str, Any]:
        return {
            "engine_family": "unknown",
            "metrics_url": self.metrics_url,
            "duration_seconds": round(duration, 3),
            "interval": self.interval,
            "sample_count": 0,
            "peaks": {},
            "engine_means": {},
            "preemption_total": None,
            "cache_config": {},
            "timeline": [],
        }
