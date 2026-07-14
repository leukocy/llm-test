"""
推理引擎运行时指标轮询器（EngineMetricsPoller）。

测试期间用独立线程按固定间隔轮询 vLLM/SGLang 的 Prometheus `/metrics` 端点，
捕获引擎自身的运行视图——这是“记录推理引擎运行”的核心：
- GPU KV cache 占用率（gpu_cache_usage_perc）——手册 F 维：KV 实况
- 调度器队列：运行中 / 等待请求数
- 抢救数（num_preemption，KV 驱逐）——稳定性信号
- 引擎侧 TTFT / TPOT（直方图均值）——对照客户端测量值
- cache_config_info：优先 kv_cache_size_tokens，旧版回退 block_size × num_gpu_blocks

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
_KV_PROBE_CACHE: dict[str, dict[str, Any]] = {}
_KV_PROBE_EVENTS: dict[str, threading.Event] = {}
_KV_PROBE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 测试前一次性 KV 容量探测（自适应测试用，不启动轮询线程）
# ---------------------------------------------------------------------------


def probe_kv_capacity(api_base_url: str | None, timeout: float = 5.0) -> dict[str, Any]:
    """测试前一次性探测 KV 缓存容量（tokens），用于自适应跳过超预算 cell。

    优先级：
      1. 引擎 /metrics 的 cache_config_info：优先使用 vLLM 的
         kv_cache_size_tokens（group-aware），旧版才回退 block_size×num_gpu_blocks。
      2. /v1/models 的 max_model_len——模型上下文上限，作 KV 预算上界（保守，
         实际 KV 池可能更小，但引擎不在跑 / /metrics 不可达时是唯一来源）。
      3. 都拿不到 → kv_capacity_tokens=None（调用方回退到不跳过，全跑）。

    纯函数、不启动线程、永不抛异常（探测失败比采集不到更糟，回退到全跑）。
    返回 {kv_capacity_tokens, source, max_model_len, metrics_ok}。
    """
    result: dict[str, Any] = {
        "kv_capacity_tokens": None,
        "source": None,  # "metrics" / "models" / None
        "max_model_len": None,
        "metrics_ok": False,
    }
    metrics_url = default_metrics_url(api_base_url)

    # 1. /metrics → cache_config
    if metrics_url:
        try:
            import httpx  # 与 poller 同源；缺 httpx 则跳过本路径

            with httpx.Client(timeout=timeout) as c:
                resp = c.get(metrics_url)
                if resp.status_code == 200:
                    parsed = parse_prometheus(resp.text)
                    family = detect_engine_family(parsed)
                    rt = (
                        extract_vllm_runtime(parsed)
                        if family != "sglang"
                        else extract_sglang_runtime(parsed)
                    )
                    cc = rt.get("cache_config") or {}
                    capacity = cc.get("kv_cache_size_tokens")
                    blocks = cc.get("num_gpu_blocks")
                    bsize = cc.get("block_size")
                    if capacity:
                        result["kv_capacity_tokens"] = int(capacity)
                        result["source"] = "metrics"
                        result["metrics_ok"] = True
                    elif blocks and bsize:
                        result["kv_capacity_tokens"] = int(blocks) * int(bsize)
                        result["source"] = "metrics"
                        result["metrics_ok"] = True
        except Exception as e:  # noqa: BLE001  探测兜底，绝不抛
            logger.debug(f"probe_kv_capacity /metrics 失败: {e}")

    # 2. /v1/models → max_model_len 兜底
    if result["kv_capacity_tokens"] is None and api_base_url:
        try:
            import httpx

            with httpx.Client(timeout=timeout) as c:
                resp = c.get(api_base_url.rstrip("/") + "/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data") if isinstance(data, dict) else None
                    if models:
                        mml = models[0].get("max_model_len")
                        if mml:
                            result["max_model_len"] = int(mml)
                            # max_model_len 是单请求上下文上限；KV 池通常 ≥ 它但
                            # 多并发时总占用受限于池大小。作保守上界用。
                            result["kv_capacity_tokens"] = int(mml)
                            result["source"] = result["source"] or "models"
        except Exception as e:  # noqa: BLE001
            logger.debug(f"probe_kv_capacity /v1/models 失败: {e}")

    return result


def warm_kv_capacity_cache(api_base_url: str | None) -> None:
    """Prefetch KV capacity so endpoint probing is off the first-request path."""
    key = (api_base_url or "").strip().rstrip("/")
    if not key:
        return

    with _KV_PROBE_LOCK:
        if key in _KV_PROBE_CACHE or key in _KV_PROBE_EVENTS:
            return
        ready = threading.Event()
        _KV_PROBE_EVENTS[key] = ready

    def _probe() -> None:
        try:
            result = probe_kv_capacity(key)
            with _KV_PROBE_LOCK:
                _KV_PROBE_CACHE[key] = result
        finally:
            ready.set()

    threading.Thread(
        target=_probe,
        name="kv-capacity-prefetch",
        daemon=True,
    ).start()


def get_cached_kv_capacity(api_base_url: str | None) -> dict[str, Any]:
    """Return a prefetched probe result, or perform the first probe synchronously."""
    key = (api_base_url or "").strip().rstrip("/")
    if not key:
        return probe_kv_capacity(api_base_url)

    with _KV_PROBE_LOCK:
        cached = _KV_PROBE_CACHE.get(key)
        ready = _KV_PROBE_EVENTS.get(key)

    if cached is not None:
        return dict(cached)

    if ready is not None:
        ready.wait()
        with _KV_PROBE_LOCK:
            return dict(_KV_PROBE_CACHE.get(key, {}))

    result = probe_kv_capacity(key)
    with _KV_PROBE_LOCK:
        _KV_PROBE_CACHE[key] = result
    return dict(result)


def estimate_kv_need(concurrency: int, context_tokens: int, max_tokens: int) -> int:
    """估算一个 cell 的 KV 占用（tokens）：并发 × (上下文 + 最大输出)。

    与 live_bench.build_phases 同口径：conc*ctx + conc*max_tokens ≤ kv_budget。
    context_tokens 是输入 prompt 长度，max_tokens 是生成上限。
    """
    return concurrency * (context_tokens + max_tokens)


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
    """轮询引擎 /metrics 的后台采样器。start()/stop() 成对，stop() 返回汇总 dict。

    若连续 _MAX_CONSECUTIVE_FAILURES 次请求均未返回 200，自动停止轮询并记录警告，
    避免引擎日志被 404 刷屏。
    """

    _MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self, metrics_url: str | None, interval: float = 5.0, timeout: float = 2.0
    ):
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
        self._client: Any = None  # httpx.Client，惰性创建
        self._consecutive_failures = 0

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
        self._thread = threading.Thread(
            target=self._run, name="EngineMetricsPoller", daemon=True
        )
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
                self._consecutive_failures = 0
                self._samples.append(sample)
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                    logger.warning(
                        "引擎 metrics 端点 %s 连续 %d 次不可达，停止轮询（不影响测试）",
                        self.metrics_url,
                        self._consecutive_failures,
                    )
                    self._stop_event.set()
                    break
            if not self._stop_event.is_set():
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
            "num_requests_swapped": rt.get("num_requests_swapped"),
            "ttft_mean_s": rt.get("ttft_mean_s"),
            "tpot_mean_s": rt.get("tpot_mean_s"),
            "gpu_prefix_cache_hit_rate": rt.get("gpu_prefix_cache_hit_rate"),
            "spec_token_acceptance_rate": rt.get("spec_token_acceptance_rate"),
        }

    def _extract(self, parsed: dict[str, Any]) -> dict[str, Any]:
        if self._engine_family == "sglang" or (
            self._engine_family == "unknown"
            and detect_engine_family(parsed) == "sglang"
        ):
            return extract_sglang_runtime(parsed)
        return extract_vllm_runtime(parsed)

    # ------------------------------------------------------------------
    def _summarize(self) -> dict[str, Any]:
        duration = (self._end_ts or time.monotonic()) - (
            self._start_ts or time.monotonic()
        )
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
        preemption_vals = [
            s["num_preemption"]
            for s in self._samples
            if s.get("num_preemption") is not None
        ]
        preemption_total = None
        if len(preemption_vals) >= 2:
            preemption_total = preemption_vals[-1] - preemption_vals[0]

        cc = self._cache_config or {}
        kv_capacity_tokens = cc.get("kv_cache_size_tokens")
        if (
            kv_capacity_tokens is None
            and cc.get("num_gpu_blocks")
            and cc.get("block_size")
        ):
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
                "num_requests_swapped": _peak("num_requests_swapped"),
            },
            "engine_means": {
                # 末采样点的直方图均值（直方图累积，末点 = 引擎整段整体 TTFT/TPOT）
                "ttft_s": _last("ttft_mean_s"),
                "tpot_s": _last("tpot_mean_s"),
                # prefix cache 命中率 / 推测解码接受率(末点 = 整体累积值)
                "gpu_prefix_cache_hit_rate": _last("gpu_prefix_cache_hit_rate"),
                "spec_token_acceptance_rate": _last("spec_token_acceptance_rate"),
            },
            "preemption_total": preemption_total,
            "cache_config": {
                "block_size": cc.get("block_size"),
                "num_gpu_blocks": cc.get("num_gpu_blocks"),
                "num_cpu_blocks": cc.get("num_cpu_blocks"),
                "kv_cache_size_tokens": cc.get("kv_cache_size_tokens"),
                "kv_cache_max_concurrency": cc.get("kv_cache_max_concurrency"),
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
