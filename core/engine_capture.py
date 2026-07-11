"""推理引擎配置自动采集(per-run,绑定每次测试)—— 多引擎适配器架构。

为什么需要:引擎配置(max_seqs/KV/并行/冷启动)是结论关键上下文,硬编码会陈旧。
vLLM/SGLang/llama.cpp/ktransformers/fastllm 日志格式与参数名各不相同,故用
**适配器模式**:通用部分(docker inspect + OpenAI /v1/models)+ 每引擎适配器
(各自的日志解析 + 参数归一化),capture_engine_config 自动探测引擎后分派。

设计:
- 通用层:launch_cmd/image/env(docker inspect)+ model_id/max_model_len(/v1/models)。
- 适配器:EngineCaptureAdapter.detect() 识别引擎;.parse_logs() 解析日志;
  .normalize_params() 把引擎特有参数名归一化到标准 C 维度 schema。
- 新增引擎:写个 Adapter 子类 + 进 _ADAPTERS 注册表即可,不动通用层。
- 全防御:docker/日志/API 不可用优雅降级,永不抛异常。
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


def _run(args: list[str], timeout: float = 12.0) -> str | None:
    try:
        r = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def find_vllm_container(api_base_url: str, hint: str | None = None) -> str | None:
    """按服务端口(从 api_base_url 解析)在 docker ps 里找容器名。"""
    out = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"])
    if not out:
        return None
    m = re.search(r":(\d+)/?", api_base_url or "")
    port = m.group(1) if m else None
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        name, ports = parts
        if port and f":{port}->" in ports:
            return name
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            name = parts[0]
            if hint and hint in name:
                return name
    return None


# ---------------------------------------------------------------------------
# 适配器基类 + 各引擎实现
# ---------------------------------------------------------------------------
class EngineCaptureAdapter:
    """引擎采集适配器基类。子类实现 detect/parse_logs/normalize_params。"""

    name = "base"
    image_keywords: tuple[str, ...] = ()
    cmd_keywords: tuple[str, ...] = ()

    @classmethod
    def detect(
        cls, image: str, launch_cmd: str, api_model: dict | None
    ) -> bool:  # noqa: ARG003
        s = f"{image} {launch_cmd}".lower()
        return any(k in s for k in cls.image_keywords) or any(
            k in s for k in cls.cmd_keywords
        )

    @classmethod
    def parse_logs(cls, logs: str) -> dict[str, Any]:  # noqa: ARG003
        """解析引擎日志 → {runtime{}, args{}}(引擎特有)。默认空。"""
        return {}

    @classmethod
    def normalize_params(
        cls, launch_cmd: str, parsed: dict[str, Any]
    ) -> dict[str, Any]:  # noqa: ARG003
        """把引擎参数归一化到标准 schema:parallel/schedule/runtime。默认空。"""
        return {}


def _last(logs: str, pat: str) -> str | None:
    ms = re.findall(pat, logs)
    return ms[-1] if ms else None


class VLLMAdapter(EngineCaptureAdapter):
    name = "vLLM"
    image_keywords = ("vllm/vllm", "/vllm-openai", "vllm")
    cmd_keywords = ("vllm serve",)

    @classmethod
    def parse_logs(cls, logs: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        # non-default args
        matches = re.findall(
            r"non-default args: (\{.*?\})\s*(?=\[|\n|$)", logs, re.DOTALL
        )
        if matches:
            raw = matches[-1]
            args: dict[str, Any] = {}
            for key in (
                "tensor_parallel_size",
                "decode_context_parallel_size",
                "enable_expert_parallel",
                "gpu_memory_utilization",
                "max_num_seqs",
                "enable_prefix_caching",
                "max_model_len",
                "pipeline_parallel_size",
                "data_parallel_size",
            ):
                m = re.search(rf"'{key}':\s*([^,}}]+)", raw)
                if m:
                    v = m.group(1).strip().strip("'\"")
                    try:
                        args[key] = (
                            int(v)
                            if re.fullmatch(r"-?\d+", v)
                            else (float(v) if re.fullmatch(r"-?\d+\.\d+", v) else v)
                        )
                    except ValueError:
                        args[key] = v
            out["args"] = args
        # runtime 分解
        runtime: dict[str, Any] = {}
        for k, pat in (
            ("weight_load_s", r"Loading weights took ([\d.]+) seconds"),
            (
                "model_load_s",
                r"Model loading took [0-9.]+ GiB memory and ([\d.]+) seconds",
            ),
            ("init_engine_s", r"init engine .* took ([\d.]+) s"),
            ("graph_capture_s", r"Graph capturing finished in (\d+) secs"),
        ):
            v = _last(logs, pat)
            if v:
                runtime[k] = float(v) if "." in v else int(v)
        kv = _last(logs, r"GPU KV cache size: ([\d,]+) tokens")
        if kv:
            runtime["kv_cache_tokens"] = int(kv.replace(",", ""))
        if runtime:
            wl, ie, gc = (
                runtime.get("weight_load_s", 0),
                runtime.get("init_engine_s", 0),
                runtime.get("graph_capture_s", 0),
            )
            if wl or ie or gc:
                runtime["cold_start_s_est"] = round(wl + ie + gc, 1)
            out["runtime"] = runtime
        return out

    @classmethod
    def normalize_params(
        cls, launch_cmd: str, parsed: dict[str, Any]
    ) -> dict[str, Any]:  # noqa: ARG003
        args = parsed.get("args") or {}
        out: dict[str, Any] = {}
        if args:
            out["schedule"] = {
                k: args[k]
                for k in (
                    "max_num_seqs",
                    "gpu_memory_utilization",
                    "enable_prefix_caching",
                    "max_model_len",
                )
                if k in args
            }
            out["parallel"] = {
                "tp": args.get("tensor_parallel_size"),
                "dcp": args.get("decode_context_parallel_size"),
                "ep": args.get("enable_expert_parallel"),
            }
        if parsed.get("runtime"):
            out["runtime"] = parsed["runtime"]
        return out


class SGLangAdapter(EngineCaptureAdapter):
    name = "SGLang"
    image_keywords = ("sglang", "lmsys/sglang")
    cmd_keywords = ("sglang.launch", "python -m sglang", "-m sglang")

    @classmethod
    def parse_logs(cls, logs: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        runtime: dict[str, Any] = {}
        for k, pat in (
            ("init_s", r"model load took ([\d.]+) s"),
            ("graph_capture_s", r"Capture cudagraph.*?(\d+\.?\d*)\s*s"),
        ):
            v = _last(logs, pat)
            if v:
                runtime[k] = float(v) if "." in v else int(v)
        kv = _last(logs, r"max_total_num_token *= *(\d+)") or _last(
            logs, r"KV cache size: (\d+)"
        )
        if kv:
            runtime["kv_cache_tokens"] = int(kv)
        if runtime:
            out["runtime"] = runtime
        return out

    @classmethod
    def normalize_params(
        cls, launch_cmd: str, parsed: dict[str, Any]
    ) -> dict[str, Any]:
        # SGLang 参数从 launch_cmd 解析(--tp, --max-running-requests, --mem-fraction-static)
        def flag(name: str) -> str | None:
            m = re.search(rf"--{name}[ =](\S+)", launch_cmd)
            return m.group(1) if m else None

        out: dict[str, Any] = {}
        sched: dict[str, Any] = {}
        max_running = flag("max-running-requests")
        if max_running:
            sched["max_running_requests"] = int(max_running)
        mem_fraction = flag("mem-fraction-static")
        if mem_fraction:
            sched["gpu_memory_utilization"] = float(mem_fraction)
        if sched:
            out["schedule"] = sched
        tp = flag("tp")
        out["parallel"] = {"tp": int(tp) if tp and tp.isdigit() else tp}
        if parsed.get("runtime"):
            out["runtime"] = parsed["runtime"]
        return out


class LlamaCppAdapter(EngineCaptureAdapter):
    """llama.cpp / llama-server。无 max_seqs 概念(同步/连续批),KV=-c 上下文。"""

    name = "llama.cpp"
    image_keywords = ("llama.cpp", "ggml", "ghcr.io/ggerganov")
    cmd_keywords = ("llama-server", "llama.cpp/server", "main -m")

    @classmethod
    def parse_logs(cls, logs: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        wl = _last(logs, r"load_model.*?([\d.]+)\s*(?:ms|s)") or _last(
            logs, r"model loaded in ([\d.]+) ms"
        )
        if wl:
            out["runtime"] = {"model_load_ms": float(wl)}
        return out

    @classmethod
    def normalize_params(
        cls, launch_cmd: str, parsed: dict[str, Any]
    ) -> dict[str, Any]:
        def flag(name: str) -> str | None:
            m = re.search(rf"-{name}\s+(\S+)", launch_cmd)
            return m.group(1) if m else None

        out: dict[str, Any] = {"schedule": {}}
        ctx = flag("c")
        if ctx and ctx.isdigit():
            out["schedule"]["context_length"] = int(ctx)
            out["runtime"] = {
                "kv_cache_tokens_est": int(ctx)
            }  # llama.cpp KV ≈ 上下文容量
        ngl = flag("ngl")
        if ngl:
            out["schedule"]["gpu_layers"] = int(ngl) if ngl.isdigit() else ngl
        out["parallel"] = {"tp": 1, "note": "llama.cpp 单进程,-ngl 控制 GPU 层"}
        if parsed.get("runtime"):
            out["runtime"].update(parsed["runtime"])
        return out


class KTransformersAdapter(EngineCaptureAdapter):
    name = "ktransformers"
    image_keywords = ("ktransformers",)
    cmd_keywords = ("ktransformers",)

    @classmethod
    def normalize_params(
        cls, launch_cmd: str, parsed: dict[str, Any]
    ) -> dict[str, Any]:  # noqa: ARG003
        # ktransformers 配置在 yaml/gguf,API 暴露有限;记 launch_cmd 即可
        return {
            "parallel": {"note": "ktransformers 配置见 yaml(本采集仅记 launch_cmd)"}
        }


class FastLLMAdapter(EngineCaptureAdapter):
    name = "fastllm"
    image_keywords = ("fastllm",)
    cmd_keywords = ("fastllm",)


# 注册表(顺序=探测优先级;vLLM 先,最常见)
_ADAPTERS: list[type[EngineCaptureAdapter]] = [
    VLLMAdapter,
    SGLangAdapter,
    LlamaCppAdapter,
    KTransformersAdapter,
    FastLLMAdapter,
]


def detect_adapter(
    image: str, launch_cmd: str, api_model: dict | None
) -> type[EngineCaptureAdapter] | None:
    for a in _ADAPTERS:
        try:
            if a.detect(image, launch_cmd, api_model):
                return a
        except Exception:  # noqa: BLE001
            continue
    return None


def get_adapters() -> list[str]:
    """已注册的引擎适配器名(UI/调试用)。"""
    return [a.name for a in _ADAPTERS]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def capture_engine_config(
    api_base_url: str, container_name: str | None = None
) -> dict[str, Any]:
    """自动采集推理引擎配置(docker inspect + 日志 + /v1/models + 引擎适配器)。永不抛异常。"""
    result: dict[str, Any] = {
        "captured_at": datetime.now().isoformat(),
        "capture_source": [],
    }

    # 1. /v1/models(最可靠,OpenAI 兼容引擎通用)
    api_model: dict | None = None
    if httpx:
        try:
            r = httpx.get(f"{api_base_url.rstrip('/')}/models", timeout=8.0)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    api_model = {
                        "model_id": data[0].get("id"),
                        "max_model_len": data[0].get("max_model_len"),
                    }
                    result["model"] = api_model
                    result["capture_source"].append("api")
        except Exception:  # noqa: BLE001
            pass

    # 2. docker inspect + logs
    container = container_name or find_vllm_container(api_base_url)
    if not container:
        result["capture_source"].append("no_container")
        return result
    result["container"] = container

    cmd = _run(
        [
            "docker",
            "inspect",
            container,
            "--format",
            "{{.Config.Cmd}}|||{{.Config.Image}}",
        ]
    )
    image = ""
    if cmd:
        parts = cmd.split("|||", 1)
        result["launch_cmd"] = parts[0].strip()
        image = parts[1].strip() if len(parts) > 1 else ""
        result["image"] = image
        mv = re.search(r":v?([\d.]+)", image)
        if mv:
            result["engine_version"] = mv.group(1)
        result["capture_source"].append("docker_inspect")

    env = _run(
        [
            "docker",
            "inspect",
            container,
            "--format",
            "{{range .Config.Env}}{{println .}}{{end}}",
        ]
    )
    if env:
        envd = {
            k: v
            for line in env.splitlines()
            if "=" in line
            for k, _, v in [line.partition("=")]
            if re.search(r"VLLM_|SGLANG_|PYTORCH_CUDA|CUDA_VERSION|KT_", k)
        }
        if envd:
            result["env"] = envd

    logs = _run(["docker", "logs", container], timeout=30.0) or ""

    # 3. 探测引擎 + 分派适配器
    adapter = detect_adapter(image, result.get("launch_cmd", ""), api_model)
    if adapter:
        result["engine"] = adapter.name
        result["adapter"] = adapter.name
        try:
            parsed = adapter.parse_logs(logs) if logs else {}
            norm = adapter.normalize_params(result.get("launch_cmd", ""), parsed)
            if norm.get("schedule"):
                result["schedule"] = norm["schedule"]
                result.update(norm["schedule"])  # 顶层提升(max_num_seqs 等)
            if norm.get("parallel"):
                result["parallel_strategy"] = norm["parallel"]
            if norm.get("runtime"):
                result["runtime"] = norm["runtime"]
            if logs:
                result["capture_source"].append(f"adapter:{adapter.name}")
        except Exception:  # noqa: BLE001  适配器失败不影响通用部分
            result["capture_source"].append(f"adapter:{adapter.name}:error")
    else:
        result["engine"] = image.split("/")[-1] if image else "unknown"
        result["adapter"] = None
        result["capture_source"].append("no_adapter(仅通用:docker+api)")

    return result


if __name__ == "__main__":  # 手动验证
    import json

    print("已注册适配器:", get_adapters())
    print(
        json.dumps(
            capture_engine_config("http://localhost:10814/v1"),
            ensure_ascii=False,
            indent=2,
        )
    )
