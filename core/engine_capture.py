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
    """按服务端口(从 api_base_url 解析)在 docker ps 里找容器名。

    四级匹配(逐级降级):
    1. 端口映射:`docker ps` 的 Ports 列含 `:PORT->`(bridge 网络模式);
    2. host 网络:容器无端口映射但 Cmd/Env 里含 `--port PORT`(host 网络模式);
    3. 引擎特征:api_base 端口经 proxy 转发(端口不直接匹配)时,通过 image/cmd
       识别推理引擎容器(vllm/sglang/tgi 等关键词 + 有 PORT= 环境变量);
    4. 名称子串:hint 匹配容器名。
    """
    out = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}\t{{.Image}}"])
    if not out:
        return None
    m = re.search(r":(\d+)/?", api_base_url or "")
    port = m.group(1) if m else None
    names = []
    _engine_image_keywords = (
        "vllm",
        "sglang",
        "tgi",
        "text-generation-inference",
        "triton",
        "tensorrt-llm",
        "lmdeploy",
    )
    port_match_candidates: list[str] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        names.append(parts[0])
        name, ports = parts[0], parts[1]
        # 1. 端口映射(bridge 网络)——收集候选,优先返回引擎容器
        if port and f":{port}->" in ports:
            image = parts[2].lower() if len(parts) >= 3 else ""
            if any(k in image for k in _engine_image_keywords):
                return name  # 引擎容器直接命中
            port_match_candidates.append(name)
    # 端口命中了但非引擎容器(proxy/gateway),不直接返回,继续降级尝试
    # 2. host 网络:容器无端口映射,但启动命令/env 含 --port PORT
    _engine_image_keywords = (
        "vllm",
        "sglang",
        "tgi",
        "text-generation-inference",
        "triton",
        "tensorrt-llm",
        "lmdeploy",
    )
    host_port_candidates: list[str] = []
    if port and names:
        for name in names:
            cmd = _run(
                [
                    "docker",
                    "inspect",
                    name,
                    "--format",
                    "{{.Config.Cmd}} {{range .Config.Env}}{{println .}}{{end}}",
                ],
                timeout=8.0,
            )
            if cmd and (
                f"--port {port}" in cmd
                or f"--port={port}" in cmd
                or f"PORT={port}" in cmd
            ):
                # 优先返回引擎容器;非引擎(如 Open-WebUI)降级为候选
                image = ""
                for line in out.splitlines():
                    cols = line.split("\t")
                    if len(cols) >= 3 and cols[0] == name:
                        image = cols[2].lower()
                        break
                if any(k in image for k in _engine_image_keywords):
                    return name
                host_port_candidates.append(name)
    # 3. 引擎特征:端口不直接匹配(可能经 proxy 转发),按 image/cmd 识别推理引擎容器
    for name in names:
        image = ""
        for line in out.splitlines():
            cols = line.split("\t")
            if len(cols) >= 3 and cols[0] == name:
                image = cols[2].lower()
                break
        if not image:
            continue
        # image 名含引擎关键词 + 无端口映射(host 网络)→ 推理引擎候选
        if any(k in image for k in _engine_image_keywords):
            ports_check = _run(
                [
                    "docker",
                    "inspect",
                    name,
                    "--format",
                    "{{json .NetworkSettings.Ports}}",
                ],
                timeout=5.0,
            )
            # host 网络模式(空端口映射)或 PORT= 环境变量 → 高置信度
            if ports_check and (ports_check.strip() in ("{}", "null", '""')):
                return name
    # 4. 名称子串
    for name in names:
        if hint and hint in name:
            return name
    # 5. 回退:端口命中的非引擎候选(proxy/gateway 转发到引擎)
    if port_match_candidates:
        return port_match_candidates[0]
    if host_port_candidates:
        return host_port_candidates[0]
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


def _first(logs: str, pat: str) -> str | None:
    """取首个匹配。用于「Loading weights took X seconds」这类多次出现的日志——
    vLLM 会先记录真实权重加载(如 127s),再记录 draft 模型/快速重载(如 1.05s);
    _last 会误取后者。取首个才反映真实加载耗时。"""
    ms = re.findall(pat, logs)
    return ms[0] if ms else None


def _max_float(logs: str, pat: str) -> str | None:
    """取所有匹配中的最大值。多 worker 并行各记一行,取 max = 最慢 worker(决定整体耗时)。"""
    ms = re.findall(pat, logs)
    if not ms:
        return None
    try:
        return str(max(float(m) for m in ms))
    except ValueError:
        return ms[0]


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
            # 标量字段:用 [^,}}]+ 提取
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
                "block_size",
                "enforce_eager",
                "quantization",
                "kv_cache_dtype",
                "attention_backend",
                "moe_backend",
                "num_speculative_tokens",
                "speculative_model",
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
            # 嵌套字典字段(speculative_config):用平衡括号提取
            m = re.search(r"'speculative_config':\s*(\{[^}]*\})", raw)
            if m:
                spec_raw = m.group(1)
                spec: dict[str, Any] = {}
                for sk in (
                    "method",
                    "num_speculative_tokens",
                    "model",
                    "moe_backend",
                    "draft_sample_method",
                ):
                    sm = re.search(rf"'{sk}':\s*([^,}}]+)", spec_raw)
                    if sm:
                        sv = sm.group(1).strip().strip("'\"")
                        try:
                            spec[sk] = int(sv) if sv.isdigit() else sv
                        except ValueError:
                            spec[sk] = sv
                if spec:
                    args["speculative_config"] = spec
            out["args"] = args
        # runtime 分解
        # 注意「Loading weights took」会多次出现(真实加载 + draft/快速重载),取首个;
        # 「Model loading took ... and X seconds」是含显存分配的完整加载(每 worker 一行),取 max(最慢 worker);
        # 「init engine took」/「Graph capturing」也取 max(多 worker)。
        runtime: dict[str, Any] = {}
        # Model loading(完整,最权威)优先;Loading weights 作 fallback
        ml = _max_float(
            logs,
            r"Model loading took [0-9.]+ (?:GiB|GB) (?:memory )?and ([\d.]+) seconds",
        )
        if ml:
            runtime["model_load_s"] = float(ml)
        wl = _first(logs, r"Loading weights took ([\d.]+) seconds")
        if wl:
            runtime["weight_load_s"] = float(wl)
        ie = _max_float(logs, r"init engine .* took ([\d.]+) s")
        if ie:
            runtime["init_engine_s"] = float(ie)
        gc = _max_float(logs, r"Graph capturing finished in (\d+) secs")
        if gc:
            runtime["graph_capture_s"] = int(float(gc))
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
                    "block_size",
                    "enforce_eager",
                )
                if k in args
            }
            out["parallel"] = {
                "tp": args.get("tensor_parallel_size"),
                "dcp": args.get("decode_context_parallel_size"),
                "ep": args.get("enable_expert_parallel"),
            }
            # 后端/量化(serving 配置关键字段)
            backends: dict[str, Any] = {}
            for k in (
                "quantization",
                "kv_cache_dtype",
                "attention_backend",
                "moe_backend",
            ):
                if k in args:
                    backends[k] = args[k]
            if backends:
                out["backends"] = backends
            # MTP/推测解码
            mtp: dict[str, Any] = {}
            if args.get("speculative_config"):
                mtp["speculative_config"] = args["speculative_config"]
                mtp["mtp_enabled"] = True
            if args.get("num_speculative_tokens"):
                mtp["num_speculative_tokens"] = args["num_speculative_tokens"]
                mtp["mtp_enabled"] = True
            if args.get("speculative_model"):
                mtp["speculative_model"] = args["speculative_model"]
            if mtp:
                out["mtp"] = mtp
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


class HeyiAdapter(EngineCaptureAdapter):
    """合一是(heyi-engine)自研推理引擎。镜像名含 heyi-engine,启动命令以
    /model/<name> 开头,参数用 --kvcache-num-tokens / --prefill-chunk-size /
    --num-decode-workers / --max-length 等 heyi 专有 flag。
    日志首行打印完整 args dict(heyi/server/main.py),可解析出 use_cuda_graph /
    max_batch_size / max_new_tokens / enable_layerwise_prefill 等运行时参数。"""

    name = "heyi-engine"
    image_keywords = ("heyi-engine", "xingyun/heyi")
    cmd_keywords = (
        "--kvcache-num-tokens",
        "--num-decode-workers",
        "--layerwise-prefill",
    )

    @classmethod
    def parse_logs(cls, logs: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        # 日志首行:heyi/server/main.py 打印完整 args dict
        m = re.search(r"\{[^}]*'kvcache_num_tokens'[^}]*\}", logs, re.DOTALL)
        if m:
            raw = m.group(0)
            args: dict[str, Any] = {}
            for key in (
                "kvcache_num_tokens",
                "kvcache_num_gpu_tokens",
                "max_length",
                "prefill_chunk_size",
                "layerwise_prefill_thresh_len",
                "layerwise_prefill_device",
                "layerwise_prefill_world_size",
                "num_decode_workers",
                "batch_sizes_per_runner",
                "max_batch_size",
                "max_new_tokens",
                "num_cpu_threads",
                "use_cuda_graph",
                "enable_layerwise_prefill",
                "thinking",
            ):
                sm = re.search(rf"'{key}':\s*([^,}}]+)", raw)
                if sm:
                    v = sm.group(1).strip().strip("'\"")
                    try:
                        args[key] = (
                            int(v)
                            if re.fullmatch(r"-?\d+", v)
                            else (float(v) if re.fullmatch(r"-?\d+\.\d+", v) else v)
                        )
                    except ValueError:
                        args[key] = v
            if args:
                out["args"] = args
        # runtime:heyi 日志无标准加载耗时格式,暂不解析
        return out

    @classmethod
    def normalize_params(
        cls, launch_cmd: str, parsed: dict[str, Any]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        args = parsed.get("args") or {}
        # 从 launch_cmd 也能解析(docker inspect 的 Cmd,--key value 格式)
        cmd_args = _parse_heyi_cmd_flags(launch_cmd)
        args = {**cmd_args, **args}  # 日志解析优先(含运行时推断值)

        schedule: dict[str, Any] = {}
        if args.get("max_length"):
            schedule["max_model_len"] = args["max_length"]
        if args.get("max_batch_size"):
            schedule["max_num_seqs"] = args["max_batch_size"]
        if args.get("kvcache_num_tokens"):
            schedule["kv_cache_tokens"] = args["kvcache_num_tokens"]
        if args.get("use_cuda_graph") is not None:
            schedule["use_cuda_graph"] = args["use_cuda_graph"]
        if schedule:
            out["schedule"] = schedule

        # heyi 无 TP/DP 概念,但记录 layerwise prefill 并行度
        parallel: dict[str, Any] = {}
        if args.get("layerwise_prefill_world_size"):
            parallel["layerwise_prefill_world_size"] = args[
                "layerwise_prefill_world_size"
            ]
        if args.get("num_decode_workers"):
            parallel["num_decode_workers"] = args["num_decode_workers"]
        if parallel:
            out["parallel"] = parallel

        backends: dict[str, Any] = {}
        if args.get("prefill_chunk_size"):
            backends["prefill_chunk_size"] = args["prefill_chunk_size"]
        if args.get("enable_layerwise_prefill") is not None:
            backends["enable_layerwise_prefill"] = args["enable_layerwise_prefill"]
        if backends:
            out["backends"] = backends

        # KV 容量(heyi 用 kvcache-num-gpu-tokens 表 GPU 侧 KV 池)
        runtime: dict[str, Any] = {}
        if args.get("kvcache_num_tokens"):
            runtime["kv_cache_tokens"] = args["kvcache_num_tokens"]
        if args.get("kvcache_num_gpu_tokens"):
            runtime["kv_cache_gpu_tokens"] = args["kvcache_num_gpu_tokens"]
        if args.get("max_new_tokens"):
            runtime["max_new_tokens"] = args["max_new_tokens"]
        if runtime:
            out["runtime"] = runtime
        return out


def _parse_heyi_cmd_flags(cmd: str) -> dict[str, Any]:
    """解析 heyi-engine 启动命令的 --key value 参数(docker inspect Cmd 格式)。"""
    args: dict[str, Any] = {}
    # Cmd 形如 ["/model/X","--kvcache-num-tokens","120000","--max-length","60000",...]
    # 也可能是空格分隔的字符串
    tokens = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", cmd)
    i = 0
    while i < len(tokens):
        t = tokens[i].strip("'\"")
        if t.startswith("--"):
            key = t[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                val = tokens[i + 1].strip("'\"")
                try:
                    args[key] = (
                        int(val)
                        if re.fullmatch(r"-?\d+", val)
                        else (float(val) if re.fullmatch(r"-?\d+\.\d+", val) else val)
                    )
                except ValueError:
                    args[key] = val
                i += 2
            else:
                args[key] = True  # flag 类型(--thinking/--auto-license)
                i += 1
        else:
            i += 1
    return args


# 注册表(顺序=探测优先级;vLLM 先,最常见)
_ADAPTERS: list[type[EngineCaptureAdapter]] = [
    VLLMAdapter,
    SGLangAdapter,
    HeyiAdapter,
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
        # 裸进程回退:按端口找监听进程,从 /proc 拿 cmdline + environ
        proc_info = _capture_bare_process(api_base_url)
        if proc_info:
            result.update(proc_info)
            result["capture_source"].append("bare_process")
            # 适配器探测 + 本地包版本
            _apply_adapter_and_versions(
                result, proc_info.get("launch_cmd", ""), api_model
            )
        else:
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
            if re.search(
                r"VLLM_|SGLANG_|PYTORCH_CUDA|CUDA_VERSION|CUDA_VISIBLE|CUDA_DEVICE|"
                r"KT_|B12X_|CUTE_DSL|NCCL_|OMP_NUM|TORCH_|TORCHINDUCTOR_|"
                r"HF_|TRANSFORMERS_|SAFETENSORS|TRITON_|FLASHINFER|XDG_CACHE",
                k,
            )
        }
        if envd:
            result["env"] = envd

    logs = _run(["docker", "logs", container], timeout=30.0) or ""

    # 3. 探测引擎 + 分派适配器 + 容器版本
    _apply_adapter_and_versions(
        result,
        result.get("launch_cmd", ""),
        api_model,
        image=image,
        logs=logs,
        container=container,
    )

    return result


def _apply_adapter_and_versions(
    result: dict,
    launch_cmd: str,
    api_model: dict | None,
    image: str = "",
    logs: str = "",
    container: str | None = None,
):
    """通用:适配器探测 + 日志解析 + 运行时版本探测。Docker 和裸进程共用。"""
    adapter = detect_adapter(image, launch_cmd, api_model)
    if adapter:
        result["engine"] = adapter.name
        result["adapter"] = adapter.name
        try:
            parsed = adapter.parse_logs(logs) if logs else {}
            norm = adapter.normalize_params(launch_cmd, parsed)
            if norm.get("schedule"):
                result["schedule"] = norm["schedule"]
                result.update(norm["schedule"])
            if norm.get("parallel"):
                result["parallel_strategy"] = norm["parallel"]
            if norm.get("backends"):
                result["backends"] = norm["backends"]
            if norm.get("mtp"):
                result["mtp"] = norm["mtp"]
            if norm.get("runtime"):
                result["runtime"] = norm["runtime"]
            if logs:
                result["capture_source"].append(f"adapter:{adapter.name}")
            elif launch_cmd:
                # 裸进程无日志,但从 launch_cmd 也能解析参数
                result["capture_source"].append(f"adapter:{adapter.name}:cmd_only")
        except Exception:  # noqa: BLE001
            result["capture_source"].append(f"adapter:{adapter.name}:error")
    else:
        result["engine"] = image.split("/")[-1] if image else "unknown"
        result["adapter"] = None
        result["capture_source"].append("no_adapter")
        # 兜底:no_adapter 时仍保留原始 launch_cmd + 关键 env,让报告至少有原始数据可查
        # (launch_cmd 已在 capture_engine_config 主流程写入 result;此处确保 env_flags 也在)
        if not result.get("env") and result.get("launch_cmd"):
            result["env_flags"] = {"_note": "未知引擎,未解析参数;见 launch_cmd 原文"}

    # 运行时版本:优先 docker exec(容器),其次 importlib.metadata(裸进程本地)
    if container:
        runtime_versions = _query_container_runtime_versions(container)
    else:
        runtime_versions = _query_local_runtime_versions()
    if runtime_versions:
        result["container_runtime"] = runtime_versions
        result["capture_source"].append(
            "container_exec" if container else "local_metadata"
        )


def _query_container_runtime_versions(container: str | None) -> dict[str, str | None]:
    """通过 docker exec 在容器内探测 torch/cuda/vllm 版本。失败返回空字典。"""
    if not container:
        return {}
    out: dict[str, str | None] = {}
    # 逐包 try/except,避免 find_spec 技巧脆弱性;torch 单独 import(warnings 抑制)
    script = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "try:\n"
        "    import torch; print('torch', torch.__version__)\n"
        "    print('cuda', torch.version.cuda or '')\n"
        "except Exception: pass\n"
        "import importlib.metadata as md\n"
        "for p in ('vllm','sglang','flashinfer-python','transformers'):\n"
        "    try: print(p, md.version(p))\n"
        "    except Exception: pass\n"
    )
    res = _run(["docker", "exec", container, "python", "-c", script], timeout=15.0)
    if res:
        for line in res.strip().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                key, val = parts[0].strip(), parts[1].strip()
                if val:
                    out[key] = val
    return out


def _query_local_runtime_versions() -> dict[str, str | None]:
    """裸进程场景:用 importlib.metadata 探测本地安装的 vllm/sglang/torch 版本。"""
    out: dict[str, str | None] = {}
    try:
        import importlib.metadata as md

        for pkg in ("vllm", "sglang", "torch", "transformers", "flashinfer-python"):
            try:
                out[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                pass
        # torch CUDA 版本
        try:
            import torch

            if torch.version.cuda:
                out["cuda"] = torch.version.cuda
        except ImportError:
            pass
    except Exception:  # noqa: BLE001
        pass
    return out


def _find_pid_by_port(port: int) -> int | None:
    """通过 /proc/net/tcp + /proc/<pid>/fd 找监听指定端口的进程 PID。
    回退:lsof / ss(需安装且有权限)。"""
    # 方法 1: /proc/net/tcp → inode → /proc/*/fd
    try:
        for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
            with open(proc_file, encoding="ascii") as f:
                lines = f.readlines()[1:]
            for line in lines:
                parts = line.split()
                if len(parts) < 4:
                    continue
                local = parts[1]
                state = parts[3]
                if state != "0A":
                    continue
                port_hex = local.split(":")[1]
                if int(port_hex, 16) == port:
                    inode = parts[9]
                    pid = _find_pid_by_inode(inode)
                    if pid:
                        return pid
    except (OSError, ValueError, IndexError):
        pass
    # 方法 2: lsof(回退,只取 LISTEN)
    out = _run(["lsof", "-i", f":{port}", "-sTCP:LISTEN", "-t"], timeout=5.0)
    if out:
        for line in out.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
    # 方法 3: ss -ltnp(回退)
    out = _run(["ss", "-ltnp"], timeout=5.0)
    if out:
        for line in out.splitlines():
            if f":{port}" in line and "pid=" in line:
                m = re.search(r"pid=(\d+)", line)
                if m:
                    return int(m.group(1))
    return None


def _find_pid_by_inode(inode: str) -> int | None:
    """遍历 /proc/<pid>/fd 找拥有指定 socket inode 的进程。"""
    import os

    try:
        for pid_dir in os.listdir("/proc"):
            if not pid_dir.isdigit():
                continue
            fd_dir = f"/proc/{pid_dir}/fd"
            try:
                for fd in os.listdir(fd_dir):
                    try:
                        link = os.readlink(f"{fd_dir}/{fd}")
                        if link.startswith("socket:[") and inode in link:
                            return int(pid_dir)
                    except OSError:
                        continue
            except (OSError, PermissionError):
                continue
    except Exception:  # noqa: BLE001
        pass
    return None


def _capture_bare_process(api_base_url: str) -> dict[str, Any]:
    """裸进程场景:按端口找监听进程,从 /proc 拿 cmdline + environ。
    返回 {launch_cmd, env, pid} 或空字典(找不到/无权限)。"""
    m = re.search(r":(\d+)/?", api_base_url or "")
    if not m:
        return {}
    port = int(m.group(1))
    pid = _find_pid_by_port(port)
    if not pid:
        return {}
    result: dict[str, Any] = {"pid": pid}
    # cmdline → launch_cmd
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        # cmdline 用 \0 分隔参数
        args = raw.decode("utf-8", errors="replace").split("\0")
        args = [a for a in args if a]
        if args:
            result["launch_cmd"] = " ".join(args)
    except (OSError, PermissionError):
        pass
    # environ → env(同 docker inspect 的 Env 过滤)
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
        env_raw = raw.decode("utf-8", errors="replace").split("\0")
        envd = {}
        for entry in env_raw:
            if "=" in entry:
                k, _, v = entry.partition("=")
                if re.search(
                    r"VLLM_|SGLANG_|PYTORCH_CUDA|CUDA_VERSION|CUDA_VISIBLE|CUDA_DEVICE|"
                    r"KT_|B12X_|CUTE_DSL|NCCL_|OMP_NUM|TORCH_|TORCHINDUCTOR_|"
                    r"HF_|TRANSFORMERS_|SAFETENSORS|TRITON_|FLASHINFER|XDG_CACHE",
                    k,
                ):
                    envd[k] = v
        if envd:
            result["env"] = envd
    except (OSError, PermissionError):
        pass
    return result if result.get("launch_cmd") else {}


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
