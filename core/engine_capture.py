"""推理引擎配置自动采集(per-run,绑定每次测试)。

为什么需要:引擎配置(max_num_seqs / KV 容量 / 并行策略 / 冷启动分解)是结论的
关键上下文。vLLM 配置一变,硬编码的报告就错(曾因 max_num_seqs 16→64、KV 变化
导致报告陈旧)。本模块在测试时 docker inspect + 日志解析 + /v1/models 自动采集
真实配置,绑到每次 TestRun 的 serving_config_json。

设计:纯函数、全防御——docker/日志不可用时优雅降级(返回能拿到的部分),永不抛异常。
与 hardware_fingerprint 同口径(subprocess + httpx)。
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore


def _run(args: list[str], timeout: float = 12.0) -> str | None:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def find_vllm_container(api_base_url: str, hint: str | None = None) -> str | None:
    """按服务端口(从 api_base_url 解析)在 docker ps 里找容器名。"""
    out = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"])
    if not out:
        return None
    # 端口:从 http://host:PORT/... 取 PORT
    m = re.search(r":(\d+)/?", api_base_url or "")
    port = m.group(1) if m else None
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        name, ports = parts
        if port and f":{port}->" in ports:
            return name
    # 回退:名字含 hint / vllm
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            name = parts[0]
            if hint and hint in name:
                return name
    return None


def _parse_nondefault_args(logs: str) -> dict[str, Any]:
    """从 vLLM 'non-default args: {...}' 行提取关键参数(取最后一条=最近启动)。"""
    matches = re.findall(r"non-default args: (\{.*?\})\s*(?=\[|\n|$)", logs, re.DOTALL)
    if not matches:
        return {}
    raw = matches[-1]
    out: dict[str, Any] = {}
    for key in ("tensor_parallel_size", "decode_context_parallel_size", "enable_expert_parallel",
                "gpu_memory_utilization", "max_num_seqs", "enable_prefix_caching",
                "max_model_len", "pipeline_parallel_size", "data_parallel_size"):
        m = re.search(rf"'{key}':\s*([^,}}]+)", raw)
        if m:
            v = m.group(1).strip().strip("'\"")
            try:
                out[key] = int(v) if re.fullmatch(r"-?\d+", v) else (float(v) if re.fullmatch(r"-?\d+\.\d+", v) else v)
            except ValueError:
                out[key] = v
    return out


def _parse_startup_breakdown(logs: str) -> dict[str, Any]:
    """从 vLLM 日志解析最近一次启动的分解(权重/init/graph/KV/冷启动)。"""
    def last(pat: str) -> str | None:
        ms = re.findall(pat, logs)
        return ms[-1] if ms else None

    out: dict[str, Any] = {}
    w = last(r"Loading weights took ([\d.]+) seconds")
    if w:
        out["weight_load_s"] = float(w)
    m = last(r"Model loading took [0-9.]+ GiB memory and ([\d.]+) seconds")
    if m:
        out["model_load_s"] = float(m)
    ie = last(r"init engine .* took ([\d.]+) s")
    if ie:
        out["init_engine_s"] = float(ie)
    gc = last(r"Graph capturing finished in (\d+) secs")
    if gc:
        out["graph_capture_s"] = int(gc)
    kv = last(r"GPU KV cache size: ([\d,]+) tokens")
    if kv:
        out["kv_cache_tokens"] = int(kv.replace(",", ""))
    return out


def capture_engine_config(api_base_url: str, container_name: str | None = None) -> dict[str, Any]:
    """自动采集推理引擎配置(docker inspect + 日志 + /v1/models)。永不抛异常。

    Returns: {engine, engine_version, image, launch_cmd, args{}, parallel{}, schedule{},
              runtime{cold_start_s, ...}, model{}, container, capture_source, captured_at}
    """
    result: dict[str, Any] = {"captured_at": datetime.now().isoformat(), "capture_source": []}

    # 1. /v1/models(最可靠,不依赖 docker)
    if httpx:
        try:
            r = httpx.get(f"{api_base_url.rstrip('/')}/models", timeout=8.0)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    m0 = data[0]
                    result["model"] = {"model_id": m0.get("id"), "max_model_len": m0.get("max_model_len")}
                    result["capture_source"].append("api")
        except Exception:  # noqa: BLE001
            pass

    # 2. docker inspect + logs
    container = container_name or find_vllm_container(api_base_url)
    if not container:
        result["capture_source"].append("no_container")
        return result
    result["container"] = container

    cmd = _run(["docker", "inspect", container, "--format", "{{.Config.Cmd}}|||{{.Config.Image}}"])
    if cmd:
        parts = cmd.split("|||", 1)
        result["launch_cmd"] = parts[0].strip()
        image = parts[1].strip() if len(parts) > 1 else ""
        result["image"] = image
        # engine_version 从 image tag(vllm/vllm-openai:v0.23.0-x86_64 → 0.23.0)
        mv = re.search(r":v?([\d.]+)", image)
        if mv:
            result["engine_version"] = mv.group(1)
        result["engine"] = "vLLM" if "vllm" in image.lower() else image.split("/")[-1]
        result["capture_source"].append("docker_inspect")

    # 关键 ENV
    env = _run(["docker", "inspect", container, "--format", "{{range .Config.Env}}{{println .}}{{end}}"])
    if env:
        envd = {}
        for line in env.splitlines():
            if "=" in line and re.search(r"VLLM_|PYTORCH_CUDA|CUDA_VERSION", line):
                k, _, v = line.partition("=")
                envd[k] = v
        if envd:
            result["env"] = envd

    # 日志:non-default args + 启动分解
    logs = _run(["docker", "logs", container], timeout=30.0) or ""
    if logs:
        args = _parse_nondefault_args(logs)
        if args:
            result["args"] = args
            result["schedule"] = {k: args[k] for k in ("max_num_seqs", "gpu_memory_utilization",
                                "enable_prefix_caching", "max_model_len") if k in args}
            result["parallel"] = {
                "tp": args.get("tensor_parallel_size"),
                "dcp": args.get("decode_context_parallel_size"),
                "ep": args.get("enable_expert_parallel"),
            }
        breakdown = _parse_startup_breakdown(logs)
        if breakdown:
            result["runtime"] = breakdown
            # 冷启动 ≈ weight_load + init + graph(粗略;精确需时间戳首末)
            wl = breakdown.get("weight_load_s", 0)
            ie = breakdown.get("init_engine_s", 0)
            gc = breakdown.get("graph_capture_s", 0)
            if wl or ie or gc:
                result["runtime"]["cold_start_s_est"] = round(wl + ie + gc, 1)
        result["capture_source"].append("docker_logs")

    return result


if __name__ == "__main__":  # 手动验证
    import json
    print(json.dumps(capture_engine_config("http://localhost:10814/v1"), ensure_ascii=False, indent=2))
