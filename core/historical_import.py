"""
历史数据回填导入器（处理 raw_data/ 里带 .csv.meta.json 的测试组）。

raw_data 是平台“前八维时代”产生的数据：CSV 是 per-request 性能（全），
.meta.json 含非结构化字符串 system_info（"1* pro 6000" / "4*48G DDR5 6400" /
"vLLM-v0.22.0-MTP3"）。本模块把字符串解析成结构化维度，创建**带维度**的 test_run
+ 导入 test_results，并明确标注：status_detail='historical_import'、
external_level='internal'、结构化维度部分缺（CPU/PCIe/CUDA/资源监控/模型规格/
引擎运行时/归因 均无）。

设计原则：解析函数纯（可测，不碰 DB）；导入函数用 db 句柄；任何一组失败只 log
不中断整体。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from core.hardware_fingerprint import _lookup_gpu_bandwidth
from core.models import TestResult, TestRun

logger = logging.getLogger(__name__)


# ===========================================================================
# 字符串 system_info 解析（纯函数）
# ===========================================================================

_ENGINE_RE = re.compile(r"(vllm|sglang|tensorrt-?llm|trtllm|lmdeploy|llama\.cpp)", re.IGNORECASE)
_VERSION_RE = re.compile(r"v?(\d+\.\d+(?:\.\d+)?)")
_MTP_RE = re.compile(r"[-_]?\s*mtp\s*(\d+)", re.IGNORECASE)


def parse_engine_name(engine_name: str | None) -> dict[str, Any]:
    """ "vLLM-v0.22.0-MTP3" → {engine, engine_version, mtp_enabled, num_speculative_tokens}。"""
    if not engine_name:
        return {}
    s = engine_name.strip()
    out: dict[str, Any] = {}
    m = _ENGINE_RE.search(s)
    if m:
        out["engine"] = m.group(1).lower().replace("-", "")
    vm = _VERSION_RE.search(s)
    if vm:
        out["engine_version"] = vm.group(1)
    mm = _MTP_RE.search(s)
    if mm:
        out["mtp_enabled"] = True
        out["num_speculative_tokens"] = int(mm.group(1))
    return out


_GPU_COUNT_RE = re.compile(r"(\d+)\s*[*×xX]\s*(.+)")


def parse_gpu_str(gpu_str: str | None) -> list[dict[str, Any]]:
    """ "1* pro 6000" / "4*H100" / "2× RTX 4090" → [{name, count, nominal_bandwidth_gbps}]。

    count 标注但**不可靠**（"1*" 可能是型号数而非卡数），故 machine_id 不依赖它。
    """
    if not gpu_str:
        return []
    s = gpu_str.strip()
    m = _GPU_COUNT_RE.match(s)
    if m:
        count = int(m.group(1))
        name = m.group(2).strip()
    else:
        count = None
        name = s
    return [
        {
            "name": name,
            "count": count,
            "nominal_bandwidth_gbps": _lookup_gpu_bandwidth(name, None),
        }
    ]


_MEM_RE = re.compile(
    r"(\d+)\s*[*×xX]\s*(\d+)\s*G[B]?\s*(DDR\d+\w*|HBM\d?|LPDDR\d+\w*)\s*(\d+)?",
    re.IGNORECASE,
)


def parse_memory_str(mem_str: str | None) -> dict[str, Any]:
    """ "4*48G DDR5 6400" → {sticks, capacity_gb_per_stick, total_gb, type, speed_mt_s}。"""
    if not mem_str:
        return {}
    m = _MEM_RE.search(mem_str.strip())
    if not m:
        return {}
    sticks = int(m.group(1))
    cap_per = int(m.group(2))
    return {
        "sticks": sticks,
        "capacity_gb_per_stick": cap_per,
        "total_gb": sticks * cap_per,
        "type": m.group(3).upper(),
        "speed_mt_s": int(m.group(4)) if m.group(4) else None,
    }


def build_legacy_fingerprint(sys_info: dict[str, Any]) -> dict[str, Any]:
    """legacy 字符串 system_info → 结构化 hardware_fingerprint（部分字段缺）。"""
    gpus = parse_gpu_str(sys_info.get("gpu"))
    mem = parse_memory_str(sys_info.get("memory"))
    fp_key = (
        "|".join(sorted(g["name"] for g in gpus if g.get("name"))) + f"|mem={mem.get('total_gb')}"
    )
    machine_id = hashlib.sha1(fp_key.encode()).hexdigest()[:16] if fp_key != "|mem=None" else None
    return {
        "machine_id": machine_id,
        "gpus": gpus,
        "memory": mem,
        "cpu": {},  # legacy 缺
        "cuda": {},  # legacy 缺
        "source": "legacy_meta",
        "note": "历史回填：CPU 拓扑/PCIe/CUDA/ECC 缺",
    }


def _parse_timestamp(test_config: dict[str, Any]) -> datetime | None:
    """从 test_config.Timestamp（'20260611_233321'）解析时间。"""
    ts = test_config.get("Timestamp")
    if not ts:
        return None
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(str(ts), fmt)
        except ValueError:
            continue
    return None


# ===========================================================================
# CSV 行 → TestResult
# ===========================================================================


def _to_int(v: Any) -> int | None:
    try:
        return int(float(v)) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def row_to_test_result(row: dict[str, Any], run_id: int, index: int) -> TestResult:
    """CSV 行 → TestResult（镜像 import_service 的列映射）。"""
    err = row.get("error")
    return TestResult(
        run_id=run_id,
        request_index=index,
        session_id=_to_int(row.get("session_id")),
        round=_to_int(row.get("round")),
        concurrency_level=_to_int(row.get("concurrency")),
        input_tokens_target=_to_int(row.get("input_tokens_target")),
        ttft=_to_float(row.get("ttft")),
        tpot=_to_float(row.get("tpot")),
        total_time=_to_float(row.get("total_time")),
        decode_time=_to_float(row.get("decode_time")),
        prefill_speed=_to_float(row.get("prefill_speed")),
        tps=_to_float(row.get("tps")),
        system_throughput=_to_float(row.get("system_throughput")),
        system_input_throughput=_to_float(row.get("system_input_throughput")),
        system_output_throughput=_to_float(row.get("system_output_throughput")),
        rps=_to_float(row.get("rps")),
        prefill_tokens=_to_int(row.get("prefill_tokens")),
        decode_tokens=_to_int(row.get("decode_tokens")),
        cache_hit_tokens=_to_int(row.get("cache_hit_tokens")),
        token_calc_method=row.get("token_calc_method"),
        start_time=_to_float(row.get("start_time")),
        end_time=_to_float(row.get("end_time")),
        error=(err if err and err != "None" else None),
        created_at=datetime.now(),
    )


# ===========================================================================
# 导入流程
# ===========================================================================


def import_meta_group(
    db, meta_path: str | Path, csv_path: str | Path | None = None
) -> dict[str, Any]:
    """导入一组（meta + 对应 csv）：创建带维度 run + 导入 test_results。返回统计。"""
    meta_path = Path(meta_path)
    if csv_path:
        csv_path = Path(csv_path)
    else:
        # meta 是 "X.csv.meta.json" → csv 是 "X.csv"（去 ".meta.json"）
        name = meta_path.name
        csv_name = name[: -len(".meta.json")] if name.endswith(".meta.json") else name
        csv_path = meta_path.with_name(csv_name)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"读 meta 失败: {e}"}

    sys_info = meta.get("system_info") or {}
    test_config = meta.get("test_config") or {}
    fp = build_legacy_fingerprint(sys_info)
    engine = parse_engine_name(sys_info.get("engine_name"))

    run = TestRun.create(
        test_type=meta.get("test_type") or "unknown",
        model_id=meta.get("model_id") or sys_info.get("model_name") or "unknown",
        provider=meta.get("provider"),
    )
    run.created_at = _parse_timestamp(test_config) or run.created_at
    run.config = dict(test_config)
    run.system_info = {"hardware_fingerprint": fp, "machine_id": fp.get("machine_id")}
    run.serving_config = engine
    run.notes = "历史回填导入（legacy_meta）；结构化维度部分缺"
    run.csv_path = str(csv_path)
    # 历史数据标注：不可对外
    run.status_detail = "historical_import"
    run.external_level = "internal"

    try:
        run_id = db.runs.insert(run)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"创建 run 失败: {e}"}
    run.id = run_id

    result_count = 0
    if csv_path.exists():
        try:
            with csv_path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                results = [row_to_test_result(row, run_id, i) for i, row in enumerate(reader)]
            if results:
                db.results.insert_batch(results)
                result_count = len(results)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"导入 CSV 失败 {csv_path}: {e}")

    # 算聚合 stats（avg_tps/avg_ttft/success_rate 等），让仓库查询/矩阵有指标
    if result_count:
        try:
            db.complete_test_run(run, success=True, calculate_stats=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"算聚合 stats 失败 run_id={run_id}: {e}")

    return {
        "ok": True,
        "run_id": run_id,
        "model_id": run.model_id,
        "test_type": run.test_type,
        "machine_id": fp.get("machine_id"),
        "engine": engine.get("engine"),
        "result_count": result_count,
    }


def import_raw_data_directory(db, raw_data_dir: str | Path) -> dict[str, Any]:
    """遍历 raw_data_dir 下所有 *.csv.meta.json，批量导入。返回汇总。"""
    raw_data_dir = Path(raw_data_dir)
    metas = sorted(raw_data_dir.rglob("*.csv.meta.json"))
    summary = {
        "total": len(metas),
        "imported": 0,
        "failed": 0,
        "results": 0,
        "errors": [],
    }
    for meta_path in metas:
        r = import_meta_group(db, meta_path)
        if r.get("ok"):
            summary["imported"] += 1
            summary["results"] += r.get("result_count", 0)
        else:
            summary["failed"] += 1
            summary["errors"].append(f"{meta_path.name}: {r.get('error')}")
    return summary
