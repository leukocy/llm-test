"""
数据仓库查询层（纯函数，无 Streamlit 依赖）。

把数据库里的 TestRun 投影成手册三套字段模板能直接吃掉的扁平行，并提供：
- WarehouseFilter：跨八维筛选（machine_id/模型/引擎/可对外等级/状态/类型/测试员/日期/搜索）
- query_runs()：筛选 + 复测折叠（supersedes_test_id，默认只留最新一版）
- distinct_values()：填筛选下拉用
- build_hardware_inventory_rows()：machine_id 维度的硬件清单（模板 #1，每台机器一行）
- build_hm_test_rows()：每次测试一行（模板 #2）
- build_cross_matrix()：硬件 × 模型透视（手册"不同硬件下的表现"）

设计原则：每个字段的解析独立 try/except，单字段缺测不影响其余；缺测即决策信息
（手册："没有硬件、没排期、跑不起来，都要在矩阵里有一格，而不是消失"）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.config_hash import compute_config_hash
from core.models import TestRun
from core.warehouse.templates import HARDWARE_INVENTORY_FIELDS, HM_TEST_FIELDS, MA_TEST_FIELDS

# ---------------------------------------------------------------------------
# WarehouseFilter
# ---------------------------------------------------------------------------


@dataclass
class WarehouseFilter:
    """跨八维的仓库筛选条件。所有字段可选；None = 不过滤。"""

    machine_id: str | None = None
    model_id: str | None = None
    engine: str | None = None  # serving_config.engine（模糊匹配）
    external_level: str | None = None  # internal / review / publishable
    status_detail: str | None = None
    test_type: str | None = None
    tester: str | None = None
    comparison_group: str | None = None
    config_hash: str | None = None  # CASE 02：同配置才能承诺
    date_from: datetime | None = None
    date_to: datetime | None = None
    search: str | None = None  # 跨字段模糊（model/notes/tester/machine_id）
    include_superseded: bool = False  # False=只留 supersedes 链最新一版
    limit: int = 500

    def matches(self, row: dict[str, Any]) -> bool:
        """对一条投影后的行做字段级匹配。"""
        if self.machine_id and row.get("machine_id") != self.machine_id:
            return False
        if self.model_id and row.get("model_name") != self.model_id:
            return False
        if self.engine:
            eng = str(row.get("engine") or "").lower()
            if self.engine.lower() not in eng:
                return False
        if self.external_level and row.get("external_level") != self.external_level:
            return False
        if self.status_detail and row.get("status") != self.status_detail:
            return False
        if self.tester and row.get("tester") != self.tester:
            return False
        if self.config_hash and row.get("config_hash") != self.config_hash:
            return False
        if self.search:
            haystack = " ".join(
                str(row.get(k) or "")
                for k in (
                    "model_name",
                    "tester",
                    "machine_id",
                    "engine",
                    "next_action",
                    "remark",
                )
            ).lower()
            if self.search.lower() not in haystack:
                return False
        return True


# ---------------------------------------------------------------------------
# 单条 TestRun → 扁平投影（覆盖三套模板全部字段）
# ---------------------------------------------------------------------------


def _num(value: Any) -> Any:
    """数值规整：None/空 → None；float → 4 位小数；其余原样。"""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return value


def _parallel_strategy(serving: dict[str, Any]) -> str:
    parts = []
    for key, label in (
        ("tp_size", "tp"),
        ("dp_size", "dp"),
        ("ep_size", "ep"),
        ("pp_size", "pp"),
    ):
        v = serving.get(key)
        if v:
            parts.append(f"{label}{v}")
    return "-".join(parts)


def _engine_params(serving: dict[str, Any], engine: dict[str, Any]) -> str:
    """把调度/MTP/runtime 关键参数压成一行可读字符串。"""
    bits = []
    schedule_keys = (
        "max_model_len",
        "gpu_memory_utilization",
        "max_num_seqs",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "block_size",
        "enforce_eager",
    )
    for k in schedule_keys:
        v = serving.get(k)
        if v is not None and v != "":
            bits.append(f"{k}={v}")
    if serving.get("attention_backend"):
        bits.append(f"attn={serving['attention_backend']}")
    if serving.get("kv_cache_dtype"):
        bits.append(f"kv={serving['kv_cache_dtype']}")
    if serving.get("mtp_enabled"):
        bits.append(f"mtp={serving.get('speculative_method') or 'on'}")
    if engine.get("engine_family") and engine.get("engine_family") != "unknown":
        bits.append(f"family={engine['engine_family']}")
    return ", ".join(bits)


def project_run(run: TestRun) -> dict[str, Any]:
    """把一条 TestRun 投影成手册三套字段全集的扁平 dict。

    缺测字段为 None（导出时落 ""）。本函数是模板导出/矩阵透视的唯一数据出口。
    """
    spec: dict[str, Any] = run.model_spec or {}
    serving: dict[str, Any] = run.serving_config or {}
    monitor: dict[str, Any] = run.resource_monitor or {}
    engine: dict[str, Any] = run.engine_metrics or {}
    sysinfo: dict[str, Any] = run.system_info or {}
    fp: dict[str, Any] = sysinfo.get("hardware_fingerprint") or {}
    config: dict[str, Any] = run.config or {}
    peaks: dict[str, Any] = monitor.get("peaks") or {}

    cpu = fp.get("cpu") or {}
    mem = fp.get("memory") or {}
    cuda = fp.get("cuda") or {}
    gpus = fp.get("gpus") or []
    gpu0 = gpus[0] if gpus else {}

    created = run.created_at
    date_str = created.strftime("%Y-%m-%d") if created else ""

    # decode_tps：优先 config 显式值，回退 avg_tps（并发测试的每流吞吐）
    decode_tps = config.get("decode_tps")
    if decode_tps is None:
        decode_tps = run.avg_tps

    row: dict[str, Any] = {
        # ---- 公共 ----
        "test_id": run.test_id or "",
        "date": date_str,
        "tester": run.tester or "",
        "machine_id": run.machine_id or fp.get("machine_id") or "",
        # ---- 引擎 / 服务 ----
        "engine": serving.get("engine") or config.get("engine") or "",
        "engine_version": serving.get("engine_version") or "",
        "engine_params": _engine_params(serving, engine),
        "parallel_strategy": _parallel_strategy(serving),
        # ---- 模型 ----
        "model_name": run.model_id or spec.get("name") or "",
        "model_version": spec.get("version") or "",
        "model_type": spec.get("architecture") or "",
        "total_params": _num(spec.get("total_params_b")),
        "active_params": _num(spec.get("active_params_b")),
        "num_experts": spec.get("num_experts"),
        "top_k": spec.get("num_experts_per_tok"),
        "quantization": serving.get("serving_quant") or spec.get("quant_method") or "",
        "dtype": spec.get("weight_dtype") or "",
        "max_context": spec.get("max_position_embeddings") or serving.get("max_model_len"),
        # ---- 测试配置 ----
        "concurrency": run.concurrency,
        "usecase_set_version": config.get("usecase_set_version") or "",
        "prompt_tokens": config.get("prompt_tokens") or config.get("total_prefill_tokens") or "",
        "output_tokens": config.get("output_tokens") or run.max_tokens,
        "load_time_s": _num(config.get("load_time_s")),
        # ---- 性能 ----
        "ttft_s": _num(run.avg_ttft),
        "prefill_tps": _num(config.get("prefill_tps")),
        "decode_tps": _num(decode_tps),
        "long_context_tps": _num(config.get("long_context_tps")),
        "p50_latency_s": _num(run.p50_ttft),
        "p95_latency_s": _num(run.p95_ttft),
        "p99_latency_s": _num(run.p99_ttft),
        # ---- 资源峰值 ----
        "gpu_vram_peak_gb": _num(run.gpu_vram_peak_gb or peaks.get("gpu_vram_gb")),
        "system_memory_peak_gb": _num(run.system_memory_peak_gb or peaks.get("system_memory_gb")),
        "effective_bandwidth_gbps": _num(run.effective_bandwidth_gbps),
        "bandwidth_utilization_pct": _num(run.bandwidth_utilization_pct),
        "cpu_threads_used": config.get("cpu_threads") or "",
        "cpu_util_pct": _num(peaks.get("cpu_percent")),
        "gpu_util_pct": _num(peaks.get("gpu_util_percent")),
        "power_w": _num(peaks.get("gpu_power_w")),
        "temp_c": _num(peaks.get("gpu_temp_c")),
        # ---- 归因 ----
        "status": run.status_detail or run.status,
        "bottleneck": run.bottleneck or "",
        "error_type": config.get("error_type") or "",
        "error_detail": config.get("error_detail") or "",
        "log_path": run.csv_path or config.get("log_path") or "",
        "screenshot_path": config.get("screenshot_path") or "",
        "external_level": run.external_level or "internal",
        "next_action": run.next_action or "",
        "supersedes_test_id": run.supersedes_test_id or "",
        # ---- 硬件盘点（模板 #1）----
        "product_line": config.get("product_line") or "",
        "cpu_model": cpu.get("model_name") or "",
        "cpu_sockets": cpu.get("sockets"),
        "cpu_cores": cpu.get("cores_per_socket"),
        "cpu_threads": cpu.get("threads_per_core"),
        "numa_nodes": cpu.get("numa_nodes"),
        "memory_type": mem.get("type") or "",
        "memory_capacity_gb": mem.get("total_gb"),
        "memory_channels_populated": mem.get("channels"),
        "memory_speed_mtps": mem.get("speed_mt_s"),
        "ecc_enabled": mem.get("ecc"),
        "gpu_model": gpu0.get("name") or "",
        "gpu_count": len(gpus) or None,
        "gpu_vram_gb": gpu0.get("vram_gb"),
        "gpu_memory_type": gpu0.get("memory_type") or "",
        "gpu_bandwidth_gbps": gpu0.get("nominal_bandwidth_gbps"),
        "pcie_gen": gpu0.get("pcie_gen"),
        "pcie_width": gpu0.get("pcie_width"),
        "ssd_model": config.get("ssd_model") or "",
        "ssd_capacity_tb": config.get("ssd_capacity_tb") or "",
        "os": fp.get("os") or sysinfo.get("os") or "",
        "driver": cuda.get("driver") or "",
        "cuda_or_rocm": cuda.get("cuda_version") or "",
        "engine_ready": config.get("engine_ready") or "",
        "power_supply_w": config.get("power_supply_w") or "",
        "cooling_note": config.get("cooling_note") or "",
        "owner": run.tester or "",
        "location": config.get("location") or "",
        "remark": run.notes or "",
        # ---- 模型×应用（模板 #3；应用质量维度，多由未来应用评估层填）----
        "case_id": config.get("case_id") or "",
        "scenario": config.get("scenario") or "",
        "task_name": config.get("task_name") or "",
        "customer_type": config.get("customer_type") or "",
        "input_tokens": config.get("input_tokens") or "",
        "context_length": config.get("context_length") or run.max_tokens,
        "retrieval_latency_s": _num(config.get("retrieval_latency_s")),
        "prefill_latency_s": _num(run.avg_ttft),
        "total_latency_s": _num(config.get("total_latency_s")),
        "quality_score": _num(config.get("quality_score")),
        "success": config.get("success"),
        "citation_score": _num(config.get("citation_score")),
        "tool_success_rate": _num(config.get("tool_success_rate")),
        "privacy_requirement": config.get("privacy_requirement") or "",
        "cost_note": config.get("cost_note") or "",
        "recommended_config": config.get("recommended_config") or "",
        "sales_summary": config.get("sales_summary") or "",
        "failure_reason": config.get("failure_reason") or "",
        "evidence_path": config.get("evidence_path") or run.csv_path or "",
    }
    # 配置指纹（CASE 02“同配置才能承诺”）：仓库内部去重/过滤键，不进手册模板字段
    row["config_hash"] = compute_config_hash(
        model_name=row["model_name"],
        engine=row["engine"],
        engine_version=serving.get("engine_version") or "",
        parallel_strategy=row["parallel_strategy"],
        quantization=row["quantization"],
        dtype=row["dtype"],
        max_context=row["max_context"],
    )
    return row


# ---------------------------------------------------------------------------
# 查询入口
# ---------------------------------------------------------------------------


def query_runs(db, flt: WarehouseFilter | None = None) -> list[TestRun]:
    """按 WarehouseFilter 筛选历史 TestRun。

    Args:
        db: 数据库管理器（需有 get_recent_runs / search_runs）。
        flt: 筛选条件；None = 取全部最近记录。

    Returns:
        匹配的 TestRun 列表（按 created_at 倒序）。supersedes 链默认只留最新一版。
    """
    flt = flt or WarehouseFilter()
    limit = max(1, int(flt.limit or 500))

    # 统一从最近记录取候选，所有过滤在 Python 里做（口径一致、可预测）。
    if not hasattr(db, "get_recent_runs"):
        return []
    runs: list[TestRun] = db.get_recent_runs(limit=limit)

    # 第一遍：run 级字段（test_type / comparison_group / 日期），无需投影。
    runs = [
        r
        for r in runs
        if (not flt.test_type or r.test_type == flt.test_type)
        and (not flt.comparison_group or r.comparison_group == flt.comparison_group)
        and (not flt.date_from or not r.created_at or r.created_at >= flt.date_from)
        and (not flt.date_to or not r.created_at or r.created_at <= flt.date_to)
    ]

    # 第二遍：投影级字段（machine_id/模型/引擎/可对外等级/状态/测试员/搜索）。
    runs = [r for r in runs if flt.matches(project_run(r))]

    if not flt.include_superseded:
        runs = _collapse_superseded(runs)

    return runs


def _collapse_superseded(runs: list[TestRun]) -> list[TestRun]:
    """把被 supersedes_test_id 指向的旧 run 从结果里剔除，只留最新一版。"""
    superseded_ids = {r.supersedes_test_id for r in runs if r.supersedes_test_id}
    if not superseded_ids:
        return runs
    return [r for r in runs if r.test_id not in superseded_ids]


def distinct_values(db, field: str, limit: int = 200) -> list[Any]:
    """取某投影字段的不重复非空值（填筛选下拉用）。

    field 可以是任意 project_run 产出的键（如 machine_id / model_name / engine /
    external_level / status / tester）。
    """
    runs = db.get_recent_runs(limit=limit) if hasattr(db, "get_recent_runs") else []
    seen: list[Any] = []
    seen_set: set = set()
    for r in runs:
        val = project_run(r).get(field)
        if val in (None, "", []):
            continue
        key = str(val)
        if key in seen_set:
            continue
        seen_set.add(key)
        seen.append(val)
    # 字符串排序更稳定；数值保持原序
    if all(isinstance(v, str) for v in seen):
        seen.sort()
    return seen


# ---------------------------------------------------------------------------
# 模板行构造
# ---------------------------------------------------------------------------


def _select_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """从投影行里按模板字段顺序挑选，缺字段填 None。"""
    return {f: row.get(f) for f in fields}


def build_hm_test_rows(runs: list[TestRun]) -> list[dict[str, Any]]:
    """硬件×模型测试模板（#2）行：每次测试一行，字段顺序与手册一致。"""
    return [_select_fields(project_run(r), HM_TEST_FIELDS) for r in runs]


def build_hardware_inventory_rows(runs: list[TestRun]) -> list[dict[str, Any]]:
    """硬件盘点模板（#1）行：每个 machine_id 一行（取该机器最新一次 run 的指纹）。

    多次测试同一台机器时，硬件指纹不变，只留一条；缺 machine_id 的 run 跳过。
    """
    latest_per_machine: dict[str, TestRun] = {}
    for r in runs:
        mid = project_run(r).get("machine_id")
        if not mid:
            continue
        prev = latest_per_machine.get(mid)
        # created_at 倒序传入时，首个即最新；否则取更晚的
        if prev is None or (
            r.created_at and (not prev.created_at or r.created_at > prev.created_at)
        ):
            latest_per_machine[mid] = r
    return [
        _select_fields(project_run(r), HARDWARE_INVENTORY_FIELDS)
        for r in latest_per_machine.values()
    ]


def build_ma_test_rows(runs: list[TestRun]) -> list[dict[str, Any]]:
    """模型×应用测试模板（#3）行：当前应用质量维度多未采集，缺测字段为 None。

    预留出口：未来的应用评估层把 case_id/scenario/quality_score 等写进 run.config，
    本函数即可自动投影出来。
    """
    return [_select_fields(project_run(r), MA_TEST_FIELDS) for r in runs]


def build_ma_test_rows_from_cases(
    db,
    scenario: str | None = None,
    model_name: str | None = None,
    machine_id: str | None = None,
    external_level: str | None = None,
    source: str | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """模型×应用测试模板（#3）行——从 application_cases 表读（数据真源）。

    ApplicationCase 的字段名与 MA_TEST_FIELDS 逐字对齐，故直接按模板字段投影。
    任一筛选参数 None = 不过滤。
    """
    if not hasattr(db, "list_application_cases"):
        return []
    cases = db.list_application_cases(
        scenario=scenario,
        model_name=model_name,
        machine_id=machine_id,
        external_level=external_level,
        source=source,
        limit=limit,
    )
    return [{f: getattr(c, f, None) for f in MA_TEST_FIELDS} for c in cases]


# ---------------------------------------------------------------------------
# 硬件 × 模型 透视矩阵（手册"不同硬件下的表现"）
# ---------------------------------------------------------------------------


@dataclass
class CrossMatrix:
    """硬件×模型透视结果。"""

    row_key: str
    col_key: str
    metric: str
    row_labels: list[str]
    col_labels: list[str]
    cells: dict[str, dict[str, Any]]  # {row: {col: value}}
    agg: str = "latest"


def build_cross_matrix(
    runs: list[TestRun],
    row_key: str = "machine_id",
    col_key: str = "model_name",
    metric: str = "decode_tps",
    agg: str = "latest",
) -> CrossMatrix:
    """构造 行×列→指标 的透视矩阵。

    Args:
        runs: 已筛选的 TestRun 列表。
        row_key / col_key: project_run 的字段名（默认 machine_id × model_name）。
        metric: 透视的指标字段（默认 decode_tps）。
        agg: 同一格多次测试的聚合方式——"latest"=取最新一次；"best"=取最大值。

    Returns:
        CrossMatrix：cells[row][col] = 聚合后的指标值（缺测为 None）。
    """
    # 收集每个 (row,col) 的候选 (created_at, value)
    candidates: dict[tuple[str, str], list[tuple[Any, Any]]] = {}
    for r in runs:
        proj = project_run(r)
        rv = proj.get(row_key)
        cv = proj.get(col_key)
        if not rv or not cv:
            continue
        val = proj.get(metric)
        if val is None and agg == "best":
            continue  # best 模式忽略缺测；latest 模式保留 None 以显"跑过但无指标"
        candidates.setdefault((str(rv), str(cv)), []).append((r.created_at, val))

    row_labels: list[str] = []
    col_labels: list[str] = []
    cells: dict[str, dict[str, Any]] = {}
    for (rv, cv), samples in candidates.items():
        if rv not in row_labels:
            row_labels.append(rv)
        if cv not in col_labels:
            col_labels.append(cv)
        if agg == "best":
            numeric = [v for _, v in samples if isinstance(v, (int, float))]
            chosen = max(numeric) if numeric else None
        else:
            # latest：按 created_at 取最新；None created_at 退化为列表最后一个
            samples_sorted = sorted(samples, key=lambda x: x[0] or datetime.min, reverse=True)
            chosen = samples_sorted[0][1] if samples_sorted else None
        cells.setdefault(rv, {})[cv] = chosen

    row_labels.sort()
    col_labels.sort()
    return CrossMatrix(
        row_key=row_key,
        col_key=col_key,
        metric=metric,
        row_labels=row_labels,
        col_labels=col_labels,
        cells=cells,
        agg=agg,
    )
