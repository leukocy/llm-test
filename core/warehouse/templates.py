"""
数据仓库字段模板（手册 #templates 的三套字段，verbatim）。

手册原话：「仓库和报告不是一回事：数据仓库是全集，保存每一行可复用、可筛选、
可追溯的数据；单次测试报告是某一次测试的切片，用来给人看。」

三套模板定义了仓库的可导出口径——导出时按这些字段顺序产出 CSV/JSON，缺测字段
留空（手册：「缺测本身就是决策信息」）。字段名逐字取自
test-standard/端侧AI硬件与模型.html 的 <pre id="hwInventory|hmTest|maTest">。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 三套字段模板（顺序、命名与手册 <pre> 完全一致）
# 每套 = 手册对应 <pre> 里那一行 CSV 字段的逐字拷贝，split 成列表。
# 用 split 而非 list literal：便于和手册 <pre> 逐字 diff（SIM905 故意保留）。
# ---------------------------------------------------------------------------

# 硬件盘点字段（一台机器一行）
HARDWARE_INVENTORY_FIELDS: list[str] = (  # noqa: SIM905
    "machine_id,product_line,cpu_model,cpu_sockets,cpu_cores,cpu_threads,"
    "numa_nodes,memory_type,memory_capacity_gb,memory_channels_populated,"
    "memory_speed_mtps,ecc_enabled,gpu_model,gpu_count,gpu_vram_gb,"
    "gpu_memory_type,gpu_bandwidth_gbps,pcie_gen,pcie_width,ssd_model,"
    "ssd_capacity_tb,os,driver,cuda_or_rocm,engine_ready,power_supply_w,"
    "cooling_note,owner,location,remark"
).split(",")

# 硬件 × 模型测试字段（一次测试一行）
HM_TEST_FIELDS: list[str] = (  # noqa: SIM905
    "test_id,date,tester,machine_id,engine,engine_version,engine_params,"
    "parallel_strategy,model_name,model_version,model_type,total_params,"
    "active_params,num_experts,top_k,quantization,dtype,max_context,"
    "concurrency,usecase_set_version,prompt_tokens,output_tokens,load_time_s,"
    "ttft_s,prefill_tps,decode_tps,long_context_tps,p50_latency_s,"
    "p95_latency_s,p99_latency_s,gpu_vram_peak_gb,system_memory_peak_gb,"
    "effective_bandwidth_gbps,bandwidth_utilization_pct,cpu_threads_used,"
    "cpu_util_pct,gpu_util_pct,power_w,temp_c,status,bottleneck,error_type,"
    "error_detail,log_path,screenshot_path,external_level,next_action,"
    "supersedes_test_id"
).split(",")

# 模型 × 应用测试字段（一个应用用例一行；应用质量维度，部分字段由未来的应用评估层填）
MA_TEST_FIELDS: list[str] = (  # noqa: SIM905
    "case_id,date,tester,scenario,task_name,customer_type,model_name,"
    "machine_id,engine,usecase_set_version,input_tokens,output_tokens,"
    "context_length,concurrency,ttft_s,retrieval_latency_s,prefill_latency_s,"
    "total_latency_s,decode_tps,quality_score,success,citation_score,"
    "tool_success_rate,privacy_requirement,cost_note,recommended_config,"
    "sales_summary,external_level,failure_reason,evidence_path,next_action"
).split(",")

# 模板注册表
TEMPLATE_FIELDS: dict[str, list[str]] = {
    "hwInventory": HARDWARE_INVENTORY_FIELDS,
    "hmTest": HM_TEST_FIELDS,
    "maTest": MA_TEST_FIELDS,
}

# 中文展示名（UI 用）
TEMPLATE_TITLES: dict[str, str] = {
    "hwInventory": "硬件盘点字段",
    "hmTest": "硬件 × 模型测试字段",
    "maTest": "模型 × 应用测试字段",
}

# 单行简介（UI / 导出文件头注释用）
TEMPLATE_DESCRIPTIONS: dict[str, str] = {
    "hwInventory": "一台机器一行：machine_id 维度的硬件清单（CPU/内存/GPU/存储/驱动）。",
    "hmTest": "一次硬件×模型测试一行：含性能、资源、归因、可对外等级。",
    "maTest": "一个模型×应用用例一行：RAG/代码/文档/Agent 场景的能力与质量评分。",
}


def template_fields(name: str) -> list[str]:
    """取某套模板的字段列表。未知模板名返回空列表。"""
    return list(TEMPLATE_FIELDS.get(name, []))


def all_template_names() -> list[str]:
    """全部模板名（稳定顺序）。"""
    return ["hwInventory", "hmTest", "maTest"]
