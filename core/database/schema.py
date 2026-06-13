"""
Database Schema 定义

包含所has表 SQL 定义andMigration语句。
"""

SCHEMA_VERSION = "1.3.0"

# ============================================
# Table schema定义
# ============================================

CREATE_TEST_RUNS = """
CREATE TABLE IF NOT EXISTS test_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id TEXT UNIQUE NOT NULL,
    test_type TEXT NOT NULL,

    -- Status
    status TEXT NOT NULL DEFAULT 'running',
    progress_percent REAL DEFAULT 0,

    -- Model Configuration
    model_id TEXT NOT NULL,
    provider TEXT,
    api_base_url TEXT,
    api_key_hash TEXT,

    -- Test Parameters
    concurrency INTEGER DEFAULT 1,
    max_tokens INTEGER DEFAULT 512,
    temperature REAL DEFAULT 0.0,
    thinking_enabled INTEGER DEFAULT 0,
    thinking_budget INTEGER,
    reasoning_effort TEXT DEFAULT 'medium',

    -- 完整Configure快照 (JSON)
    config_json TEXT,

    -- 时间信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,

    -- Statistics摘要
    total_requests INTEGER DEFAULT 0,
    completed_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    success_rate REAL,

    -- Aggregate指标
    avg_ttft REAL,
    avg_tps REAL,
    avg_tpot REAL,
    p50_ttft REAL,
    p95_ttft REAL,
    p99_ttft REAL,
    total_tokens INTEGER,

    -- 环境信息
    system_info_json TEXT,
    python_version TEXT,
    git_hash TEXT,

    -- 元Data
    tags TEXT,
    notes TEXT,
    csv_path TEXT,

    -- ===== 1.2.0 数据仓库扩展（手册：报告是切片，仓库是全集）=====
    -- 硬件指纹分组 / 测试人 / 可对外等级 / 瓶颈归因（一等列，便于筛选）
    machine_id TEXT,
    tester TEXT,
    external_level TEXT DEFAULT 'internal',  -- internal / review / publishable
    bottleneck TEXT,
    next_action TEXT,
    supersedes_test_id TEXT,   -- 复测指向原 test_id
    comparison_group TEXT,     -- MTP on/off、A/B 对照分组
    mtp_enabled INTEGER,       -- 推测解码开关（便于直接筛选）

    -- 资源监控 / 等效带宽 头条指标（一等列，便于筛选/排序）
    effective_bandwidth_gbps REAL,
    bandwidth_utilization_pct REAL,
    gpu_vram_peak_gb REAL,
    system_memory_peak_gb REAL,

    -- 变长 JSON 字段（富模型规格 / 服务配置 / 监控时序 / 状态明细）
    model_spec_json TEXT,        -- 模型架构规格（架构/MoE/KV精度/MTP...）
    serving_config_json TEXT,    -- 服务配置（引擎/并行/注意力&MoE后端/调度/MTP/runtime）
    resource_monitor_json TEXT,  -- 资源监控汇总 + 时序
    status_detail TEXT,          -- 状态明细：未测/异常/已通过/需复测/可对外

    -- ===== 1.3.0 推理引擎运行时（手册 F 维 KV 实况 + 引擎自身视图）=====
    engine_metrics_json TEXT,            -- 引擎 /metrics 时序 + peaks + cache_config
    gpu_kv_cache_usage_peak_pct REAL,    -- 引擎 GPU KV cache 占用峰值(0~1)
    num_preemption_total INTEGER,        -- 测试窗口内 KV 抢救数（稳定性信号）
    engine_running_requests_peak INTEGER,  -- 调度器运行队列峰值
    kv_cache_capacity_tokens INTEGER      -- 引擎 KV 容量(tokens)，来自 cache_config/log
);
"""

CREATE_TEST_RESULTS = """
CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,

    -- 请求标识
    session_id INTEGER,
    request_index INTEGER,
    round INTEGER,
    concurrency_level INTEGER,
    batch_id TEXT,

    -- onunder文/输入
    input_tokens_target INTEGER,
    context_length_target INTEGER,

    -- Performance Metrics - Latency
    ttft REAL,
    tpot REAL,
    tpot_p95 REAL,
    tpot_p99 REAL,
    total_time REAL,
    decode_time REAL,
    prefill_speed REAL,

    -- Performance Metrics - 吞吐
    tps REAL,
    system_throughput REAL,
    system_input_throughput REAL,
    system_output_throughput REAL,
    system_total_throughput REAL,
    rps REAL,

    -- Token Statistics
    prefill_tokens INTEGER,
    decode_tokens INTEGER,
    cache_hit_tokens INTEGER,
    api_prefill INTEGER,
    api_decode INTEGER,
    effective_prefill_tokens INTEGER,
    effective_decode_tokens INTEGER,
    token_source TEXT,
    token_calc_method TEXT,
    cache_hit_source TEXT,

    -- 时间戳
    start_time REAL,
    end_time REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Error message
    error TEXT,
    error_type TEXT,

    -- Prompt and输出（用于可复现性）
    prompt_text TEXT,
    output_text TEXT,

    -- 扩展字段 (JSON)
    extra_metrics TEXT,

    FOREIGN KEY (run_id) REFERENCES test_runs(id) ON DELETE CASCADE
);
"""

CREATE_API_LOGS = """
CREATE TABLE IF NOT EXISTS api_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id TEXT UNIQUE NOT NULL,
    run_id INTEGER,

    -- 基本信息
    session_id TEXT,
    test_type TEXT,
    status TEXT NOT NULL,

    -- Model Configuration
    provider TEXT,
    model_id TEXT,
    api_base_url TEXT,

    -- Performance Metrics
    ttft REAL,
    total_time REAL,

    -- 详细信息 (JSON)
    request_json TEXT,
    response_json TEXT,
    metrics_json TEXT,

    -- Error
    error TEXT,

    -- 时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id) REFERENCES test_runs(id) ON DELETE SET NULL
);
"""

CREATE_EXECUTION_LOGS = """
CREATE TABLE IF NOT EXISTS execution_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,

    -- Log Content
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    session_id TEXT,

    -- 指标 (JSON)
    metrics_json TEXT,

    -- Error
    error TEXT,

    -- 时间
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id) REFERENCES test_runs(id) ON DELETE CASCADE
);
"""

CREATE_REPORTS = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT UNIQUE NOT NULL,
    run_id INTEGER,

    -- 基本信息
    report_type TEXT NOT NULL,
    version TEXT DEFAULT '1.0',

    -- Model信息
    model_id TEXT NOT NULL,
    model_type TEXT,
    provider TEXT,

    -- 报告内容 (JSON)
    model_info_json TEXT,
    environment_json TEXT,
    config_json TEXT,
    results_json TEXT,
    aggregate_json TEXT,
    failure_analysis_json TEXT,

    -- 质量Evaluation特has
    latency_metrics_json TEXT,
    token_metrics_json TEXT,
    quality_metrics_json TEXT,
    cost_metrics_json TEXT,

    -- ExportFile path
    json_path TEXT,
    html_path TEXT,
    markdown_path TEXT,
    excel_path TEXT,

    -- 元Data
    tags TEXT,
    notes TEXT,

    -- 时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id) REFERENCES test_runs(id) ON DELETE SET NULL
);
"""

CREATE_DB_META = """
CREATE TABLE IF NOT EXISTS db_meta (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# ============================================
# Index定义
# ============================================

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_test_runs_type ON test_runs(test_type);",
    "CREATE INDEX IF NOT EXISTS idx_test_runs_model ON test_runs(model_id);",
    "CREATE INDEX IF NOT EXISTS idx_test_runs_status ON test_runs(status);",
    "CREATE INDEX IF NOT EXISTS idx_test_runs_created ON test_runs(created_at);",
    # 注：machine_id / external_level / comparison_group 上的索引引用 1.2.0 新列，
    # 不能放在 create_tables（对老库会在迁移加列之前执行而崩溃），统一由 1.2.0 迁移创建。

    "CREATE INDEX IF NOT EXISTS idx_results_run_id ON test_results(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_results_session ON test_results(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_results_error ON test_results(error) WHERE error IS NOT NULL;",

    "CREATE INDEX IF NOT EXISTS idx_api_logs_run ON api_logs(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_api_logs_status ON api_logs(status);",
    "CREATE INDEX IF NOT EXISTS idx_api_logs_model ON api_logs(model_id);",
    "CREATE INDEX IF NOT EXISTS idx_api_logs_created ON api_logs(created_at);",

    "CREATE INDEX IF NOT EXISTS idx_exec_logs_run ON execution_logs(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_exec_logs_level ON execution_logs(level);",
    "CREATE INDEX IF NOT EXISTS idx_exec_logs_time ON execution_logs(timestamp);",

    "CREATE INDEX IF NOT EXISTS idx_reports_run ON reports(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_reports_model ON reports(model_id);",
    "CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);",
    "CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at);",
]


def get_schema_sql() -> str:
    """Get完整 Schema SQL"""
    tables = [
        CREATE_TEST_RUNS,
        CREATE_TEST_RESULTS,
        CREATE_API_LOGS,
        CREATE_EXECUTION_LOGS,
        CREATE_REPORTS,
        CREATE_DB_META,
    ]
    return "\n".join(tables + CREATE_INDEXES)


def create_tables(conn) -> None:
    """
    Create所has表andIndex

    Args:
        conn: sqlite3 Connect对象
    """
    cursor = conn.cursor()

    # Create表
    for sql in [
        CREATE_TEST_RUNS,
        CREATE_TEST_RESULTS,
        CREATE_API_LOGS,
        CREATE_EXECUTION_LOGS,
        CREATE_REPORTS,
        CREATE_DB_META,
    ]:
        cursor.execute(sql)

    # CreateIndex
    for sql in CREATE_INDEXES:
        cursor.execute(sql)

    # 注：不在此写入 schema_version，让 run_migrations 成为版本唯一来源。
    # 这样全新库也会执行幂等迁移（含 1.2.0 新列索引），老库则补齐历史迁移。
    conn.commit()
