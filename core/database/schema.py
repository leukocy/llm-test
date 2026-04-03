"""
Database Schema 定义

包含所has表 SQL 定义andMigration语句。
"""

SCHEMA_VERSION = "1.1.0"

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
    csv_path TEXT
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

    # Initialize元Data
    cursor.execute("""
        INSERT OR IGNORE INTO db_meta (key, value) VALUES ('schema_version', ?)
    """, (SCHEMA_VERSION,))

    conn.commit()
