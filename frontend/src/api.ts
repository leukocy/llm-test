export type Endpoint = {
  id: string;
  label: string;
  provider: string;
  api_base_url: string;
  model_id: string;
  tokenizer_option: string;
  credential_configured: boolean;
  source: "managed" | "file";
};
export type Preset = {
  tags: string[];
  source_metadata?: Record<string, unknown> | null;
  preset_id: string;
  name: string;
  description: string;
  endpoint_id: string;
  test_type: string;
  parameters: Record<string, unknown>;
  run_config: Record<string, unknown>;
  created_at: number;
  updated_at: number;
};
export type WarehouseRow = {
  run_id: number;
  test_id: string;
  test_type: string;
  date: string;
  machine_id: string;
  model_name: string;
  engine: string;
  parallel_strategy: string;
  concurrency: number | null;
  decode_tps: number | null;
  ttft_s: number | null;
  effective_bandwidth_gbps: number | null;
  gpu_vram_peak_gb: number | null;
  bottleneck: string;
  status: string;
  external_level: string;
  tester: string;
  config_hash: string;
  tags: string | null;
};
export type WarehouseData = {
  scope: {
    matched_total: number;
    scanned: number;
    shown: number;
    scan_limit: number;
    truncated: boolean;
    invalid_rows: number;
  };
  kpis: {
    runs: number;
    machines: number;
    models: number;
    completed: number;
    publishable: number;
  };
  filters: Record<string, string[]>;
  rows: WarehouseRow[];
  matrix: {
    metric: string;
    agg: string;
    row_labels: string[];
    col_labels: string[];
    cells: Record<string, Record<string, number | null>>;
  };
  inventory: Record<string, string | number | null>[];
  scaling: {
    model_name: string;
    tp_size: number;
    decode_tps: number;
    speedup_vs_tp1: number | null;
    efficiency: number | null;
    interpretation: string;
  }[];
};
export type WarehouseDetail = {
  run_id: number;
  test_id: string;
  test_type: string;
  fields: Record<string, string | number | boolean | null>;
};
export type Job = {
  job_id: string;
  status: string;
  test_type: string;
  endpoint_id: string;
  model_id: string;
  parent_job_id: string | null;
  parameters: Record<string, unknown>;
  saved_progress_planned?: number;
  saved_progress_unit?: string;
  saved_progress_committed?: number;
  saved_progress_at?: number;
  progress_completed: number;
  progress_total: number;
  pause_count: number;
  paused_seconds: number;
  pause_started_at: number | null;
  result_run_id: number | null;
  result_artifact: string | null;
  error_code: string | null;
  error_message: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
};
export type BatchSummary = {
  batch_id: string;
  name: string;
  description: string;
  default_endpoint_id: string;
  requested_items: number;
  submitted_items: number;
  max_parallel: number;
  stop_on_error: boolean;
  created_at: number;
  status_counts?: Record<string, number>;
};
export type Metric = {
  count: number;
  mean: number | null;
  median: number | null;
  p95: number | null;
  p99: number | null;
  min: number | null;
  max: number | null;
};
export type Slice = {
  label?: string;
  requests: number;
  successes: number;
  failures: number;
  unknown_outcomes?: number;
  success_rate: number | null;
  success_rate_ci95: number[] | null;
  metrics: Record<string, Metric>;
  planned_requests?: number | null;
  input_tokens?: {
    target: number | null;
    count: number;
    median: number | null;
    target_deviation_pct: number | null;
  };
};
export type MeasurementPlan = {
  protocol_version: string | null;
  workload_model: string;
  measured_requests: number;
  warmup_requests: number;
  warmup_recorded?: number;
  warmup_failures?: number;
  total_requests: number;
  configured_input_token_volume?: number | null;
  maximum_output_token_volume?: number | null;
  cells: {
    label: string;
    measured_requests: number;
    warmup_requests: number;
    input_tokens_target?: number;
  }[];
  warnings: string[];
};
export type ReportEnvironment = {
  source: "user_reported";
  scope: "model_server" | "test_client" | "unspecified";
  fields: Partial<
    Record<
      "processor" | "mainboard" | "memory" | "gpu" | "system" | "engine_name",
      string
    >
  >;
};
export type StabilityTimeSeries = {
  live: boolean;
  window_state: string | null;
  contract: string;
  timed_requests: number;
  missing_requests: number;
  invalid_requests: number;
  complete: boolean;
  window_seconds: number | null;
  planned_seconds: number | null;
  admission_budget_seconds?: number | null;
  attempt?: number | null;
  bin_seconds: number | null;
  notes: string[];
  cross_interruption?: boolean;
  segments?: StabilityTimeSeries[];
  bins: {
    start_seconds: number;
    end_seconds: number;
    requests: number;
    failures: number;
    successes: number;
    metrics: Record<string, Metric>;
  }[];
};

export type Summary = {
  stability_budget?: {
    planned_seconds: number;
    saved_scheduling_seconds: number;
    remaining_seconds: number;
  } | null;
  time_series?: StabilityTimeSeries;
  extended_observations?: {
    system: {
      contract: string;
      valid_batches: number;
      invalid_batches: number;
      untagged_requests: number;
    };
    cache: {
      sources: Record<string, number>;
      unknown_successes: number;
      invalid_observations: number;
    };
    phase?: {
      contract: string;
      valid_batches: number;
      missing_clock_batches: number;
      invalid_clock_batches: number;
      no_success_batches: number;
      missing_first_token_batches: number;
      nonpositive_windows: { input: number; output: number; total: number };
    };
  };
  scenario_analysis?: {
    cards?: { key: string; label: string; value: string; note: string }[];
    title: string;
    axis: string;
    notes: string[];
    observations: { metric: string; text: string }[];
  };
  metric_contract_version: string;
  integrity: {
    verified: boolean;
    expected_requests: number | null;
    recorded_requests: number;
    reasons: string[];
  };
  run: Record<string, string | number | null>;
  overall: Slice;
  group_axis: string;
  groups: Slice[];
  measurement_protocol: MeasurementPlan | null;
  execution_control: {
    batch_id?: string | null;
    pause_policy?: string | null;
    pause_count?: number;
    paused_seconds?: number;
    max_parallel?: number;
    stop_on_error?: boolean;
  };
  report_environment?: ReportEnvironment | null;
  tokenizer_installation?: {
    name: string;
    repo_id: string;
    revision: string;
    files: { name: string; size: number; sha256: string }[];
  } | null;
  data_quality: { warnings: string[] };
  provenance: { token_sources: string[]; token_methods: string[] };
  notes: string[];
};
export type RequestResult = {
  id: number;
  session_id: number | null;
  concurrency_level: number | null;
  ttft: number | null;
  tps: number | null;
  total_time: number | null;
  prefill_tokens: number | null;
  decode_tokens: number | null;
  token_source: string | null;
  prompt_sha256: string | null;
  error: string | null;
};
export type JobEvent = {
  id: number;
  event: string;
  actor: string;
  to_status: string;
  created_at: number;
};
export type ReasoningAssessment = {
  version: string;
  source: string;
  correctness_basis: string;
  dimensions: Record<string, number | null>;
  overall: number | null;
  input_sha256: string;
  weights: Record<string, number>;
  status: string;
};
export type ReasoningAnalysisSummary = {
  version: string;
  source: string;
  total: number;
  ignored: number;
  dimensions: Record<string, { label: string; n: number; mean: number | null }>;
  radar_n: number;
  note: string;
  figure: import("plotly.js-dist-min").Figure | null;
};

export type QualityReport = {
  job_id: string;
  model_id: string;
  report_environment?: ReportEnvironment | null;
  analysis?: QualityReportAnalysis;
  datasets: Record<
    string,
    {
      accuracy: number;
      correct_samples: number;
      standard_correct_samples?: number | null;
      judge_corrected_samples?: number | null;
      total_samples: number;
      duration_seconds: number;
      config: Record<string, unknown>;
      by_category: Record<string, { accuracy: number | null; count: number }>;
      performance_stats: Record<string, number>;
      extended_metrics: Record<string, unknown>;
      details: {
        sample_id: string;
        question: string;
        prompt: string;
        correct_answer: string;
        predicted_answer: string;
        model_response: string;
        latency_ms?: number | null;
        reasoning_content?: string;
        reasoning_quality?: number | null;
        reasoning_quality_overall?: number | null;
        reasoning_assessment?: ReasoningAssessment | null;
        is_correct: boolean;
        is_judge_corrected?: boolean;
        judge_verdict?: string | null;
        category?: string;
        failure_category?: string;
        failure_analysis?: string;
        failure_confidence?: number | null;
        failure_root_cause?: string;
        failure_suggestions?: string[];
        execution_error?: string | null;
        answer_parse_method?: string;
        answer_parse_confidence?: number;
        evaluation_method?: string;
        measurement_provenance?: {
          ttft_scope?: string;
          output_token_scope?: string;
          provider_usage?: Record<string, unknown> | null;
          finish_reason?: string | null;
        };
        error?: string | null;
      }[];
    }
  >;
};

export type QualityMetricStats = {
  count: number;
  eligible: number;
  mean: number | null;
  median: number | null;
  p95: number | null;
  p99: number | null;
  min: number | null;
  max: number | null;
  total: number | null;
  sources: Record<string, number>;
};
export type QualityReportAnalysis = {
  version: string;
  notes: string[];
  figures: Record<string, import("plotly.js-dist-min").Figure>;
  datasets: Record<
    string,
    {
      name: string;
      model: string;
      timestamp: string;
      reasoning_analysis?: ReasoningAnalysisSummary;
      failure_analysis: {
        status: string;
        source: string;
        failed_samples: number;
        response_samples: number;
        request_errors: number;
        failure_rate: number | null;
        distribution: Record<string, number>;
        top_issues: string[];
        suggestions: string[];
        warnings: string[];
        note: string;
        figure: import("plotly.js-dist-min").Figure | null;
      };
      total: number | null;
      correct: number | null;
      accuracy: number | null;
      ci95: number[] | null;
      standard_correct: number | null;
      judge_corrected: number | null;
      duration_seconds: number | null;
      detail_count: number;
      complete: boolean;
      cache_count: number;
      unknown_cache: number;
      request_errors: number;
      non_error_fraction: number | null;
      metrics: Record<string, QualityMetricStats>;
      methods: Record<string, number>;
      parsers: Record<string, number>;
      warnings: string[];
      confidence: {
        count: number;
        mean: number | null;
        high: number;
        low: number;
      };
      category_figure: import("plotly.js-dist-min").Figure;
    }
  >;
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(
  token: string,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(init.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init.headers,
    },
  });
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = body.detail
          .slice(0, 3)
          .map(
            (item: { loc?: string[]; msg?: string }) =>
              `${item.loc?.slice(1).join(".") || "参数"}: ${item.msg || "校验失败"}`,
          )
          .join("；");
      }
    } catch {
      /* keep status */
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function downloadFile(
  token: string,
  path: string,
  filename: string,
) {
  const response = await fetch(path, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    let detail = `下载失败（${response.status}）`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep status */
    }
    throw new ApiError(response.status, detail);
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export type EvaluationCheckpoint = {
  unit_label?: string;
  supported: boolean;
  available: boolean;
  planned_units: number;
  committed_units: number;
  issued_unit_attempts: number;
  repeated_unit_attempts: number;
  recoveries: number;
  can_recover: boolean;
  can_delete: boolean;
  revision: string | null;
  saved_at: number | null;
  notes: string[];
};
