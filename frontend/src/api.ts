export type Endpoint = {
  id: string;
  label: string;
  provider: string;
  model_id: string;
};
export type Preset = {
  preset_id: string;
  name: string;
  endpoint_id: string;
  test_type: string;
  parameters: Record<string, unknown>;
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
  parameters: Record<string, unknown>;
  progress_completed: number;
  progress_total: number;
  result_run_id: number | null;
  result_artifact: string | null;
  error_code: string | null;
  error_message: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
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
  success_rate: number | null;
  success_rate_ci95: number[] | null;
  metrics: Record<string, Metric>;
};
export type Summary = {
  metric_contract_version: string;
  run: Record<string, string | number | null>;
  overall: Slice;
  group_axis: string;
  groups: Slice[];
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
  error: string | null;
};
export type JobEvent = {
  id: number;
  event: string;
  actor: string;
  to_status: string;
  created_at: number;
};
export type QualityReport = {
  job_id: string;
  model_id: string;
  datasets: Record<
    string,
    {
      accuracy: number;
      correct_samples: number;
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
        is_correct: boolean;
        category?: string;
        failure_category?: string;
        failure_analysis?: string;
        answer_parse_method?: string;
        answer_parse_confidence?: number;
        evaluation_method?: string;
        error?: string | null;
      }[];
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
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
    },
  });
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : detail;
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
