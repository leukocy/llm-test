export type Endpoint = {
  id: string;
  label: string;
  provider: string;
  model_id: string;
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
      extended_metrics: Record<string, unknown>;
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
  return response.json() as Promise<T>;
}
