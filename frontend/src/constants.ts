export type View = "overview" | "runs" | "new" | "warehouse";
export type JobType =
  | "concurrency"
  | "prefill"
  | "segmented_prefill"
  | "long_context"
  | "matrix"
  | "stability"
  | "custom_text"
  | "dataset"
  | "quality"
  | "robustness";

export const scenarios: {
  id: JobType;
  label: string;
  description: string;
  parameters: Record<string, unknown>;
}[] = [
  {
    id: "concurrency",
    label: "并发阶梯",
    description: "逐级测量首字延迟、吞吐与失败率",
    parameters: {
      selected_concurrencies: [1, 4, 8],
      rounds_per_level: 3,
      max_tokens: 512,
      input_tokens_target: 1024,
      warmup_rounds_per_level: 1,
    },
  },
  {
    id: "prefill",
    label: "输入长度",
    description: "对比不同输入规模的 prefill 特性",
    parameters: {
      token_levels: [512, 2048, 8192],
      requests_per_level: 5,
      max_tokens: 256,
      warmup_requests_per_level: 1,
    },
  },
  {
    id: "segmented_prefill",
    label: "分段缓存",
    description: "测量前缀复用与分段输入",
    parameters: {
      segment_levels: [512, 2048],
      requests_per_segment: 3,
      max_tokens: 256,
      cumulative_mode: true,
      total_rounds: 2,
      per_round_unique: false,
      concurrency: 1,
    },
  },
  {
    id: "long_context",
    label: "长上下文",
    description: "观察上下文长度对延迟的影响",
    parameters: {
      context_lengths: [4096, 16384],
      rounds_per_level: 3,
      max_tokens: 256,
    },
  },
  {
    id: "matrix",
    label: "吞吐矩阵",
    description: "并发与上下文的二维扫描",
    parameters: {
      concurrencies: [1, 4, 8],
      context_lengths: [512, 2048],
      rounds: 2,
      max_tokens: 512,
      enable_warmup: true,
    },
  },
  {
    id: "stability",
    label: "稳定性",
    description: "持续运行观察性能波动",
    parameters: {
      concurrency: 4,
      duration_seconds: 120,
      max_tokens: 512,
      input_tokens_target: 1024,
    },
  },
  {
    id: "custom_text",
    label: "自定义提示词",
    description: "对业务提示词进行可重复测量",
    parameters: {
      selected_concurrencies: [1, 4],
      rounds_per_level: 3,
      base_prompt: "请用三句话解释向量检索。",
      suffix_instruction: "",
      max_tokens: 256,
      avoid_cache: true,
    },
  },
  {
    id: "dataset",
    label: "数据集压测",
    description: "对一组提示词逐行发起请求并测量性能",
    parameters: {
      rows: [{ prompt: "请用一句话介绍向量数据库。" }],
      concurrency: 4,
      max_tokens: 256,
      rounds: 1,
    },
  },
  {
    id: "robustness",
    label: "鲁棒性",
    description: "样本经扰动后的答案保持度与一致性",
    parameters: {
      samples: [{ question: "1+1 等于几?", correct_answer: "2" }],
      perturbation_types: ["typo", "case", "punctuation"],
      max_tokens: 256,
    },
  },
  {
    id: "quality",
    label: "质量评估",
    description: "使用带数据指纹的标准数据集",
    parameters: {
      datasets: ["mmlu", "gsm8k"],
      max_samples: 100,
      num_shots: 5,
      max_tokens: 8192,
      concurrency: 4,
      temperature: 0,
      use_cache: true,
    },
  },
];

export type MeasurementProfile =
  "quick" | "standard" | "thorough" | "original" | "custom";

// Defaults transcribed from the first repository revision.
export const firstCommitParameters: Partial<
  Record<JobType, Record<string, unknown>>
> = {
  concurrency: {
    selected_concurrencies: [1, 2],
    rounds_per_level: 1,
    input_tokens_target: 64,
    max_tokens: 512,
    warmup_rounds_per_level: 0,
  },
  prefill: {
    token_levels: [4096, 8192, 16384, 32768, 65536, 130000],
    requests_per_level: 1,
    max_tokens: 1,
    warmup_requests_per_level: 0,
  },
  segmented_prefill: {
    segment_levels: [2000, 8000, 20000, 40000, 60000],
    requests_per_segment: 1,
    max_tokens: 512,
    cumulative_mode: true,
    total_rounds: 1,
    per_round_unique: false,
    concurrency: 1,
  },
  long_context: {
    context_lengths: [4096, 8192, 16384, 32768, 65536, 130000],
    rounds_per_level: 1,
    max_tokens: 512,
  },
  matrix: {
    concurrencies: [1, 2],
    context_lengths: [1024, 4096, 16384, 65536],
    rounds: 1,
    max_tokens: 256,
    enable_warmup: true,
  },
  stability: {
    concurrency: 1,
    duration_seconds: 60,
    max_tokens: 512,
    input_tokens_target: 64,
  },
  custom_text: {
    selected_concurrencies: [1, 2, 4],
    rounds_per_level: 1,
    max_tokens: 512,
    avoid_cache: true,
    suffix_instruction: "Please summarize the above content.",
  },
};

export const profileOverrides: Record<
  JobType,
  { quick: Record<string, unknown>; thorough: Record<string, unknown> }
> = {
  concurrency: {
    quick: {
      selected_concurrencies: [1, 2],
      rounds_per_level: 1,
      max_tokens: 128,
      input_tokens_target: 256,
      warmup_rounds_per_level: 1,
    },
    thorough: {
      selected_concurrencies: [1, 2, 4, 8, 16],
      rounds_per_level: 5,
      warmup_rounds_per_level: 1,
    },
  },
  prefill: {
    quick: {
      token_levels: [512, 2048],
      requests_per_level: 2,
      max_tokens: 128,
      warmup_requests_per_level: 0,
    },
    thorough: {
      token_levels: [512, 2048, 8192, 16384],
      requests_per_level: 10,
      warmup_requests_per_level: 2,
    },
  },
  segmented_prefill: {
    quick: { segment_levels: [512], requests_per_segment: 1, total_rounds: 1 },
    thorough: {
      segment_levels: [512, 2048, 8192],
      requests_per_segment: 5,
      total_rounds: 3,
    },
  },
  long_context: {
    quick: { context_lengths: [4096], rounds_per_level: 1, max_tokens: 128 },
    thorough: { context_lengths: [4096, 16384, 32768], rounds_per_level: 5 },
  },
  matrix: {
    quick: {
      concurrencies: [1, 4],
      context_lengths: [512],
      rounds: 1,
      max_tokens: 128,
      enable_warmup: false,
    },
    thorough: {
      concurrencies: [1, 4, 8, 16],
      context_lengths: [512, 2048, 8192],
      rounds: 3,
    },
  },
  stability: {
    quick: { concurrency: 1, duration_seconds: 30, max_tokens: 128 },
    thorough: { duration_seconds: 600 },
  },
  custom_text: {
    quick: {
      selected_concurrencies: [1],
      rounds_per_level: 1,
      max_tokens: 128,
    },
    thorough: { selected_concurrencies: [1, 4, 8], rounds_per_level: 5 },
  },
  dataset: {
    quick: { concurrency: 1, rounds: 1, max_tokens: 128 },
    thorough: { concurrency: 8, rounds: 3 },
  },
  quality: {
    quick: { max_samples: 10, concurrency: 1 },
    thorough: { max_samples: 100, concurrency: 8 },
  },
  robustness: {
    quick: { perturbation_types: ["typo"], max_tokens: 128 },
    thorough: {
      perturbation_types: [
        "typo",
        "case",
        "punctuation",
        "whitespace",
        "synonym",
      ],
    },
  },
};

export function profileParameters(
  type: JobType,
  profile: Exclude<MeasurementProfile, "custom">,
) {
  const base = scenarios.find((item) => item.id === type)?.parameters || {};
  if (profile === "original")
    return { ...base, ...firstCommitParameters[type] };
  return profile === "standard"
    ? { ...base }
    : { ...base, ...profileOverrides[type][profile] };
}
export const labels: Record<string, string> = Object.fromEntries(
  scenarios.map((item) => [item.id, item.label]),
);
labels.throughput_matrix = labels.matrix;
labels.dataset_prepare = "数据准备";
export const statusLabels: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  pausing: "暂停中",
  paused: "已暂停",
  cancelling: "取消中",
  cancelled: "已取消",
  completed: "已完成",
  failed: "失败",
};
export const activeStates = new Set([
  "queued",
  "running",
  "pausing",
  "paused",
  "cancelling",
]);
export const pausableTypes = new Set([
  "quality",
  "robustness",
  "stability",
  "concurrency",
  "prefill",
  "segmented_prefill",
  "long_context",
  "matrix",
  "custom_text",
  "dataset",
]);

export function formatNumber(value: number | null | undefined, digits = 2) {
  return value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : value.toLocaleString("zh-CN", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      });
}
export function formatPercent(value: number | null | undefined) {
  return value == null ? "—" : `${formatNumber(value * 100, 1)}%`;
}
export function date(value: number | null | undefined) {
  return value
    ? new Date(value * 1000).toLocaleString("zh-CN", { hour12: false })
    : "—";
}
export function shortId(value: string) {
  return value.slice(0, 8);
}

// ---- 趋势/对比目录（与后端 ui/warehouse_charts.py 同口径） ----
export const trendMetrics: Record<string, string> = {
  decode_tps: "decode TPS",
  prefill_tps: "prefill TPS",
  ttft_s: "TTFT (s)",
  p50_latency_s: "p50 (s)",
  p95_latency_s: "p95 (s)",
  p99_latency_s: "p99 (s)",
  effective_bandwidth_gbps: "等效带宽 (GB/s)",
  bandwidth_utilization_pct: "带宽利用率 (%)",
  gpu_vram_peak_gb: "显存峰值 (GB)",
  system_memory_peak_gb: "内存峰值 (GB)",
  gpu_util_pct: "GPU 利用率 (%)",
  cpu_util_pct: "CPU 利用率 (%)",
};
export const trendDims: Record<string, string> = {
  model_name: "模型",
  machine_id: "硬件",
  engine: "引擎",
  parallel_strategy: "并行",
  quantization: "量化",
  tester: "测试员",
};
export const compareGroups = ["性能", "资源峰值"];

// ---- spec 字段中文标签（SchemaForm 表头用，缺省回退字段名） ----
export const fieldLabels: Record<string, string> = {
  selected_concurrencies: "并发档位（逗号分隔）",
  rounds_per_level: "每档轮数",
  warmup_rounds_per_level: "每档预热轮数（不计入正式样本）",
  max_tokens: "最大输出 tokens",
  input_tokens_target: "输入 tokens 目标（0 = 不加压）",
  token_levels: "输入长度档位",
  requests_per_level: "每档请求数",
  warmup_requests_per_level: "每档预热请求数（不计入正式样本）",
  segment_levels: "分段长度档位",
  requests_per_segment: "每段轮数（每轮请求数等于并发数）",
  cumulative_mode: "累积模式（前缀复用）",
  total_rounds: "总轮数",
  per_round_unique: "每轮独立内容",
  concurrency: "并发数",
  context_lengths: "上下文长度档位",
  concurrencies: "并发档位",
  rounds: "轮数",
  enable_warmup: "每个矩阵条件预热一轮",
  duration_seconds: "持续时长（秒）",
  base_prompt: "基础提示词",
  suffix_instruction: "附加指令",
  avoid_cache: "避开缓存",
  datasets: "数据集（逗号分隔）",
  max_samples: "每数据集样本数",
  num_shots: "few-shot 数",
  model_type: "模型类型",
  use_llm_judge: "AI Judge 错题二次复核",
  ceval_split: "C-Eval 评分分区",
  temperature: "温度",
  use_cache: "使用缓存",
  thinking_enabled: "启用思考模式",
  thinking_budget: "思考预算 tokens",
  reasoning_effort: "推理强度",
  random_seed: "随机种子",
  skip_first_token_for_tps: "TPS 不计首 token",
  template_tokens: "模板 tokens",
  latency_offset: "延迟校准（秒）",
  tokenizer_option: "Tokenizer 选择",
  hf_tokenizer_model_id: "HF Tokenizer 模型 ID",
};
