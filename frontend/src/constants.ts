export type View = "overview" | "runs" | "new";
export type JobType =
  | "concurrency"
  | "prefill"
  | "segmented_prefill"
  | "long_context"
  | "matrix"
  | "stability"
  | "custom_text"
  | "quality";

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
      enable_warmup: false,
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
    id: "quality",
    label: "质量评估",
    description: "使用带数据指纹的标准数据集",
    parameters: {
      datasets: ["gsm8k"],
      max_samples: 30,
      num_shots: 0,
      max_tokens: 512,
      concurrency: 4,
      temperature: 0,
      use_cache: false,
    },
  },
];
export const labels: Record<string, string> = Object.fromEntries(
  scenarios.map((item) => [item.id, item.label]),
);
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
