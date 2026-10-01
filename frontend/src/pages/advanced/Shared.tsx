import { formatNumber } from "../../constants";

export type Snapshot = {
  tool_version: string;
  perturbation_generator_version?: string;
  input_sha256: string;
  input: Record<string, unknown>;
  warnings: string[];
};
export type ParseResult = Snapshot & {
  extracted_answer: string;
  confidence: number | null;
  method: string;
  answer_type: string;
  normalized_value: string | null;
  error: string | null;
  is_correct?: boolean | null;
  comparison_method?: string;
};
export type ConsistencyResult = Snapshot & {
  total: number;
  parsed: number;
  unparsed: number;
  consistency_rate: number | null;
  stable: boolean | null;
  tied_majority: boolean;
  reference_match_rate: number | null;
  reference_matched: number | null;
  reference_undetermined: number | null;
  groups: { answer: string; count: number; normalized_key: string }[];
  results: {
    index: number;
    response: string;
    extracted_answer: string;
    error: string | null;
    is_correct: boolean | null;
    group_key: string | null;
  }[];
};
export type PerturbResult = Snapshot & {
  label: string;
  original_text: string;
  perturbed_text: string;
  details: string;
  status: "applied" | "unchanged" | "unsupported";
};
export type ReasoningResult = Snapshot & {
  reasoning_analysis?: import("../../api").ReasoningAnalysisSummary;
  final_answer_correct: boolean | null;
  quality_score: Record<string, number | null>;
  steps: { step_number: number; content: string; step_type: string }[];
  failure_category: string;
  failure_analysis: string;
};
export type Catalog = {
  datasets: string[];
  perturbations: {
    id: string;
    label: string;
    supported: boolean;
    description: string;
  }[];
};

const DATASETS = [
  "auto",
  "mmlu",
  "gsm8k",
  "math500",
  "humaneval",
  "gpqa",
  "truthfulqa",
  "longbench",
];
export const TABS = ["答案解析", "回答一致性", "文本扰动", "推理评分"];
export const SAMPLE_RESPONSE =
  "Let me solve this step by step.\n\n15 + 27 = 42\n\nTherefore, the answer is \\boxed{42}.";
export const SAMPLE_TEXT =
  "If you have 1000 apples and buy 500 more, how many do you have in total?";
export const DIM_LABELS: Record<string, string> = {
  coherence: "连贯性",
  completeness: "完整性",
  relevance: "相关性",
  correctness: "正确性",
  efficiency: "效率",
  overall: "综合",
};

export function judgment(value: boolean | null | undefined) {
  return value == null ? "未能判定" : value ? "匹配" : "不匹配";
}
export function percent(value: number | null) {
  return value == null ? "未能判定" : `${formatNumber(value * 100, 1)}%`;
}
export function ToolError({ text }: { text: string }) {
  return text ? (
    <div className="alert" role="alert">
      {text}
    </div>
  ) : null;
}
export function DatasetPicker({
  id,
  value,
  onChange,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="schema-field" htmlFor={id}>
      <span className="input-label">数据集类型</span>
      <select
        aria-label="数据集类型"
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {DATASETS.map((dataset) => (
          <option key={dataset} value={dataset}>
            {dataset === "auto" ? "auto · 自动识别" : dataset}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ToolRecord({
  result,
  name,
}: {
  result: Snapshot;
  name: string;
}) {
  function download() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `${name}-${result.input_sha256.slice(0, 12)}.json`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <div className="advanced-record">
      <ul>
        {result.warnings.map((warning) => (
          <li key={warning}>{warning}</li>
        ))}
      </ul>
      <div className="advanced-record-footer">
        <div>
          <span>
            {result.tool_version}
            {result.perturbation_generator_version
              ? ` · ${result.perturbation_generator_version}`
              : ""}{" "}
            · 输入 SHA-256
          </span>
          <code>{result.input_sha256}</code>
        </div>
        <button className="button secondary" onClick={download}>
          下载分析 JSON
        </button>
      </div>
    </div>
  );
}
