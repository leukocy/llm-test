import type { ReportEnvironment } from "../api";

const fields = [
  ["processor", "Processor · 处理器", "例如 AMD EPYC 9654"],
  ["mainboard", "Mainboard · 主板", "例如服务器型号 / 主板型号"],
  ["memory", "Memory · 内存", "例如 512 GB DDR5"],
  ["gpu", "GPU · 加速器", "例如 2 × NVIDIA H100 80 GB"],
  ["system", "System · 系统", "例如 Ubuntu 24.04 / CUDA 12.8"],
  ["engine_name", "Engine · 推理引擎", "例如 vLLM 版本、精度与并行配置"],
] as const;

const scopes = {
  model_server: "受测模型服务器",
  test_client: "测试客户端",
  unspecified: "未明确对象",
};

export function ReportEnvironmentEditor({
  runConfig,
  onChange,
  idPrefix = "report-env",
  disabled = false,
}: {
  runConfig: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
  idPrefix?: string;
  disabled?: boolean;
}) {
  const raw = runConfig.report_environment;
  const value =
    raw && typeof raw === "object" && !Array.isArray(raw)
      ? (raw as Record<string, unknown>)
      : {};
  const filled = fields.filter(([key]) =>
    String(value[key] || "").trim(),
  ).length;
  function update(key: string, text: string) {
    const next = {
      ...value,
      scope: value.scope || "model_server",
      [key]: text,
    };
    onChange({ ...runConfig, report_environment: next });
  }
  return (
    <details className="knobs-section report-environment-editor">
      <summary>
        报告环境信息（六项手填）{filled ? ` · 已填 ${filled} 项` : " · 可选"}
      </summary>
      <p className="field-help">
        填写本次测试的硬件与软件条件，可随方案保存。内容标记为用户填写，自动采集的执行端硬件快照独立保留。
      </p>
      <fieldset disabled={disabled}>
        <label className="input-label" htmlFor={`${idPrefix}-scope`}>
          信息对应的环境
        </label>
        <select
          id={`${idPrefix}-scope`}
          value={String(value.scope || "model_server")}
          onChange={(event) => update("scope", event.target.value)}
        >
          {Object.entries(scopes).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
        <div className="report-environment-fields">
          {fields.map(([key, label, placeholder]) => (
            <div key={key}>
              <label className="input-label" htmlFor={`${idPrefix}-${key}`}>
                {label}
              </label>
              <input
                id={`${idPrefix}-${key}`}
                value={String(value[key] || "")}
                maxLength={240}
                placeholder={placeholder}
                onChange={(event) => update(key, event.target.value)}
              />
            </div>
          ))}
        </div>
        <button
          type="button"
          className="text-action"
          onClick={() =>
            onChange({ ...runConfig, report_environment: undefined })
          }
        >
          清空环境信息
        </button>
      </fieldset>
    </details>
  );
}

export function ReportEnvironmentCard({
  environment,
}: {
  environment?: ReportEnvironment | null;
}) {
  if (!environment || !Object.keys(environment.fields).length) return null;
  return (
    <section className="surface report-environment-card">
      <div className="section-head">
        <h2>报告环境信息</h2>
        <span className="minor-tag">用户填写</span>
      </div>
      <p className="field-help">
        对象：{scopes[environment.scope]}
        。未经自动核验；自动硬件快照来自执行端，与本表分别记录。
      </p>
      <dl className="env-kv">
        {fields.map(([key, label]) =>
          environment.fields[key] ? (
            <div key={key}>
              <dt>{label}</dt>
              <dd>{environment.fields[key]}</dd>
            </div>
          ) : null,
        )}
      </dl>
    </section>
  );
}
