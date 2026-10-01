import { useEffect, useRef, useState } from "react";
import { api, type Endpoint, type Job } from "../api";
import { activeStates, scenarios, statusLabels } from "../constants";
import { templates, originalModels } from "../endpointTemplates";
import { QualityParameters } from "./QualityParameters";
import type { JsonSchema } from "./SchemaForm";

export function OnlineComparison({
  token,
  onReady,
}: {
  token: string;
  onReady: (a: string, b: string) => void;
}) {
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [parameters, setParameters] = useState<Record<string, unknown>>(() => ({
    ...scenarios.find((item) => item.id === "quality")!.parameters,
    datasets: [],
    max_samples: 50,
    num_shots: 0,
  }));
  const [schema, setSchema] = useState<JsonSchema | null>(null);
  const [batch, setBatch] = useState(
    () => sessionStorage.getItem("llm-test-online-comparison") || "",
  );
  const [jobs, setJobs] = useState<Job[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [direct, setDirect] = useState({
    provider: "OpenAI",
    api_base_url: "https://api.openai.com/v1",
    model_id: "",
    api_key: "",
  });
  const submitted = useRef<string | null>(null);
  const delivered = useRef("");
  const restored = useRef("");
  const ready = useRef(onReady);
  ready.current = onReady;
  useEffect(() => {
    submitted.current = null;
  }, [a, b, parameters]);
  async function refreshEndpoints() {
    const result = await api<{ items: Endpoint[] }>(token, "/api/v1/endpoints");
    setEndpoints(result.items);
    setA((current) => current || result.items[0]?.id || "");
  }
  useEffect(() => {
    let alive = true;
    void refreshEndpoints().catch((exc) => {
      if (alive) setError(String(exc));
    });
    void api<{ items: Record<string, { schema: JsonSchema }> }>(
      token,
      "/api/v1/specs",
    )
      .then((result) => {
        if (alive) setSchema(result.items.quality.schema);
      })
      .catch((exc) => {
        if (alive) setError(String(exc));
      });
    return () => {
      alive = false;
    };
  }, [token]);
  useEffect(() => {
    if (!batch) return;
    const controller = new AbortController();
    async function poll() {
      try {
        const result = await api<{ items: Job[] }>(
          token,
          `/api/v1/jobs?parent_job_id=${encodeURIComponent(batch)}&limit=10`,
          { signal: controller.signal },
        );
        if (controller.signal.aborted) return;
        setJobs(result.items);
        const first = result.items.find(
          (item) =>
            (item.parameters._comparison as { role: string } | undefined)
              ?.role === "A",
        );
        const second = result.items.find(
          (item) =>
            (item.parameters._comparison as { role: string } | undefined)
              ?.role === "B",
        );
        if (first && second && restored.current !== batch) {
          restored.current = batch;
          setA(first.endpoint_id);
          setB(second.endpoint_id);
          setParameters(
            Object.fromEntries(
              Object.entries(first.parameters).filter(
                ([name]) => !name.startsWith("_"),
              ),
            ),
          );
        }
        if (
          first?.status === "completed" &&
          second?.status === "completed" &&
          delivered.current !== batch
        ) {
          delivered.current = batch;
          ready.current(first.job_id, second.job_id);
        }
      } catch (exc) {
        if (!controller.signal.aborted) setError(String(exc));
      }
    }
    void poll();
    const timer = window.setInterval(() => void poll(), 3000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [batch, token]);
  async function saveB() {
    setBusy(true);
    setError("");
    try {
      const result = await api<Endpoint>(token, "/api/v1/endpoints", {
        method: "POST",
        body: JSON.stringify({
          ...direct,
          label: `B · ${direct.model_id}`.slice(0, 80),
          tokenizer_option: "auto",
        }),
      });
      setDirect((current) => ({ ...current, api_key: "" }));
      await refreshEndpoints();
      setB(result.id);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }
  async function run() {
    setBusy(true);
    setError("");
    submitted.current ||= crypto.randomUUID();
    try {
      const result = await api<{ batch_id: string; jobs: Job[] }>(
        token,
        "/api/v1/comparisons",
        {
          method: "POST",
          headers: { "Idempotency-Key": submitted.current },
          body: JSON.stringify({
            endpoint_id_a: a,
            endpoint_id_b: b,
            parameters,
          }),
        },
      );
      setBatch(result.batch_id);
      setJobs(result.jobs);
      sessionStorage.setItem("llm-test-online-comparison", result.batch_id);
      submitted.current = null;
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }
  async function cancel() {
    try {
      await api(token, `/api/v1/jobs/batch/${batch}/cancel`, {
        method: "POST",
      });
    } catch (exc) {
      setError(String(exc));
    }
  }
  const active = jobs.some((job) => activeStates.has(job.status));
  return (
    <section className="surface online-comparison">
      <div className="section-head">
        <div>
          <span className="eyebrow">SHARED SAMPLE PLAN</span>
          <h2>在线双模型对比</h2>
        </div>
      </div>
      <p className="field-help">
        先冻结全部题目和 few-shot，再顺序执行 A、B。B
        复用原计划；本地数据不足请先准备，不会在评分过程中自动下载。两侧完成后自动生成配对分析，关闭页面不停止任务。
      </p>
      <div className="compare-picker">
        {[
          ["A", a, setA],
          ["B", b, setB],
        ].map(([role, value, setValue]) => (
          <label key={String(role)}>
            模型 {String(role)} 端点
            <select
              aria-label={`在线模型 ${role}`}
              value={value as string}
              onChange={(event) =>
                (setValue as (value: string) => void)(event.target.value)
              }
            >
              <option value="">选择已配置端点…</option>
              {endpoints.map((endpoint) => (
                <option key={endpoint.id} value={endpoint.id}>
                  {endpoint.label} · {endpoint.model_id}
                  {!endpoint.credential_configured && "（缺少凭证）"}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <details className="knobs-section">
        <summary>直接设置模型 B 的 API</summary>
        <div className="schema-form">
          <label className="schema-field">
            服务商预设
            <select
              aria-label="模型 B 服务商"
              defaultValue=""
              onChange={(event) => {
                const item = templates.find(
                  (item) => item.label === event.target.value,
                );
                if (item)
                  setDirect((current) => ({
                    ...current,
                    provider: item.provider,
                    api_base_url: item.url,
                  }));
              }}
            >
              <option value="">自定义</option>
              {templates.map((item) => (
                <option key={item.label}>{item.label}</option>
              ))}
            </select>
          </label>
          <label className="schema-field">
            接口协议
            <select
              aria-label="模型 B 协议"
              value={direct.provider}
              onChange={(event) =>
                setDirect((current) => ({
                  ...current,
                  provider: event.target.value,
                }))
              }
            >
              <option value="OpenAI">OpenAI 兼容</option>
              <option value="Gemini">Gemini 原生</option>
            </select>
          </label>
          <label className="schema-field">
            Base URL
            <input
              aria-label="模型 B Base URL"
              value={direct.api_base_url}
              onChange={(event) =>
                setDirect((current) => ({
                  ...current,
                  api_base_url: event.target.value,
                }))
              }
            />
          </label>
          <label className="schema-field">
            Model ID
            <input
              aria-label="模型 B Model ID"
              list="comparison-models"
              value={direct.model_id}
              onChange={(event) =>
                setDirect((current) => ({
                  ...current,
                  model_id: event.target.value,
                }))
              }
            />
            <datalist id="comparison-models">
              {originalModels.map((model) => (
                <option key={model}>{model}</option>
              ))}
            </datalist>
          </label>
          <label className="schema-field">
            API key
            <input
              aria-label="模型 B API key"
              type="password"
              autoComplete="off"
              value={direct.api_key}
              onChange={(event) =>
                setDirect((current) => ({
                  ...current,
                  api_key: event.target.value,
                }))
              }
            />
          </label>
          <button
            className="button subtle"
            disabled={busy || !direct.model_id || !direct.api_key}
            onClick={() => void saveB()}
          >
            保存并选择模型 B
          </button>
        </div>
      </details>
      <details className="knobs-section">
        <summary>共同质量参数</summary>
        {schema && (
          <QualityParameters
            token={token}
            schema={schema}
            value={parameters}
            onChange={setParameters}
          />
        )}
      </details>
      {error && (
        <div role="alert" className="alert">
          {error}
        </div>
      )}
      <div className="compare-actions">
        <button
          className="button primary"
          disabled={
            busy ||
            active ||
            !schema ||
            !Array.isArray(parameters.datasets) ||
            !parameters.datasets.length ||
            !a ||
            !b ||
            a === b ||
            !endpoints.find((item) => item.id === a)?.credential_configured ||
            !endpoints.find((item) => item.id === b)?.credential_configured
          }
          onClick={() => void run()}
        >
          {busy ? "提交中…" : "运行在线 A/B →"}
        </button>
        {active && (
          <button
            className="button subtle danger"
            onClick={() => void cancel()}
          >
            取消整组对比
          </button>
        )}
      </div>
      {batch && (
        <>
          <p className="chart-caption">
            对比组 <a href={`/runs?batch=${batch}`}>{batch}</a> ·
            顺序执行，不代表并发公平性或连续性能。准备后以实际样本数量为准；两侧恢复分别遵循原检查点。
          </p>
          <div className="quality-grid">
            {jobs.map((job) => (
              <div className="quality-card" key={job.job_id}>
                <h3>
                  模型{" "}
                  {String(
                    (job.parameters._comparison as { role: string } | undefined)
                      ?.role || "",
                  )}{" "}
                  · {job.model_id}
                </h3>
                <p>
                  {statusLabels[job.status] || job.status} ·{" "}
                  {job.progress_completed} / {job.progress_total || "待准备"}{" "}
                  样本
                </p>
                <a href={`/runs/${job.job_id}`}>查看报告与恢复</a>
                {job.error_message && <p>{job.error_message}</p>}
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
