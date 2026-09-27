import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Endpoint, type Job, type MeasurementPlan } from "../api";
import { scenarios, type JobType } from "../constants";
import { SchemaForm, type JsonSchema } from "../components/SchemaForm";

type Item = {
  test_type: JobType;
  endpoint_id: string;
  parameters: Record<string, unknown>;
  raw: string;
  mode: "form" | "json";
};

type PlanState = { plan?: MeasurementPlan; error?: string };

function makeItem(
  testType: JobType,
  parameters?: Record<string, unknown>,
): Item {
  const chosen =
    parameters || scenarios.find((item) => item.id === testType)!.parameters;
  return {
    test_type: testType,
    endpoint_id: "",
    parameters: { ...chosen },
    raw: JSON.stringify(chosen, null, 2),
    mode: "form",
  };
}

function parametersFor(item: Item): Record<string, unknown> {
  if (item.mode === "form") return item.parameters;
  const value: unknown = JSON.parse(item.raw);
  if (!value || Array.isArray(value) || typeof value !== "object") {
    throw new Error("参数必须是 JSON 对象");
  }
  return value as Record<string, unknown>;
}

export function Batch({
  endpoints,
  token,
  onSubmitted,
}: {
  endpoints: Endpoint[];
  token: string;
  onSubmitted: (batchId: string) => void;
}) {
  const navigate = useNavigate();
  const [endpoint, setEndpoint] = useState(endpoints[0]?.id || "");
  const [items, setItems] = useState<Item[]>([makeItem("concurrency")]);
  const [schemas, setSchemas] = useState<Record<string, JsonSchema>>({});
  const [plans, setPlans] = useState<PlanState[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [configName, setConfigName] = useState("");
  const [configNotice, setConfigNotice] = useState("");

  useEffect(() => {
    if (!endpoints.some((item) => item.id === endpoint)) {
      setEndpoint(endpoints[0]?.id || "");
    }
  }, [endpoints, endpoint]);

  useEffect(() => {
    let alive = true;
    api<{ items: Record<string, { schema: JsonSchema }> }>(
      token,
      "/api/v1/specs",
    )
      .then((data) => {
        if (alive) {
          setSchemas(
            Object.fromEntries(
              Object.entries(data.items).map(([name, value]) => [
                name,
                value.schema,
              ]),
            ),
          );
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [token]);

  useEffect(() => {
    let alive = true;
    setPlans([]);
    if (!endpoint) return;
    const timer = window.setTimeout(() => {
      void Promise.all(
        items.map(async (item): Promise<PlanState> => {
          try {
            const plan = await api<MeasurementPlan>(
              token,
              "/api/v1/jobs/plan",
              {
                method: "POST",
                body: JSON.stringify({
                  endpoint_id: item.endpoint_id || endpoint,
                  test_type: item.test_type,
                  parameters: parametersFor(item),
                }),
              },
            );
            return { plan };
          } catch (exc) {
            return {
              error: exc instanceof Error ? exc.message : "工作负载校验失败",
            };
          }
        }),
      ).then((result) => {
        if (alive) setPlans(result);
      });
    }, 350);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [endpoint, items, token]);

  function updateItem(index: number, update: (item: Item) => Item) {
    setItems((current) =>
      current.map((item, position) =>
        position === index ? update(item) : item,
      ),
    );
    setError("");
  }

  function loadAllTests() {
    setItems([
      makeItem("concurrency", {
        selected_concurrencies: [1, 2],
        rounds_per_level: 1,
        input_tokens_target: 64,
        max_tokens: 512,
        warmup_rounds_per_level: 0,
      }),
      makeItem("prefill", {
        token_levels: [4096, 8192],
        requests_per_level: 1,
        max_tokens: 1,
        warmup_requests_per_level: 0,
      }),
      makeItem("long_context", {
        context_lengths: [4096, 8192, 16384, 32768, 65536],
        rounds_per_level: 1,
        max_tokens: 512,
      }),
    ]);
    setError("");
  }

  function exportConfiguration() {
    setError("");
    try {
      const content = JSON.stringify(
        {
          schema_version: 1,
          name: configName.trim(),
          endpoint_id: endpoint,
          items: items.map((item) => ({
            test_type: item.test_type,
            endpoint_id: item.endpoint_id || undefined,
            parameters: parametersFor(item),
          })),
        },
        null,
        2,
      );
      const url = URL.createObjectURL(
        new Blob([content], { type: "application/json" }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = "llm-test-batch.json";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "配置导出失败");
    }
  }

  async function importConfiguration(file: File | undefined) {
    if (!file) return;
    setError("");
    setConfigNotice("");
    try {
      if (file.size > 1_000_000) throw new Error("配置文件不能超过 1 MB");
      const value: unknown = JSON.parse(await file.text());
      if (!value || Array.isArray(value) || typeof value !== "object") {
        throw new Error("配置必须是 JSON 对象");
      }
      const config = value as Record<string, unknown>;
      if (
        config.schema_version !== 1 ||
        !Array.isArray(config.items) ||
        config.items.length < 1 ||
        config.items.length > 10
      ) {
        throw new Error("只支持 1~10 个子任务的 v1 批量配置");
      }
      if (
        typeof config.endpoint_id !== "string" ||
        !endpoints.some((target) => target.id === config.endpoint_id)
      ) {
        throw new Error("配置中的默认端点不存在，请先添加受测 API");
      }
      const imported = config.items.map((entry, index) => {
        if (!entry || Array.isArray(entry) || typeof entry !== "object") {
          throw new Error(`第 ${index + 1} 项格式无效`);
        }
        const row = entry as Record<string, unknown>;
        if (
          !scenarios.some((scenario) => scenario.id === row.test_type) ||
          !row.parameters ||
          Array.isArray(row.parameters) ||
          typeof row.parameters !== "object"
        ) {
          throw new Error(`第 ${index + 1} 项类型或参数无效`);
        }
        if (
          row.endpoint_id != null &&
          (typeof row.endpoint_id !== "string" ||
            !endpoints.some((target) => target.id === row.endpoint_id))
        ) {
          throw new Error(`第 ${index + 1} 项端点不存在，请先添加受测 API`);
        }
        return {
          ...makeItem(
            row.test_type as JobType,
            row.parameters as Record<string, unknown>,
          ),
          endpoint_id: String(row.endpoint_id || ""),
        };
      });
      setEndpoint(config.endpoint_id);
      setItems(imported);
      setConfigName(
        typeof config.name === "string" ? config.name.slice(0, 80) : "",
      );
      setConfigNotice("已导入配置；请核对每项工作负载计划后再提交。");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "配置导入失败");
    }
  }

  async function submit() {
    setError("");
    setBusy(true);
    try {
      const parsed = items.map((item, index) => {
        try {
          return {
            test_type: item.test_type,
            parameters: parametersFor(item),
            ...(item.endpoint_id ? { endpoint_id: item.endpoint_id } : {}),
          };
        } catch {
          throw new Error(`第 ${index + 1} 项参数不是合法 JSON 对象`);
        }
      });
      const result = await api<{ batch_id: string; items: Job[] }>(
        token,
        "/api/v1/jobs/batch",
        {
          method: "POST",
          headers: { "Idempotency-Key": crypto.randomUUID() },
          body: JSON.stringify({ endpoint_id: endpoint, items: parsed }),
        },
      );
      onSubmitted(result.batch_id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "批量提交失败");
    } finally {
      setBusy(false);
    }
  }

  const validPlans =
    plans.length === items.length && plans.every((item) => item.plan);
  const totalRequests = plans.reduce(
    (sum, item) => sum + (item.plan?.total_requests || 0),
    0,
  );

  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">BATCH MEASUREMENT</span>
          <h1>批量测量</h1>
          <p>逐项选择端点和参数，提交前核对样本预算。worker 按顺序执行。</p>
        </div>
      </div>
      <section className="surface batch-all-tests">
        <div className="section-head">
          <div>
            <span className="eyebrow">ORIGINAL ALL TESTS</span>
            <h2>一键配置基础三项</h2>
          </div>
          <button className="button primary" onClick={loadAllTests}>
            载入并发 + Prefill + 长上下文
          </button>
        </div>
        <p>
          对应初版 All Tests 的三阶段顺序。Prefill 默认只生成 1 token
          以隔离输入处理；上下文长度遵守当前 131,072 token 上限，可在下方调整。
        </p>
      </section>
      <section className="surface">
        <div className="section-head">
          <div>
            <span className="section-index">01</span>
            <h2>默认目标端点</h2>
          </div>
        </div>
        <select
          aria-label="批量默认端点"
          value={endpoint}
          disabled={!endpoints.length}
          onChange={(event) => setEndpoint(event.target.value)}
        >
          {!endpoints.length && <option value="">尚未配置受测 API</option>}
          {endpoints.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label} · {item.model_id}
            </option>
          ))}
        </select>
        <button
          className="text-action"
          onClick={() => navigate("/settings/api")}
        >
          配置受测 API →
        </button>
      </section>
      <section className="surface batch-config-tools">
        <div className="section-head">
          <h2>批量配置文件</h2>
        </div>
        <div className="batch-config-actions">
          <input
            aria-label="批量方案名称"
            value={configName}
            maxLength={80}
            placeholder="方案名称（用于配置文件）"
            onChange={(event) => setConfigName(event.target.value)}
          />
          <button className="button subtle" onClick={exportConfiguration}>
            保存为 JSON ↓
          </button>
          <label className="button subtle" htmlFor="batch-config-file">
            导入 JSON ↑
          </label>
          <input
            id="batch-config-file"
            type="file"
            accept=".json,application/json"
            onChange={(event) => {
              void importConfiguration(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
        </div>
        {configNotice && (
          <p className="form-success" role="status">
            {configNotice}
          </p>
        )}
      </section>
      <section className="surface">
        <div className="section-head">
          <div>
            <span className="section-index">02</span>
            <h2>子任务（{items.length} / 10）</h2>
          </div>
          <button
            className="button subtle"
            onClick={() =>
              setItems((current) => [...current, makeItem("concurrency")])
            }
            disabled={items.length >= 10}
          >
            ＋ 添加子任务
          </button>
        </div>
        {items.map((item, index) => (
          <div className="batch-item" key={index}>
            <div className="batch-item-head">
              <strong>#{index + 1}</strong>
              <select
                aria-label={`子任务 ${index + 1} 类型`}
                value={item.test_type}
                onChange={(event) =>
                  updateItem(index, (current) => ({
                    ...makeItem(event.target.value as JobType),
                    endpoint_id: current.endpoint_id,
                  }))
                }
              >
                {scenarios.map((scenario) => (
                  <option key={scenario.id} value={scenario.id}>
                    {scenario.label}
                  </option>
                ))}
              </select>
              {items.length > 1 && (
                <button
                  className="text-button danger"
                  onClick={() =>
                    setItems((current) => current.filter((_, i) => i !== index))
                  }
                >
                  移除
                </button>
              )}
            </div>
            <label className="input-label" htmlFor={`batch-target-${index}`}>
              此项受测 API / 模型
            </label>
            <select
              id={`batch-target-${index}`}
              value={item.endpoint_id}
              onChange={(event) =>
                updateItem(index, (current) => ({
                  ...current,
                  endpoint_id: event.target.value,
                }))
              }
            >
              <option value="">使用批量默认端点</option>
              {endpoints.map((target) => (
                <option key={target.id} value={target.id}>
                  {target.label} · {target.model_id}
                </option>
              ))}
            </select>
            <div
              className="mode-toggle"
              role="tablist"
              aria-label={`子任务 ${index + 1} 编辑方式`}
            >
              <button
                role="tab"
                aria-selected={item.mode === "form"}
                className={item.mode === "form" ? "active" : ""}
                onClick={() => {
                  try {
                    const parameters = parametersFor(item);
                    updateItem(index, (current) => ({
                      ...current,
                      parameters,
                      mode: "form",
                    }));
                  } catch {
                    setError(`第 ${index + 1} 项参数不是合法 JSON 对象`);
                  }
                }}
              >
                表单
              </button>
              <button
                role="tab"
                aria-selected={item.mode === "json"}
                className={item.mode === "json" ? "active" : ""}
                onClick={() =>
                  updateItem(index, (current) => ({
                    ...current,
                    raw: JSON.stringify(current.parameters, null, 2),
                    mode: "json",
                  }))
                }
              >
                JSON
              </button>
            </div>
            {item.mode === "form" &&
            schemas[item.test_type] &&
            !["dataset", "quality", "robustness"].includes(item.test_type) ? (
              <SchemaForm
                schema={schemas[item.test_type]}
                value={item.parameters}
                onChange={(parameters) =>
                  updateItem(index, (current) => ({ ...current, parameters }))
                }
                idPrefix={`batch-${index}`}
              />
            ) : (
              <textarea
                className="json-editor batch-item-editor"
                spellCheck={false}
                aria-label={`子任务 ${index + 1} 参数`}
                value={item.raw}
                onChange={(event) =>
                  updateItem(index, (current) => ({
                    ...current,
                    raw: event.target.value,
                    mode: "json",
                  }))
                }
              />
            )}
            {plans[index]?.error && (
              <p className="form-error" role="alert">
                {plans[index].error}
              </p>
            )}
            {plans[index]?.plan && (
              <p className="batch-plan">
                计划 {plans[index].plan!.measured_requests} 次正式请求 +{" "}
                {plans[index].plan!.warmup_requests} 次预热
              </p>
            )}
          </div>
        ))}
        <div className="form-footer">
          <span>
            总计 {validPlans ? totalRequests : "校验中"}{" "}
            次请求；按子任务顺序排队。
          </span>
          <button
            className="button primary"
            disabled={busy || !endpoint || !validPlans}
            onClick={() => void submit()}
          >
            {busy ? "提交中…" : `提交 ${items.length} 个子任务 →`}
          </button>
        </div>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
      </section>
    </div>
  );
}
