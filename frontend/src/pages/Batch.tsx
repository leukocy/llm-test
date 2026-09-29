import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Endpoint, type Job, type MeasurementPlan } from "../api";
import { scenarios, type JobType } from "../constants";
import { SchemaForm, type JsonSchema } from "../components/SchemaForm";
import { ReportEnvironmentEditor } from "../components/ReportEnvironment";

type Item = {
  enabled: boolean;
  test_type: JobType;
  endpoint_id: string;
  parameters: Record<string, unknown>;
  raw: string;
  mode: "form" | "json";
  run_config: Record<string, unknown>;
};

type PlanState = { plan?: MeasurementPlan; error?: string };

function makeItem(
  testType: JobType,
  parameters?: Record<string, unknown>,
): Item {
  const chosen =
    parameters || scenarios.find((item) => item.id === testType)!.parameters;
  return {
    enabled: true,
    test_type: testType,
    endpoint_id: "",
    parameters: { ...chosen },
    raw: JSON.stringify(chosen, null, 2),
    mode: "form",
    run_config: {},
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
  const planKey = JSON.stringify({ endpoint, items });
  const [planState, setPlanState] = useState<{
    key: string;
    items: PlanState[];
  }>({ key: "", items: [] });
  const plans = planState.key === planKey ? planState.items : [];
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [configName, setConfigName] = useState("");
  const [description, setDescription] = useState("");
  const [maxParallel, setMaxParallel] = useState(1);
  const [stopOnError, setStopOnError] = useState(false);
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
    setPlanState({ key: planKey, items: [] });
    if (!endpoint) return;
    const timer = window.setTimeout(() => {
      void Promise.all(
        items.map(async (item): Promise<PlanState> => {
          if (!item.enabled) return {};
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
                  run_config: item.run_config,
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
        if (alive) setPlanState({ key: planKey, items: result });
      });
    }, 350);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [endpoint, items, planKey, token]);

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
          description: description.trim(),
          max_parallel: maxParallel,
          stop_on_error: stopOnError,
          endpoint_id: endpoint,
          items: items.map((item) => ({
            enabled: item.enabled,
            test_type: item.test_type,
            endpoint_id: item.endpoint_id || undefined,
            parameters: parametersFor(item),
            run_config: item.run_config,
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
      if (
        config.max_parallel != null &&
        (typeof config.max_parallel !== "number" ||
          !Number.isInteger(config.max_parallel) ||
          config.max_parallel < 1 ||
          config.max_parallel > 8)
      ) {
        throw new Error("并行上限必须是 1~8 的整数");
      }
      if (
        config.stop_on_error != null &&
        typeof config.stop_on_error !== "boolean"
      ) {
        throw new Error("失败即停必须为布尔值");
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
        if (row.enabled != null && typeof row.enabled !== "boolean") {
          throw new Error(`第 ${index + 1} 项启用状态必须为布尔值`);
        }
        if (
          row.run_config != null &&
          (typeof row.run_config !== "object" || Array.isArray(row.run_config))
        ) {
          throw new Error(`第 ${index + 1} 项运行设置必须是 JSON 对象`);
        }
        return {
          ...makeItem(
            row.test_type as JobType,
            row.parameters as Record<string, unknown>,
          ),
          endpoint_id: String(row.endpoint_id || ""),
          enabled: row.enabled !== false,
          run_config:
            (row.run_config as Record<string, unknown> | undefined) || {},
        };
      });
      setEndpoint(config.endpoint_id);
      setItems(imported);
      setConfigName(
        typeof config.name === "string" ? config.name.slice(0, 80) : "",
      );
      setDescription(
        typeof config.description === "string"
          ? config.description.slice(0, 500)
          : "",
      );
      setMaxParallel(
        typeof config.max_parallel === "number" ? config.max_parallel : 1,
      );
      setStopOnError(config.stop_on_error === true);
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
        if (!item.enabled) {
          return {
            enabled: false,
            run_config: item.run_config,
            test_type: item.test_type,
            parameters: item.parameters,
            ...(item.endpoint_id ? { endpoint_id: item.endpoint_id } : {}),
          };
        }
        try {
          return {
            enabled: true,
            run_config: item.run_config,
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
          body: JSON.stringify({
            name: configName.trim() || "批量测量",
            description: description.trim(),
            max_parallel: maxParallel,
            stop_on_error: stopOnError,
            endpoint_id: endpoint,
            items: parsed,
          }),
        },
      );
      onSubmitted(result.batch_id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "批量提交失败");
    } finally {
      setBusy(false);
    }
  }

  const enabledCount = items.filter((item) => item.enabled).length;
  const validPlans =
    enabledCount > 0 &&
    plans.length === items.length &&
    items.every((item, index) => !item.enabled || plans[index]?.plan);
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
          <p>逐项选择端点和参数，提交前核对样本预算与执行策略。</p>
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
          <h2>批次信息与配置文件</h2>
        </div>
        <div className="batch-config-actions">
          <input
            aria-label="批量方案名称"
            value={configName}
            maxLength={80}
            placeholder="批次名称"
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
        <textarea
          aria-label="批次说明"
          value={description}
          maxLength={500}
          placeholder="测量目的、运行条件或备注（可选）"
          onChange={(event) => setDescription(event.target.value)}
        />
        <div className="batch-policy">
          <label>
            <input
              type="checkbox"
              checked={maxParallel > 1}
              onChange={(event) => setMaxParallel(event.target.checked ? 2 : 1)}
            />
            并行执行
          </label>
          <label>
            同时执行上限
            <input
              type="number"
              min={2}
              max={8}
              value={maxParallel}
              disabled={maxParallel === 1}
              onChange={(event) =>
                setMaxParallel(
                  Math.min(
                    8,
                    Math.max(2, Math.trunc(Number(event.target.value)) || 2),
                  ),
                )
              }
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={stopOnError}
              onChange={(event) => setStopOnError(event.target.checked)}
            />
            失败即停
          </label>
        </div>
        <p className="batch-plan">
          {maxParallel > 1
            ? "并行任务共享受测服务容量，实际并行数受可用执行资源限制；结果会记录此条件。"
            : "串行模式会等待前一项完成，即使有多个执行资源也保持顺序。"}
          {stopOnError &&
            " 任务执行失败或正式性能请求出现错误时，取消尚未执行的同批次任务，并通知正在执行的任务停止。答题错误不触发此策略。"}
        </p>
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
              <label>
                <input
                  type="checkbox"
                  aria-label={`启用子任务 ${index + 1}`}
                  checked={item.enabled}
                  onChange={(event) =>
                    updateItem(index, (current) => ({
                      ...current,
                      enabled: event.target.checked,
                    }))
                  }
                />
                启用
              </label>
              <select
                aria-label={`子任务 ${index + 1} 类型`}
                value={item.test_type}
                onChange={(event) =>
                  updateItem(index, (current) => ({
                    ...makeItem(event.target.value as JobType),
                    endpoint_id: current.endpoint_id,
                    enabled: current.enabled,
                    run_config: current.run_config,
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
            <ReportEnvironmentEditor
              runConfig={item.run_config}
              onChange={(run_config) =>
                updateItem(index, (current) => ({ ...current, run_config }))
              }
              idPrefix={`batch-env-${index}`}
              disabled={!item.enabled}
            />
            {item.enabled && plans[index]?.error && (
              <p className="form-error" role="alert">
                {plans[index].error}
              </p>
            )}
            {item.enabled && plans[index]?.plan && (
              <p className="batch-plan">
                计划 {plans[index].plan!.measured_requests} 次正式请求 +{" "}
                {plans[index].plan!.warmup_requests} 次预热
              </p>
            )}
          </div>
        ))}
        <div className="form-footer">
          <span>
            总计 {validPlans ? totalRequests : "校验中"} 次请求；已启用{" "}
            {enabledCount} / {items.length} 项，
            {maxParallel === 1 ? "串行执行" : `最多 ${maxParallel} 项并行`}。
          </span>
          <button
            className="button primary"
            disabled={busy || !endpoint || !validPlans}
            onClick={() => void submit()}
          >
            {busy ? "提交中…" : `提交 ${enabledCount} 个子任务 →`}
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
