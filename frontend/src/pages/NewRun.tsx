import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, type Endpoint, type MeasurementPlan, type Preset } from "../api";
import {
  profileParameters,
  scenarios,
  type JobType,
  type MeasurementProfile,
} from "../constants";
import {
  SchemaForm,
  schemaDefaults,
  type JsonSchema,
} from "../components/SchemaForm";
import { TokenizerTools } from "../components/TokenizerTools";

type SpecCatalog = {
  items: Record<string, { label: string; schema: JsonSchema }>;
  run_config_schema: JsonSchema;
};

export function NewRun({
  endpoints,
  token,
  onSubmit,
  busy,
}: {
  endpoints: Endpoint[];
  token: string;
  onSubmit: (
    endpoint: string,
    type: JobType,
    params: Record<string, unknown>,
    runConfig?: Record<string, unknown>,
  ) => Promise<void>;
  busy: boolean;
}) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const requestedEndpoint = searchParams.get("endpoint");
  const requestedOffset = Number(searchParams.get("latency_offset"));
  const initialOffset =
    searchParams.has("latency_offset") &&
    Number.isFinite(requestedOffset) &&
    requestedOffset >= 0 &&
    requestedOffset <= 5
      ? requestedOffset
      : null;
  const [endpoint, setEndpoint] = useState(
    requestedEndpoint || endpoints[0]?.id || "",
  );
  const [type, setType] = useState<JobType>("concurrency");
  const [profile, setProfile] = useState<MeasurementProfile>("standard");
  const [params, setParams] = useState<Record<string, unknown>>(
    () => scenarios[0].parameters,
  );
  const [runConfig, setRunConfig] = useState<Record<string, unknown>>(
    initialOffset == null ? {} : { latency_offset: initialOffset },
  );
  const [textFile, setTextFile] = useState("");
  const [specs, setSpecs] = useState<SpecCatalog | null>(null);
  const [mode, setMode] = useState<"form" | "json">("form");
  const [raw, setRaw] = useState(
    JSON.stringify(scenarios[0].parameters, null, 2),
  );
  const [error, setError] = useState("");
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetId, setPresetId] = useState("");
  const [presetName, setPresetName] = useState("");
  const [presetError, setPresetError] = useState("");
  const [presetBusy, setPresetBusy] = useState(false);
  const [plan, setPlan] = useState<MeasurementPlan | null>(null);
  const [planError, setPlanError] = useState("");
  const specSchema = specs?.items[type]?.schema;
  const selectedEndpoint = endpoints.find((item) => item.id === endpoint);
  const performanceRun = type !== "quality" && type !== "robustness";

  useEffect(() => {
    setEndpoint((current) => {
      if (
        requestedEndpoint &&
        endpoints.some((item) => item.id === requestedEndpoint)
      ) {
        return requestedEndpoint;
      }
      return endpoints.some((item) => item.id === current)
        ? current
        : endpoints[0]?.id || "";
    });
  }, [endpoints, requestedEndpoint]);

  useEffect(() => {
    let active = true;
    api<SpecCatalog>(token, "/api/v1/specs")
      .then((data) => {
        if (active) setSpecs(data);
      })
      .catch(() => {
        /* 无 schema 时退回纯 JSON 模式 */
      });
    api<{ items: Preset[] }>(token, "/api/v1/presets")
      .then((data) => {
        if (active) setPresets(data.items);
      })
      .catch((exc) => {
        if (active)
          setPresetError(exc instanceof Error ? exc.message : "无法读取方案");
      });
    return () => {
      active = false;
    };
  }, [token]);

  useEffect(() => {
    if (specs && specs.run_config_schema) {
      setRunConfig((current) => ({
        ...schemaDefaults(specs.run_config_schema),
        ...current,
      }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [specs]);

  const knobDefaults = useMemo(
    () => (specs ? schemaDefaults(specs.run_config_schema) : {}),
    [specs],
  );
  const commonKnobSchema = useMemo(() => {
    if (!specs) return null;
    const properties = Object.fromEntries(
      Object.entries(specs.run_config_schema.properties || {}).filter(
        ([name]) =>
          [
            "temperature",
            "thinking_enabled",
            "reasoning_effort",
            "random_seed",
          ].includes(name),
      ),
    );
    return { ...specs.run_config_schema, properties };
  }, [specs]);
  const advancedKnobSchema = useMemo(() => {
    if (!specs) return null;
    const properties = Object.fromEntries(
      Object.entries(specs.run_config_schema.properties || {}).filter(
        ([name]) =>
          ![
            "temperature",
            "thinking_enabled",
            "reasoning_effort",
            "random_seed",
          ].includes(name),
      ),
    );
    return { ...specs.run_config_schema, properties };
  }, [specs]);

  function applyProfile(next: Exclude<MeasurementProfile, "custom">) {
    const value = profileParameters(type, next);
    setParams(value);
    setRaw(JSON.stringify(value, null, 2));
    setProfile(next);
    setPresetId("");
    setError("");
  }

  function updateParams(next: Record<string, unknown>) {
    setParams(next);
    setProfile("custom");
  }

  async function loadTextFile(file: File | undefined) {
    if (!file) return;
    setError("");
    try {
      if (!file.name.toLowerCase().endsWith(".txt")) {
        throw new Error("请选择 .txt 文本文件");
      }
      if (file.size > 200_000) {
        throw new Error("TXT 文件不能超过 200 KB");
      }
      const text = new TextDecoder("utf-8", { fatal: true }).decode(
        await file.arrayBuffer(),
      );
      if (!text.trim() || text.length > 50_000) {
        throw new Error("文本需为非空 UTF-8，最多 50,000 字符");
      }
      const next = { ...params, base_prompt: text };
      updateParams(next);
      setRaw(JSON.stringify(next, null, 2));
      setTextFile(file.name);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "读取 TXT 失败");
    }
  }

  // Use the same server validation and workload calculator as actual submission.
  useEffect(() => {
    let active = true;
    setPlan(null);
    setPlanError("");
    if (!endpoint) return;
    const timer = window.setTimeout(() => {
      let parameters: Record<string, unknown>;
      try {
        const parsed = mode === "json" ? JSON.parse(raw) : params;
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
          throw new Error("参数必须是 JSON 对象");
        }
        parameters = parsed as Record<string, unknown>;
      } catch (exc) {
        if (active)
          setPlanError(exc instanceof Error ? exc.message : "JSON 格式错误");
        return;
      }
      const runConfigDiff = performanceRun
        ? Object.fromEntries(
            Object.entries(runConfig).filter(
              ([key, value]) =>
                value !== undefined &&
                value !== "" &&
                value !== knobDefaults[key],
            ),
          )
        : {};
      api<MeasurementPlan>(token, "/api/v1/jobs/plan", {
        method: "POST",
        body: JSON.stringify({
          endpoint_id: endpoint,
          test_type: type,
          parameters,
          ...(Object.keys(runConfigDiff).length
            ? { run_config: runConfigDiff }
            : {}),
        }),
      })
        .then((value) => {
          if (active) setPlan(value);
        })
        .catch((exc) => {
          if (active)
            setPlanError(
              exc instanceof Error ? exc.message : "无法核对工作负载",
            );
        });
    }, 300);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [
    endpoint,
    type,
    params,
    raw,
    mode,
    runConfig,
    knobDefaults,
    token,
    performanceRun,
  ]);

  function currentParameters(): Record<string, unknown> {
    if (mode === "json") {
      const parsed = JSON.parse(raw) as Record<string, unknown>;
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object")
        throw new Error("参数必须是 JSON 对象");
      return parsed;
    }
    return params;
  }

  function currentRunConfig(): Record<string, unknown> | undefined {
    if (!specs || !performanceRun) return undefined;
    // 与默认值完全一致时不随提交携带, 避免噪音覆盖
    const diff = Object.fromEntries(
      Object.entries(runConfig).filter(
        ([key, v]) => v !== undefined && v !== "" && v !== knobDefaults[key],
      ),
    );
    return Object.keys(diff).length ? diff : undefined;
  }

  async function savePreset() {
    setPresetError("");
    setPresetBusy(true);
    try {
      if (!presetName.trim()) throw new Error("请输入方案名称");
      const saved = await api<Preset>(
        token,
        presetId ? `/api/v1/presets/${presetId}` : "/api/v1/presets",
        {
          method: presetId ? "PUT" : "POST",
          body: JSON.stringify({
            name: presetName.trim(),
            schema_version: 1,
            endpoint_id: endpoint,
            test_type: type,
            parameters: currentParameters(),
            run_config: currentRunConfig(),
          }),
        },
      );
      setPresets((current) => [
        saved,
        ...current.filter((item) => item.preset_id !== saved.preset_id),
      ]);
      setPresetId(saved.preset_id);
      setPresetName(saved.name);
    } catch (exc) {
      setPresetError(exc instanceof Error ? exc.message : "保存方案失败");
    } finally {
      setPresetBusy(false);
    }
  }

  async function deletePreset() {
    if (!presetId) return;
    setPresetError("");
    setPresetBusy(true);
    try {
      await api<void>(token, `/api/v1/presets/${presetId}`, {
        method: "DELETE",
      });
      setPresets((current) =>
        current.filter((item) => item.preset_id !== presetId),
      );
      setPresetId("");
      setPresetName("");
    } catch (exc) {
      setPresetError(exc instanceof Error ? exc.message : "删除方案失败");
    } finally {
      setPresetBusy(false);
    }
  }

  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">NEW MEASUREMENT</span>
          <h1>创建测量</h1>
          <p>先选用例与目标端点，再检查本次执行参数。</p>
        </div>
      </div>
      <section className="surface preset-surface">
        <div className="section-head">
          <div>
            <span className="eyebrow">REUSABLE PLAN</span>
            <h2>测试方案</h2>
          </div>
          <span className="minor-tag">保存在平台数据库</span>
        </div>
        <div className="preset-controls">
          <select
            aria-label="已保存方案"
            value={presetId}
            onChange={(event) => {
              const id = event.target.value;
              setPresetId(id);
              const preset = presets.find((item) => item.preset_id === id);
              if (!preset) return;
              if (
                !scenarios.some((item) => item.id === preset.test_type) ||
                !endpoints.some((item) => item.id === preset.endpoint_id)
              ) {
                setPresetError("方案引用的测试类型或端点已不可用");
                return;
              }
              setPresetError("");
              setPresetName(preset.name);
              setEndpoint(preset.endpoint_id);
              setType(preset.test_type as JobType);
              setParams(preset.parameters);
              setRaw(JSON.stringify(preset.parameters, null, 2));
              setRunConfig({ ...knobDefaults, ...preset.run_config });
              setProfile("custom");
            }}
          >
            <option value="">选择已保存方案</option>
            {presets.map((item) => (
              <option key={item.preset_id} value={item.preset_id}>
                {item.name}
              </option>
            ))}
          </select>
          <input
            aria-label="方案名称"
            value={presetName}
            onChange={(event) => setPresetName(event.target.value)}
            maxLength={80}
            placeholder="输入方案名称"
          />
          <button
            className="button subtle"
            onClick={() => void savePreset()}
            disabled={presetBusy || !endpoint}
          >
            {presetId ? "更新方案" : "保存方案"}
          </button>
          {presetId && (
            <>
              <button
                className="button subtle"
                onClick={() => {
                  setPresetId("");
                  setPresetName("");
                }}
              >
                另存为
              </button>
              <button
                className="button subtle danger"
                onClick={() => void deletePreset()}
                disabled={presetBusy}
              >
                删除方案
              </button>
            </>
          )}
        </div>
        {presetError && (
          <p className="form-error" role="alert">
            {presetError}
          </p>
        )}
      </section>
      <div className="new-layout">
        <section className="surface form-surface">
          <div className="section-head">
            <div>
              <span className="section-index">01</span>
              <h2>测试类型</h2>
            </div>
          </div>
          <div className="scenario-grid">
            {scenarios.map((item) => (
              <button
                key={item.id}
                className={`scenario ${type === item.id ? "selected" : ""}`}
                onClick={() => {
                  setType(item.id);
                  setParams(item.parameters);
                  setRaw(JSON.stringify(item.parameters, null, 2));
                  setProfile("standard");
                  setPresetId("");
                  setError("");
                }}
              >
                <strong>{item.label}</strong>
                <span>{item.description}</span>
              </button>
            ))}
          </div>
          <button className="text-action" onClick={() => navigate("/batch")}>
            需要依次运行并发、Prefill、长上下文？打开一键基础三项 →
          </button>
          <div className="section-head split">
            <div>
              <span className="section-index">02</span>
              <h2>目标端点</h2>
            </div>
          </div>
          <div className="endpoint-select-head">
            <label className="input-label" htmlFor="endpoint">
              受测 API
            </label>
            <button
              className="text-action"
              onClick={() => navigate("/settings/api")}
            >
              配置 API →
            </button>
          </div>
          <select
            id="endpoint"
            value={endpoint}
            disabled={!endpoints.length}
            onChange={(event) => {
              setEndpoint(event.target.value);
              setPresetId("");
            }}
          >
            {!endpoints.length && <option value="">尚未配置受测 API</option>}
            {endpoints.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label} · {item.model_id}
              </option>
            ))}
          </select>
          {selectedEndpoint && !selectedEndpoint.credential_configured && (
            <p className="form-error" role="alert">
              此端点缺少 API key，请先配置凭证。
            </p>
          )}
          <div className="section-head split">
            <div>
              <span className="section-index">03</span>
              <h2>运行参数</h2>
            </div>
            <div
              className="mode-toggle"
              role="tablist"
              aria-label="参数编辑方式"
            >
              <button
                role="tab"
                aria-selected={mode === "form"}
                className={mode === "form" ? "active" : ""}
                onClick={() => {
                  if (mode === "json") {
                    try {
                      setParams(currentParameters());
                      setError("");
                    } catch (exc) {
                      setError(
                        exc instanceof Error ? exc.message : "JSON 格式错误",
                      );
                      return;
                    }
                  }
                  setMode("form");
                }}
              >
                表单
              </button>
              <button
                role="tab"
                aria-selected={mode === "json"}
                className={mode === "json" ? "active" : ""}
                onClick={() => {
                  setRaw(JSON.stringify(params, null, 2));
                  setMode("json");
                }}
              >
                JSON
              </button>
            </div>
          </div>
          <div className="profile-picker" aria-label="测量强度">
            <div>
              <strong>一键设置参数</strong>
              <small>先选强度，仍可在下方逐项调整</small>
            </div>
            <div className="profile-actions">
              {(
                [
                  ["quick", "快速检查"],
                  ["standard", "标准测量"],
                  ["thorough", "深入测量"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  className={profile === id ? "active" : ""}
                  aria-pressed={profile === id}
                  onClick={() => applyProfile(id)}
                >
                  {label}
                </button>
              ))}
              {profile === "custom" && (
                <span className="minor-tag">已自定义</span>
              )}
            </div>
          </div>
          {profile === "quick" && (
            <p className="profile-note">
              快速检查用于确认端点和流程可用；正式性能对比请选择标准或深入测量。
            </p>
          )}
          {mode === "form" && type === "segmented_prefill" && (
            <div className="scenario-shortcuts">
              <strong>分段长度策略</strong>
              {(
                [
                  ["渐进", [2000, 8000, 20000, 40000, 60000]],
                  ["快速增长", [4000, 16000, 32000, 64000]],
                  ["细粒度", [1000, 2000, 4000, 8000, 16000, 32000, 64000]],
                ] as const
              ).map(([label, levels]) => (
                <button
                  key={label}
                  type="button"
                  className="button subtle"
                  onClick={() =>
                    updateParams({ ...params, segment_levels: [...levels] })
                  }
                >
                  {label}
                </button>
              ))}
            </div>
          )}
          {mode === "form" && type === "prefill" && (
            <label className="checkbox-label prefill-isolation">
              <input
                type="checkbox"
                checked={params.max_tokens === 1}
                onChange={(event) =>
                  updateParams({
                    ...params,
                    max_tokens: event.target.checked ? 1 : 256,
                  })
                }
              />
              只生成 1 token，隔离输入处理延迟
            </label>
          )}
          {type === "custom_text" && (
            <div className="text-file-import">
              <label className="input-label" htmlFor="custom-text-file">
                从 TXT 导入提示词正文
              </label>
              <input
                id="custom-text-file"
                type="file"
                accept=".txt,text/plain"
                onChange={(event) => {
                  void loadTextFile(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
              <small>
                UTF-8 · 最多 200 KB / 50,000 字符 · 导入后仍可编辑
                {textFile ? ` · 已导入 ${textFile}` : ""}
              </small>
            </div>
          )}
          {initialOffset != null && (
            <p className="profile-note">
              已将 GET /models 响应首部参考耗时 {initialOffset.toFixed(3)}{" "}
              秒填入本次延迟偏移；该值含服务商处理时间，可在更多运行设置中调整。
            </p>
          )}
          {mode === "form" && type === "dataset" ? (
            <DatasetParams value={params} onChange={updateParams} />
          ) : mode === "form" && type === "robustness" ? (
            <RobustnessParams value={params} onChange={updateParams} />
          ) : mode === "form" && specSchema ? (
            <SchemaForm
              schema={specSchema}
              value={params}
              onChange={updateParams}
              idPrefix={`spec-${type}`}
            />
          ) : (
            <>
              <p className="field-help">
                {specSchema
                  ? "已切换到 JSON 直编模式。"
                  : "未取到参数规格（/api/v1/specs 不可达），以 JSON 编辑。"}
                服务端会拒绝超出边界的请求。
              </p>
              <textarea
                className="json-editor"
                spellCheck={false}
                aria-label="运行参数 JSON"
                value={raw}
                onChange={(event) => {
                  setRaw(event.target.value);
                  setProfile("custom");
                }}
              />
            </>
          )}
          {performanceRun && commonKnobSchema && (
            <div className="common-knobs">
              <strong>常用生成设置</strong>
              <SchemaForm
                schema={commonKnobSchema}
                value={runConfig}
                onChange={setRunConfig}
                idPrefix="common-config"
              />
            </div>
          )}
          {performanceRun && advancedKnobSchema && (
            <details className="knobs-section">
              <summary>更多运行设置（tokenizer、思考预算和偏移等）</summary>
              <TokenizerTools
                token={token}
                modelId={selectedEndpoint?.model_id || ""}
                selected={String(runConfig.hf_tokenizer_model_id || "").replace(
                  "./tokenizers/",
                  "",
                )}
                onSelect={(name) =>
                  setRunConfig((current) => ({
                    ...current,
                    tokenizer_option: name
                      ? "HuggingFace Tokenizer"
                      : undefined,
                    hf_tokenizer_model_id: name
                      ? `./tokenizers/${name}`
                      : undefined,
                  }))
                }
              />
              <SchemaForm
                schema={advancedKnobSchema}
                value={runConfig}
                onChange={setRunConfig}
                idPrefix="run-config"
              />
            </details>
          )}
          <div className="form-footer">
            <span>提交后由独立 worker 执行；页面关闭不影响任务。</span>
            <button
              className="button primary"
              disabled={
                busy ||
                !endpoint ||
                !selectedEndpoint?.credential_configured ||
                !plan ||
                Boolean(planError)
              }
              onClick={async () => {
                setError("");
                try {
                  await onSubmit(
                    endpoint,
                    type,
                    currentParameters(),
                    currentRunConfig(),
                  );
                } catch (exc) {
                  setError(exc instanceof Error ? exc.message : "提交失败");
                }
              }}
            >
              {busy ? "提交中…" : "开始测量 →"}
            </button>
          </div>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
        </section>
        <aside className="new-aside">
          <div className="aside-card measurement-plan" role="status">
            <span className="eyebrow">WORKLOAD PLAN</span>
            <h3>提交前核对</h3>
            {planError ? (
              <p className="form-error">{planError}</p>
            ) : plan ? (
              <>
                <p className="plan-model">
                  {plan.workload_model === "closed_loop_fixed_concurrency"
                    ? "闭环固定并发"
                    : plan.workload_model === "sequential_fixed_input_targets"
                      ? "顺序固定输入长度"
                      : "按测试类型定义的负载"}
                  {plan.protocol_version ? ` · ${plan.protocol_version}` : ""}
                </p>
                <div className="plan-totals">
                  <div>
                    <strong>
                      {!plan.protocol_version && plan.measured_requests === 0
                        ? "动态"
                        : plan.measured_requests}
                    </strong>
                    <span>正式请求</span>
                  </div>
                  <div>
                    <strong>{plan.warmup_requests}</strong>
                    <span>预热请求</span>
                  </div>
                  <div>
                    <strong>
                      {!plan.protocol_version && plan.total_requests === 0
                        ? "动态"
                        : plan.total_requests}
                    </strong>
                    <span>总请求预算</span>
                  </div>
                </div>
                {plan.cells.length > 0 && (
                  <div className="plan-cells">
                    {plan.cells.map((cell) => (
                      <div key={cell.label}>
                        <span>{cell.label}</span>
                        <strong>
                          {cell.measured_requests} + {cell.warmup_requests}
                        </strong>
                      </div>
                    ))}
                  </div>
                )}
                {plan.warnings.length > 0 && (
                  <ul className="plan-warnings">
                    {plan.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <p>正在校验参数和请求预算…</p>
            )}
          </div>
          <div className="aside-card">
            <span className="eyebrow">MEASUREMENT NOTES</span>
            <h3>结果可信，从输入开始</h3>
            <ul>
              <li>每次运行都有独立 ID、状态轨迹和结果目录。</li>
              <li>端点密钥加密保存在平台数据库中，提交内容不包含密钥。</li>
              <li>统计基于逐请求数据，失败请求计入成功率。</li>
              <li>质量评估保留数据来源与样本指纹。</li>
            </ul>
          </div>
          <div className="aside-tip">
            任务创建成功后可在「运行记录」里查看实时状态与报告。
          </div>
        </aside>
      </div>
    </div>
  );
}

/** dataset 场景专用参数面板: 行源二选一(内联 JSON rows / 已存数据集名) + 标量参数。 */
function DatasetParams({
  value,
  onChange,
}: {
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}) {
  const source = value.dataset ? "stored" : "inline";
  const rowsText = JSON.stringify(value.rows ?? [{ prompt: "" }], null, 2);
  const [rawRows, setRawRows] = useState(rowsText);
  const [rowsError, setRowsError] = useState("");
  useEffect(
    () => setRawRows(JSON.stringify(value.rows ?? [{ prompt: "" }], null, 2)),
    [value.rows],
  );

  function setScalar(key: string, v: unknown) {
    onChange({ ...value, [key]: v });
  }

  function commitRows(text: string) {
    setRawRows(text);
    try {
      const parsed = JSON.parse(text) as { prompt?: string }[];
      if (!Array.isArray(parsed) || !parsed.length) throw new Error("bad");
      setRowsError("");
      onChange({ ...value, dataset: undefined, rows: parsed });
    } catch {
      setRowsError('rows 必须是 JSON 数组，如 [{"prompt": "问题"}]');
    }
  }

  return (
    <div className="dataset-params">
      <div className="mode-toggle" role="tablist" aria-label="数据集来源">
        <button
          role="tab"
          aria-selected={source === "inline"}
          className={source === "inline" ? "active" : ""}
          onClick={() =>
            onChange({ ...value, dataset: undefined, rows: value.rows ?? [] })
          }
        >
          内联行
        </button>
        <button
          role="tab"
          aria-selected={source === "stored"}
          className={source === "stored" ? "active" : ""}
          onClick={() => onChange({ ...value, rows: undefined, dataset: "" })}
        >
          已存数据集
        </button>
      </div>
      {source === "inline" ? (
        <label className="schema-field">
          <span className="input-label">
            rows（JSON 数组，每行一个 prompt）
            <em className="required-mark">*</em>
          </span>
          <textarea
            className="json-editor"
            spellCheck={false}
            value={rawRows}
            onChange={(event) => commitRows(event.target.value)}
            aria-label="数据集 rows JSON"
          />
          {rowsError && <small className="form-error">{rowsError}</small>}
        </label>
      ) : (
        <label className="schema-field">
          <span className="input-label">
            已存数据集文件名（datasets/ 目录下）
            <em className="required-mark">*</em>
          </span>
          <input
            value={String(value.dataset ?? "")}
            placeholder="如 my_prompts.json"
            onChange={(event) => setScalar("dataset", event.target.value)}
          />
        </label>
      )}
      <div className="schema-form">
        <label className="schema-field">
          <span className="input-label">并发数</span>
          <input
            type="number"
            min={1}
            max={128}
            value={Number(value.concurrency ?? 4)}
            onChange={(event) =>
              setScalar("concurrency", Number(event.target.value))
            }
          />
        </label>
        <label className="schema-field">
          <span className="input-label">最大输出 tokens</span>
          <input
            type="number"
            min={1}
            max={8192}
            value={Number(value.max_tokens ?? 256)}
            onChange={(event) =>
              setScalar("max_tokens", Number(event.target.value))
            }
          />
        </label>
        <label className="schema-field">
          <span className="input-label">轮数</span>
          <input
            type="number"
            min={1}
            max={20}
            value={Number(value.rounds ?? 1)}
            onChange={(event) =>
              setScalar("rounds", Number(event.target.value))
            }
          />
        </label>
      </div>
    </div>
  );
}

/** robustness 场景专用面板: samples JSON 编辑 + 扰动类型多选 + max_tokens。 */
const PERTURBATION_OPTIONS: [string, string][] = [
  ["synonym", "同义词替换"],
  ["typo", "拼写错误"],
  ["reorder", "词序调整"],
  ["case", "大小写变化"],
  ["punctuation", "标点变化"],
  ["whitespace", "空白符变化"],
  ["number_format", "数字格式"],
  ["paraphrase", "同义改写"],
  ["context_add", "添加无关上下文"],
  ["rephrase", "问题重述"],
];

function RobustnessParams({
  value,
  onChange,
}: {
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}) {
  const [rawSamples, setRawSamples] = useState(() =>
    JSON.stringify(
      value.samples ?? [{ question: "", correct_answer: "" }],
      null,
      2,
    ),
  );
  const [samplesError, setSamplesError] = useState("");
  useEffect(
    () =>
      setRawSamples(
        JSON.stringify(
          value.samples ?? [{ question: "", correct_answer: "" }],
          null,
          2,
        ),
      ),
    [value.samples],
  );

  const pickedTypes = (value.perturbation_types as string[] | undefined) ?? [];

  function commitSamples(text: string) {
    setRawSamples(text);
    try {
      const parsed = JSON.parse(text) as {
        question?: string;
        correct_answer?: string;
      }[];
      if (!Array.isArray(parsed) || !parsed.length) throw new Error("bad");
      if (parsed.some((row) => !(row.question || "").trim()))
        throw new Error("bad");
      setSamplesError("");
      onChange({ ...value, samples: parsed });
    } catch {
      setSamplesError(
        'samples 必须是 JSON 数组，如 [{"question": "…", "correct_answer": "…"}]',
      );
    }
  }

  function toggleType(id: string) {
    const next = pickedTypes.includes(id)
      ? pickedTypes.filter((item) => item !== id)
      : [...pickedTypes, id];
    onChange({ ...value, perturbation_types: next.length ? next : undefined });
  }

  return (
    <div className="dataset-params">
      <label className="schema-field case-form-wide">
        <span className="input-label">
          samples（JSON 数组，含 question 与 correct_answer）
          <em className="required-mark">*</em>
        </span>
        <textarea
          className="json-editor"
          spellCheck={false}
          value={rawSamples}
          onChange={(event) => commitSamples(event.target.value)}
          aria-label="鲁棒性 samples JSON"
        />
        {samplesError && <small className="form-error">{samplesError}</small>}
      </label>
      <div className="schema-field case-form-wide">
        <span className="input-label">扰动类型（不选 = 默认 5 类）</span>
        <div className="perturbation-grid">
          {PERTURBATION_OPTIONS.map(([id, label]) => (
            <label className="checkbox-label" key={id}>
              <input
                type="checkbox"
                checked={pickedTypes.includes(id)}
                onChange={() => toggleType(id)}
              />
              {label}
            </label>
          ))}
        </div>
      </div>
      <label className="schema-field">
        <span className="input-label">最大输出 tokens</span>
        <input
          type="number"
          min={1}
          max={8192}
          value={Number(value.max_tokens ?? 256)}
          onChange={(event) =>
            onChange({ ...value, max_tokens: Number(event.target.value) })
          }
        />
      </label>
    </div>
  );
}
