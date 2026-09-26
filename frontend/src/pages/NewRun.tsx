import { useEffect, useMemo, useState } from "react";
import { api, type Endpoint, type Preset } from "../api";
import { scenarios, type JobType } from "../constants";
import { SchemaForm, schemaDefaults, type JsonSchema } from "../components/SchemaForm";

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
  const [endpoint, setEndpoint] = useState(endpoints[0]?.id || "");
  const [type, setType] = useState<JobType>("concurrency");
  const [params, setParams] = useState<Record<string, unknown>>(
    () => scenarios[0].parameters,
  );
  const [runConfig, setRunConfig] = useState<Record<string, unknown>>({});
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
  const specSchema = specs?.items[type]?.schema;

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

  // 切换类型：表单值重置为该类型推荐起点（schema 默认值 × 内置预设）
  useEffect(() => {
    const base = scenarios.find((item) => item.id === type)!.parameters;
    setParams(base);
    setRaw(JSON.stringify(base, null, 2));
  }, [type]);

  useEffect(() => {
    if (specs && !Object.keys(runConfig).length && specs.run_config_schema) {
      setRunConfig(schemaDefaults(specs.run_config_schema));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [specs]);

  const knobDefaults = useMemo(
    () => (specs ? schemaDefaults(specs.run_config_schema) : {}),
    [specs],
  );

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
    if (!specs) return undefined;
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
                  setPresetId("");
                  setError("");
                }}
              >
                <strong>{item.label}</strong>
                <span>{item.description}</span>
              </button>
            ))}
          </div>
          <div className="section-head split">
            <div>
              <span className="section-index">02</span>
              <h2>目标端点</h2>
            </div>
          </div>
          <label className="input-label" htmlFor="endpoint">
            管理员预设端点
          </label>
          <select
            id="endpoint"
            value={endpoint}
            onChange={(event) => {
              setEndpoint(event.target.value);
              setPresetId("");
            }}
          >
            {endpoints.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label} · {item.model_id}
              </option>
            ))}
          </select>
          <div className="section-head split">
            <div>
              <span className="section-index">03</span>
              <h2>运行参数</h2>
            </div>
            <div className="mode-toggle" role="tablist" aria-label="参数编辑方式">
              <button
                role="tab"
                aria-selected={mode === "form"}
                className={mode === "form" ? "active" : ""}
                onClick={() => setMode("form")}
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
          {mode === "form" && specSchema ? (
            <SchemaForm
              schema={specSchema}
              value={params}
              onChange={setParams}
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
                onChange={(event) => setRaw(event.target.value)}
              />
            </>
          )}
          {specs && (
            <details className="knobs-section">
              <summary>
                运行配置旋钮（温度 / 思考 / 种子 / tokenizer 等，可选）
              </summary>
              <SchemaForm
                schema={specs.run_config_schema}
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
              disabled={busy || !endpoint}
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
          <div className="aside-card">
            <span className="eyebrow">MEASUREMENT NOTES</span>
            <h3>结果可信，从输入开始</h3>
            <ul>
              <li>每次运行都有独立 ID、状态轨迹和结果目录。</li>
              <li>端点地址与密钥由管理员配置，提交内容不包含密钥。</li>
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
