import { useEffect, useState } from "react";
import { api } from "../api";
import { Empty } from "../components";
import { formatNumber } from "../constants";

export type Case = {
  case_id: string;
  source: string;
  date: string;
  scenario: string;
  task_name: string;
  customer_type: string;
  model_name: string;
  machine_id: string;
  engine: string;
  tester: string;
  quality_score: number | null;
  citation_score: number | null;
  success: boolean | null;
  external_level: string;
  failure_reason: string;
  next_action: string;
  sales_summary: string;
};

const EMPTY_FORM = {
  scenario: "",
  model_name: "",
  machine_id: "",
  engine: "",
  tester: "",
  task_name: "",
  customer_type: "",
  quality_score: "",
  external_level: "internal",
  failure_reason: "",
  next_action: "",
  sales_summary: "",
};

export function WarehouseCases({ token }: { token: string }) {
  const [cases, setCases] = useState<Case[]>([]);
  const [scenario, setScenario] = useState("");
  const [modelName, setModelName] = useState("");
  const [level, setLevel] = useState("");
  const [error, setError] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [formBusy, setFormBusy] = useState(false);
  const [formMsg, setFormMsg] = useState("");

  function load() {
    const params = new URLSearchParams();
    if (scenario) params.set("scenario", scenario);
    if (modelName) params.set("model_name", modelName);
    if (level) params.set("external_level", level);
    api<{ items: Case[] }>(token, `/api/v1/cases?${params}`)
      .then((data) => {
        setCases(data.items);
        setError("");
      })
      .catch((exc) =>
        setError(exc instanceof Error ? exc.message : "用例读取失败"),
      );
  }
  useEffect(load, [token, scenario, modelName, level]);

  function setField(key: keyof typeof EMPTY_FORM) {
    return (event: { target: { value: string } }) =>
      setForm((current) => ({ ...current, [key]: event.target.value }));
  }

  async function submitCase() {
    setFormMsg("");
    setFormBusy(true);
    try {
      if (!form.scenario.trim() || !form.model_name.trim())
        throw new Error("场景与模型名为必填");
      const body: Record<string, unknown> = {
        ...form,
        quality_score:
          form.quality_score === "" ? null : Number(form.quality_score),
      };
      delete body.quality_score;
      if (form.quality_score !== "")
        body.quality_score = Number(form.quality_score);
      await api(token, "/api/v1/cases", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setForm(EMPTY_FORM);
      setFormMsg("已录入");
      load();
    } catch (exc) {
      setFormMsg(exc instanceof Error ? exc.message : "录入失败");
    } finally {
      setFormBusy(false);
    }
  }

  async function removeCase(caseId: string) {
    try {
      await api(token, `/api/v1/cases/${encodeURIComponent(caseId)}`, {
        method: "DELETE",
      });
      load();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "删除失败");
    }
  }

  return (
    <div>
      <div className="warehouse-matrix-controls">
        <label>
          场景
          <input
            aria-label="筛选场景"
            value={scenario}
            onChange={(event) => setScenario(event.target.value)}
            placeholder="精确匹配"
          />
        </label>
        <label>
          模型
          <input
            aria-label="筛选模型"
            value={modelName}
            onChange={(event) => setModelName(event.target.value)}
            placeholder="精确匹配"
          />
        </label>
        <label>
          对外等级
          <select
            aria-label="筛选等级"
            value={level}
            onChange={(event) => setLevel(event.target.value)}
          >
            <option value="">全部</option>
            <option value="internal">internal</option>
            <option value="review">review</option>
            <option value="publishable">publishable</option>
          </select>
        </label>
      </div>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {cases.length ? (
        <div className="table-scroll">
          <table className="data-table stats-table">
            <thead>
              <tr>
                <th>日期</th>
                <th>场景</th>
                <th>模型</th>
                <th>硬件</th>
                <th>质量分</th>
                <th>成功</th>
                <th>等级</th>
                <th>来源</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {cases.map((item) => (
                <tr key={item.case_id}>
                  <td>{item.date || "—"}</td>
                  <td>
                    <strong>{item.scenario}</strong>
                    <small>{item.task_name || ""}</small>
                  </td>
                  <td>{item.model_name}</td>
                  <td>{item.machine_id || "—"}</td>
                  <td>{formatNumber(item.quality_score, 1)}</td>
                  <td>
                    {item.success === null ? "—" : item.success ? "是" : "否"}
                  </td>
                  <td>{item.external_level}</td>
                  <td>{item.source}</td>
                  <td>
                    <button
                      className="text-button danger"
                      onClick={() => void removeCase(item.case_id)}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty title="暂无应用用例" text="质量评估自动采集或在下方手动录入。" />
      )}

      <details className="knobs-section" open={!cases.length}>
        <summary>手动录入用例</summary>
        <div className="schema-form case-form">
          <label className="schema-field">
            <span className="input-label">
              场景<em className="required-mark">*</em>
            </span>
            <input value={form.scenario} onChange={setField("scenario")} />
          </label>
          <label className="schema-field">
            <span className="input-label">
              模型名<em className="required-mark">*</em>
            </span>
            <input value={form.model_name} onChange={setField("model_name")} />
          </label>
          <label className="schema-field">
            <span className="input-label">硬件</span>
            <input value={form.machine_id} onChange={setField("machine_id")} />
          </label>
          <label className="schema-field">
            <span className="input-label">引擎</span>
            <input value={form.engine} onChange={setField("engine")} />
          </label>
          <label className="schema-field">
            <span className="input-label">任务名</span>
            <input value={form.task_name} onChange={setField("task_name")} />
          </label>
          <label className="schema-field">
            <span className="input-label">客户类型</span>
            <input
              value={form.customer_type}
              onChange={setField("customer_type")}
            />
          </label>
          <label className="schema-field">
            <span className="input-label">质量分（0-100）</span>
            <input
              type="number"
              value={form.quality_score}
              onChange={setField("quality_score")}
            />
          </label>
          <label className="schema-field">
            <span className="input-label">对外等级</span>
            <select
              value={form.external_level}
              onChange={setField("external_level")}
            >
              <option value="internal">internal</option>
              <option value="review">review</option>
              <option value="publishable">publishable</option>
            </select>
          </label>
          <label className="schema-field">
            <span className="input-label">失败原因</span>
            <input
              value={form.failure_reason}
              onChange={setField("failure_reason")}
            />
          </label>
          <label className="schema-field">
            <span className="input-label">下一步</span>
            <input
              value={form.next_action}
              onChange={setField("next_action")}
            />
          </label>
          <label className="schema-field case-form-wide">
            <span className="input-label">销售摘要</span>
            <input
              value={form.sales_summary}
              onChange={setField("sales_summary")}
            />
          </label>
        </div>
        <div className="form-footer">
          <span>{formMsg}</span>
          <button
            className="button primary"
            disabled={formBusy}
            onClick={() => void submitCase()}
          >
            录入用例
          </button>
        </div>
      </details>
    </div>
  );
}

export function WarehouseCapability({ token }: { token: string }) {
  const [minLevel, setMinLevel] = useState("review");
  const [capability, setCapability] = useState<{
    items: Record<string, string | number | null>[];
    markdown: string;
  } | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api<{ items: Record<string, string | number | null>[]; markdown: string }>(
      token,
      `/api/v1/cases/capability?min_level=${minLevel}`,
    )
      .then((data) => {
        setCapability(data);
        setError("");
      })
      .catch((exc) => {
        setCapability(null);
        setError(exc instanceof Error ? exc.message : "能力表生成失败");
      });
  }, [token, minLevel]);

  const columns = capability?.items.length
    ? Object.keys(capability.items[0])
    : [];

  return (
    <div>
      <div className="warehouse-matrix-controls">
        <label>
          对外口径下限
          <select
            aria-label="对外口径下限"
            value={minLevel}
            onChange={(event) => setMinLevel(event.target.value)}
          >
            <option value="internal">internal</option>
            <option value="review">review 起</option>
            <option value="publishable">publishable</option>
          </select>
        </label>
        <button
          className="button subtle"
          disabled={!capability?.markdown}
          onClick={() => {
            const blob = new Blob([capability!.markdown], {
              type: "text/markdown",
            });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "capability_sheet.md";
            link.click();
            window.setTimeout(() => URL.revokeObjectURL(url), 1000);
          }}
        >
          导出 Markdown ↓
        </button>
      </div>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {capability?.items.length ? (
        <div className="table-scroll">
          <table className="data-table stats-table">
            <thead>
              <tr>
                {columns.map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {capability.items.map((row, index) => (
                <tr key={index}>
                  {columns.map((key) => (
                    <td key={key}>
                      {typeof row[key] === "number"
                        ? formatNumber(row[key] as number, 2)
                        : String(row[key] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !error && (
          <Empty
            title={`当前口径下无能力切片`}
            text="降低口径下限或录入更多应用用例。"
          />
        )
      )}
      {capability?.markdown && (
        <details className="knobs-section">
          <summary>预览对外 Markdown</summary>
          <pre className="markdown-preview">{capability.markdown}</pre>
        </details>
      )}
    </div>
  );
}
