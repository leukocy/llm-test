import { useEffect, useState } from "react";
import { api, type Job } from "../api";
import { statusLabels } from "../constants";
import { SchemaForm, type JsonSchema } from "./SchemaForm";

type Dataset = {
  id: string;
  group: string;
  available: boolean;
  local_available?: boolean | null;
  bytes?: number;
  downloadable?: boolean;
  preparation?: Job | null;
  preparation_stage?: { fraction: number; message: string } | null;
};

export function QualityParameters({
  token,
  schema,
  value,
  onChange,
}: {
  token: string;
  schema: JsonSchema;
  value: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
}) {
  const [catalog, setCatalog] = useState<Dataset[]>([]);
  const [error, setError] = useState("");
  const [customSampling, setCustomSampling] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [preparing, setPreparing] = useState(false);
  const active = catalog.some(
    (item) =>
      item.preparation &&
      ["queued", "running", "cancelling"].includes(item.preparation.status),
  );
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(
      () => setRefresh((current) => current + 1),
      3000,
    );
    return () => window.clearInterval(timer);
  }, [active]);
  useEffect(() => {
    const controller = new AbortController();
    void api<{ items: Dataset[] }>(token, "/api/v1/quality/datasets", {
      signal: controller.signal,
    })
      .then((result) => {
        if (!controller.signal.aborted) {
          setCatalog(result.items);
          setError("");
        }
      })
      .catch((exc) => {
        if (!controller.signal.aborted) setError(String(exc));
      });
    return () => controller.abort();
  }, [token, refresh]);
  async function prepare(names?: string[]) {
    setPreparing(true);
    setError("");
    try {
      await api(token, "/api/v1/quality/datasets/preparations", {
        method: "POST",
        body: JSON.stringify({ names: names || null }),
      });
      setRefresh((current) => current + 1);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setPreparing(false);
    }
  }
  async function cancel(id: string) {
    try {
      await api(token, `/api/v1/jobs/${id}/cancel`, { method: "POST" });
      setRefresh((current) => current + 1);
    } catch (exc) {
      setError(String(exc));
    }
  }
  const selected = Array.isArray(value.datasets)
    ? (value.datasets as string[])
    : [];
  const sampling =
    value.max_samples === null
      ? "full"
      : customSampling
        ? "custom"
        : value.max_samples === 100
          ? "quick"
          : value.max_samples === 500
            ? "medium"
            : "custom";
  const modelType = String(value.model_type || "standard");
  const fields = Object.fromEntries(
    Object.entries(schema.properties || {}).filter(
      ([name]) =>
        !["datasets", "max_samples", "model_type"].includes(name) &&
        (name !== "ceval_split" || selected.includes("ceval")) &&
        (modelType === "thinking" ||
          !["thinking_enabled", "thinking_budget", "reasoning_effort"].includes(
            name,
          )),
    ),
  );
  function update(fields: Record<string, unknown>) {
    onChange({ ...value, ...fields });
  }
  return (
    <div className="quality-parameters">
      <details className="dataset-status">
        <summary>数据集状态与准备</summary>
        <p className="field-help">
          下载由 worker
          在测量结束后执行。页面关闭不影响任务；取消在当前下载阶段结束后生效。文件校验后发布，本地文件存在不代表评测内容已经核验。
        </p>
        <div className="compare-actions">
          <button
            type="button"
            className="button subtle"
            onClick={() => setRefresh((current) => current + 1)}
          >
            刷新数据集状态
          </button>
          <button
            type="button"
            className="button primary"
            disabled={
              preparing ||
              !catalog.some(
                (item) => item.downloadable && !item.local_available,
              )
            }
            onClick={() => void prepare()}
          >
            下载全部缺失数据集
          </button>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>数据集</th>
                <th>本地文件</th>
                <th>大小</th>
                <th>准备状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {catalog.map((item) => (
                <tr key={item.id}>
                  <td>{item.id}</td>
                  <td>
                    {item.local_available == null
                      ? "本地或生成"
                      : item.local_available
                        ? "已存在"
                        : "缺失"}
                  </td>
                  <td>
                    {item.bytes
                      ? `${(item.bytes / 1024 ** 2).toFixed(1)} MiB`
                      : "—"}
                  </td>
                  <td>
                    {item.preparation ? (
                      <>
                        <a href={`/runs/${item.preparation.job_id}`}>
                          {statusLabels[item.preparation.status] ||
                            item.preparation.status}
                        </a>
                        {item.preparation_stage && (
                          <small>
                            {" "}
                            ·{" "}
                            {(item.preparation_stage.fraction * 100).toFixed(0)}
                            % · {item.preparation_stage.message}
                          </small>
                        )}
                        {item.preparation.error_message && (
                          <small> · {item.preparation.error_message}</small>
                        )}
                      </>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td>
                    {item.downloadable && !item.local_available && (
                      <button
                        type="button"
                        className="button subtle"
                        disabled={
                          preparing ||
                          Boolean(
                            item.preparation &&
                            ["queued", "running", "cancelling"].includes(
                              item.preparation.status,
                            ),
                          )
                        }
                        onClick={() => void prepare([item.id])}
                      >
                        下载 {item.id}
                      </button>
                    )}
                    {item.preparation &&
                      ["queued", "running"].includes(
                        item.preparation.status,
                      ) && (
                        <button
                          type="button"
                          className="button subtle danger"
                          onClick={() => void cancel(item.preparation!.job_id)}
                        >
                          取消 {item.id} 下载
                        </button>
                      )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <h3>评测数据集</h3>
      {selected.includes("cmmlu") && (
        <p className="field-help">
          CMMLU 使用带标签的 test 评分，同科目的 dev 提供 few-shot 示例（每科 5
          题）。缺少标签或示例时停止评测；报告保留实际样本及分区来源。
        </p>
      )}
      {selected.includes("ceval") && (
        <p className="field-help">
          C-Eval 可选 test / val，默认使用已发布标签的
          test；报告记录实际分区。few-shot 来自同科目的
          dev，不借用评分题作示例。官方 dev 每科 5 题；缺标签或示例将报错。
        </p>
      )}
      <p className="field-help">
        使用本地真实数据；准备阶段冻结样本顺序、few-shot
        与内容指纹。全量模式使用全部实际样本。
      </p>
      {error && (
        <p role="alert" className="form-error">
          无法读取数据集目录：{error}；可切换 JSON 模式。
        </p>
      )}
      {[...new Set(catalog.map((item) => item.group))].map((group) => (
        <fieldset key={group}>
          <legend>{group}</legend>
          <div className="quality-dataset-grid">
            {catalog
              .filter((item) => item.group === group)
              .map((item) => (
                <label key={item.id}>
                  <input
                    type="checkbox"
                    checked={selected.includes(item.id)}
                    disabled={!item.available}
                    onChange={(event) =>
                      update({
                        datasets: event.target.checked
                          ? [...selected, item.id]
                          : selected.filter((name) => name !== item.id),
                      })
                    }
                  />
                  {item.id}
                  {!item.available && "（评测器未实现）"}
                </label>
              ))}
          </div>
        </fieldset>
      ))}
      <div className="scenario-controls">
        <label>
          模型类型
          <select
            aria-label="质量模型类型"
            value={modelType}
            onChange={(event) =>
              update({
                model_type: event.target.value,
                thinking_enabled: event.target.value === "thinking",
                num_shots: event.target.value === "thinking" ? 0 : 5,
              })
            }
          >
            <option value="standard">普通模型</option>
            <option value="thinking">思考模型（CoT）</option>
            <option value="code">代码模型</option>
          </select>
        </label>
        <label>
          采样模式
          <select
            aria-label="质量采样模式"
            value={sampling}
            onChange={(event) => {
              const mode = event.target.value;
              setCustomSampling(mode === "custom");
              update({
                max_samples:
                  mode === "full"
                    ? null
                    : mode === "medium"
                      ? 500
                      : mode === "quick"
                        ? 100
                        : 200,
              });
            }}
          >
            <option value="quick">快速：每数据集 100 题</option>
            <option value="medium">中等：每数据集 500 题</option>
            <option value="custom">自定义数量</option>
            <option value="full">全量：全部实际样本</option>
          </select>
        </label>
        {sampling === "custom" && (
          <label>
            每数据集样本数
            <input
              aria-label="质量自定义样本数"
              type="number"
              min={1}
              max={10000}
              value={Number(value.max_samples || 1)}
              onChange={(event) =>
                update({ max_samples: Number(event.target.value) })
              }
            />
          </label>
        )}
      </div>
      <p className="field-help">
        抽样按固定种子 42；数据不足时使用实际数量。单作业最多 100,000
        个冻结样本，超出时拆分数据集。缺数据将报错，不生成替代题。
      </p>
      <SchemaForm
        schema={{ ...schema, properties: fields }}
        value={value}
        onChange={onChange}
        idPrefix="quality"
      />
      <p className="field-help">
        AI Judge 使用同一模型复核规则判错且有回答的样本，仅接受完整 YES /
        NO。报告区分规则成绩与复核后成绩；启用后会增加请求与
        token。部分专用评测器有自己的评分流程，不能据此认为所有数据集均执行复核。响应缓存最长保留
        7 天，测量新响应性能时应关闭。
      </p>
    </div>
  );
}
