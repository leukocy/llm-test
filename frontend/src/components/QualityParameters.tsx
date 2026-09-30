import { useEffect, useState } from "react";
import { api } from "../api";
import { SchemaForm, type JsonSchema } from "./SchemaForm";

type Dataset = { id: string; group: string; available: boolean };

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
  useEffect(() => {
    const controller = new AbortController();
    void api<{ items: Dataset[] }>(token, "/api/v1/quality/datasets", {
      signal: controller.signal,
    })
      .then((result) => {
        setCatalog(result.items);
        setError("");
      })
      .catch((exc) => {
        if (!controller.signal.aborted) setError(String(exc));
      });
    return () => controller.abort();
  }, [token]);
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
      <h3>评测数据集</h3>
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
