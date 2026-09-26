import { useState } from "react";
import { api, type Endpoint, type Job } from "../api";
import { scenarios, type JobType } from "../constants";

type Item = { test_type: JobType; raw: string };

export function Batch({
  endpoints,
  token,
  onSubmitted,
}: {
  endpoints: Endpoint[];
  token: string;
  onSubmitted: (batchId: string) => void;
}) {
  const [endpoint, setEndpoint] = useState(endpoints[0]?.id || "");
  const [items, setItems] = useState<Item[]>([
    { test_type: "concurrency", raw: JSON.stringify(scenarios[0].parameters, null, 2) },
  ]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function updateItem(index: number, patch: Partial<Item>) {
    setItems((current) => current.map((item, i) => (i === index ? { ...item, ...patch } : item)));
  }

  function addItem() {
    setItems((current) => [
      ...current,
      { test_type: "concurrency", raw: JSON.stringify(scenarios[0].parameters, null, 2) },
    ]);
  }

  function removeItem(index: number) {
    setItems((current) => current.filter((_, i) => i !== index));
  }

  async function submit() {
    setError("");
    setBusy(true);
    try {
      const parsed = items.map((item, index) => {
        let parameters: Record<string, unknown>;
        try {
          parameters = JSON.parse(item.raw) as Record<string, unknown>;
        } catch {
          throw new Error(`第 ${index + 1} 项参数不是合法 JSON`);
        }
        return { test_type: item.test_type, parameters };
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

  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">BATCH MEASUREMENT</span>
          <h1>批量测量</h1>
          <p>
            一次提交 1~10 个子任务，同一批次 ID 关联；worker 逐个串行执行。
            每项可用不同的测试类型与参数。
          </p>
        </div>
      </div>
      <section className="surface">
        <div className="section-head">
          <div>
            <span className="section-index">01</span>
            <h2>目标端点</h2>
          </div>
        </div>
        <select
          aria-label="批量端点"
          value={endpoint}
          onChange={(event) => setEndpoint(event.target.value)}
        >
          {endpoints.map((item) => (
            <option key={item.id} value={item.id}>
              {item.label} · {item.model_id}
            </option>
          ))}
        </select>
      </section>
      <section className="surface">
        <div className="section-head">
          <div>
            <span className="section-index">02</span>
            <h2>子任务（{items.length} / 10）</h2>
          </div>
          <button className="button subtle" onClick={addItem} disabled={items.length >= 10}>
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
                onChange={(event) => {
                  const next = event.target.value as JobType;
                  updateItem(index, {
                    test_type: next,
                    raw: JSON.stringify(
                      scenarios.find((s) => s.id === next)!.parameters,
                      null,
                      2,
                    ),
                  });
                }}
              >
                {scenarios.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
              {items.length > 1 && (
                <button className="text-button danger" onClick={() => removeItem(index)}>
                  移除
                </button>
              )}
            </div>
            <textarea
              className="json-editor batch-item-editor"
              spellCheck={false}
              aria-label={`子任务 ${index + 1} 参数`}
              value={item.raw}
              onChange={(event) => updateItem(index, { raw: event.target.value })}
            />
          </div>
        ))}
        <div className="form-footer">
          <span>提交后可在「运行记录」按批次查看整体进度。</span>
          <button
            className="button primary"
            disabled={busy || !endpoint || !items.length}
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
