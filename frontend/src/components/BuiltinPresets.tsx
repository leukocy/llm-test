import { useEffect, useState } from "react";
import { api } from "../api";

export type BuiltinTemplate = {
  id: string;
  name: string;
  display_name: string;
  description: string;
  tags: string[];
  test_type: string;
  parameters: Record<string, unknown>;
  run_config: Record<string, unknown>;
  source_config: Record<string, unknown>;
  baseline_commit: string;
  notes: string[];
};
export function BuiltinPresets({
  token,
  busy,
  canApply,
  onApply,
}: {
  token: string;
  busy: boolean;
  canApply: boolean;
  onApply: (template: BuiltinTemplate) => void;
}) {
  const [items, setItems] = useState<BuiltinTemplate[]>([]);
  const [id, setId] = useState("");
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setItems([]);
    setError("");
    void api<{ items: BuiltinTemplate[] }>(token, "/api/v1/presets/templates", {
      signal: controller.signal,
    })
      .then((result) => {
        if (!controller.signal.aborted) setItems(result.items);
      })
      .catch((exc) => {
        if (!controller.signal.aborted) setError(String(exc));
      });
    return () => controller.abort();
  }, [token, retry]);
  const selected = items.find((item) => item.id === id);
  return (
    <details className="builtin-presets">
      <summary>原版内置预设</summary>
      <p className="chart-caption">
        选择当前端点后填入原版配置；不会覆盖已保存方案，也不会启动测量。可以检查参数后另存为方案。
      </p>
      <div className="api-form-actions">
        <select
          aria-label="原版内置预设"
          value={id}
          disabled={busy || !items.length}
          onChange={(event) => setId(event.target.value)}
        >
          <option value="">选择内置预设</option>
          {items.map((item) => (
            <option value={item.id} key={item.id}>
              {item.display_name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="button subtle"
          disabled={busy || !selected || !canApply}
          onClick={() => selected && onApply(selected)}
        >
          填入内置预设
        </button>
      </div>
      {selected && (
        <>
          <p>{selected.description}</p>
          {selected.notes.map((note) => (
            <p className="chart-caption" key={note}>
              {note}
            </p>
          ))}
          <details>
            <summary>查看原版参数</summary>
            <pre className="builtin-preset-source">
              {JSON.stringify(selected.source_config, null, 2)}
            </pre>
          </details>
        </>
      )}
      {error && (
        <p className="form-error" role="alert">
          {error}
          <button
            type="button"
            className="text-button"
            onClick={() => setRetry((value) => value + 1)}
          >
            重新读取内置预设
          </button>
        </p>
      )}
    </details>
  );
}
