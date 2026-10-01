import { useRef, useState } from "react";
import { api, downloadFile, type Endpoint, type Preset } from "../api";

type Envelope = {
  format: "llm-test-preset";
  version: 1;
  preset: Record<string, unknown>;
};
export function PresetTransfer({
  token,
  presets,
  endpoints,
  onImported,
}: {
  token: string;
  presets: Preset[];
  endpoints: Endpoint[];
  onImported: (preset: Preset) => void;
}) {
  const [draft, setDraft] = useState<Envelope | null>(null);
  const [name, setName] = useState("");
  const [endpoint, setEndpoint] = useState("");
  const [exportId, setExportId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const generation = useRef(0);
  async function read(file?: File) {
    const request = ++generation.current;
    setDraft(null);
    setError("");
    setNotice("");
    if (!file) return;
    try {
      if (file.size > 1024 * 1024) throw new Error("配置文件最多 1 MiB");
      const decoded: unknown = JSON.parse(
        (await file.text()).replace(/^\uFEFF/, ""),
      );
      if (!decoded || typeof decoded !== "object" || Array.isArray(decoded))
        throw new Error("配置必须是 JSON 对象");
      const envelope = decoded as Envelope;
      if (
        envelope.format !== "llm-test-preset" ||
        envelope.version !== 1 ||
        !envelope.preset ||
        typeof envelope.preset !== "object" ||
        Array.isArray(envelope.preset)
      )
        throw new Error("请选择平台导出的 llm-test-preset v1 文件");
      if (
        typeof envelope.preset.name !== "string" ||
        typeof envelope.preset.endpoint_id !== "string" ||
        typeof envelope.preset.test_type !== "string"
      )
        throw new Error("缺少方案名称、端点或测试类型");
      if (request !== generation.current) return;
      setDraft(envelope);
      setName(envelope.preset.name);
      setEndpoint(
        endpoints.some((item) => item.id === envelope.preset.endpoint_id)
          ? envelope.preset.endpoint_id
          : "",
      );
    } catch (exc) {
      if (request === generation.current) setError(`未导入：${String(exc)}`);
    }
  }
  async function importPreset() {
    if (!draft) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const saved = await api<Preset>(token, "/api/v1/presets/import", {
        method: "POST",
        body: JSON.stringify({
          ...draft,
          preset: { ...draft.preset, name: name.trim(), endpoint_id: endpoint },
        }),
      });
      onImported(saved);
      setDraft(null);
      setNotice(`已导入 ${saved.name}，选择方案以应用。`);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="preset-transfer">
      <summary>配置导入／导出</summary>
      <p className="chart-caption">
        文件格式 llm-test-preset v1，最多 1
        MiB。导出含端点引用，不含端点凭证；跨平台导入可重新选择端点。导入只创建新方案，重名不会覆盖。
      </p>
      <label className="input-label">
        选择配置 JSON
        <input
          type="file"
          accept=".json,application/json"
          aria-label="导入方案 JSON"
          disabled={busy}
          onChange={(event) => {
            void read(event.target.files?.[0]);
            event.currentTarget.value = "";
          }}
        />
      </label>
      {draft && (
        <div className="schema-form">
          <label className="schema-field">
            导入方案名称
            <input
              aria-label="导入方案名称"
              value={name}
              maxLength={80}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <label className="schema-field">
            导入端点
            <select
              aria-label="导入端点"
              value={endpoint}
              onChange={(event) => setEndpoint(event.target.value)}
            >
              <option value="">选择已配置端点</option>
              {endpoints.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <p className="chart-caption">
            测试类型：{String(draft.preset.test_type)}
            。参数、标签与说明将在导入时进行服务端完整校验。
          </p>
          <button
            className="button subtle"
            type="button"
            disabled={busy || !name.trim() || !endpoint}
            onClick={() => void importPreset()}
          >
            导入为新方案
          </button>
        </div>
      )}
      <div className="api-form-actions">
        <select
          aria-label="选择要导出的方案"
          value={exportId}
          onChange={(event) => setExportId(event.target.value)}
        >
          <option value="">选择要导出的方案</option>
          {presets.map((item) => (
            <option value={item.preset_id} key={item.preset_id}>
              {item.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="button subtle"
          disabled={
            !exportId ||
            busy ||
            !presets.some((item) => item.preset_id === exportId)
          }
          onClick={() => {
            setError("");
            void downloadFile(
              token,
              `/api/v1/presets/${exportId}/export`,
              "llm-test-preset.json",
            ).catch((exc) => setError(String(exc)));
          }}
        >
          导出方案 JSON
        </button>
      </div>
      {error && (
        <p role="alert" className="form-error">
          {error}
        </p>
      )}
      {notice && <p role="status">{notice}</p>}
    </details>
  );
}
