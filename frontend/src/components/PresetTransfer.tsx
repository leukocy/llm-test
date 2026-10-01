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
  const [conversion, setConversion] = useState<Record<string, unknown> | null>(
    null,
  );
  const [parametersText, setParametersText] = useState("");
  const [parametersError, setParametersError] = useState("");
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
    setConversion(null);
    setParametersError("");
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
      let envelope = decoded as Envelope;
      let convertedLegacy = false;
      const legacy = decoded as Record<string, unknown>;
      if (
        legacy.config &&
        typeof legacy.config === "object" &&
        !Array.isArray(legacy.config) &&
        typeof legacy.name === "string" &&
        !("format" in legacy)
      ) {
        const excluded = new Set([
          "api_key",
          "apiKey",
          "token",
          "access_token",
          "authorization",
          "password",
          "secret",
          "provider",
          "model_id",
          "api_base_url",
        ]);
        const isExcluded = (field: string) =>
          excluded.has(field) ||
          /^(api[_-]?key|access[_-]?token|token|authorization|password|secret)$/i.test(
            field,
          );
        const config = Object.fromEntries(
          Object.entries(legacy.config).filter(([field]) => !isExcluded(field)),
        );
        const fields = Object.keys(legacy.config).filter(isExcluded);
        const sanitized = {
          name: legacy.name,
          description: legacy.description || "",
          tags: legacy.tags || [],
          created_at: legacy.created_at || null,
          config,
          excluded_fields: fields,
        };
        const converted = await api<
          Envelope & { conversion: Record<string, unknown> }
        >(token, "/api/v1/presets/convert", {
          method: "POST",
          body: JSON.stringify(sanitized),
        });
        if (request !== generation.current) return;
        convertedLegacy = true;
        envelope = {
          format: converted.format,
          version: converted.version,
          preset: converted.preset,
        };
        setConversion(converted.conversion);
        setParametersText(JSON.stringify(converted.preset.parameters, null, 2));
      }
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
        !convertedLegacy &&
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
      setConversion(null);
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
        支持新版 llm-test-preset v1 与首次提交的 ConfigPreset 文件，最多 1
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
          {conversion && (
            <section className="case-form-wide legacy-preset-preview">
              <h3>旧版配置转换预览</h3>
              {(conversion.notes as string[]).map((note) => (
                <p className="chart-caption" key={note}>
                  {note}
                </p>
              ))}
              <p className="chart-caption">
                原创建时间：{String(conversion.source_created_at || "未记录")} ·
                排除字段：
                {(conversion.excluded_fields as string[]).join(", ") || "无"}
              </p>
              <p className="chart-caption">
                原校验问题：{JSON.stringify(conversion.errors)}
                。编辑后以保存时的完整校验为准。
              </p>
              <label className="schema-field">
                转换后的测试参数
                <textarea
                  className="json-editor"
                  aria-label="转换后的测试参数"
                  maxLength={1024 * 1024}
                  value={parametersText}
                  onChange={(event) => {
                    const text = event.target.value;
                    setParametersText(text);
                    try {
                      const parsed: unknown = JSON.parse(text);
                      if (
                        !parsed ||
                        typeof parsed !== "object" ||
                        Array.isArray(parsed)
                      )
                        throw new Error("参数必须是对象");
                      setParametersError("");
                      setDraft((current) =>
                        current
                          ? {
                              ...current,
                              preset: { ...current.preset, parameters: parsed },
                            }
                          : current,
                      );
                    } catch (exc) {
                      setParametersError(String(exc));
                    }
                  }}
                />
              </label>
              {parametersError && (
                <p role="alert" className="form-error">
                  {parametersError}；不能提交旧的有效参数。
                </p>
              )}
            </section>
          )}
          <button
            className="button subtle"
            type="button"
            disabled={busy || !name.trim() || !endpoint || !!parametersError}
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
