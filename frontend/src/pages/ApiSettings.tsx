import { templates, originalModels } from "../endpointTemplates";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type Endpoint } from "../api";

type ApiForm = {
  label: string;
  provider: "OpenAI" | "Gemini" | "Anthropic";
  api_base_url: string;
  model_id: string;
  tokenizer_option: string;
  api_key: string;
};

const emptyForm: ApiForm = {
  label: "",
  provider: "OpenAI",
  api_base_url: "",
  model_id: "",
  tokenizer_option: "auto",
  api_key: "",
};

export function ApiSettings({
  endpoints,
  token,
  onChanged,
}: {
  endpoints: Endpoint[];
  token: string;
  onChanged: () => Promise<void>;
}) {
  const navigate = useNavigate();
  const [editingId, setEditingId] = useState("");
  const [form, setForm] = useState<ApiForm>(emptyForm);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [probe, setProbe] = useState<Record<string, string>>({});
  const [probingId, setProbingId] = useState("");
  const [referenceLatency, setReferenceLatency] = useState<
    Record<string, number>
  >({});
  const [measuringId, setMeasuringId] = useState("");
  const [discoveredModels, setDiscoveredModels] = useState<string[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const savedEndpoint = endpoints.find((item) => item.id === editingId);
  const matchesSavedEndpoint =
    !!savedEndpoint &&
    savedEndpoint.provider === form.provider &&
    savedEndpoint.api_base_url === form.api_base_url.trim();
  const canDiscoverModels =
    !!form.api_base_url.trim() &&
    (!!form.api_key.trim() || matchesSavedEndpoint);

  function change<K extends keyof ApiForm>(key: K, value: ApiForm[K]) {
    setForm((current) => ({ ...current, [key]: value }));
    if (key !== "model_id") setDiscoveredModels([]);
    setError("");
    setNotice("");
  }

  function edit(item: Endpoint) {
    if (item.source !== "managed") return;
    setEditingId(item.id);
    setForm({
      label: item.label,
      provider:
        item.provider === "Anthropic"
          ? "Anthropic"
          : item.provider === "Gemini"
            ? "Gemini"
            : "OpenAI",
      api_base_url: item.api_base_url,
      model_id: item.model_id,
      tokenizer_option: item.tokenizer_option,
      api_key: "",
    });
    setError("");
    setNotice("");
    setDiscoveredModels([]);
  }

  function reset() {
    setEditingId("");
    setForm(emptyForm);
    setError("");
    setNotice("");
    setDiscoveredModels([]);
  }

  async function save() {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const payload = {
        label: form.label.trim(),
        provider: form.provider,
        api_base_url: form.api_base_url.trim(),
        model_id: form.model_id.trim(),
        tokenizer_option: form.tokenizer_option.trim() || "auto",
        ...(form.api_key ? { api_key: form.api_key } : {}),
      };
      const result = await api<Endpoint>(
        token,
        editingId
          ? `/api/v1/endpoints/${encodeURIComponent(editingId)}`
          : "/api/v1/endpoints",
        { method: editingId ? "PUT" : "POST", body: JSON.stringify(payload) },
      );
      await onChanged();
      setEditingId(result.id);
      setForm((current) => ({ ...current, api_key: "" }));
      setNotice("已保存。可以测试连接，或直接用此端点创建测量。");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "保存受测 API 失败");
    } finally {
      setBusy(false);
    }
  }

  async function testConnection(item: Endpoint) {
    setProbingId(item.id);
    setProbe((current) => ({ ...current, [item.id]: "正在检测…" }));
    try {
      const result = await api<{
        ok: boolean;
        message: string;
        latency_ms?: number;
      }>(token, `/api/v1/endpoints/${encodeURIComponent(item.id)}/probe`, {
        method: "POST",
      });
      setProbe((current) => ({
        ...current,
        [item.id]: `${result.ok ? "✓" : "×"} ${result.message}${
          result.latency_ms == null ? "" : ` · ${result.latency_ms} ms`
        }`,
      }));
    } catch (exc) {
      setProbe((current) => ({
        ...current,
        [item.id]: exc instanceof Error ? exc.message : "连接检测失败",
      }));
    } finally {
      setProbingId("");
    }
  }

  async function measureLatency(item: Endpoint) {
    setMeasuringId(item.id);
    try {
      const result = await api<{ reference_ms: number; method: string }>(
        token,
        `/api/v1/endpoints/${encodeURIComponent(item.id)}/reference-latency`,
        { method: "POST" },
      );
      setReferenceLatency((current) => ({
        ...current,
        [item.id]: result.reference_ms,
      }));
    } catch (exc) {
      setProbe((current) => ({
        ...current,
        [item.id]: exc instanceof Error ? exc.message : "参考耗时测量失败",
      }));
    } finally {
      setMeasuringId("");
    }
  }

  async function refreshModels() {
    if (!canDiscoverModels) return;
    setDiscovering(true);
    setError("");
    setNotice("");
    try {
      const result =
        matchesSavedEndpoint && !form.api_key.trim()
          ? await api<{ items: string[]; truncated: boolean }>(
              token,
              `/api/v1/endpoints/${encodeURIComponent(editingId)}/models`,
            )
          : await api<{ items: string[]; truncated: boolean }>(
              token,
              "/api/v1/endpoints/models",
              {
                method: "POST",
                body: JSON.stringify({
                  provider: form.provider,
                  api_base_url: form.api_base_url.trim(),
                  api_key: form.api_key,
                }),
              },
            );
      setDiscoveredModels(result.items);
      if (result.items.length === 1 && !form.model_id.trim()) {
        change("model_id", result.items[0]);
        setNotice(`已发现并填入模型 ${result.items[0]}。`);
      } else if (result.items.length) {
        setNotice(
          `已读取 ${result.items.length} 个模型${result.truncated ? "（仅第一页）" : ""}；可从下方列表选择。`,
        );
      } else {
        setNotice("端点返回空模型列表，请手动填写模型 ID。");
      }
    } catch (exc) {
      setDiscoveredModels([]);
      setError(exc instanceof Error ? exc.message : "无法读取模型列表");
    } finally {
      setDiscovering(false);
    }
  }

  async function remove(item: Endpoint) {
    if (
      !window.confirm(`删除受测 API「${item.label}」？历史测量记录仍会保留。`)
    )
      return;
    setBusy(true);
    setError("");
    try {
      await api<void>(
        token,
        `/api/v1/endpoints/${encodeURIComponent(item.id)}`,
        {
          method: "DELETE",
        },
      );
      await onChanged();
      if (editingId === item.id) reset();
      setProbe((current) => {
        const next = { ...current };
        delete next[item.id];
        return next;
      });
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "删除失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">MEASUREMENT API</span>
          <h1>受测 API 设置</h1>
          <p>
            集中设置模型地址、模型 ID 和凭证。保存后，API 与 worker
            会使用同一份配置。
          </p>
        </div>
      </div>
      <div className="api-settings-layout">
        <section className="surface api-endpoint-list">
          <div className="section-head">
            <div>
              <span className="eyebrow">AVAILABLE TARGETS</span>
              <h2>可用端点</h2>
            </div>
            <span className="minor-tag">{endpoints.length} 个</span>
          </div>
          {!endpoints.length && (
            <p className="empty-state">
              尚未配置受测 API。使用右侧表单添加第一个端点。
            </p>
          )}
          {endpoints.map((item) => (
            <article className="api-endpoint-card" key={item.id}>
              <div className="api-endpoint-heading">
                <strong>{item.label}</strong>
                <span className="minor-tag">
                  {item.source === "managed" ? "界面管理" : "文件预设"}
                </span>
              </div>
              <p>{item.model_id}</p>
              <small className="api-endpoint-url">{item.api_base_url}</small>
              <div className="api-endpoint-footer">
                <span
                  className={
                    item.credential_configured
                      ? "credential-ready"
                      : "credential-missing"
                  }
                >
                  {item.credential_configured ? "凭证已配置" : "缺少凭证"}
                </span>
                <div>
                  <button
                    className="button subtle"
                    onClick={() =>
                      navigate(`/new?endpoint=${encodeURIComponent(item.id)}`)
                    }
                  >
                    开始测量
                  </button>
                  <button
                    className="button subtle"
                    disabled={
                      probingId === item.id || !item.credential_configured
                    }
                    onClick={() => void testConnection(item)}
                  >
                    {probingId === item.id ? "检测中…" : "测试连接"}
                  </button>
                  <button
                    className="button subtle"
                    disabled={
                      measuringId === item.id || !item.credential_configured
                    }
                    onClick={() => void measureLatency(item)}
                  >
                    {measuringId === item.id ? "测量中…" : "测参考网络耗时"}
                  </button>
                  {item.source === "managed" && (
                    <>
                      <button
                        className="button subtle"
                        onClick={() => edit(item)}
                      >
                        编辑
                      </button>
                      <button
                        className="button subtle danger"
                        disabled={busy}
                        onClick={() => void remove(item)}
                      >
                        删除
                      </button>
                    </>
                  )}
                </div>
              </div>
              {probe[item.id] && (
                <p className="api-probe-result" role="status">
                  {probe[item.id]}
                </p>
              )}
              {referenceLatency[item.id] != null && (
                <p className="api-probe-result" role="status">
                  参考耗时 {referenceLatency[item.id]} ms（平台到服务商 GET
                  /models 响应首部，包含服务商处理时间）。
                  <button
                    className="text-action"
                    onClick={() =>
                      navigate(
                        `/new?endpoint=${encodeURIComponent(item.id)}&latency_offset=${Math.min(5, referenceLatency[item.id] / 1000).toFixed(3)}`,
                      )
                    }
                  >
                    用作本次延迟偏移 →
                  </button>
                </p>
              )}
            </article>
          ))}
        </section>
        <section className="surface api-settings-form">
          <div className="section-head">
            <div>
              <span className="eyebrow">CONFIGURE TARGET</span>
              <h2>{editingId ? "编辑端点" : "添加端点"}</h2>
            </div>
            {editingId && (
              <button className="text-action" onClick={reset}>
                新建端点
              </button>
            )}
          </div>
          <label htmlFor="api-template">快速选择服务商</label>
          <select
            id="api-template"
            defaultValue=""
            onChange={(event) => {
              const template = templates.find(
                (item) => item.label === event.target.value,
              );
              if (!template) return;
              setForm((current) => ({
                ...current,
                label: current.label || template.label,
                provider: template.provider,
                api_base_url: template.url,
              }));
              setDiscoveredModels([]);
              setError("");
              setNotice("");
            }}
          >
            <option value="">自定义 / 选择常用服务商</option>
            {templates.map((item) => (
              <option key={item.label}>{item.label}</option>
            ))}
          </select>
          <label htmlFor="api-label">显示名称</label>
          <input
            id="api-label"
            value={form.label}
            maxLength={80}
            placeholder="如：DeepSeek 生产环境"
            onChange={(event) => change("label", event.target.value)}
          />
          <label htmlFor="api-provider">接口协议</label>
          <select
            id="api-provider"
            value={form.provider}
            onChange={(event) =>
              change("provider", event.target.value as ApiForm["provider"])
            }
          >
            <option value="OpenAI">OpenAI 兼容</option>
            <option value="Gemini">Gemini 原生</option>
            <option value="Anthropic">Anthropic 原生</option>
          </select>
          {form.provider === "Anthropic" && (
            <p className="form-hint">
              使用原生 Messages API。启用思考时将温度设为 1；手动思考预算至少
              1024，且必须小于最大输出 Token。模型支持范围以服务商返回为准。
            </p>
          )}
          <label htmlFor="api-base-url">API Base URL</label>
          <input
            id="api-base-url"
            type="url"
            value={form.api_base_url}
            placeholder="https://api.example.com/v1"
            onChange={(event) => change("api_base_url", event.target.value)}
          />
          <p className="api-secret-note">
            填写接口根路径，例如以 /v1 结尾；不要包含 /chat/completions。
          </p>
          <label htmlFor="api-model-id">模型 ID</label>
          <select
            aria-label="初版模型快选"
            value=""
            onChange={(event) => change("model_id", event.target.value)}
          >
            <option value="">从初版模型列表快选</option>
            {originalModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
          <div className="api-model-id-row">
            <input
              id="api-model-id"
              value={form.model_id}
              placeholder="服务商提供的模型名称"
              onChange={(event) => change("model_id", event.target.value)}
            />
            <button
              className="button subtle"
              disabled={discovering || busy || !canDiscoverModels}
              onClick={() => void refreshModels()}
            >
              {discovering ? "读取中…" : "自动获取"}
            </button>
          </div>
          {discoveredModels.length > 0 && (
            <div className="api-model-discovery">
              <select
                aria-label="服务商返回的模型"
                value=""
                onChange={(event) => change("model_id", event.target.value)}
              >
                <option value="">选择服务商返回的模型</option>
                {discoveredModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          )}
          <p className="api-secret-note">
            可直接输入或自动获取。新端点需填写地址和 API
            key；编辑已保存端点且地址未变时可直接读取。
          </p>
          <label htmlFor="api-key">API key</label>
          <input
            id="api-key"
            type="password"
            autoComplete="new-password"
            value={form.api_key}
            placeholder={
              editingId ? "留空则保持当前凭证" : "粘贴受测 API 的凭证"
            }
            onChange={(event) => change("api_key", event.target.value)}
          />
          <p className="api-secret-note">
            密钥加密保存在平台数据库中，不回显，也不写入测量任务或报告。
          </p>
          <details className="knobs-section">
            <summary>Tokenizer 与私有端点设置</summary>
            <label htmlFor="api-tokenizer">Tokenizer 选项</label>
            <input
              id="api-tokenizer"
              value={form.tokenizer_option}
              onChange={(event) =>
                change("tokenizer_option", event.target.value)
              }
            />
            <p>
              自定义主机需加入 <code>LLM_TEST_TRUSTED_API_HOSTS</code>
              ；内网地址还需启用 <code>LLM_TEST_ALLOW_PRIVATE_ENDPOINTS=1</code>
              ，然后重启容器。
            </p>
          </details>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          {notice && (
            <p className="form-success" role="status">
              {notice}
            </p>
          )}
          <div className="api-form-actions">
            <button
              className="button primary"
              disabled={
                busy ||
                !form.label.trim() ||
                !form.api_base_url.trim() ||
                !form.model_id.trim() ||
                (!editingId && !form.api_key)
              }
              onClick={() => void save()}
            >
              {busy ? "保存中…" : editingId ? "保存修改" : "添加受测 API"}
            </button>
            {editingId && (
              <button
                className="button subtle"
                onClick={() =>
                  navigate(`/new?endpoint=${encodeURIComponent(editingId)}`)
                }
              >
                使用此端点测量 →
              </button>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
