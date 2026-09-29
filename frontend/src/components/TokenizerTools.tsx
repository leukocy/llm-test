import { useEffect, useState } from "react";
import { api } from "../api";

type Catalog = {
  items: {
    name: string;
    available: boolean;
    repo_id: string;
    source: "installed" | "local" | null;
    revision: string | null;
    installation: Installation | null;
  }[];
  matched_name: string | null;
};

type Installation = {
  install_id: string;
  status: string;
  downloaded_bytes: number;
  total_bytes: number;
  completed_files: number;
  total_files: number;
  message: string;
};

const activeStates = new Set([
  "queued",
  "downloading",
  "validating",
  "cancelling",
]);
const stateLabels: Record<string, string> = {
  queued: "排队中",
  downloading: "下载中",
  validating: "校验中",
  cancelling: "取消中",
  cancelled: "已取消",
  completed: "安装完成",
  failed: "下载失败",
};

export function TokenizerTools({
  token,
  modelId,
  selected,
  onSelect,
}: {
  token: string;
  modelId: string;
  selected: string;
  onSelect: (name: string) => void;
}) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [text, setText] = useState("");
  const [mode, setMode] = useState<"local" | "tiktoken" | "characters">(
    "local",
  );
  const [result, setResult] = useState<{
    count: number;
    unit: string;
    method: string;
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [installBusy, setInstallBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const active =
    catalog?.items.some((item) =>
      activeStates.has(item.installation?.status || ""),
    ) || false;

  useEffect(() => {
    setResult(null);
  }, [text, mode, selected]);

  async function refresh() {
    setCatalog(
      await api<Catalog>(
        token,
        `/api/v1/tokenizers?model_id=${encodeURIComponent(modelId)}`,
      ),
    );
    setError("");
  }

  useEffect(() => {
    let alive = true;
    setCatalog(null);
    api<Catalog>(
      token,
      `/api/v1/tokenizers?model_id=${encodeURIComponent(modelId)}`,
    )
      .then((value) => {
        if (alive) setCatalog(value);
      })
      .catch((exc) => {
        if (alive)
          setError(
            exc instanceof Error ? exc.message : "无法读取 Tokenizer 列表",
          );
      });
    return () => {
      alive = false;
    };
  }, [token, modelId]);

  useEffect(() => {
    if (!active) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        if (!document.hidden) {
          const value = await api<Catalog>(
            token,
            `/api/v1/tokenizers?model_id=${encodeURIComponent(modelId)}`,
          );
          if (alive) setCatalog(value);
        }
      } catch (exc) {
        if (alive)
          setError(exc instanceof Error ? exc.message : "下载进度读取失败");
      } finally {
        if (alive) timer = setTimeout(() => void poll(), 1500);
      }
    }
    timer = setTimeout(() => void poll(), 1500);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [active, token, modelId]);

  async function install(names?: string[]) {
    setInstallBusy(true);
    setError("");
    setNotice("");
    try {
      const value = await api<{
        items: Installation[];
        skipped_names: string[];
      }>(token, "/api/v1/tokenizers/installations", {
        method: "POST",
        body: JSON.stringify(names ? { names } : {}),
      });
      setNotice(
        value.items.length
          ? `已提交 ${value.items.length} 项下载，页面会自动更新进度。`
          : "所选 Tokenizer 已安装。",
      );
      await refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "提交下载失败");
    } finally {
      setInstallBusy(false);
    }
  }

  async function cancel(installId: string) {
    setInstallBusy(true);
    setError("");
    try {
      await api(token, `/api/v1/tokenizers/installations/${installId}/cancel`, {
        method: "POST",
      });
      await refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "取消失败");
    } finally {
      setInstallBusy(false);
    }
  }

  async function count() {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const value = await api<{ count: number; unit: string; method: string }>(
        token,
        "/api/v1/tokenizers/count",
        {
          method: "POST",
          body: JSON.stringify({
            text,
            mode,
            ...(mode === "local" ? { name: selected } : {}),
          }),
        },
      );
      setResult(value);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "计数失败");
    } finally {
      setBusy(false);
    }
  }

  const match = catalog?.items.find(
    (item) => item.name === catalog.matched_name,
  );
  return (
    <div className="tokenizer-tools">
      <strong>Tokenizer 快选与文本计数</strong>
      <p>
        {catalog?.matched_name
          ? `模型自动映射：${catalog.matched_name}（${match?.available ? "已安装" : match ? "未安装" : "本地目录，未登记下载源"}）`
          : "当前模型没有自动映射，可从已安装列表选择。"}
      </p>
      <label className="input-label" htmlFor="tokenizer-select">
        本次测量使用的本地 Tokenizer
      </label>
      <select
        id="tokenizer-select"
        value={selected}
        onChange={(event) => onSelect(event.target.value)}
      >
        <option value="">使用端点 / 引擎默认设置</option>
        {catalog?.items.map((item) => (
          <option key={item.name} value={item.name} disabled={!item.available}>
            {item.name} · {item.available ? "已安装" : "未安装"}
          </option>
        ))}
      </select>
      <p className="field-help">
        本地 Tokenizer 仅加载已安装文件。下载仅取公开仓库的 Tokenizer
        文件，固定版本并离线校验后才启用。
      </p>
      <div className="tokenizer-count-actions">
        {match?.available && (
          <button
            className="button subtle"
            onClick={() => onSelect(match.name)}
          >
            使用自动匹配
          </button>
        )}
        {match && !match.available && (
          <button
            className="button subtle"
            disabled={
              installBusy || activeStates.has(match.installation?.status || "")
            }
            onClick={() => void install([match.name])}
          >
            下载匹配项
          </button>
        )}
        <button
          className="button subtle"
          disabled={
            installBusy ||
            !catalog ||
            !catalog.items.some(
              (item) =>
                !item.available &&
                !activeStates.has(item.installation?.status || ""),
            )
          }
          onClick={() => void install()}
        >
          下载全部缺失项（
          {catalog?.items.filter((item) => !item.available).length || 0}）
        </button>
        <button
          className="button subtle"
          disabled={installBusy}
          onClick={() =>
            void refresh().catch((exc: unknown) =>
              setError(exc instanceof Error ? exc.message : "刷新失败"),
            )
          }
        >
          刷新状态
        </button>
      </div>
      <p className="field-help">
        下载保存在持久化缓存，容器重启后可继续使用。测量优先执行，下载排队等待；公开下载源需要
        worker 能访问 Hugging Face。
      </p>
      {notice && <p role="status">{notice}</p>}
      <details className="tokenizer-inventory">
        <summary>
          本地 Tokenizer 与下载进度{active ? " · 有下载任务" : ""}
        </summary>
        <ul>
          {catalog?.items.map((item) => {
            const task = item.installation;
            const pending = activeStates.has(task?.status || "");
            return (
              <li key={item.name}>
                <div className="tokenizer-item-info">
                  <strong>{item.name}</strong>
                  <small>{item.repo_id}</small>
                  <small>
                    {item.available
                      ? `${item.source === "installed" ? "已安装 · 校验通过" : "已有本地文件"}${item.revision ? ` · ${item.revision.slice(0, 12)}` : ""}`
                      : task
                        ? stateLabels[task.status] || task.status
                        : "未安装"}
                  </small>
                  {task && !item.available && (
                    <small
                      className={task.status === "failed" ? "form-error" : ""}
                    >
                      {task.message}
                    </small>
                  )}
                  {task && pending && (
                    <>
                      <progress
                        aria-label={`${item.name} 下载进度`}
                        max={task.total_bytes || 1}
                        value={
                          task.total_bytes ? task.downloaded_bytes : undefined
                        }
                      />
                      <small>
                        {task.completed_files} / {task.total_files || "—"}{" "}
                        个文件 · {(task.downloaded_bytes / 1048576).toFixed(1)}{" "}
                        /{" "}
                        {task.total_bytes
                          ? (task.total_bytes / 1048576).toFixed(1)
                          : "—"}{" "}
                        MiB
                      </small>
                    </>
                  )}
                </div>
                {item.available ? (
                  <button
                    className="button subtle"
                    onClick={() => onSelect(item.name)}
                  >
                    用于测量
                  </button>
                ) : pending && task ? (
                  <button
                    className="button subtle"
                    disabled={installBusy || task.status === "cancelling"}
                    onClick={() => void cancel(task.install_id)}
                  >
                    {task.status === "cancelling" ? "取消中…" : "取消下载"}
                  </button>
                ) : (
                  <button
                    className="button subtle"
                    disabled={installBusy}
                    onClick={() => void install([item.name])}
                  >
                    {task?.status === "failed" ? "重新下载" : "下载"}
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      </details>
      <label className="input-label" htmlFor="tokenizer-count-text">
        待计数文本
      </label>
      <p className="field-help">
        参考编码首次使用可能下载其内置词表，计数可能与目标模型不同。
      </p>
      <textarea
        id="tokenizer-count-text"
        value={text}
        maxLength={10000}
        rows={3}
        onChange={(event) => setText(event.target.value)}
      />
      <div className="tokenizer-count-actions">
        <select
          aria-label="计数方式"
          value={mode}
          onChange={(event) => setMode(event.target.value as typeof mode)}
        >
          <option value="local">已安装的本地 Tokenizer</option>
          <option value="tiktoken">cl100k_base 参考计数</option>
          <option value="characters">Unicode 字符数</option>
        </select>
        <button
          className="button subtle"
          disabled={busy || !text || (mode === "local" && !selected)}
          onClick={() => void count()}
        >
          {busy ? "计数中…" : "计算"}
        </button>
      </div>
      {result && (
        <p role="status">
          <strong>{result.count}</strong> {result.unit} · {result.method}
        </p>
      )}
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
