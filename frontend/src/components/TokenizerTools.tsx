import { useEffect, useState } from "react";
import { api } from "../api";

type Catalog = {
  items: { name: string; available: boolean }[];
  matched_name: string | null;
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

  useEffect(() => {
    let alive = true;
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
          ? `模型自动映射：${catalog.matched_name}（${match?.available ? "已安装" : "未安装"}）`
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
        仅加载平台已登记的本地文件；计数工具不会下载模型或执行远程代码。
      </p>
      <label className="input-label" htmlFor="tokenizer-count-text">
        待计数文本
      </label>
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
