import { useEffect, useRef, useState } from "react";

type Sample = Record<string, string>;

function parseSamples(text: string): Sample[] {
  const decoded: unknown = JSON.parse(text);
  const samples =
    decoded && !Array.isArray(decoded) && typeof decoded === "object"
      ? (decoded as { samples?: unknown }).samples
      : decoded;
  if (!Array.isArray(samples) || !samples.length || samples.length > 50)
    throw new Error("需要 1–50 个样本");
  for (const [index, row] of samples.entries()) {
    if (
      !row ||
      typeof row !== "object" ||
      Array.isArray(row) ||
      Object.values(row).some((value) => typeof value !== "string")
    )
      throw new Error(`第 ${index + 1} 个样本必须是文本字段对象`);
    if (typeof row.question !== "string" || !row.question.trim())
      throw new Error(`第 ${index + 1} 个样本缺少问题`);
    if (row.question.length > 50000 || (row.correct_answer || "").length > 5000)
      throw new Error(`第 ${index + 1} 个样本超出文本长度限制`);
  }
  return samples;
}

export function RobustnessSamplesEditor({
  value,
  onChange,
}: {
  value: unknown;
  onChange: (samples: Sample[]) => void;
}) {
  const samples: Sample[] = Array.isArray(value) ? value : [];
  const [mode, setMode] = useState<"form" | "json">("form");
  const [text, setText] = useState(() => JSON.stringify(samples, null, 2));
  const [error, setError] = useState("");
  const [fileError, setFileError] = useState("");
  const dirty = useRef(false);
  const owned = useRef<unknown>(undefined);
  const latest = useRef(samples);
  const lastValid = useRef(samples);
  latest.current = samples;
  useEffect(() => {
    if (value !== owned.current) {
      dirty.current = false;
      setError("");
      lastValid.current = Array.isArray(value) ? value : [];
    }
    if (!dirty.current)
      setText(JSON.stringify(Array.isArray(value) ? value : [], null, 2));
  }, [value]);
  function publish(next: Sample[]) {
    owned.current = next;
    if (next.length && next.every((sample) => sample.question?.trim()))
      lastValid.current = next;
    onChange(next);
  }
  function commit(text: string) {
    setText(text);
    dirty.current = true;
    try {
      const next = parseSamples(text);
      setError("");
      publish(next);
    } catch (exc) {
      setError(String(exc));
      publish([]); // Never submit a previous valid sample set while the draft is invalid.
    }
  }
  function update(index: number, field: string, text: string) {
    publish(
      latest.current.map((sample, position) =>
        position === index ? { ...sample, [field]: text } : sample,
      ),
    );
  }
  async function importFile(file?: File) {
    if (!file) return;
    setFileError("");
    try {
      if (file.size > 1024 * 1024) throw new Error("样本文件最多 1 MiB");
      const next = parseSamples((await file.text()).replace(/^\uFEFF/, ""));
      dirty.current = false;
      setError("");
      setText(JSON.stringify(next, null, 2));
      publish(next);
    } catch (exc) {
      setFileError(`未导入：${String(exc)}`);
    }
  }
  function exportFile() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(samples, null, 2) + "\n"], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "robustness-samples.json";
    link.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section className="robustness-samples-editor">
      <div className="section-head">
        <h3>鲁棒性样本 · {samples.length} 条</h3>
        <div className="api-form-actions">
          <button
            type="button"
            className="button subtle"
            disabled={mode === "json" && !!error}
            onClick={() => {
              dirty.current = false;
              setMode("form");
              setError("");
            }}
          >
            逐条编辑
          </button>
          <button
            type="button"
            className="button subtle"
            onClick={() => {
              if (!dirty.current) setText(JSON.stringify(samples, null, 2));
              setMode("json");
            }}
          >
            JSON 编辑
          </button>
        </div>
      </div>
      <p className="field-help">
        最多 50 条，问题最长 50,000 字符，标准答案最长 5,000
        字符；导入支持样本数组或含 samples
        字段的配置对象。每条样本的其他文本字段会保留。
      </p>
      <div className="api-form-actions">
        <label className="input-label">
          导入样本 JSON
          <input
            aria-label="导入鲁棒性样本 JSON"
            type="file"
            accept=".json,application/json"
            onChange={(event) => {
              void importFile(event.target.files?.[0]);
              event.currentTarget.value = "";
            }}
          />
        </label>
        <button
          type="button"
          className="button subtle"
          disabled={!samples.length || !!error}
          onClick={exportFile}
        >
          导出样本 JSON
        </button>
      </div>
      {fileError && (
        <p className="form-error" role="alert">
          {fileError}，原样本已保留。
        </p>
      )}
      {mode === "json" ? (
        <label className="schema-field case-form-wide">
          样本 JSON
          <textarea
            className="json-editor"
            spellCheck={false}
            value={text}
            maxLength={1024 * 1024}
            aria-label="鲁棒性 samples JSON"
            onChange={(event) => commit(event.target.value)}
          />
          {error && (
            <p className="form-error" role="alert">
              {error}；当前不能保存或提交。
            </p>
          )}
          {error && (
            <button
              type="button"
              className="button subtle"
              onClick={() => {
                const restored = lastValid.current;
                dirty.current = false;
                setError("");
                setText(JSON.stringify(restored, null, 2));
                publish(restored);
                setMode("form");
              }}
            >
              撤销 JSON 修改
            </button>
          )}
        </label>
      ) : (
        <>
          {!samples.length && (
            <p className="chart-caption">添加一条样本开始填写。</p>
          )}
          {samples.map((sample, index) => (
            <fieldset key={index} className="robustness-sample-row">
              <legend>样本 {index + 1}</legend>
              <label>
                问题
                <textarea
                  aria-label={`鲁棒性问题 ${index + 1}`}
                  rows={3}
                  maxLength={50000}
                  value={sample.question || ""}
                  onChange={(event) =>
                    update(index, "question", event.target.value)
                  }
                />
              </label>
              <label>
                标准答案
                <textarea
                  aria-label={`鲁棒性标准答案 ${index + 1}`}
                  rows={2}
                  maxLength={5000}
                  value={sample.correct_answer || ""}
                  onChange={(event) =>
                    update(index, "correct_answer", event.target.value)
                  }
                />
              </label>
              <button
                type="button"
                className="text-button"
                aria-label={`删除鲁棒性样本 ${index + 1}`}
                onClick={() =>
                  publish(
                    latest.current.filter((_, position) => position !== index),
                  )
                }
              >
                删除样本
              </button>
            </fieldset>
          ))}
          <button
            type="button"
            className="button subtle"
            disabled={samples.length >= 50}
            onClick={() =>
              publish([...latest.current, { question: "", correct_answer: "" }])
            }
          >
            添加鲁棒性样本
          </button>
        </>
      )}
    </section>
  );
}
