import { useEffect, useRef, useState } from "react";

type Kind = "string" | "number" | "boolean" | "json" | "null";
type Row = {
  id: number;
  name: string;
  location: string;
  kind: Kind;
  text: string;
  value: unknown;
  valid: boolean;
};

function kindOf(value: unknown): Kind {
  if (value === null) return "null";
  if (typeof value === "number") return "number";
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "object") return "json";
  return "string";
}
function finiteJson(value: unknown): boolean {
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(finiteJson);
  if (value && typeof value === "object")
    return Object.values(value).every(finiteJson);
  return true;
}
function serialize(rows: Row[]) {
  return rows.map((row) => ({
    name: row.name.trim(),
    location: row.location,
    ...(row.valid ? { value: row.value } : {}),
  }));
}

export function CustomParametersEditor({
  value,
  onChange,
}: {
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const nextId = useRef(0);
  function decode(value: unknown): Row[] {
    return (Array.isArray(value) ? value : []).map((raw) => {
      const kind = kindOf(raw.value);
      return {
        id: nextId.current++,
        name: String(raw.name || ""),
        location: raw.location || "top_level",
        kind,
        text:
          kind === "json"
            ? JSON.stringify(raw.value, null, 2)
            : kind === "null"
              ? ""
              : String(raw.value ?? ""),
        value: raw.value,
        valid: Object.hasOwn(raw, "value"),
      };
    });
  }
  const [rows, setRows] = useState<Row[]>(() => decode(value));
  const latest = useRef(rows);
  useEffect(() => {
    if (
      JSON.stringify(serialize(latest.current)) !== JSON.stringify(value || [])
    ) {
      const next = decode(value);
      latest.current = next;
      setRows(next);
    }
    // The parent is authoritative only when its serialized value changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  function publish(next: Row[]) {
    latest.current = next;
    setRows(next);
    onChange(next.length ? serialize(next) : null);
  }
  function edit(id: number, patch: Partial<Row>) {
    publish(
      latest.current.map((row) => (row.id === id ? { ...row, ...patch } : row)),
    );
  }
  function changeKind(row: Row, kind: Kind) {
    const value =
      kind === "number"
        ? 0
        : kind === "boolean"
          ? false
          : kind === "json"
            ? {}
            : kind === "null"
              ? null
              : "";
    edit(row.id, {
      kind,
      value,
      valid: true,
      text: kind === "json" ? "{}" : kind === "null" ? "" : String(value),
    });
  }
  function changeText(row: Row, text: string) {
    try {
      const value =
        row.kind === "json"
          ? JSON.parse(text)
          : row.kind === "number"
            ? Number(text)
            : text;
      if ((row.kind === "number" && !text.trim()) || !finiteJson(value))
        throw new Error("invalid value");
      edit(row.id, { text, value, valid: true });
    } catch {
      // Missing required value makes preview/save/submit fail; retain the user's draft.
      edit(row.id, { text, value: undefined, valid: false });
    }
  }
  return (
    <section className="custom-parameters-editor">
      <div className="section-head">
        <h3>额外 API 参数</h3>
        <button
          type="button"
          className="button subtle"
          disabled={rows.length >= 20}
          onClick={() =>
            publish([
              ...latest.current,
              {
                id: nextId.current++,
                name: "",
                location: "top_level",
                kind: "string",
                text: "",
                value: "",
                valid: true,
              },
            ])
          }
        >
          添加参数
        </button>
      </div>
      <p className="field-help">
        设置提供方支持的其他字段。温度、输出长度和思考设置使用专用控件；最多 20
        项，每项值最多 16 KiB。
      </p>
      {!rows.length && <p className="chart-caption">未设置额外参数。</p>}
      {rows.map((row, index) => (
        <fieldset key={row.id} className="custom-parameter-row">
          <legend>参数 {index + 1}</legend>
          <div className="custom-parameter-fields">
            <label>
              参数名
              <input
                aria-label={`额外参数名 ${index + 1}`}
                value={row.name}
                maxLength={64}
                placeholder="如 top_p"
                onChange={(event) => edit(row.id, { name: event.target.value })}
              />
            </label>
            <label>
              发送位置
              <select
                aria-label={`额外参数位置 ${index + 1}`}
                value={row.location}
                onChange={(event) =>
                  edit(row.id, { location: event.target.value })
                }
              >
                <option value="top_level">请求参数</option>
                <option value="extra_body">扩展请求体</option>
              </select>
            </label>
            <label>
              值类型
              <select
                aria-label={`额外参数类型 ${index + 1}`}
                value={row.kind}
                onChange={(event) =>
                  changeKind(row, event.target.value as Kind)
                }
              >
                <option value="string">文本</option>
                <option value="number">数字</option>
                <option value="boolean">开关</option>
                <option value="json">JSON 对象或数组</option>
                <option value="null">空值</option>
              </select>
            </label>
            <label>
              参数值
              {row.kind === "boolean" ? (
                <input
                  type="checkbox"
                  aria-label={`额外参数值 ${index + 1}`}
                  checked={row.value === true}
                  onChange={(event) =>
                    edit(row.id, { value: event.target.checked, valid: true })
                  }
                />
              ) : row.kind === "null" ? (
                <span>空值（null）</span>
              ) : row.kind === "json" ? (
                <textarea
                  aria-label={`额外参数值 ${index + 1}`}
                  rows={3}
                  maxLength={16384}
                  value={row.text}
                  onChange={(event) => changeText(row, event.target.value)}
                />
              ) : (
                <input
                  aria-label={`额外参数值 ${index + 1}`}
                  value={row.text}
                  inputMode={row.kind === "number" ? "decimal" : "text"}
                  maxLength={16384}
                  onChange={(event) => changeText(row, event.target.value)}
                />
              )}
            </label>
          </div>
          {!row.valid && (
            <p className="form-error" role="alert">
              请输入有效的{row.kind === "number" ? "有限数字" : "JSON 值"}
              ；当前不能保存或提交。
            </p>
          )}
          <button
            type="button"
            className="text-button"
            aria-label={`删除额外参数 ${index + 1}`}
            onClick={() =>
              publish(latest.current.filter((item) => item.id !== row.id))
            }
          >
            删除参数
          </button>
        </fieldset>
      ))}
    </section>
  );
}
