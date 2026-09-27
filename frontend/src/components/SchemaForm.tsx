import { useEffect, useMemo, useState } from "react";
import { fieldLabels } from "../constants";

/** 后端 pydantic JSON Schema 的最小可用子集。 */
export type JsonSchema = {
  type?: string;
  title?: string;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  minItems?: number;
  maxItems?: number;
  enum?: (string | number)[];
  anyOf?: JsonSchema[];
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
  required?: string[];
};

type FieldDef = {
  name: string;
  schema: JsonSchema;
  required: boolean;
  kind:
    | "integer"
    | "number"
    | "boolean"
    | "string"
    | "int-list"
    | "string-list"
    | "enum"
    | "unknown";
};

function resolveNullable(schema: JsonSchema): JsonSchema {
  // pydantic 的可选字段常表达为 anyOf: [<T>, {type: "null"}]
  if (schema.anyOf) {
    const nonNull = schema.anyOf.find((item) => item.type !== "null");
    if (nonNull)
      return { ...nonNull, default: schema.default ?? nonNull.default };
  }
  return schema;
}

function fieldKind(schema: JsonSchema): FieldDef["kind"] {
  const s = resolveNullable(schema);
  if (s.enum) return "enum";
  if (s.type === "integer") return "integer";
  if (s.type === "number") return "number";
  if (s.type === "boolean") return "boolean";
  if (s.type === "string") return "string";
  if (s.type === "array") {
    const itemType = resolveNullable(s.items || {}).type;
    return itemType === "integer" || itemType === "number"
      ? "int-list"
      : "string-list";
  }
  return "unknown";
}

export function schemaDefaults(schema: JsonSchema): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [name, prop] of Object.entries(schema.properties || {})) {
    const s = resolveNullable(prop);
    if (s.default !== undefined && s.default !== null) out[name] = s.default;
  }
  return out;
}

function ListField({
  def,
  value,
  numeric,
  onChange,
}: {
  def: FieldDef;
  value: unknown;
  numeric: boolean;
  onChange: (v: (number | string)[]) => void;
}) {
  const display = Array.isArray(value) ? value.join(", ") : "";
  const [text, setText] = useState(display);
  const [invalid, setInvalid] = useState(false);
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setText(Array.isArray(value) ? value.join(", ") : "");
  }, [value, focused]);

  function commit(raw: string) {
    setText(raw);
    const parts = raw
      .split(/[,，\s]+/)
      .map((item) => item.trim())
      .filter(Boolean);
    if (!parts.length) {
      setInvalid(false);
      onChange([]);
      return;
    }
    if (numeric) {
      const nums = parts.map(Number);
      if (nums.some((n) => !Number.isFinite(n))) {
        setInvalid(true);
        onChange([]);
        return;
      }
      setInvalid(false);
      onChange(nums);
    } else {
      setInvalid(false);
      onChange(parts);
    }
  }

  return (
    <>
      <input
        className={invalid ? "input-invalid" : ""}
        value={text}
        placeholder="逗号分隔，如 1, 4, 8"
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onChange={(event) => commit(event.target.value)}
        aria-label={def.name}
      />
      {invalid && <small className="form-error">存在非数字项，未生效</small>}
    </>
  );
}

export function SchemaForm({
  schema,
  value,
  onChange,
  idPrefix,
}: {
  schema: JsonSchema;
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  idPrefix: string;
}) {
  const defs = useMemo<FieldDef[]>(() => {
    const required = new Set(schema.required || []);
    return Object.entries(schema.properties || {}).map(([name, prop]) => ({
      name,
      schema: resolveNullable(prop),
      required: required.has(name),
      kind: fieldKind(prop),
    }));
  }, [schema]);

  function setField(name: string, v: unknown) {
    onChange({ ...value, [name]: v });
  }

  return (
    <div className="schema-form">
      {defs.map((def) => {
        const id = `${idPrefix}-${def.name}`;
        const label = fieldLabels[def.name] || def.schema.title || def.name;
        const v = value[def.name];
        return (
          <label key={def.name} className="schema-field" htmlFor={id}>
            <span className="input-label">
              {label}
              {def.required && <em className="required-mark">*</em>}
            </span>
            {def.kind === "boolean" ? (
              <input
                id={id}
                type="checkbox"
                checked={Boolean(v)}
                onChange={(event) => setField(def.name, event.target.checked)}
              />
            ) : def.kind === "integer" || def.kind === "number" ? (
              <input
                id={id}
                type="number"
                value={v === undefined || v === null ? "" : Number(v)}
                min={def.schema.minimum}
                max={def.schema.maximum}
                step={def.kind === "integer" ? 1 : "any"}
                onChange={(event) => {
                  const raw = event.target.value;
                  setField(def.name, raw === "" ? undefined : Number(raw));
                }}
              />
            ) : def.kind === "enum" ? (
              <select
                id={id}
                value={String(v ?? "")}
                onChange={(event) => setField(def.name, event.target.value)}
              >
                <option value="">（默认）</option>
                {(def.schema.enum || []).map((item) => (
                  <option key={String(item)} value={String(item)}>
                    {String(item)}
                  </option>
                ))}
              </select>
            ) : def.kind === "int-list" || def.kind === "string-list" ? (
              <ListField
                def={def}
                value={v}
                numeric={def.kind === "int-list"}
                onChange={(list) => setField(def.name, list)}
              />
            ) : (
              <input
                id={id}
                value={v === undefined || v === null ? "" : String(v)}
                maxLength={def.schema.maxLength}
                onChange={(event) => setField(def.name, event.target.value)}
              />
            )}
          </label>
        );
      })}
    </div>
  );
}
