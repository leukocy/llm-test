import { useId, useState } from "react";
import type { Preset } from "../api";

export function PresetLibrary({
  presets,
  busy,
  onApply,
  onDelete,
}: {
  presets: Preset[];
  busy: boolean;
  onApply: (preset: Preset) => void;
  onDelete: (id: string) => void;
}) {
  const id = useId();
  const [tab, setTab] = useState(0);
  const [filter, setFilter] = useState("");
  const tags = [
    ...new Set(presets.flatMap((preset) => preset.tags || [])),
  ].sort();
  return (
    <details className="preset-library">
      <summary>浏览已保存方案</summary>
      <div className="warehouse-tabs" role="tablist" aria-label="预设浏览方式">
        {["全部预设", "按标签浏览"].map((label, index) => (
          <button
            type="button"
            key={label}
            id={`${id}-tab-${index}`}
            role="tab"
            aria-selected={tab === index}
            aria-controls={`${id}-panel-${index}`}
            tabIndex={tab === index ? 0 : -1}
            className={tab === index ? "active" : ""}
            onClick={() => setTab(index)}
            onKeyDown={(event) => {
              const next =
                event.key === "ArrowRight" || event.key === "ArrowLeft"
                  ? 1 - index
                  : event.key === "Home"
                    ? 0
                    : event.key === "End"
                      ? 1
                      : null;
              if (next !== null) {
                event.preventDefault();
                setTab(next);
                document.getElementById(`${id}-tab-${next}`)?.focus();
              }
            }}
          >
            {label}
          </button>
        ))}
      </div>
      <section
        role="tabpanel"
        id={`${id}-panel-0`}
        aria-labelledby={`${id}-tab-0`}
        hidden={tab !== 0}
      >
        {!presets.length && <p>暂无已保存方案。</p>}
        {presets.map((preset) => (
          <details
            className="preset-library-card"
            key={preset.preset_id}
            data-preset-id={preset.preset_id}
          >
            <summary>{preset.name}</summary>
            <p className="preset-library-description">
              {preset.description || "暂无说明"}
            </p>
            {preset.source_metadata && (
              <details>
                <summary>导入来源说明</summary>
                <p className="preset-library-description">
                  旧文件创建时间：
                  {String(preset.source_metadata.created_at || "未记录")}
                  。这是导入声明，不是测量证据。
                </p>
                {Array.isArray(preset.source_metadata.notes) &&
                  preset.source_metadata.notes
                    .filter((note): note is string => typeof note === "string")
                    .map((note) => (
                      <p className="preset-library-description" key={note}>
                        {note}
                      </p>
                    ))}
                <p className="preset-library-description">
                  去除凭证后的配置指纹：
                  {String(
                    preset.source_metadata.sanitized_config_sha256 || "未记录",
                  )}
                </p>
              </details>
            )}
            <p className="chart-caption">
              测试类型：{preset.test_type} · 端点：{preset.endpoint_id}
            </p>
            <div className="preset-library-tags">
              {(preset.tags || []).map((tag) => (
                <span className="minor-tag" key={tag}>
                  {tag}
                </span>
              ))}
            </div>
            <div className="api-form-actions">
              <button
                type="button"
                className="button subtle"
                disabled={busy}
                aria-label={`应用预设 ${preset.name}`}
                onClick={() => onApply(preset)}
              >
                应用
              </button>
              <button
                type="button"
                className="button subtle danger"
                disabled={busy}
                aria-label={`删除预设 ${preset.name}`}
                onClick={() => onDelete(preset.preset_id)}
              >
                删除
              </button>
            </div>
          </details>
        ))}
      </section>
      <section
        role="tabpanel"
        id={`${id}-panel-1`}
        aria-labelledby={`${id}-tab-1`}
        hidden={tab !== 1}
      >
        <label className="input-label">
          方案标签筛选
          <select
            aria-label="方案标签筛选"
            value={tags.includes(filter) ? filter : ""}
            onChange={(event) => setFilter(event.target.value)}
          >
            <option value="">全部标签</option>
            {tags.map((tag) => (
              <option key={tag}>{tag}</option>
            ))}
          </select>
        </label>
        {!tags.length && <p>暂无带标签的方案。</p>}
        {tags
          .filter((tag) => !tags.includes(filter) || !filter || tag === filter)
          .map((tag) => (
            <section key={tag} aria-label={`预设标签 ${tag}`}>
              <h4>{tag}</h4>
              <div className="api-form-actions">
                {presets
                  .filter((preset) => preset.tags?.includes(tag))
                  .map((preset) => (
                    <button
                      className="button subtle"
                      type="button"
                      key={preset.preset_id}
                      disabled={busy}
                      onClick={() => onApply(preset)}
                    >
                      {preset.name}
                    </button>
                  ))}
              </div>
            </section>
          ))}
      </section>
    </details>
  );
}
