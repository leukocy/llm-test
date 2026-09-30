import { useState } from "react";
import { useOfflineTool } from "../../hooks/useOfflineTool";
import {
  ToolError,
  ToolRecord,
  SAMPLE_TEXT,
  type Catalog,
  type PerturbResult,
} from "./Shared";

export function PerturbDemo({
  token,
  catalog,
  catalogError,
  onRetry,
}: {
  token: string;
  catalog: Catalog | null;
  catalogError: string;
  onRetry: () => void;
}) {
  const [text, setText] = useState(SAMPLE_TEXT);
  const [kind, setKind] = useState("synonym");
  const [seed, setSeed] = useState("42");
  const tool = useOfflineTool<PerturbResult>(token, "/api/v1/advanced/perturb");
  const capability = catalog?.perturbations.find((item) => item.id === kind);
  const validSeed = /^\d+$/.test(seed) && Number(seed) <= 2 ** 32 - 1;
  const result = tool.result;
  return (
    <section className="surface">
      <div className="section-head">
        <div>
          <span className="section-index">03</span>
          <h2>文本扰动演示</h2>
        </div>
        <span className="minor-tag">十种原始选项 · 独立随机种子</span>
      </div>
      <label className="input-label" htmlFor="perturb-text">
        原始文本
      </label>
      <textarea
        id="perturb-text"
        className="json-editor"
        value={text}
        maxLength={10000}
        onChange={(event) => {
          tool.reset();
          setText(event.target.value);
        }}
      />
      <div className="schema-form">
        <label className="schema-field">
          <span className="input-label">扰动类型</span>
          <select
            aria-label="扰动类型"
            value={kind}
            disabled={!catalog}
            onChange={(event) => {
              tool.reset();
              setKind(event.target.value);
            }}
          >
            {catalog ? (
              catalog.perturbations.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label} · {item.id}
                  {!item.supported ? "（未实现）" : ""}
                </option>
              ))
            ) : (
              <option value="synonym">加载工具清单…</option>
            )}
          </select>
        </label>
        <label className="schema-field">
          <span className="input-label">随机种子</span>
          <input
            type="number"
            min={0}
            max={2 ** 32 - 1}
            step={1}
            value={seed}
            onChange={(event) => {
              tool.reset();
              setSeed(event.target.value);
            }}
          />
        </label>
      </div>
      {capability && <p className="advanced-hint">{capability.description}</p>}
      <ToolError
        text={
          catalogError ||
          tool.error ||
          (!validSeed ? "种子必须是 0～4294967295 的整数。" : "")
        }
      />
      {catalogError && (
        <button className="button secondary" onClick={onRetry}>
          重试加载清单
        </button>
      )}
      <div className="form-footer">
        <span>变化不等于语义保持，也不等于模型鲁棒性。</span>
        <button
          className="button primary"
          disabled={!text.trim() || !catalog || !validSeed || tool.busy}
          onClick={() =>
            void tool.run({ text, perturbation_type: kind, seed: Number(seed) })
          }
        >
          {tool.busy ? "生成中…" : "应用扰动"}
        </button>
      </div>
      {result && (
        <div className="advanced-result" aria-live="polite">
          <div className="gate-result">
            <strong>
              {result.status === "unsupported"
                ? "未实现此规则 · 保留原文"
                : result.status === "unchanged"
                  ? "未产生变化 · 保留原文"
                  : "已生成扰动文本"}
            </strong>
            <p>{result.details}</p>
          </div>
          <div className="advanced-text-pair">
            <div>
              <h3>原始文本</h3>
              <pre>{result.original_text}</pre>
            </div>
            <div>
              <h3>扰动后文本</h3>
              <pre>{result.perturbed_text}</pre>
            </div>
          </div>
          <ToolRecord result={result} name="text-perturbation" />
        </div>
      )}
    </section>
  );
}
