import { useState } from "react";
import { useOfflineTool } from "../../hooks/useOfflineTool";
import { MetricCard } from "../../components";
import { formatNumber } from "../../constants";
import {
  DatasetPicker,
  ToolError,
  ToolRecord,
  judgment,
  SAMPLE_RESPONSE,
  type ParseResult,
} from "./Shared";

export function ParserDemo({ token }: { token: string }) {
  const [response, setResponse] = useState(SAMPLE_RESPONSE);
  const [mode, setMode] = useState("dataset");
  const [dataset, setDataset] = useState("auto");
  const [answerType, setAnswerType] = useState("number");
  const [expected, setExpected] = useState("42");
  const tool = useOfflineTool<ParseResult>(token, "/api/v1/advanced/parse");
  const result = tool.result;
  return (
    <section className="surface">
      <div className="section-head">
        <div>
          <span className="section-index">01</span>
          <h2>智能答案解析</h2>
        </div>
        <span className="minor-tag">八种数据集 · 六种答案类型</span>
      </div>
      <label className="input-label" htmlFor="parse-response">
        模型响应文本
      </label>
      <textarea
        id="parse-response"
        className="json-editor"
        maxLength={50000}
        value={response}
        onChange={(event) => {
          tool.reset();
          setResponse(event.target.value);
        }}
      />
      <div className="schema-form">
        <label className="schema-field">
          <span className="input-label">解析方式</span>
          <select
            aria-label="解析方式"
            value={mode}
            onChange={(event) => {
              tool.reset();
              setMode(event.target.value);
            }}
          >
            <option value="dataset">按数据集解析（初版）</option>
            <option value="answer_type">按答案类型解析</option>
          </select>
        </label>
        {mode === "dataset" ? (
          <DatasetPicker
            id="parse-dataset"
            value={dataset}
            onChange={(value) => {
              tool.reset();
              setDataset(value);
            }}
          />
        ) : (
          <label className="schema-field">
            <span className="input-label">答案类型</span>
            <select
              aria-label="答案类型"
              value={answerType}
              onChange={(event) => {
                tool.reset();
                setAnswerType(event.target.value);
              }}
            >
              {Object.entries({
                number: "数值",
                choice: "选择题",
                text: "文本",
                boolean: "是/否",
                code: "代码",
                math: "数学表达式",
              }).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="schema-field case-form-wide">
          <span className="input-label">参考答案（可选）</span>
          <input
            maxLength={10000}
            value={expected}
            onChange={(event) => {
              tool.reset();
              setExpected(event.target.value);
            }}
            placeholder="留空仅解析，不判分"
          />
        </label>
      </div>
      <div className="form-footer">
        <span>提取规则不使用参考答案提示。</span>
        <button
          className="button primary"
          disabled={!response.trim() || tool.busy}
          onClick={() =>
            void tool.run({
              response,
              ...(mode === "dataset"
                ? { dataset_type: dataset }
                : { answer_type: answerType }),
              expected_answer: expected.trim() ? expected : null,
            })
          }
        >
          {tool.busy ? "解析中…" : "解析答案"}
        </button>
      </div>
      <ToolError text={tool.error} />
      {result && (
        <div className="advanced-result" aria-live="polite">
          <div className="metric-grid">
            <MetricCard
              label="解析器"
              value={result.method}
              note={`类型：${result.answer_type || "文本"}`}
            />
            <MetricCard
              label="参考答案判定"
              value={
                result.is_correct === undefined
                  ? "未提供参考"
                  : judgment(result.is_correct)
              }
              note={result.comparison_method || "仅提取答案"}
            />
          </div>
          <div className="gate-result">
            <strong>提取答案</strong>
            <pre>{result.extracted_answer || "（未提取到）"}</pre>
            <p>
              规范化值：<code>{result.normalized_value ?? "未能规范化"}</code>
            </p>
            {result.confidence != null && (
              <p>
                规则匹配强度 {formatNumber(result.confidence, 2)}
                （不是正确概率）
              </p>
            )}
            {result.error && <p className="text-danger">{result.error}</p>}
          </div>
          <ToolRecord result={result} name="answer-parse" />
        </div>
      )}
    </section>
  );
}
