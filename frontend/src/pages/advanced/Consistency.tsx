import { useState } from "react";
import { useOfflineTool } from "../../hooks/useOfflineTool";
import { MetricCard } from "../../components";
import {
  DatasetPicker,
  ToolError,
  ToolRecord,
  judgment,
  percent,
  type ConsistencyResult,
} from "./Shared";

export function ConsistencyDemo({ token }: { token: string }) {
  const [question, setQuestion] = useState("What is 15 + 27?");
  const [runs, setRuns] = useState(5);
  const [threshold, setThreshold] = useState(0.8);
  const [dataset, setDataset] = useState("auto");
  const [expected, setExpected] = useState("42");
  const [responses, setResponses] = useState("");
  const tool = useOfflineTool<ConsistencyResult>(
    token,
    "/api/v1/advanced/consistency",
  );
  let values: string[] = [];
  let inputError = "";
  if (responses.trim()) {
    try {
      const parsed: unknown = JSON.parse(responses);
      if (
        !Array.isArray(parsed) ||
        !parsed.every((item) => typeof item === "string")
      )
        inputError = '请输入 JSON 字符串数组，例如 ["42", "43"]。';
      else if (parsed.length !== runs)
        inputError = `当前 ${parsed.length} 条回答，需要 ${runs} 条；空回答用 \"\" 记录。`;
      else values = parsed;
    } catch {
      inputError = "回答格式不是有效 JSON；请使用字符串数组。";
    }
  }
  const result = tool.result;
  return (
    <section className="surface">
      <div className="section-head">
        <div>
          <span className="section-index">02</span>
          <h2>回答一致性分析</h2>
        </div>
        <span className="minor-tag">已有回答 · 单问题描述统计</span>
      </div>
      <p className="advanced-intro">
        粘贴同一问题的多次回答。原版重复次数与阈值保留；这里分析提供的文本，来源及采样独立性未经核验。
      </p>
      <div className="schema-form">
        <label className="schema-field case-form-wide">
          <span className="input-label">测试问题</span>
          <input
            maxLength={10000}
            value={question}
            onChange={(event) => {
              tool.reset();
              setQuestion(event.target.value);
            }}
          />
        </label>
        <label className="schema-field">
          <span className="input-label">重复次数：{runs}</span>
          <input
            type="range"
            min={2}
            max={10}
            step={1}
            value={runs}
            onChange={(event) => {
              tool.reset();
              setRuns(Number(event.target.value));
            }}
          />
        </label>
        <label className="schema-field">
          <span className="input-label">
            稳定性阈值：{threshold.toFixed(2)}
          </span>
          <input
            type="range"
            min={0.5}
            max={1}
            step={0.05}
            value={threshold}
            onChange={(event) => {
              tool.reset();
              setThreshold(Number(event.target.value));
            }}
          />
        </label>
        <DatasetPicker
          id="consistency-dataset"
          value={dataset}
          onChange={(value) => {
            tool.reset();
            setDataset(value);
          }}
        />
        <label className="schema-field">
          <span className="input-label">参考答案（可选）</span>
          <input
            maxLength={10000}
            value={expected}
            onChange={(event) => {
              tool.reset();
              setExpected(event.target.value);
            }}
          />
        </label>
      </div>
      <div className="advanced-input-head">
        <label className="input-label" htmlFor="consistency-responses">
          重复回答（JSON 字符串数组）
        </label>
        <button
          className="button secondary"
          onClick={() => {
            tool.reset();
            setResponses(
              JSON.stringify(
                Array.from({ length: runs }, (_, index) =>
                  index === runs - 1 ? "43" : "42",
                ),
                null,
                2,
              ),
            );
          }}
        >
          载入示例
        </button>
      </div>
      <textarea
        id="consistency-responses"
        className="json-editor"
        maxLength={80000}
        value={responses}
        placeholder={'["42", "42", "42", "42", "43"]'}
        onChange={(event) => {
          tool.reset();
          setResponses(event.target.value);
        }}
      />
      <p className="advanced-hint">
        示例为合成数据。最多 10 条，每条 10,000 字符，总计 50,000
        字符；空回答计入分母。
      </p>
      <ToolError text={inputError || tool.error} />
      <div className="form-footer">
        <span>一致性与答案正确性分别报告。</span>
        <button
          className="button primary"
          disabled={
            !question.trim() || !values.length || !!inputError || tool.busy
          }
          onClick={() =>
            void tool.run({
              question,
              runs,
              threshold,
              dataset_type: dataset,
              expected_answer: expected.trim() ? expected : null,
              responses: values,
            })
          }
        >
          {tool.busy ? "分析中…" : "分析回答一致性"}
        </button>
      </div>
      {result && (
        <div className="advanced-result" aria-live="polite">
          <div className="metric-grid">
            <MetricCard
              label="一致率"
              value={percent(result.consistency_rate)}
              note={`最多相同答案 / 全部 ${result.total} 次`}
              accent
            />
            <MetricCard
              label="阈值判定"
              value={
                result.stable == null
                  ? "未能判定"
                  : result.stable
                    ? "达到阈值"
                    : result.tied_majority
                      ? "并列最多"
                      : "未达阈值"
              }
              note={`阈值 ${percent(Number(result.input.threshold))}`}
            />
            <MetricCard
              label="可分组 / 未解析"
              value={`${result.parsed} / ${result.unparsed}`}
              note="未解析回答仍在分母中"
            />
            <MetricCard
              label="参考匹配率"
              value={
                result.reference_matched == null
                  ? "未提供参考"
                  : percent(result.reference_match_rate)
              }
              note={
                result.reference_undetermined == null
                  ? "不进行判分"
                  : `匹配 ${result.reference_matched} 次 · 未判定 ${result.reference_undetermined} 次`
              }
            />
          </div>
          <h3>答案分组</h3>
          {result.groups.length ? (
            <div className="advanced-groups">
              {result.groups.map((group) => (
                <div key={group.normalized_key}>
                  <pre>{group.answer}</pre>
                  <span>
                    {group.count} / {result.total}
                  </span>
                  <progress
                    max={result.total}
                    value={group.count}
                    aria-label={`答案 ${group.answer} 出现次数`}
                  />
                </div>
              ))}
            </div>
          ) : (
            <p>全部回答无法解析，没有有效分组。</p>
          )}
          <details>
            <summary>逐条回答与解析结果（{result.total} 条）</summary>
            <div className="advanced-observations">
              {result.results.map((item) => (
                <article key={item.index}>
                  <strong>
                    #{item.index} ·{" "}
                    {item.group_key == null ? "未解析" : "已分组"}
                    {result.reference_matched != null
                      ? ` · ${judgment(item.is_correct)}`
                      : ""}
                  </strong>
                  <pre>{item.response || "（空回答）"}</pre>
                  <p>
                    提取：{item.extracted_answer || "无"}
                    {item.error ? ` · ${item.error}` : ""}
                  </p>
                </article>
              ))}
            </div>
          </details>
          <ToolRecord result={result} name="answer-consistency" />
        </div>
      )}
    </section>
  );
}
