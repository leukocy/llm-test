import { ReasoningAnalysis } from "../../components/ReasoningAnalysis";
import { useState } from "react";
import { useOfflineTool } from "../../hooks/useOfflineTool";
import { MetricCard } from "../../components";
import { formatNumber } from "../../constants";
import {
  ToolError,
  ToolRecord,
  DIM_LABELS,
  judgment,
  type ReasoningResult,
} from "./Shared";

export function ReasoningDemo({ token }: { token: string }) {
  const [question, setQuestion] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [finalAnswer, setFinalAnswer] = useState("");
  const [correctAnswer, setCorrectAnswer] = useState("");
  const tool = useOfflineTool<ReasoningResult>(
    token,
    "/api/v1/advanced/reasoning",
  );
  const result = tool.result;
  return (
    <section className="surface">
      <div className="section-head">
        <div>
          <span className="section-index">04</span>
          <h2>推理过程质量评估</h2>
        </div>
        <span className="minor-tag">规则法 · 五个维度</span>
      </div>
      <div className="schema-form">
        <label className="schema-field case-form-wide">
          <span className="input-label">原始问题</span>
          <input
            maxLength={20000}
            value={question}
            onChange={(event) => {
              tool.reset();
              setQuestion(event.target.value);
            }}
            placeholder="如：小明有 3 个苹果，又买了 5 个，一共几个？"
          />
        </label>
        <label className="schema-field case-form-wide">
          <span className="input-label">推理过程（可空）</span>
          <textarea
            className="json-editor"
            maxLength={100000}
            value={reasoning}
            onChange={(event) => {
              tool.reset();
              setReasoning(event.target.value);
            }}
            placeholder="粘贴模型输出的推理过程"
          />
        </label>
        <label className="schema-field">
          <span className="input-label">模型最终答案</span>
          <input
            maxLength={20000}
            value={finalAnswer}
            onChange={(event) => {
              tool.reset();
              setFinalAnswer(event.target.value);
            }}
          />
        </label>
        <label className="schema-field">
          <span className="input-label">参考答案（可选）</span>
          <input
            maxLength={20000}
            value={correctAnswer}
            onChange={(event) => {
              tool.reset();
              setCorrectAnswer(event.target.value);
            }}
          />
        </label>
      </div>
      <div className="form-footer">
        <span>规则得分不代表经校准的推理能力。</span>
        <button
          className="button primary"
          disabled={!question.trim() || tool.busy}
          onClick={() =>
            void tool.run({
              question,
              reasoning,
              final_answer: finalAnswer,
              correct_answer: correctAnswer,
            })
          }
        >
          {tool.busy ? "评估中…" : "评估推理过程"}
        </button>
      </div>
      <ToolError text={tool.error} />
      {result && (
        <div className="advanced-result" aria-live="polite">
          <div className="metric-grid">
            {Object.entries(DIM_LABELS).map(([dim, label]) => (
              <MetricCard
                key={dim}
                label={label}
                value={
                  result.quality_score[dim] == null
                    ? "未能判定"
                    : formatNumber(result.quality_score[dim], 1)
                }
                note="/ 10 · 启发式评分"
              />
            ))}
            <MetricCard
              label="最终答案参考判定"
              value={judgment(result.final_answer_correct)}
              note="文本或有界数学规则比较"
            />
          </div>
          {result.reasoning_analysis && (
            <ReasoningAnalysis summary={result.reasoning_analysis} />
          )}
          {result.failure_analysis && (
            <div className="gate-result">
              <strong>规则分析（{result.failure_category}）</strong>
              <p>{result.failure_analysis}</p>
            </div>
          )}
          {result.steps.length > 0 && (
            <details>
              <summary>推理步骤（{result.steps.length}）</summary>
              <div className="advanced-observations">
                {result.steps.map((step) => (
                  <article key={step.step_number}>
                    <strong>
                      #{step.step_number} · {step.step_type}
                    </strong>
                    <pre>{step.content}</pre>
                  </article>
                ))}
              </div>
            </details>
          )}
          <ToolRecord result={result} name="reasoning-analysis" />
        </div>
      )}
    </section>
  );
}
