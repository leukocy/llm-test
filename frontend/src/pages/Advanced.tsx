import { useState } from "react";
import { api } from "../api";
import { formatNumber } from "../constants";

type ParseResultView = {
  extracted_answer: string;
  confidence: number;
  method: string;
  normalized_value: unknown;
  raw_match: string;
  error: string | null;
  is_correct?: boolean;
  score?: number;
};

type ReasoningResultView = {
  final_answer_correct: boolean;
  answer_confidence: number;
  quality_score: Record<string, number>;
  steps: { step_number: number; content: string; step_type: string }[];
  failure_category: string;
  failure_analysis: string;
  evaluation_method: string;
  evaluation_confidence: number;
};

const DIM_LABELS: Record<string, string> = {
  coherence: "连贯性",
  completeness: "完整性",
  relevance: "相关性",
  correctness: "正确性",
  efficiency: "效率",
  overall: "综合",
};

export function Advanced({ token }: { token: string }) {
  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">ADVANCED EVALUATION</span>
          <h1>高级评估</h1>
          <p>
            离线分析工具：答案解析演示与推理过程质量评估（规则法，不消耗模型调用）。
          </p>
        </div>
      </div>
      <ParserDemo token={token} />
      <ReasoningDemo token={token} />
    </div>
  );
}

function ParserDemo({ token }: { token: string }) {
  const [response, setResponse] = useState("");
  const [answerType, setAnswerType] = useState("number");
  const [expected, setExpected] = useState("");
  const [result, setResult] = useState<ParseResultView | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    setError("");
    try {
      const payload = await api<ParseResultView>(
        token,
        "/api/v1/advanced/parse",
        {
          method: "POST",
          body: JSON.stringify({
            response,
            answer_type: answerType,
            ...(expected ? { expected_answer: expected } : {}),
          }),
        },
      );
      setResult(payload);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "解析失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="surface">
      <div className="section-head">
        <div>
          <span className="section-index">01</span>
          <h2>Smart Parser 演示</h2>
        </div>
        <span className="minor-tag">规则解析 · 可选判分</span>
      </div>
      <label className="input-label" htmlFor="parse-response">
        模型响应文本
      </label>
      <textarea
        id="parse-response"
        className="json-editor"
        value={response}
        onChange={(event) => setResponse(event.target.value)}
        placeholder="粘贴一段模型输出, 如: 计算过程略。最终答案是 42。"
      />
      <div className="warehouse-matrix-controls">
        <label>
          答案类型
          <select
            aria-label="答案类型"
            value={answerType}
            onChange={(event) => setAnswerType(event.target.value)}
          >
            <option value="number">数值</option>
            <option value="choice">选择题</option>
            <option value="text">文本</option>
            <option value="boolean">是/否</option>
            <option value="code">代码</option>
            <option value="math">数学表达式</option>
          </select>
        </label>
        <label>
          参考答案（可选）
          <input
            aria-label="参考答案"
            value={expected}
            onChange={(event) => setExpected(event.target.value)}
            placeholder="给出即判分"
          />
        </label>
        <button
          className="button primary"
          disabled={!response.trim() || busy}
          onClick={() => void run()}
        >
          {busy ? "解析中…" : "解析 →"}
        </button>
      </div>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {result && (
        <div className="gate-result">
          <p>
            提取答案：
            <strong>{result.extracted_answer || "（未提取到）"}</strong>
          </p>
          <p>
            置信度 {formatNumber(result.confidence, 2)} · 方法 {result.method}
            {result.raw_match ? ` · 匹配 ${result.raw_match}` : ""}
          </p>
          {result.is_correct !== undefined && (
            <p className={result.is_correct ? "text-good" : "text-danger"}>
              判分：{result.is_correct ? "正确" : "错误"}
              {result.score !== undefined
                ? `（${formatNumber(result.score, 2)}）`
                : ""}
            </p>
          )}
          {result.error && <p className="text-danger">{result.error}</p>}
        </div>
      )}
    </section>
  );
}

function ReasoningDemo({ token }: { token: string }) {
  const [question, setQuestion] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [finalAnswer, setFinalAnswer] = useState("");
  const [correctAnswer, setCorrectAnswer] = useState("");
  const [result, setResult] = useState<ReasoningResultView | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    setError("");
    try {
      const payload = await api<ReasoningResultView>(
        token,
        "/api/v1/advanced/reasoning",
        {
          method: "POST",
          body: JSON.stringify({
            question,
            reasoning,
            final_answer: finalAnswer,
            correct_answer: correctAnswer,
          }),
        },
      );
      setResult(payload);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "评估失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="surface">
      <div className="section-head">
        <div>
          <span className="section-index">02</span>
          <h2>推理过程质量评估</h2>
        </div>
        <span className="minor-tag">规则法五维评分</span>
      </div>
      <div className="schema-form">
        <label className="schema-field case-form-wide">
          <span className="input-label">原始问题</span>
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="如: 小明有 3 个苹果, 又买了 5 个, 一共几个?"
          />
        </label>
        <label className="schema-field case-form-wide">
          <span className="input-label">推理过程（可空）</span>
          <textarea
            className="json-editor"
            value={reasoning}
            onChange={(event) => setReasoning(event.target.value)}
            placeholder="粘贴模型的思维链 / reasoning 内容"
          />
        </label>
        <label className="schema-field">
          <span className="input-label">模型最终答案</span>
          <input
            value={finalAnswer}
            onChange={(event) => setFinalAnswer(event.target.value)}
          />
        </label>
        <label className="schema-field">
          <span className="input-label">参考答案</span>
          <input
            value={correctAnswer}
            onChange={(event) => setCorrectAnswer(event.target.value)}
          />
        </label>
      </div>
      <div className="form-footer">
        <span />
        <button
          className="button primary"
          disabled={!question.trim() || busy}
          onClick={() => void run()}
        >
          {busy ? "评估中…" : "评估 →"}
        </button>
      </div>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {result && (
        <>
          <div className="metric-grid">
            {Object.entries(result.quality_score).map(([dim, value]) => (
              <div className="metric-card" key={dim}>
                <div className="metric-label">{DIM_LABELS[dim] || dim}</div>
                <strong>{formatNumber(value, 1)}</strong>
                <div className="metric-note">/ 10</div>
              </div>
            ))}
            <div className="metric-card">
              <div className="metric-label">最终答案</div>
              <strong
                className={
                  result.final_answer_correct ? "text-good" : "text-danger"
                }
              >
                {result.final_answer_correct ? "正确" : "错误"}
              </strong>
              <div className="metric-note">
                评估置信度 {formatNumber(result.evaluation_confidence, 2)}
              </div>
            </div>
          </div>
          {result.failure_analysis && (
            <div className="gate-result">
              <strong>失败分析（{result.failure_category}）</strong>
              <p>{result.failure_analysis}</p>
            </div>
          )}
          {result.steps.length > 0 && (
            <div className="table-scroll">
              <table className="data-table stats-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>类型</th>
                    <th>推理步骤</th>
                  </tr>
                </thead>
                <tbody>
                  {result.steps.map((step) => (
                    <tr key={step.step_number}>
                      <td>{step.step_number}</td>
                      <td>{step.step_type}</td>
                      <td>{step.content}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}
