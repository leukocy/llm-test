import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Job } from "../api";
import { Empty } from "../components";
import { formatPercent, shortId } from "../constants";
import { OnlineComparison } from "../components/OnlineComparison";

type ComparePayload = {
  job_a: { job_id: string; model_id: string; endpoint_id: string };
  job_b: { job_id: string; model_id: string; endpoint_id: string };
  datasets: Record<
    string,
    {
      samples: number;
      accuracy_a: number | null;
      accuracy_b: number | null;
      statistic: number | null;
      p_value: number | null;
      significant: boolean | null;
      verified: boolean;
      p_value_label: string;
      adjusted_p_value_label: string;
      adjusted_significant: boolean | null;
      test_family_size: number;
      score_basis: string;
      accuracy_difference: number;
      unpaired_a: number;
      unpaired_b: number;
      excluded_a: Record<string, number>;
      excluded_b: Record<string, number>;
      warnings: string[];
      b01_count: number;
      b10_count: number;
      interpretation: string;
    }
  >;
  skipped_datasets: string[];
};

export function Compare({ jobs, token }: { jobs: Job[]; token: string }) {
  const candidates = useMemo(
    () =>
      jobs.filter(
        (job) => job.test_type === "quality" && job.status === "completed",
      ),
    [jobs],
  );
  const [idA, setIdA] = useState("");
  const [idB, setIdB] = useState("");
  const [result, setResult] = useState<ComparePayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [basis, setBasis] = useState("standard");
  const [autoCompare, setAutoCompare] = useState(false);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => {
    pending.current?.abort();
    setResult(null);
    setError("");
    setBusy(false);
    return () => pending.current?.abort();
  }, [idA, idB, basis]);
  useEffect(() => {
    if (autoCompare && idA && idB) {
      setAutoCompare(false);
      void run();
    }
  }, [autoCompare, idA, idB]);

  async function run() {
    setBusy(true);
    setError("");
    setResult(null);
    const controller = new AbortController();
    pending.current?.abort();
    pending.current = controller;
    try {
      const payload = await api<ComparePayload>(token, "/api/v1/compare", {
        method: "POST",
        body: JSON.stringify({
          job_id_a: idA,
          job_id_b: idB,
          score_basis: basis,
        }),
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setResult(payload);
    } catch (exc) {
      if (!controller.signal.aborted)
        setError(exc instanceof Error ? exc.message : "对比失败");
    } finally {
      if (pending.current === controller) setBusy(false);
    }
  }

  function jobLabel(job: Job) {
    return `${job.model_id} · #${shortId(job.job_id)}`;
  }
  function exportResult() {
    if (!result) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `comparison-${idA}-${idB}-${basis}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">A/B COMPARISON</span>
          <h1>模型对比</h1>
          <p>
            对两个已完成质量作业核对同一题目、参考答案与提示词。成绩来自有效配对样本；完整来源与条件通过核验后使用双侧精确
            McNemar 检验，并对本次已核验数据集做 Holm
            校正。证据不足时保留描述，不给出显著性结论。
          </p>
        </div>
      </div>
      <OnlineComparison
        token={token}
        onReady={(a, b) => {
          setIdA(a);
          setIdB(b);
          setAutoCompare(true);
        }}
      />
      <section className="surface">
        <label className="compare-basis">
          评分口径
          <select
            aria-label="对比评分口径"
            value={basis}
            onChange={(event) => setBasis(event.target.value)}
          >
            <option value="standard">规则成绩（排除 Judge 改判）</option>
            <option value="final">最终成绩（含 Judge 改判）</option>
          </select>
        </label>
        <div className="compare-picker">
          <label>
            作业 A
            <select
              aria-label="作业 A"
              value={idA}
              onChange={(event) => setIdA(event.target.value)}
            >
              <option value="">选择质量作业…</option>
              {candidates.map((job) => (
                <option key={job.job_id} value={job.job_id}>
                  {jobLabel(job)}
                </option>
              ))}
            </select>
          </label>
          <label>
            作业 B
            <select
              aria-label="作业 B"
              value={idB}
              onChange={(event) => setIdB(event.target.value)}
            >
              <option value="">选择质量作业…</option>
              {candidates.map((job) => (
                <option key={job.job_id} value={job.job_id}>
                  {jobLabel(job)}
                </option>
              ))}
            </select>
          </label>
          <div className="compare-actions">
            <p>
              {candidates.length
                ? `可对比作业 ${candidates.length} 个`
                : "暂无已完成的质量作业"}
            </p>
            <button
              className="button primary"
              disabled={!idA || !idB || idA === idB || busy}
              onClick={() => void run()}
            >
              {busy ? "检验中…" : "开始检验 →"}
            </button>
          </div>
        </div>
        {idA && idA === idB && <p className="form-error">两个作业不能相同</p>}
      </section>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {result && (
        <section className="surface">
          <div className="section-head">
            <div>
              <span className="eyebrow">RESULT</span>
              <h2>
                {result.job_a.model_id} vs {result.job_b.model_id}
              </h2>
            </div>
            <span className="minor-tag">
              #{shortId(result.job_a.job_id)} vs #{shortId(result.job_b.job_id)}
            </span>
          </div>
          <button className="button subtle" onClick={exportResult}>
            导出对比 JSON ↓
          </button>
          <div className="table-scroll">
            <table className="data-table stats-table">
              <thead>
                <tr>
                  <th>数据集</th>
                  <th>对齐样本</th>
                  <th>A 配对准确率</th>
                  <th>B 配对准确率</th>
                  <th>A 对 B 错</th>
                  <th>A 错 B 对</th>
                  <th>精确 p</th>
                  <th>Holm p</th>
                  <th>校正后检出差异</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.datasets).map(([name, entry]) => (
                  <tr key={name}>
                    <td>
                      <strong>{name}</strong>
                    </td>
                    <td>{entry.samples}</td>
                    <td>{formatPercent(entry.accuracy_a)}</td>
                    <td>{formatPercent(entry.accuracy_b)}</td>
                    <td>{entry.b01_count}</td>
                    <td>{entry.b10_count}</td>
                    <td>{entry.p_value_label}</td>
                    <td>{entry.adjusted_p_value_label}</td>
                    <td
                      className={entry.adjusted_significant ? "text-good" : ""}
                    >
                      {entry.adjusted_significant == null
                        ? "未检验"
                        : entry.adjusted_significant
                          ? "检出"
                          : "未检出"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {Object.entries(result.datasets).map(([name, entry]) => (
            <div key={name}>
              <p className="chart-caption">
                {name}：{entry.interpretation} A−B 差值{" "}
                {formatPercent(entry.accuracy_difference)}；未配对 A{" "}
                {entry.unpaired_a} / B {entry.unpaired_b}，排除 A{" "}
                {Object.values(entry.excluded_a).reduce((a, b) => a + b, 0)} / B{" "}
                {Object.values(entry.excluded_b).reduce((a, b) => a + b, 0)}
                。校正检验数 {entry.test_family_size}
                ；未检出不代表等效，独立性仍依赖试验设计。
                {basis === "final" && "同一模型自评 Judge 不构成独立验证。"}
              </p>
              {entry.warnings.length > 0 && (
                <ul className="chart-caption">
                  {entry.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              )}
            </div>
          ))}
          {result.skipped_datasets.length > 0 && (
            <p className="chart-caption">
              仅单侧存在的数据集已跳过：{result.skipped_datasets.join("、")}
            </p>
          )}
        </section>
      )}
      {!result && !error && (
        <section className="surface">
          <Empty
            title="选择两个质量作业开始对比"
            text="检验只读不写库；样本对不齐的作业会给出明确提示。"
          />
        </section>
      )}
    </div>
  );
}
