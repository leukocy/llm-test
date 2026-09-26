import { useMemo, useState } from "react";
import { api, type Job } from "../api";
import { Empty } from "../components";
import { formatNumber, formatPercent, shortId } from "../constants";

type ComparePayload = {
  job_a: { job_id: string; model_id: string; endpoint_id: string };
  job_b: { job_id: string; model_id: string; endpoint_id: string };
  datasets: Record<
    string,
    {
      samples: number;
      accuracy_a: number | null;
      accuracy_b: number | null;
      statistic: number;
      p_value: number;
      significant: boolean;
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

  async function run() {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const payload = await api<ComparePayload>(token, "/api/v1/compare", {
        method: "POST",
        body: JSON.stringify({ job_id_a: idA, job_id_b: idB }),
      });
      setResult(payload);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "对比失败");
    } finally {
      setBusy(false);
    }
  }

  function jobLabel(job: Job) {
    return `${job.model_id} · #${shortId(job.job_id)}`;
  }

  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">A/B COMPARISON</span>
          <h1>模型对比</h1>
          <p>
            对两个已完成的质量评估作业做 McNemar 显著性检验：共同数据集内按
            sample_id 对齐逐样本比对，判定差异是否统计显著。
          </p>
        </div>
      </div>
      <section className="surface">
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
          <div className="table-scroll">
            <table className="data-table stats-table">
              <thead>
                <tr>
                  <th>数据集</th>
                  <th>对齐样本</th>
                  <th>A 准确率</th>
                  <th>B 准确率</th>
                  <th>A 对 B 错</th>
                  <th>A 错 B 对</th>
                  <th>p 值</th>
                  <th>显著 (p&lt;0.05)</th>
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
                    <td>{formatNumber(entry.p_value, 4)}</td>
                    <td className={entry.significant ? "text-good" : ""}>
                      {entry.significant ? "显著" : "不显著"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {Object.entries(result.datasets).map(([name, entry]) => (
            <p className="chart-caption" key={name}>
              {name}：{entry.interpretation}
            </p>
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
