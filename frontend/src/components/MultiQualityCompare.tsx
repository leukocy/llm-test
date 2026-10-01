import { useEffect, useRef, useState } from "react";
import type { Figure } from "plotly.js-dist-min";
import { api, type Job } from "../api";
import { formatPercent, shortId } from "../constants";
import { PlotlyFigure } from "./PlotlyFigure";

type Matrix = {
  version: string;
  score_basis: string;
  datasets: string[];
  note: string;
  rows: {
    job_id: string;
    model_id: string;
    endpoint_id: string;
    cells: Record<
      string,
      {
        accuracy: number | null;
        correct: number | null;
        total: number | null;
        warnings: string[];
      } | null
    >;
  }[];
  figure: Figure;
  warnings: Record<string, string>;
  exports: Record<string, string>;
};

export function MultiQualityCompare({
  jobs,
  token,
}: {
  jobs: Job[];
  token: string;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const [basis, setBasis] = useState("final");
  const [result, setResult] = useState<Matrix | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => {
    pending.current?.abort();
    setResult(null);
    setError("");
    setBusy(false);
    return () => pending.current?.abort();
  }, [selected, basis]);
  async function run() {
    const controller = new AbortController();
    pending.current?.abort();
    pending.current = controller;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const data = await api<Matrix>(token, "/api/v1/compare/matrix", {
        method: "POST",
        body: JSON.stringify({ job_ids: selected, score_basis: basis }),
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setResult(data);
    } catch (exc) {
      if (!controller.signal.aborted) setError(String(exc));
    } finally {
      if (pending.current === controller) setBusy(false);
    }
  }
  function save(format: string) {
    if (!result) return;
    const content =
      format === "json"
        ? JSON.stringify(result, null, 2)
        : result.exports[format];
    const url = URL.createObjectURL(
      new Blob([content], {
        type:
          format === "html"
            ? "text/html"
            : format === "csv"
              ? "text/csv;charset=utf-8"
              : format === "json"
                ? "application/json"
                : "text/markdown;charset=utf-8",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `quality-matrix-${result.score_basis}.${format === "markdown" ? "md" : format}`;
    link.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section className="surface multi-quality-comparison">
      <div className="section-head">
        <div>
          <span className="eyebrow">MULTI MODEL QUALITY</span>
          <h2>多模型质量对照</h2>
        </div>
      </div>
      <p className="chart-caption">
        选择 2–8
        个已完成质量作业。保留每格样本数；同名模型的不同运行分别展示，缺测不填零。
      </p>
      <div className="quality-analysis-grid">
        {jobs.map((job) => (
          <label key={job.job_id} className="input-label">
            <input
              type="checkbox"
              aria-label={`多模型作业 ${job.model_id} ${shortId(job.job_id)}`}
              checked={selected.includes(job.job_id)}
              disabled={!selected.includes(job.job_id) && selected.length >= 8}
              onChange={(event) =>
                setSelected((current) =>
                  event.target.checked
                    ? [...current, job.job_id]
                    : current.filter((id) => id !== job.job_id),
                )
              }
            />{" "}
            {job.model_id} · #{shortId(job.job_id)}
          </label>
        ))}
      </div>
      <div className="api-form-actions">
        <label>
          多模型评分口径{" "}
          <select
            aria-label="多模型评分口径"
            value={basis}
            onChange={(event) => setBasis(event.target.value)}
          >
            <option value="final">最终成绩（含 Judge 改判）</option>
            <option value="standard">规则成绩（排除 Judge 改判）</option>
          </select>
        </label>
        <button
          className="button primary"
          disabled={busy || selected.length < 2}
          onClick={() => void run()}
        >
          {busy ? "生成中…" : "生成多模型对照 →"}
        </button>
      </div>
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      {result && (
        <>
          <p className="chart-caption">{result.note}</p>
          <div className="table-scroll">
            <table className="data-table multi-quality-table">
              <thead>
                <tr>
                  <th>模型 / 作业</th>
                  {result.datasets.map((name) => (
                    <th key={name}>{name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row) => (
                  <tr key={row.job_id}>
                    <td>
                      {row.model_id} · #{shortId(row.job_id)}
                    </td>
                    {result.datasets.map((name) => {
                      const cell = row.cells[name];
                      return (
                        <td key={name}>
                          {cell ? (
                            <>
                              {formatPercent(cell.accuracy)}
                              <small>
                                {" "}
                                · {cell.correct ?? "—"}/{cell.total ?? "—"}
                              </small>
                            </>
                          ) : (
                            "未记录"
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <PlotlyFigure
            figure={result.figure}
            ariaLabel="多模型数据集分组柱图"
          />
          {Object.entries(result.warnings).map(([name, warning]) => (
            <p className="form-warning" key={name}>
              {name}：{warning}
            </p>
          ))}
          {result.rows.flatMap((row) =>
            Object.entries(row.cells).flatMap(
              ([name, cell]) =>
                cell?.warnings.map((warning, index) => (
                  <p
                    className="form-warning"
                    key={`${row.job_id}-${name}-${index}`}
                  >
                    {row.model_id} / {name}：{warning}
                  </p>
                )) || [],
            ),
          )}
          <div className="api-form-actions">
            {["csv", "json", "markdown", "html"].map((format) => (
              <button
                className="button subtle"
                key={format}
                onClick={() => save(format)}
              >
                导出多模型 {format.toUpperCase()} ↓
              </button>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
