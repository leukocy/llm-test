import { useEffect, useState } from "react";
import type { Figure } from "plotly.js-dist-min";
import { api, type Job, type Metric, type Summary } from "../api";
import { PlotlyFigure } from "./PlotlyFigure";
import { metrics, statistics } from "./ScenarioAnalysis";

const requestMetrics = [
  "ttft",
  "tpot",
  "tps",
  "prefill_speed",
  "total_time",
] as const;

export function StabilityTimeline({
  job,
  summary,
  token,
}: {
  job: Job;
  summary: Summary;
  token: string;
}) {
  const [metric, setMetric] = useState<string>("ttft");
  const [statistic, setStatistic] = useState("median");
  const [result, setResult] = useState<{ key: string; figure: Figure } | null>(
    null,
  );
  const [error, setError] = useState<{ key: string; text: string } | null>(
    null,
  );
  const timeline = summary.time_series;
  const key = `${job.job_id}:${metric}:${statistic}`;
  useEffect(() => {
    const controller = new AbortController();
    setResult(null);
    setError(null);
    if (!timeline?.bins.length) return () => controller.abort();
    void api<{ figure: Figure }>(
      token,
      `/api/v1/jobs/${job.job_id}/figure?${new URLSearchParams({ view: "timeline", metric, statistic })}`,
      { signal: controller.signal },
    )
      .then(({ figure }) => {
        if (!controller.signal.aborted) setResult({ key, figure });
      })
      .catch((exc) => {
        if (!controller.signal.aborted)
          setError({
            key,
            text: exc instanceof Error ? exc.message : "时间序列加载失败",
          });
      });
    return () => controller.abort();
  }, [token, job.job_id, key, metric, statistic, timeline]);
  if (!timeline) return null;
  const current = result?.key === key ? result.figure : null;
  const currentError = error?.key === key ? error.text : "";
  const figure = current && {
    ...current,
    layout: {
      ...current.layout,
      title: { text: "" },
      width: undefined,
      height: 530,
      autosize: true,
      annotations: [],
      margin: { l: 70, r: 30, t: 30, b: 90 },
      legend: { orientation: "h" as const, y: -0.2 },
    },
  };
  return (
    <section className="surface stability-timeline">
      <div className="section-head">
        <div>
          <span className="eyebrow">STABILITY / TIME SERIES</span>
          <h2>稳定性完成时间序列</h2>
        </div>
        <span className="minor-tag">单调时钟 · 完整记录分窗</span>
      </div>
      <p className="muted">
        有效计时 {timeline.timed_requests}；缺失 {timeline.missing_requests}
        ；无效 {timeline.invalid_requests}。计时记录完整：
        {timeline.complete ? "是" : "否"}。每窗 {timeline.bin_seconds ?? "未知"}{" "}
        秒，计划发起 {timeline.planned_seconds ?? "未知"} 秒，调度至排空{" "}
        {timeline.window_seconds?.toFixed(3) ?? "未知"} 秒。
      </p>
      {!timeline.bins.length ? (
        <p role="status">
          缺少可核验的单调时钟记录，无法绘制时间序列。数据库写入时间不能替代请求时间。
        </p>
      ) : (
        <>
          <div className="scenario-controls">
            <label>
              时间序列指标
              <select
                aria-label="时间序列指标"
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
              >
                {requestMetrics.map((m) => (
                  <option key={m} value={m}>
                    {metrics[m]}
                  </option>
                ))}
              </select>
            </label>
            <label>
              时间窗统计量
              <select
                aria-label="时间窗统计量"
                value={statistic}
                onChange={(e) => setStatistic(e.target.value)}
              >
                {Object.entries(statistics).map(([s, label]) => (
                  <option key={s} value={s}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="muted">
            上图为成功请求的指标，下图为全部已计时完成请求与其中的失败请求。空窗保留空值；失败指标不计入延迟分位数。
          </p>
          {currentError ? (
            <p role="alert" className="form-error">
              {currentError}
            </p>
          ) : !current ? (
            <p role="status">正在加载时间序列…</p>
          ) : (
            <PlotlyFigure
              figure={figure}
              exportFigure={current}
              ariaLabel={`稳定性-${metric}-timeline-${statistic}`}
            />
          )}
          <details>
            <summary>查看完整时间窗数值</summary>
            <div className="table-scroll">
              <table className="data-table stats-table">
                <thead>
                  <tr>
                    <th>时间窗 (s)</th>
                    <th>完成 / 失败</th>
                    <th>有效 n</th>
                    <th>{statistics[statistic as keyof typeof statistics]}</th>
                  </tr>
                </thead>
                <tbody>
                  {timeline.bins.map((b) => {
                    const stats = b.metrics[metric];
                    const value = stats[statistic as keyof Metric];
                    return (
                      <tr key={b.start_seconds}>
                        <th scope="row">
                          {b.start_seconds.toFixed(3)}–
                          {b.end_seconds.toFixed(3)}
                        </th>
                        <td>
                          {b.requests} / {b.failures}
                        </td>
                        <td>{stats.count}</td>
                        <td>{value === null ? "—" : value.toPrecision(4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
      <details>
        <summary>计时方法与解释边界</summary>
        <ul>
          {timeline.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </details>
    </section>
  );
}
