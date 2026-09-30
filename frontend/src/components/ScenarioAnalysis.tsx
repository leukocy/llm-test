import { useEffect, useState } from "react";
import type { Figure } from "plotly.js-dist-min";
import { api, type Job, type Summary, type Metric } from "../api";
import { PlotlyFigure } from "./PlotlyFigure";

export const metrics = {
  ttft: "首字延迟 TTFT · s",
  tpot: "每 token 延迟 TPOT · s",
  tps: "逐请求生成速度 TPS · token/s",
  prefill_speed: "逐请求输入处理速度 · token/s",
  total_time: "请求总耗时 · s",
  system_input_wall: "系统输入吞吐（含缓存）· token/s",
  system_output_wall: "系统输出吞吐 · token/s",
  system_total_wall: "系统总吞吐 · token/s",
  system_qpm: "成功请求处理速率 QPM · req/min",
  phase_input: "客户端输入阶段估计（含缓存）· token/s",
  phase_input_uncached_api: "客户端未缓存输入阶段估计（API）· token/s",
  phase_input_uncached_inferred:
    "客户端未缓存输入阶段估计（含 TTFT 推断）· token/s",
  phase_output: "客户端输出阶段估计 · token/s",
  phase_total: "客户端请求窗口总吞吐 · token/s",
  phase_qpm: "客户端请求窗口成功处理速率 · req/min",
  cache_tokens_api: "API 缓存命中 token · token",
  cache_rate_api: "API 缓存命中比例 · %",
  cache_tokens_inferred: "TTFT 推断缓存 token · token",
  cache_rate_inferred: "TTFT 推断缓存比例 · %",
  ttft_zero_cache_api: "API 明确零命中 TTFT · s",
  ttft_cache_api: "API 有命中 TTFT · s",
};
export const statistics = {
  median: "p50 · 中位数",
  mean: "均值",
  p95: "p95 · 尾部",
  p99: "p99 · 尾部",
  min: "观测最小值",
  max: "观测最大值",
};
type Analysis = {
  title: string;
  axis: string;
  notes: string[];
  observations: { metric: string; text: string }[];
};
type Result = { figure: Figure; analysis: Analysis };

export function ScenarioAnalysis({
  job,
  summary,
  token,
  figureEndpoint,
  revision,
}: {
  job: Pick<Job, "job_id">;
  summary: Summary;
  token: string;
  figureEndpoint?: string;
  revision?: string;
}) {
  const [metric, setMetric] = useState("ttft");
  const [view, setView] = useState("profile");
  const [statistic, setStatistic] = useState("median");
  const [result, setResult] = useState<{ key: string; data: Result } | null>(
    null,
  );
  const [error, setError] = useState<{ key: string; text: string } | null>(
    null,
  );
  const endpoint = figureEndpoint || `/api/v1/jobs/${job.job_id}/figure`;
  const key = `${endpoint}:${revision || ""}:${metric}:${view}:${statistic}`;
  useEffect(() => {
    setMetric("ttft");
    setView("profile");
    setStatistic("median");
  }, [endpoint]);
  useEffect(() => {
    const controller = new AbortController();
    setResult(null);
    setError(null);
    void api<Result>(
      token,
      `${endpoint}?${new URLSearchParams({ metric, view, statistic, ...(revision ? { revision } : {}) })}`,
      { signal: controller.signal },
    )
      .then((data) => {
        if (!controller.signal.aborted) setResult({ key, data });
      })
      .catch((exc) => {
        if (!controller.signal.aborted)
          setError({
            key,
            text: exc instanceof Error ? exc.message : "图形加载失败",
          });
      });
    return () => controller.abort();
  }, [token, endpoint, revision, metric, view, statistic, key, summary]);
  const current = result?.key === key ? result.data : null;
  const currentError = error?.key === key ? error.text : "";
  const analysis = summary.scenario_analysis || current?.analysis;
  const figure = current && {
    ...current.figure,
    layout: {
      ...current.figure.layout,
      title: { text: "" },
      width: undefined,
      height: 430,
      autosize: true,
      margin: { l: 70, r: 40, t: 30, b: 90 },
      annotations: [],
      legend: { orientation: "h" as const, y: -0.2 },
    },
  };
  return (
    <section className="surface scenario-analysis">
      <div className="section-head">
        <div>
          <span className="eyebrow">SCENARIO ANALYSIS</span>
          <h2>{analysis?.title || "场景指标分析"}</h2>
        </div>
        <span className="minor-tag">完整分组 · 非预览样本</span>
      </div>
      <div className="scenario-controls">
        <label>
          图形指标
          <select
            aria-label="图形指标"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            {Object.entries(metrics).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
                {!summary.overall.metrics[value]?.count ? "（无有效样本）" : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          图形视图
          <select
            aria-label="图形视图"
            value={view}
            onChange={(e) => setView(e.target.value)}
          >
            <option value="profile">条件曲线</option>
            <option value="comparison">p50 / p95 对照</option>
            {summary.run.test_type === "throughput_matrix" && (
              <option value="heatmap">矩阵热力图</option>
            )}
          </select>
        </label>
        <label>
          统计量
          <select
            aria-label="图形统计量"
            value={statistic}
            disabled={view === "comparison"}
            onChange={(e) => setStatistic(e.target.value)}
          >
            {Object.entries(statistics).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </div>
      {summary.extended_observations &&
        (metric.startsWith("system_") ? (
          <p className="muted">
            完整测量批次 {summary.extended_observations.system.valid_batches}
            ；排除不完整/冲突批次{" "}
            {summary.extended_observations.system.invalid_batches}
            ；缺少批次来源的请求{" "}
            {summary.extended_observations.system.untagged_requests}
            。系统速率使用墙钟窗口，包含失败等待与客户端开销。
          </p>
        ) : metric.startsWith("phase_") ? (
          <p className="muted" role="status">
            阶段时钟有效批次{" "}
            {summary.extended_observations.phase?.valid_batches || 0}
            ；缺少时钟{" "}
            {summary.extended_observations.phase?.missing_clock_batches || 0}
            ；无效时钟{" "}
            {summary.extended_observations.phase?.invalid_clock_batches || 0}
            ；首 Token 记录不齐{" "}
            {summary.extended_observations.phase?.missing_first_token_batches ||
              0}
            ；无成功请求{" "}
            {summary.extended_observations.phase?.no_success_batches || 0}
            ；非正窗口（输入/输出/总计）{" "}
            {summary.extended_observations.phase?.nonpositive_windows.input ||
              0}
            /
            {summary.extended_observations.phase?.nonpositive_windows.output ||
              0}
            /
            {summary.extended_observations.phase?.nonpositive_windows.total ||
              0}
            ；缺少批次来源的请求{" "}
            {summary.extended_observations.system.untagged_requests}
            ；排除不完整/冲突批次{" "}
            {summary.extended_observations.system.invalid_batches}。
            客户端成功请求窗口包含网络和排队，失败等待不进入；不是引擎内部阶段速率。
            非正窗口留空；输入含缓存、API 未缓存与 TTFT 推断分别统计。
          </p>
        ) : metric.startsWith("cache_") || metric.startsWith("ttft_") ? (
          <p className="muted">
            已采集 API 缓存记录{" "}
            {summary.extended_observations.cache.sources.API || 0}；TTFT
            推断记录{" "}
            {summary.extended_observations.cache.sources.TTFT_inferred || 0}
            ；成功请求中缓存来源未知{" "}
            {
              summary.extended_observations.cache.unknown_successes
            }；无效观测{" "}
            {summary.extended_observations.cache.invalid_observations}。记录数为
            0 表示未采集。
          </p>
        ) : null)}
      <p className="muted">
        {metrics[metric as keyof typeof metrics]} ·{" "}
        {view === "comparison"
          ? "p50 / p95"
          : statistics[statistic as keyof typeof statistics]}
        ；n 为
        {metric.startsWith("system_")
          ? "测量批次数（batch-wall-v1）"
          : metric.startsWith("phase_")
            ? "测量批次数（client-phase-v1）"
            : "有效请求数"}
        。PNG 保留样本数、完整性、指标口径与来源。
      </p>
      {currentError ? (
        <p role="alert" className="form-error">
          {currentError}
        </p>
      ) : !current ? (
        <p role="status">正在计算完整分组图形…</p>
      ) : (
        <PlotlyFigure
          figure={figure}
          exportFigure={current.figure}
          ariaLabel={`${analysis?.title || "测量"}-${metric}-${view}-${statistic}`}
        />
      )}
      {analysis && (
        <div className="scenario-explanation">
          <details>
            <summary>查看当前指标的完整分组数值</summary>
            <div className="table-scroll">
              <table className="data-table stats-table">
                <thead>
                  <tr>
                    <th>条件</th>
                    <th>请求 / 失败</th>
                    <th>有效 n</th>
                    <th>
                      {view === "comparison"
                        ? "p50"
                        : statistics[statistic as keyof typeof statistics]}
                    </th>
                    {view === "comparison" && <th>p95</th>}
                  </tr>
                </thead>
                <tbody>
                  {summary.groups.map((group) => {
                    const stats = group.metrics[metric];
                    if (!stats) return null;
                    const value =
                      stats[
                        view === "comparison"
                          ? "median"
                          : (statistic as keyof Metric)
                      ];
                    return (
                      <tr key={group.label}>
                        <th scope="row">{group.label}</th>
                        <td>
                          {group.requests} / {group.failures}
                        </td>
                        <td>{stats.count}</td>
                        <td>{value === null ? "—" : value.toPrecision(4)}</td>
                        {view === "comparison" && (
                          <td>
                            {stats.p95 === null
                              ? "—"
                              : stats.p95.toPrecision(4)}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </details>
          <h3>p50 观测与解释</h3>
          <ul>
            {analysis.observations
              .filter((item) => item.metric === metric)
              .map((item) => (
                <li key={item.metric}>{item.text}</li>
              ))}
          </ul>
          <details>
            <summary>测量方法与解释边界</summary>
            <ul>
              {analysis.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </details>
        </div>
      )}
    </section>
  );
}
