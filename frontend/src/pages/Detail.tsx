import { useEffect, useRef, useState } from "react";
import {
  api,
  downloadFile,
  type Job,
  type JobEvent,
  type QualityReport,
  type RequestResult,
  type Summary,
} from "../api";
import { Status, MetricCard, SlicesChart, Empty } from "../components";
import { QualityAnalysis } from "./QualityAnalysis";
import {
  activeStates,
  labels,
  statusLabels,
  date,
  shortId,
  formatNumber,
  formatPercent,
} from "../constants";

export function Detail({
  job,
  token,
  onBack,
  onCancel,
}: {
  job: Job;
  token: string;
  onBack: () => void;
  onCancel: () => Promise<void>;
}) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [quality, setQuality] = useState<QualityReport | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [results, setResults] = useState<RequestResult[]>([]);
  const [error, setError] = useState("");
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [robustness, setRobustness] = useState<RobustnessReport | null>(null);
  const resultsRef = useRef<RequestResult[]>([]);
  const logsRef = useRef<LogLine[]>([]);
  resultsRef.current = results;

  type LogLine = {
    id: number;
    timestamp: number | null;
    level: string;
    message: string;
  };
  type RobustnessReport = {
    job_id: string;
    model_id: string;
    robustness: {
      total_samples: number;
      perturbations_per_sample: number;
      original_accuracy: number;
      perturbed_accuracy: number;
      accuracy_drop: number;
      overall_robustness: number;
      overall_consistency: number;
      sensitivity_by_type: Record<string, number>;
      most_sensitive_perturbation: string;
      results: {
        sample_id: string;
        original_correct: boolean;
        robustness_score: number;
        consistency_score: number;
        perturbed_results: { type?: string; correct?: boolean }[];
      }[];
      recommendations: string[];
    };
  };
  useEffect(() => {
    let alive = true;
    setSummary(null);
    setQuality(null);
    setRobustness(null);
    setResults([]);
    setEvents([]);
    api<{ items: JobEvent[] }>(token, `/api/v1/jobs/${job.job_id}/events`)
      .then((data) => {
        if (alive) setEvents(data.items);
      })
      .catch(() => {});
    if (job.result_run_id) {
      api<Summary>(token, `/api/v1/jobs/${job.job_id}/summary`)
        .then((data) => {
          if (alive) setSummary(data);
        })
        .catch((exc) => {
          if (alive) setError(exc.message);
        });
      api<{ items: RequestResult[] }>(
        token,
        `/api/v1/jobs/${job.job_id}/results?limit=50`,
      )
        .then((data) => {
          if (alive) setResults(data.items);
        })
        .catch(() => {});
    } else if (job.result_artifact) {
      api<QualityReport | RobustnessReport>(
        token,
        `/api/v1/jobs/${job.job_id}/report`,
      )
        .then((data) => {
          if (!alive) return;
          if ((data as RobustnessReport).robustness) {
            setRobustness(data as RobustnessReport);
          } else {
            setQuality(data as QualityReport);
          }
        })
        .catch((exc) => {
          if (alive) setError(exc.message);
        });
    }
    return () => {
      alive = false;
    };
  }, [job.job_id, job.result_run_id, job.result_artifact, job.status, token]);

  // 运行中：since_id 游标增量轮询逐请求样本（去重追加）
  useEffect(() => {
    if (!job.result_run_id || !activeStates.has(job.status)) return;
    let alive = true;
    const timer = window.setInterval(() => {
      const current = resultsRef.current;
      const lastId = current.length ? current[current.length - 1].id : 0;
      api<{ items: RequestResult[] }>(
        token,
        `/api/v1/jobs/${job.job_id}/results?limit=200&since_id=${lastId}`,
      )
        .then((data) => {
          if (!alive || !data.items.length) return;
          setResults((prev) => {
            const seen = new Set(prev.map((row) => row.id));
            const fresh = data.items.filter((row) => !seen.has(row.id));
            return fresh.length ? [...prev, ...fresh] : prev;
          });
        })
        .catch(() => {});
    }, 2000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [job.job_id, job.result_run_id, job.status, token]);

  // 执行日志：行号游标增量轮询（活动任务每 2 秒, 完成的任务拉一次全量）
  useEffect(() => {
    let alive = true;
    setLogs([]);
    logsRef.current = [];
    const completed = !activeStates.has(job.status);

    async function fetchLogs(since: number) {
      if (!alive) return;
      try {
        const data = await api<{ items: LogLine[] }>(
          token,
          `/api/v1/jobs/${job.job_id}/logs?since_id=${since}&limit=500`,
        );
        if (!alive || !data.items.length) return;
        setLogs((prev) => {
          const seen = new Set(prev.map((line) => line.id));
          const fresh = data.items.filter((line) => !seen.has(line.id));
          return fresh.length ? [...prev, ...fresh] : prev;
        });
        logsRef.current = [...logsRef.current, ...data.items];
      } catch {
        /* 日志不可达不阻断页面 */
      }
    }

    void fetchLogs(0);
    if (completed) {
      return () => {
        alive = false;
      };
    }
    const timer = window.setInterval(
      () => void fetchLogs(logsRef.current.length),
      2000,
    );
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [job.job_id, job.status, token]);

  async function download(format: "json" | "html" | "csv") {
    try {
      const path =
        format === "csv"
          ? `/api/v1/jobs/${job.job_id}/export.csv`
          : `/api/v1/jobs/${job.job_id}/report?format=${format}`;
      await downloadFile(
        token,
        path,
        `llm-test-${shortId(job.job_id)}.${format}`,
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "下载失败");
    }
  }

  async function downloadWarmup() {
    try {
      await downloadFile(
        token,
        `/api/v1/jobs/${job.job_id}/warmup.csv`,
        `llm-test-${shortId(job.job_id)}-warmup.csv`,
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "预热记录下载失败");
    }
  }

  const overall = summary?.overall;
  return (
    <div className="page-grid">
      <div className="detail-head">
        <button className="back-button" onClick={onBack}>
          ← 返回运行记录
        </button>
        <div className="detail-title">
          <div>
            <span className="eyebrow">RUN / {shortId(job.job_id)}</span>
            <h1>{labels[job.test_type] || job.test_type}</h1>
            <p>
              {job.model_id} · {job.endpoint_id}
            </p>
          </div>
          <Status value={job.status} />
        </div>
        <div className="detail-actions">
          <span>创建于 {date(job.created_at)}</span>
          {activeStates.has(job.status) && (
            <button
              className="button subtle danger"
              onClick={() => void onCancel()}
            >
              取消运行
            </button>
          )}
          {(summary || quality) && (
            <button
              className="button subtle"
              onClick={() => void download("json")}
            >
              导出 JSON ↓
            </button>
          )}
          {summary && (
            <button
              className="button subtle"
              onClick={() => void download("csv")}
            >
              导出 CSV ↓
            </button>
          )}
          {summary?.measurement_protocol?.warmup_requests ? (
            <button
              className="button subtle"
              onClick={() => void downloadWarmup()}
            >
              预热记录 ↓
            </button>
          ) : null}
          {(summary || quality) && (
            <button
              className="button primary"
              onClick={() => void download("html")}
            >
              导出报告 ↓
            </button>
          )}
        </div>
      </div>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {job.error_message && (
        <div className="alert" role="alert">
          {job.error_code}: {job.error_message}
        </div>
      )}
      <div className="detail-progress surface">
        <div>
          <strong>任务进度</strong>
          <span>
            {job.progress_total
              ? `${job.progress_completed} / ${job.progress_total} 请求`
              : "等待测量引擎报告进度"}
          </span>
        </div>
        <div className="progress-track large">
          <i
            style={{
              width: `${job.progress_total ? Math.min(100, (job.progress_completed / job.progress_total) * 100) : 0}%`,
            }}
          />
        </div>
      </div>
      {summary && (
        <>
          <div
            className={`report-integrity ${summary.integrity.verified ? "ready" : "limited"}`}
            role="status"
          >
            <strong>
              {summary.integrity.verified
                ? "单次运行完整性核验通过"
                : "仅供诊断 · 未通过完整性核验"}
            </strong>
            <span>
              已记录 {summary.integrity.recorded_requests} 次请求
              {summary.integrity.expected_requests !== null
                ? ` / 计划 ${summary.integrity.expected_requests} 次`
                : " / 计划请求数未知"}
            </span>
            {summary.integrity.reasons.length > 0 && (
              <ul>
                {summary.integrity.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            )}
          </div>
          {summary.measurement_protocol && (
            <section className="surface protocol-surface">
              <div className="section-head">
                <div>
                  <span className="eyebrow">MEASUREMENT PROTOCOL</span>
                  <h2>测量方法与样本计划</h2>
                </div>
                <span className="minor-tag">
                  {summary.measurement_protocol.protocol_version}
                </span>
              </div>
              <p>
                {summary.measurement_protocol.workload_model ===
                "closed_loop_fixed_concurrency"
                  ? "闭环固定并发负载"
                  : "顺序固定输入长度负载"}
                ：正式 {summary.measurement_protocol.measured_requests}{" "}
                次，预热已记录{" "}
                {summary.measurement_protocol.warmup_recorded ?? "—"} /{" "}
                {summary.measurement_protocol.warmup_requests}{" "}
                次。预热不进入统计。
              </p>
              {summary.data_quality.warnings.length > 0 && (
                <ul className="protocol-warnings">
                  {summary.data_quality.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              )}
            </section>
          )}
          <div className="metric-grid">
            <MetricCard
              label="总请求"
              value={String(overall!.requests)}
              note={`${overall!.failures} 次失败`}
              accent
            />
            <MetricCard
              label="成功率"
              value={formatPercent(overall!.success_rate)}
              note={
                overall!.success_rate_ci95
                  ? `95% Wilson 区间 ${formatPercent(overall!.success_rate_ci95[0])}–${formatPercent(overall!.success_rate_ci95[1])}`
                  : "无有效样本"
              }
            />
            <MetricCard
              label="TTFT p50"
              value={formatNumber(overall!.metrics.ttft.median, 3)}
              note="秒 · 成功请求"
            />
            <MetricCard
              label="TPS p50"
              value={formatNumber(overall!.metrics.tps.median, 1)}
              note="tokens/s · 成功请求"
            />
          </div>
          <section className="surface chart-surface">
            <div className="section-head">
              <div>
                <span className="eyebrow">LATENCY PROFILE</span>
                <h2>延迟随{summary.group_axis}变化</h2>
              </div>
              <span className="minor-tag">
                指标契约 v{summary.metric_contract_version}
              </span>
            </div>
            <SlicesChart summary={summary} />
          </section>
          <section className="surface">
            <div className="section-head">
              <div>
                <span className="eyebrow">DISTRIBUTION</span>
                <h2>分组统计</h2>
              </div>
            </div>
            <div className="table-scroll">
              <table className="data-table stats-table">
                <thead>
                  <tr>
                    <th>{summary.group_axis}</th>
                    <th>正式 / 计划</th>
                    <th>失败</th>
                    <th>成功率</th>
                    <th>输入 token 中位数 / 目标</th>
                    <th>TTFT p50</th>
                    <th>TTFT p95</th>
                    <th>TTFT p99</th>
                    <th>TPS p50</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.groups.map((slice) => (
                    <tr key={slice.label}>
                      <td>
                        <strong>{slice.label}</strong>
                      </td>
                      <td>
                        {slice.requests} / {slice.planned_requests ?? "—"}
                      </td>
                      <td>{slice.failures}</td>
                      <td>{formatPercent(slice.success_rate)}</td>
                      <td>
                        {formatNumber(slice.input_tokens?.median, 1)} /{" "}
                        {slice.input_tokens?.target || "—"}
                      </td>
                      <td>{formatNumber(slice.metrics.ttft.median, 3)} s</td>
                      <td>{formatNumber(slice.metrics.ttft.p95, 3)} s</td>
                      <td>{formatNumber(slice.metrics.ttft.p99, 3)} s</td>
                      <td>{formatNumber(slice.metrics.tps.median, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="surface">
            <div className="section-head">
              <div>
                <span className="eyebrow">REQUEST OBSERVATIONS</span>
                <h2>逐请求样本</h2>
              </div>
              <span className="minor-tag">最近 50 条 · 不展示提示词正文</span>
            </div>
            <div className="table-scroll">
              <table className="data-table stats-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>并发</th>
                    <th>TTFT</th>
                    <th>TPS</th>
                    <th>总时长</th>
                    <th>输入 / 输出 token</th>
                    <th>Token 来源</th>
                    <th>提示词指纹</th>
                    <th>结果</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((row) => (
                    <tr key={row.id}>
                      <td>#{row.id}</td>
                      <td>{row.concurrency_level ?? "—"}</td>
                      <td>{formatNumber(row.ttft, 3)} s</td>
                      <td>{formatNumber(row.tps, 1)}</td>
                      <td>{formatNumber(row.total_time, 2)} s</td>
                      <td>
                        {row.prefill_tokens ?? "—"} / {row.decode_tokens ?? "—"}
                      </td>
                      <td>{row.token_source || "—"}</td>
                      <td title={row.prompt_sha256 || undefined}>
                        {row.prompt_sha256
                          ? row.prompt_sha256.slice(0, 12)
                          : "—"}
                      </td>
                      <td>
                        {row.error ? (
                          <span className="text-danger" title={row.error}>
                            失败
                          </span>
                        ) : (
                          <span className="text-good">成功</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <div className="methodology">
            <strong>统计口径</strong>
            <p>
              {summary.notes.join(" ")} Token 来源：
              {summary.provenance.token_sources.join("、") || "未记录"}。Token
              算法：{summary.provenance.token_methods.join("、") || "未记录"}。
            </p>
          </div>
        </>
      )}
      {quality && (
        <section className="surface">
          <div className="section-head">
            <div>
              <span className="eyebrow">QUALITY EVALUATION</span>
              <h2>数据集评估</h2>
            </div>
          </div>
          <div className="quality-grid">
            {Object.entries(quality.datasets).map(([name, value]) => (
              <div className="quality-card" key={name}>
                <div>
                  <span className="eyebrow">DATASET</span>
                  <h3>{name}</h3>
                </div>
                <strong>{formatPercent(value.accuracy)}</strong>
                <p>
                  {value.correct_samples} / {value.total_samples} 正确 ·{" "}
                  {formatNumber(value.duration_seconds, 1)} 秒
                </p>
                <small>
                  样本指纹：
                  {String(
                    (
                      value.config.dataset_provenance as
                        Record<string, unknown> | undefined
                    )?.sample_sha256 || "未记录",
                  )}
                </small>
              </div>
            ))}
          </div>
          <p className="chart-caption">
            准确率以数据集原始评分口径计算。导出 JSON
            可查看逐样本结果与完整来源信息。
          </p>
        </section>
      )}
      {quality && (
        <QualityAnalysis
          report={quality}
          onExportErrors={() => {
            void downloadFile(
              token,
              `/api/v1/jobs/${job.job_id}/report/errors.csv`,
              `llm-test-${shortId(job.job_id)}-errors.csv`,
            ).catch((exc) =>
              setError(exc instanceof Error ? exc.message : "下载失败"),
            );
          }}
        />
      )}
      {!summary && !quality && (
        <section className="surface">
          <Empty
            title={
              activeStates.has(job.status)
                ? "测量正在准备或执行"
                : "暂无可用结果"
            }
            text={
              activeStates.has(job.status)
                ? "数据写入后会自动显示；离开页面不影响任务。"
                : "检查任务事件或 worker 日志了解原因。"
            }
          />
        </section>
      )}
      {robustness && (
        <section className="surface">
          <div className="section-head">
            <div>
              <span className="eyebrow">ROBUSTNESS EVALUATION</span>
              <h2>鲁棒性评估</h2>
            </div>
            <span className="minor-tag">
              {robustness.robustness.total_samples} 样本 ×{" "}
              {robustness.robustness.perturbations_per_sample} 扰动
            </span>
          </div>
          <div className="metric-grid">
            <MetricCard
              label="原始准确率"
              value={formatPercent(robustness.robustness.original_accuracy)}
              note="未扰动"
              accent
            />
            <MetricCard
              label="扰动后准确率"
              value={formatPercent(robustness.robustness.perturbed_accuracy)}
              note={`落差 ${formatPercent(robustness.robustness.accuracy_drop)}`}
            />
            <MetricCard
              label="鲁棒性"
              value={formatPercent(robustness.robustness.overall_robustness)}
              note="扰动后保持正确比例"
            />
            <MetricCard
              label="一致性"
              value={formatPercent(robustness.robustness.overall_consistency)}
              note="扰动后答案一致比例"
            />
          </div>
          {Object.keys(robustness.robustness.sensitivity_by_type).length >
            0 && (
            <div className="table-scroll">
              <table className="data-table stats-table">
                <thead>
                  <tr>
                    <th>扰动类型</th>
                    <th>保持率（越低越敏感）</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(robustness.robustness.sensitivity_by_type)
                    .sort((a, b) => a[1] - b[1])
                    .map(([name, value]) => (
                      <tr key={name}>
                        <td>
                          <strong>{name}</strong>
                          {name ===
                            robustness.robustness
                              .most_sensitive_perturbation && (
                            <small className="text-danger"> · 最敏感</small>
                          )}
                        </td>
                        <td>{formatPercent(value)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="table-scroll">
            <table className="data-table stats-table">
              <thead>
                <tr>
                  <th>样本</th>
                  <th>原始正确</th>
                  <th>鲁棒性</th>
                  <th>一致性</th>
                </tr>
              </thead>
              <tbody>
                {robustness.robustness.results.map((row) => (
                  <tr key={row.sample_id}>
                    <td>#{row.sample_id}</td>
                    <td
                      className={
                        row.original_correct ? "text-good" : "text-danger"
                      }
                    >
                      {row.original_correct ? "正确" : "错误"}
                    </td>
                    <td>{formatPercent(row.robustness_score)}</td>
                    <td>{formatPercent(row.consistency_score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {robustness.robustness.recommendations.length > 0 && (
            <div className="gate-result">
              <strong>改进建议</strong>
              <ul>
                {robustness.robustness.recommendations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
      {logs.length > 0 && (
        <section className="surface">
          <div className="section-head">
            <div>
              <span className="eyebrow">EXECUTION LOG</span>
              <h2>执行日志</h2>
            </div>
            <span className="minor-tag">{logs.length} 条</span>
          </div>
          <div className="log-console" role="log" aria-label="执行日志">
            {logs.map((line) => (
              <div
                className={`log-line log-${line.level.toLowerCase()}`}
                key={line.id}
              >
                <span className="log-level">{line.level}</span>
                <span className="log-message">{line.message}</span>
              </div>
            ))}
          </div>
        </section>
      )}
      <section className="surface">
        <div className="section-head">
          <div>
            <span className="eyebrow">AUDIT TRAIL</span>
            <h2>运行事件</h2>
          </div>
        </div>
        <div className="event-list">
          {events.map((event) => (
            <div className="event" key={event.id}>
              <span className="event-dot" />
              <div>
                <strong>
                  {statusLabels[event.to_status] || event.to_status}
                </strong>
                <small>
                  {event.event} · {event.actor}
                </small>
              </div>
              <time>{date(event.created_at)}</time>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
