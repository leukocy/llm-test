import type { Job, Summary } from "./api";
import { labels, statusLabels, date, shortId, formatNumber } from "./constants";

export function Mark() {
  return (
    <span className="mark" aria-hidden="true">
      <svg viewBox="0 0 40 40">
        <path
          d="M5 28h7l5-16 6 21 5-12h7"
          fill="none"
          stroke="currentColor"
          strokeWidth="3.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="33" cy="21" r="3" fill="currentColor" />
      </svg>
    </span>
  );
}

export function Status({ value }: { value: string }) {
  return (
    <span className={`status status-${value}`}>
      <i />
      {statusLabels[value] || value}
    </span>
  );
}

export function MetricCard({
  label,
  value,
  note,
  accent,
}: {
  label: string;
  value: string;
  note: string;
  accent?: boolean;
}) {
  return (
    <div className={`metric-card ${accent ? "metric-card-accent" : ""}`}>
      <div className="metric-label">{label}</div>
      <strong>{value}</strong>
      <div className="metric-note">{note}</div>
    </div>
  );
}

export function SlicesChart({ summary }: { summary: Summary }) {
  const slices = summary.groups.filter(
    (row) => row.metrics.ttft?.median != null,
  );
  if (!slices.length)
    return (
      <Empty
        title="尚无可绘制的延迟数据"
        text="完成测量后将显示每个并发级别的首字延迟。"
      />
    );
  const max = Math.max(
    ...slices.map((row) => row.metrics.ttft.p95 || 0),
    0.001,
  );
  return (
    <div className="chart-wrap">
      <div className="chart-legend">
        <span>
          <i className="legend-median" />
          TTFT p50
        </span>
        <span>
          <i className="legend-p95" />
          TTFT p95
        </span>
        <em>单位：秒 · 成功请求</em>
      </div>
      <div
        className="bar-chart"
        role="img"
        aria-label={`各${summary.group_axis}条件的首字延迟中位数与第95百分位`}
      >
        {slices.map((row) => (
          <div className="bar-group" key={row.label}>
            <div className="bars">
              <div
                className="bar bar-p95"
                style={{
                  height: `${Math.max(2, ((row.metrics.ttft.p95 || 0) / max) * 100)}%`,
                }}
                title={`p95 ${formatNumber(row.metrics.ttft.p95, 3)} 秒`}
              />
              <div
                className="bar bar-median"
                style={{
                  height: `${Math.max(2, ((row.metrics.ttft.median || 0) / max) * 100)}%`,
                }}
                title={`p50 ${formatNumber(row.metrics.ttft.median, 3)} 秒`}
              />
            </div>
            <span>{row.label}</span>
          </div>
        ))}
      </div>
      <p className="chart-caption">
        每组独立计算分位数。样本数与失败请求见下方统计表。
      </p>
    </div>
  );
}

export function Empty({ title, text }: { title: string; text: string }) {
  return (
    <div className="empty">
      <span className="empty-symbol">◇</span>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

export function JobTable({
  jobs,
  onSelect,
  compact = false,
  savedProgress = false,
}: {
  jobs: Job[];
  onSelect: (job: Job) => void;
  compact?: boolean;
  savedProgress?: boolean;
}) {
  if (!jobs.length)
    return (
      <Empty
        title="还没有运行记录"
        text="创建一次测量，结果会自动出现在这里。"
      />
    );
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>测试任务</th>
            <th>目标模型</th>
            <th>状态</th>
            <th>进度</th>
            {savedProgress && <th>保存进度</th>}
            <th>创建时间</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.slice(0, compact ? 6 : undefined).map((job) => (
            <tr
              key={job.job_id}
              onClick={() => onSelect(job)}
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter") onSelect(job);
              }}
            >
              <td>
                <strong>{labels[job.test_type] || job.test_type}</strong>
                <small>#{shortId(job.job_id)}</small>
              </td>
              <td className="model-cell">{job.model_id}</td>
              <td>
                <Status value={job.status} />
              </td>
              <td>
                <div className="progress-cell">
                  <div className="progress-track">
                    <i
                      style={{
                        width: `${job.progress_total ? Math.min(100, (job.progress_completed / job.progress_total) * 100) : 0}%`,
                      }}
                    />
                  </div>
                  <small>
                    {job.progress_total
                      ? `${job.progress_completed}/${job.progress_total}`
                      : "待开始"}
                  </small>
                </div>
              </td>
              {savedProgress && (
                <td>
                  <strong>
                    {job.saved_progress_committed ?? 0} /{" "}
                    {job.saved_progress_planned ?? 0} 个样本
                  </strong>
                  <small>保存于 {date(job.saved_progress_at)}</small>
                </td>
              )}
              <td className="muted-cell">{date(job.created_at)}</td>
              <td className="row-arrow">↗</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
