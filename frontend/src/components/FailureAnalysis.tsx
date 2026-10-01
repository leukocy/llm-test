import type { QualityReportAnalysis } from "../api";
import { formatPercent } from "../constants";
import { PlotlyFigure } from "./PlotlyFigure";

export function FailureAnalysis({
  summary,
}: {
  summary: QualityReportAnalysis["datasets"][string]["failure_analysis"];
}) {
  return (
    <details className="surface automated-failure-analysis" open>
      <summary>自动失败分析报告</summary>
      <p>
        最终回答错误：{summary.failed_samples} / {summary.response_samples} ·
        错误率 {formatPercent(summary.failure_rate)} · 请求错误另列{" "}
        {summary.request_errors}
      </p>
      <p className="chart-caption">
        来源：{summary.source}。{summary.note}
      </p>
      {summary.status === "none" && <p>完整样本记录中未发现最终回答错误。</p>}
      {summary.status === "unavailable" && (
        <p>自动分类未记录或覆盖未经核验；保留原始样本供人工查看。</p>
      )}
      <div className="quality-analysis-grid">
        <div>
          <h3>主要问题</h3>
          <ul>
            {summary.top_issues.map((issue, index) => (
              <li key={index}>{issue}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3>改进建议</h3>
          <ul>
            {summary.suggestions.map((suggestion, index) => (
              <li key={index}>{suggestion}</li>
            ))}
          </ul>
        </div>
      </div>
      <PlotlyFigure figure={summary.figure} ariaLabel="自动失败分类分布" />
      {summary.warnings.map((warning, index) => (
        <p key={index} className="form-warning">
          {warning}
        </p>
      ))}
    </details>
  );
}
