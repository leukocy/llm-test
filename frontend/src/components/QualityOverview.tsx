import type { QualityReport } from "../api";
import { formatNumber, formatPercent } from "../constants";
import { PlotlyFigure } from "./PlotlyFigure";

export const qualityMetrics: Record<string, string> = {
  ttft_ms: "首内容延迟 / ms",
  tps: "单次响应速度 / tokens/s",
  total_time_ms: "响应总时间 / ms",
  latency_ms: "样本调用耗时 / ms",
  input_tokens: "输入 Token",
  output_tokens: "输出 Token",
};

export function QualityOverview({
  report,
  onExportSummary,
}: {
  report: QualityReport;
  onExportSummary: () => void;
}) {
  const analysis = report.analysis;
  if (!analysis) return null;
  const rows = Object.values(analysis.datasets);
  return (
    <section className="surface quality-overview">
      <div className="section-head">
        <div>
          <span className="eyebrow">QUALITY SUMMARY</span>
          <h2>质量汇总与性能对照</h2>
        </div>
        <button className="button subtle" onClick={onExportSummary}>
          下载质量汇总 CSV ↓
        </button>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>数据集</th>
              <th>模型</th>
              <th>最终准确率</th>
              <th>正确 / 总数</th>
              <th>规则正确</th>
              <th>Judge 改判</th>
              <th>无错误标记</th>
              <th>95% Wilson</th>
              <th>耗时 / 秒</th>
              <th>评测时间</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}>
                <td>{row.name}</td>
                <td>{row.model}</td>
                <td>{formatPercent(row.accuracy)}</td>
                <td>
                  {row.correct ?? "—"} / {row.total ?? "—"}
                </td>
                <td>{row.standard_correct ?? "未记录"}</td>
                <td>{row.judge_corrected ?? "未记录"}</td>
                <td>{formatPercent(row.non_error_fraction)}</td>
                <td>
                  {row.ci95
                    ? `${formatPercent(row.ci95[0])}–${formatPercent(row.ci95[1])}`
                    : "未计算"}
                </td>
                <td>{formatNumber(row.duration_seconds, 1)}</td>
                <td>{row.timestamp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="quality-analysis-grid">
        <PlotlyFigure
          figure={analysis.figures.accuracy}
          ariaLabel="数据集准确率柱图"
        />
        {analysis.figures.radar ? (
          <PlotlyFigure
            figure={analysis.figures.radar}
            ariaLabel="数据集成绩雷达图"
          />
        ) : (
          <p className="chart-caption">至少三个有成绩的数据集才显示雷达图。</p>
        )}
      </div>
      <div className="quality-analysis-grid">
        <PlotlyFigure
          figure={analysis.figures.ttft_ms}
          ariaLabel="质量首内容延迟对照"
        />
        <PlotlyFigure
          figure={analysis.figures.tps}
          ariaLabel="质量响应生成速度对照"
        />
      </div>
      <h3>逐样本性能统计</h3>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>数据集</th>
              <th>指标</th>
              <th>观测 / 可用</th>
              <th>均值</th>
              <th>中位数</th>
              <th>P95</th>
              <th>P99</th>
              <th>最小</th>
              <th>最大</th>
              <th>Token 合计</th>
              <th>Token 来源</th>
            </tr>
          </thead>
          <tbody>
            {rows.flatMap((row) =>
              Object.entries(qualityMetrics).map(([key, label]) => {
                const stat = row.metrics[key];
                return (
                  <tr key={`${row.name}-${key}`}>
                    <td>{row.name}</td>
                    <td>{label}</td>
                    <td>
                      {stat.count} / {stat.eligible}
                    </td>
                    {[
                      stat.mean,
                      stat.median,
                      stat.p95,
                      stat.p99,
                      stat.min,
                      stat.max,
                    ].map((value, index) => (
                      <td key={index}>{formatNumber(value, 2)}</td>
                    ))}
                    <td>
                      {key.endsWith("tokens")
                        ? formatNumber(stat.total, 0)
                        : "—"}
                    </td>
                    <td>
                      {Object.entries(stat.sources)
                        .map(
                          ([source, count]) =>
                            `${source === "api" ? "API 用量" : source === "tokenizer" ? "本地估计" : "未记录"} ${count}`,
                        )
                        .join(" · ") || "—"}
                    </td>
                  </tr>
                );
              }),
            )}
          </tbody>
        </table>
      </div>
      {analysis.notes.map((note) => (
        <p key={note} className="chart-caption">
          {note}
        </p>
      ))}
      {rows.flatMap((row) =>
        row.warnings.map((warning) => (
          <p className="form-warning" key={`${row.name}-${warning}`}>
            {row.name}：{warning}
          </p>
        )),
      )}
    </section>
  );
}
