import type { ReasoningAnalysisSummary } from "../api";
import { formatNumber } from "../constants";
import { PlotlyFigure } from "./PlotlyFigure";

export function ReasoningAnalysis({
  summary,
}: {
  summary: ReasoningAnalysisSummary;
}) {
  return (
    <section className="surface reasoning-analysis">
      <h3>五维推理规则诊断</h3>
      <p className="chart-caption">
        来源：{summary.version} · {summary.source} · 雷达共同样本 n=
        {summary.radar_n} / {summary.total}。{summary.note}
      </p>
      {summary.figure ? (
        <PlotlyFigure figure={summary.figure} ariaLabel="五维推理规则雷达图" />
      ) : (
        <p>没有五维均有记录的共同样本，不绘制完整雷达图。</p>
      )}
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>维度</th>
              <th>有记录 n</th>
              <th>均值 / 10</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(summary.dimensions).map(([dimension, stat]) => (
              <tr key={dimension}>
                <td>{stat.label}</td>
                <td>{stat.n}</td>
                <td>
                  {stat.mean === null ? "未记录" : formatNumber(stat.mean, 2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="chart-caption">
        未记录或版本不支持的样本不使用历史默认零推算评分；本图没有独立验证或显著性结论。
      </p>
    </section>
  );
}
