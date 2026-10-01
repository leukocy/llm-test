import { useMemo } from "react";
import type { Figure } from "plotly.js-dist-min";
import { PlotlyFigure } from "./PlotlyFigure";

export function RobustnessSensitivity({
  sensitivity,
}: {
  sensitivity: Record<string, number>;
}) {
  const figure = useMemo<Figure | null>(() => {
    const rows = Object.entries(sensitivity)
      .filter(([, value]) => Number.isFinite(value) && value >= 0 && value <= 1)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    if (!rows.length) return null;
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          y: rows.map(([name]) => name),
          x: rows.map(([, value]) => value * 100),
          marker: {
            color: rows.map(([, value]) =>
              value > 0.3 ? "#ef4444" : value > 0.1 ? "#eab308" : "#22c55e",
            ),
          },
          text: rows.map(([, value]) => `${(value * 100).toFixed(1)}%`),
          textposition: "auto",
          textangle: 0,
          hovertemplate: "%{y}<br>错误率 %{x:.1f}%<extra></extra>",
        },
      ],
      layout: {
        autosize: true,
        height: Math.max(320, rows.length * 36 + 100),
        margin: { l: 115, r: 25, t: 20, b: 55 },
        xaxis: {
          range: [0, 100],
          title: { text: "错误率 (%)" },
          ticksuffix: "%",
        },
        yaxis: { autorange: "reversed", automargin: true },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "system-ui", color: "#334155" },
      },
      config: { responsive: true, displaylogo: false },
    };
  }, [sensitivity]);
  return (
    <section aria-label="扰动敏感性分析">
      <h3>扰动敏感性分析</h3>
      <p className="chart-caption">
        按错误率从高到低排列，越高越敏感。颜色沿用原版展示阈值：≤10% 绿、10%–30%
        黄、&gt;30% 红；不是显著性或合格标准。
      </p>
      {figure ? (
        <PlotlyFigure figure={figure} ariaLabel="鲁棒性扰动敏感性" />
      ) : (
        <p className="chart-caption">
          没有有效的按类型评分数据，不绘制零值图。
        </p>
      )}
    </section>
  );
}
