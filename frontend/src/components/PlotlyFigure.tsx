import { useEffect, useRef, useState } from "react";
import type { Figure } from "plotly.js-dist-min";

type PlotlyModule = typeof import("plotly.js-dist-min").default;

let plotlyPromise: Promise<PlotlyModule> | null = null;

function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) {
    // 动态引入：plotly(~4MB)拆成异步 chunk, 只在有图表的页面加载
    plotlyPromise = import("plotly.js-dist-min").then((mod) => mod.default);
  }
  return plotlyPromise;
}

/** 后端 fig.to_plotly_json() 的直接渲染包装（plotly.js 懒加载）。 */
export function PlotlyFigure({
  figure,
  ariaLabel,
}: {
  figure: Figure | null | undefined;
  ariaLabel?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [plotly, setPlotly] = useState<PlotlyModule | null>(null);

  useEffect(() => {
    let active = true;
    if (figure && !plotly) {
      void loadPlotly().then((loaded) => {
        if (active) setPlotly(() => loaded);
      });
    }
    return () => {
      active = false;
    };
  }, [figure, plotly]);

  useEffect(() => {
    const el = ref.current;
    if (!el || !plotly || !figure) return;
    void plotly.react(el, figure);
    return () => {
      plotly.purge(el);
    };
  }, [plotly, figure]);

  if (!figure) return null;
  return (
    <div
      ref={ref}
      className="plotly-figure"
      role="img"
      aria-label={ariaLabel || "图表"}
      style={{ width: "100%", minHeight: 320 }}
    />
  );
}
