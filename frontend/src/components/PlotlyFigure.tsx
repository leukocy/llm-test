import { useEffect, useRef, useState } from "react";
import type { Figure } from "plotly.js-dist-min";

type PlotlyModule = typeof import("plotly.js-dist-min").default;

let plotlyPromise: Promise<PlotlyModule> | null = null;

function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) {
    // 动态引入：plotly(~4MB)拆成异步 chunk, 只在有图表的页面加载
    plotlyPromise = import("plotly.js-dist-min")
      .then((mod) => mod.default)
      .catch((error) => {
        plotlyPromise = null;
        throw error;
      });
  }
  return plotlyPromise;
}

/** Render an isolated export so downloading never resizes the visible chart. */
export async function downloadFigurePng(figure: Figure, filename: string) {
  const plotly = await loadPlotly();
  const el = document.createElement("div");
  const width = Math.min(
    2400,
    Math.max(1400, Number(figure.layout.width) || 1500),
  );
  const height = Math.min(
    2400,
    Math.max(800, Number(figure.layout.height) || 900),
  );
  el.style.cssText = `position:fixed;left:-10000px;top:0;width:${width}px;height:${height}px;`;
  document.body.appendChild(el);
  try {
    const exported = structuredClone(figure);
    exported.layout = { ...exported.layout, width, height, autosize: false };
    await plotly.react(el, exported);
    const url = await plotly.toImage(el, {
      format: "png",
      width,
      height,
      scale: 2,
    });
    const link = document.createElement("a");
    const name =
      filename.replace(/[<>:"/\\|?*\x00-\x1f\x7f]/g, "-").slice(0, 100) ||
      "llm-test-chart";
    link.href = url;
    link.download = `${name.replace(/\.png$/i, "")}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  } finally {
    try {
      plotly.purge(el);
    } finally {
      el.remove();
    }
  }
}

/** 后端 fig.to_plotly_json() 的直接渲染包装（plotly.js 懒加载）。 */
export function PlotlyFigure({
  figure,
  exportFigure,
  ariaLabel,
}: {
  figure: Figure | null | undefined;
  exportFigure?: Figure;
  ariaLabel?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [plotly, setPlotly] = useState<PlotlyModule | null>(null);
  const [error, setError] = useState("");
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    let active = true;
    if (figure && !plotly) {
      void loadPlotly()
        .then((loaded) => {
          if (active) setPlotly(() => loaded);
        })
        .catch((exc) => {
          if (active)
            setError(exc instanceof Error ? exc.message : "图表加载失败");
        });
    }
    return () => {
      active = false;
    };
  }, [figure, plotly]);

  useEffect(() => {
    const el = ref.current;
    if (!el || !plotly || !figure) return;
    let active = true;
    setError("");
    void plotly.react(el, figure).catch((exc) => {
      if (active) setError(exc instanceof Error ? exc.message : "图表渲染失败");
    });
    const observer = new ResizeObserver(() => {
      if (active && el.clientWidth > 0)
        void plotly.Plots.resize(el).catch((exc) => {
          if (active)
            setError(exc instanceof Error ? exc.message : "图表缩放失败");
        });
    });
    observer.observe(el);
    return () => {
      active = false;
      observer.disconnect();
      plotly.purge(el);
    };
  }, [plotly, figure]);

  if (!figure) return null;
  return (
    <div className="plotly-container">
      <div className="chart-export-actions">
        <button
          type="button"
          className="text-action"
          disabled={exporting}
          aria-label={`导出${ariaLabel || "图表"} PNG`}
          onClick={async () => {
            setExporting(true);
            setError("");
            try {
              await downloadFigurePng(
                exportFigure || figure,
                `llm-test-${ariaLabel || "chart"}`,
              );
            } catch (exc) {
              setError(exc instanceof Error ? exc.message : "PNG 导出失败");
            } finally {
              setExporting(false);
            }
          }}
        >
          {exporting ? "生成 PNG…" : "导出 PNG ↓"}
        </button>
      </div>
      <div
        ref={ref}
        className="plotly-figure"
        role="img"
        aria-label={ariaLabel || "图表"}
        style={{ width: "100%", minHeight: 320 }}
      />
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
