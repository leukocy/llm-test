import { useEffect, useState } from "react";
import { api } from "../api";
import { Empty } from "../components";
import { PlotlyFigure } from "../components/PlotlyFigure";
import type { Figure } from "plotly.js-dist-min";
import {
  compareGroups,
  formatNumber,
  trendDims,
  trendMetrics,
} from "../constants";

type TrendPayload = { figure: Figure; count: number };
type ComparePayload = {
  table: Record<string, Record<string, string | number | null>>;
  figures: Record<string, Figure>;
  missing_test_ids: string[];
};

export function WarehouseTrend({
  token,
  query,
  candidateIds,
}: {
  token: string;
  query: URLSearchParams;
  candidateIds: { test_id: string; label: string }[];
}) {
  const [metric, setMetric] = useState("decode_tps");
  const [dim, setDim] = useState("model_name");
  const [publishableOnly, setPublishableOnly] = useState(false);
  const [trend, setTrend] = useState<TrendPayload | null>(null);
  const [trendError, setTrendError] = useState("");

  const [picked, setPicked] = useState<string[]>([]);
  const [compare, setCompare] = useState<ComparePayload | null>(null);
  const [compareError, setCompareError] = useState("");
  const [compareBusy, setCompareBusy] = useState(false);

  useEffect(() => {
    let active = true;
    const params = new URLSearchParams(query);
    params.set("metric", metric);
    params.set("group_dim", dim);
    if (publishableOnly) params.set("publishable_only", "true");
    api<TrendPayload>(token, `/api/v1/warehouse/figures/trend?${params}`)
      .then((data) => {
        if (active) {
          setTrend(data);
          setTrendError("");
        }
      })
      .catch((exc) => {
        if (active)
          setTrendError(exc instanceof Error ? exc.message : "趋势图加载失败");
      });
    return () => {
      active = false;
    };
  }, [token, query, metric, dim, publishableOnly]);

  async function runCompare() {
    setCompareBusy(true);
    setCompareError("");
    try {
      const data = await api<ComparePayload>(
        token,
        "/api/v1/warehouse/figures/compare",
        {
          method: "POST",
          body: JSON.stringify({ test_ids: picked }),
        },
      );
      setCompare(data);
    } catch (exc) {
      setCompare(null);
      setCompareError(exc instanceof Error ? exc.message : "对比失败");
    } finally {
      setCompareBusy(false);
    }
  }

  return (
    <div className="warehouse-trend">
      <div className="warehouse-matrix-controls">
        <label>
          指标
          <select
            aria-label="趋势指标"
            value={metric}
            onChange={(event) => setMetric(event.target.value)}
          >
            {Object.entries(trendMetrics).map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          分组维度
          <select
            aria-label="趋势分组维度"
            value={dim}
            onChange={(event) => setDim(event.target.value)}
          >
            {Object.entries(trendDims).map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={publishableOnly}
            onChange={(event) => setPublishableOnly(event.target.checked)}
          />
          仅可发布
        </label>
      </div>
      {trendError && (
        <div className="alert" role="alert">
          {trendError}
        </div>
      )}
      {trend && trend.figure.data.length ? (
        <>
          <PlotlyFigure figure={trend.figure} ariaLabel="指标趋势图" />
          <p className="chart-caption">
            {trend.count} 条记录参与绘图；按日期升序，hover 可见 test_id。
          </p>
        </>
      ) : (
        !trendError && (
          <Empty title="当前筛选下无可绘数据" text="换个指标或松开筛选条件。" />
        )
      )}

      <div className="section-head split">
        <div>
          <span className="eyebrow">COMPARE</span>
          <h3>运行对比</h3>
        </div>
      </div>
      <div className="compare-picker">
        <select
          aria-label="选择对比运行"
          multiple
          size={Math.min(8, Math.max(3, candidateIds.length))}
          value={picked}
          onChange={(event) => {
            const values = Array.from(event.target.selectedOptions).map(
              (option) => option.value,
            );
            setPicked(values.slice(0, 8));
          }}
        >
          {candidateIds.map((item) => (
            <option key={item.test_id} value={item.test_id}>
              {item.label}
            </option>
          ))}
        </select>
        <div className="compare-actions">
          <p>已选 {picked.length} / 8 条（2 条起）。按住 Ctrl/Cmd 多选。</p>
          <button
            className="button primary"
            disabled={picked.length < 2 || compareBusy}
            onClick={() => void runCompare()}
          >
            {compareBusy ? "对比中…" : "生成对比 →"}
          </button>
        </div>
      </div>
      {compareError && (
        <div className="alert" role="alert">
          {compareError}
        </div>
      )}
      {compare && (
        <>
          <div className="table-scroll">
            <table className="data-table stats-table">
              <thead>
                <tr>
                  <th>指标</th>
                  {Object.keys(
                    compare.table[Object.keys(compare.table)[0]] || {},
                  ).map((label) => (
                    <th key={label}>{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(compare.table).map(([metricName, byRun]) => (
                  <tr key={metricName}>
                    <td>
                      <strong>{metricName}</strong>
                    </td>
                    {Object.entries(byRun).map(([label, value]) => (
                      <td key={label}>
                        {typeof value === "number"
                          ? formatNumber(value, 2)
                          : "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {compareGroups.map((group) =>
            compare.figures[group] ? (
              <PlotlyFigure
                key={group}
                figure={compare.figures[group]}
                ariaLabel={`${group}对比柱状图`}
              />
            ) : null,
          )}
        </>
      )}
    </div>
  );
}
