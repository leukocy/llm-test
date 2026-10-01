import { useEffect, useState } from "react";
import type { Figure } from "plotly.js-dist-min";
import { api, downloadFile } from "../api";
import { formatNumber } from "../constants";
import { PlotlyFigure } from "./PlotlyFigure";

type Visuals = {
  sample_id: string;
  dataset: string;
  index: number;
  latencies: {
    key: string;
    label: string;
    value: number | null;
    scope: string;
    figure: Figure | null;
  }[];
  tokens: { label: string; value: number | null }[];
  token_source: string;
  token_figure: Figure | null;
  notes: string[];
};
export function SampleVisuals({
  jobId,
  dataset,
  index,
  token,
}: {
  jobId: string;
  dataset: string;
  index: number;
  token: string;
}) {
  const url = `/api/v1/jobs/${jobId}/report/sample-visuals?dataset=${encodeURIComponent(dataset)}&index=${index}`;
  const [state, setState] = useState<{
    url: string;
    data?: Visuals;
    error?: string;
  } | null>(null);
  const [downloadError, setDownloadError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setDownloadError("");
    void api<Visuals>(token, url, { signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) setState({ url, data });
      })
      .catch((error) => {
        if (!controller.signal.aborted) setState({ url, error: String(error) });
      });
    return () => controller.abort();
  }, [token, url]);
  const current = state?.url === url ? state : null;
  const data = current?.data;
  return (
    <section className="sample-visuals" aria-label="逐样本计时与用量">
      <h3>逐样本计时与用量</h3>
      {current?.error ? (
        <p role="alert">{current.error}</p>
      ) : !data ? (
        <p role="status">正在读取样本测量…</p>
      ) : (
        <>
          <p className="chart-caption">
            样本 {data.sample_id} · 所有数值读取自原始报告，不重新调用模型。
          </p>
          <div className="quality-analysis-grid">
            {data.latencies.map((row) => (
              <section key={row.key}>
                <h4>{row.label}</h4>
                <p className="chart-caption">{row.scope}</p>
                {row.figure ? (
                  <PlotlyFigure
                    figure={row.figure}
                    ariaLabel={`逐样本-${row.label}-仪表`}
                  />
                ) : (
                  <p>{row.label} 未记录</p>
                )}
              </section>
            ))}
          </div>
          <h4>Token 分布</h4>
          <p className="chart-caption">来源：{data.token_source}</p>
          {data.token_figure ? (
            <PlotlyFigure
              figure={data.token_figure}
              ariaLabel="逐样本 Token 分布"
            />
          ) : (
            <p>缺少完整分区或全部计数为零，不绘制比例图。</p>
          )}
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>分区</th>
                  <th>Token</th>
                </tr>
              </thead>
              <tbody>
                {data.tokens.map((row) => (
                  <tr key={row.label}>
                    <td>{row.label}</td>
                    <td>
                      {row.value === null
                        ? "未记录"
                        : formatNumber(row.value, 0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.notes.map((note) => (
            <p className="chart-caption" key={note}>
              {note}
            </p>
          ))}
          <div className="api-form-actions">
            {(["json", "html"] as const).map((format) => (
              <button
                type="button"
                className="button subtle"
                key={format}
                onClick={() => {
                  setDownloadError("");
                  void downloadFile(
                    token,
                    `${url}&format=${format}`,
                    `llm-test-sample-${index}.${format}`,
                  ).catch((error) => setDownloadError(String(error)));
                }}
              >
                下载样本测量 {format.toUpperCase()}
              </button>
            ))}
          </div>
          {downloadError && <p role="alert">{downloadError}</p>}
        </>
      )}
    </section>
  );
}
