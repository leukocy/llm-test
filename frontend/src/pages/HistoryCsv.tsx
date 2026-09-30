import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, downloadFile, type Summary } from "../api";
import { Empty, MetricCard } from "../components";
import { ScenarioAnalysis } from "../components/ScenarioAnalysis";
import { date, formatNumber, formatPercent, labels } from "../constants";

type Entry = {
  id: string;
  filename: string;
  model_id: string;
  provider: string;
  test_type: string;
  modified_at: number;
  bytes: number;
  revision: string;
  warnings: string[];
};
type Catalog = { items: Entry[]; truncated: boolean; skipped: number };
type SavedReport = {
  job: { job_id: string; test_type: string };
  entry: Entry;
  summary: Summary & {
    origin: {
      filename: string;
      csv_sha256: string;
      metadata_sha256: string | null;
      metadata: Record<string, unknown>;
    };
  };
  preview: {
    columns: string[];
    items: Record<string, string>[];
    total: number;
    offset: number;
    limit: number;
  };
};
export function HistoryCsv({ token }: { token: string }) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [report, setReport] = useState<SavedReport | null>(null);
  const [params, setParams] = useSearchParams();
  const [refresh, setRefresh] = useState(0);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [metadataFile, setMetadataFile] = useState<File | null>(null);
  const [model, setModel] = useState("");
  const [provider, setProvider] = useState("");
  const [testType, setTestType] = useState("unknown");
  const [uploadVersion, setUploadVersion] = useState(0);
  const identifier = params.get("file") || catalog?.items[0]?.id || "";
  const loaded = params.get("loaded") === "1";
  const selected = catalog?.items.find((item) => item.id === identifier);

  const reload = useCallback(async () => {
    setConfirmed(false);
    const result = await api<Catalog>(token, "/api/v1/history");
    setCatalog(result);
    return result;
  }, [token]);

  useEffect(() => {
    void reload().catch((exc) => setError(exc.message));
  }, [reload]);

  useEffect(() => {
    setConfirmed(false);
    setReport(null);
    setError("");
    if (!identifier || !loaded) {
      setLoading(false);
      return;
    }
    let active = true;
    setLoading(true);
    const query = new URLSearchParams({ offset: String(offset), limit: "50" });
    void api<SavedReport>(token, `/api/v1/history/${identifier}?${query}`)
      .then((data) => {
        if (active) setReport(data);
      })
      .catch((exc) => {
        if (active) setError(exc.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [token, identifier, loaded, offset, refresh]);

  async function upload() {
    if (!csvFile) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      if (
        csvFile.size > 10 * 1024 * 1024 ||
        (metadataFile && metadataFile.size > 65536)
      )
        throw new Error("CSV 上限 10 MiB，元数据上限 64 KiB");
      const body = new FormData();
      body.append("file", csvFile);
      body.append(
        "metadata",
        metadataFile ||
          new File(
            [
              JSON.stringify({
                model_id: model.trim(),
                provider: provider.trim(),
                test_type: testType,
              }),
            ],
            "metadata.json",
            { type: "application/json" },
          ),
      );
      const entry = await api<Entry>(token, "/api/v1/history/uploads", {
        method: "POST",
        body,
      });
      await reload();
      setOffset(0);
      setParams({ file: entry.id, loaded: "1" });
      setRefresh((value) => value + 1);
      setCsvFile(null);
      setMetadataFile(null);
      setUploadVersion((value) => value + 1);
      setMessage(
        "CSV 已保存并重新计算报告；相同文件与元数据再次上传会复用现有记录。",
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "上传失败");
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!selected || !confirmed) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await api(token, `/api/v1/history/${selected.id}/delete`, {
        method: "POST",
        body: JSON.stringify({ confirmed: true, revision: selected.revision }),
      });
      setParams({});
      setReport(null);
      setOffset(0);
      await reload();
      setMessage("CSV 与配套元数据已从历史列表移除，回收副本已保留。");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "删除失败");
    } finally {
      setBusy(false);
      setConfirmed(false);
    }
  }

  async function download(format: "csv" | "json" | "html" | "markdown") {
    if (!report) return;
    setBusy(true);
    setError("");
    try {
      const query = new URLSearchParams({
        format,
        revision: report.entry.revision,
      });
      await downloadFile(
        token,
        `/api/v1/history/${report.entry.id}/report?${query}`,
        `llm-test-history-${report.entry.id.slice(0, 12)}.${format === "markdown" ? "md" : format}`,
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "报告下载失败");
    } finally {
      setBusy(false);
    }
  }

  const overall = report?.summary.overall;
  return (
    <div className="page-grid history-csv">
      <div className="page-head">
        <div>
          <span className="eyebrow">SAVED MEASUREMENTS</span>
          <h1>历史 CSV</h1>
          <p>选择已保存的观测文件，重新生成统计、图表和报告。</p>
        </div>
      </div>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {message && (
        <p className="history-message" role="status">
          {message}
        </p>
      )}
      <section className="surface">
        <div className="section-head">
          <div>
            <h2>保存结果</h2>
            <p>
              {catalog ? `${catalog.items.length} 个 CSV` : "正在读取列表"} ·
              原始文件独立于测量任务
            </p>
          </div>
          <button
            className="text-button"
            disabled={busy || loading}
            onClick={() => {
              setError("");
              void reload()
                .then(() => setRefresh((value) => value + 1))
                .catch((exc) => setError(exc.message));
            }}
          >
            刷新列表
          </button>
        </div>
        {(catalog?.truncated || !!catalog?.skipped) && (
          <p className="note">
            {catalog.truncated
              ? "目录扫描达到上限，当前列表不代表全部文件。"
              : ""}
            {catalog.skipped
              ? ` 跳过 ${catalog.skipped} 个软链接、不可读或不安全的条目。`
              : ""}
          </p>
        )}
        {catalog?.items.length ? (
          <>
            <label>
              选择保存结果
              <select
                aria-label="选择保存结果"
                value={identifier}
                disabled={busy || loading}
                onChange={(event) => {
                  setOffset(0);
                  setParams({ file: event.target.value });
                  setMessage("");
                }}
              >
                {catalog.items.map((item) => (
                  <option key={item.id} value={item.id}>
                    {date(item.modified_at)} | {item.model_id} |{" "}
                    {labels[item.test_type] || "未知类型"} | {item.filename}
                  </option>
                ))}
              </select>
            </label>
            {selected && (
              <p className="history-file-name">
                {selected.filename} · {selected.provider} ·{" "}
                {(selected.bytes / 1024).toFixed(1)} KiB
              </p>
            )}
            <div className="history-actions">
              <button
                className="button primary"
                disabled={busy || loading || !selected}
                onClick={() => {
                  setOffset(0);
                  setParams({ file: identifier, loaded: "1" });
                  setRefresh((value) => value + 1);
                }}
              >
                {loading ? "正在重算报告…" : "加载并重绘报告"}
              </button>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={confirmed}
                  disabled={busy || loading}
                  onChange={(event) => setConfirmed(event.target.checked)}
                />
                确认移除所选 CSV 与元数据
              </label>
              <button
                className="button danger"
                disabled={busy || loading || !confirmed || !selected}
                onClick={() => void remove()}
              >
                删除所选结果
              </button>
            </div>
            <p className="note">
              删除会保留回收副本。加载与导出只处理保存的观测；不会发起新的模型请求。
            </p>
          </>
        ) : (
          catalog && (
            <Empty
              title="尚无保存的 CSV"
              text="可上传旧结果与配套元数据，再从此处加载报告。"
            />
          )
        )}
      </section>
      <section className="surface">
        <div className="section-head">
          <div>
            <h2>上传历史结果</h2>
            <p>
              支持首次版本与当前平台导出的原始性能指标
              CSV；缺失的单位和状态不会被猜测。
            </p>
          </div>
        </div>
        <div className="history-upload-grid" key={uploadVersion}>
          <label>
            CSV 文件 · 最多 10 MiB
            <input
              aria-label="历史 CSV 文件"
              type="file"
              accept=".csv,text/csv"
              disabled={busy}
              onChange={(event) => setCsvFile(event.target.files?.[0] || null)}
            />
          </label>
          <label>
            配套元数据 JSON · 可选，最多 64 KiB
            <input
              aria-label="历史元数据 JSON"
              type="file"
              accept=".json,application/json"
              disabled={busy}
              onChange={(event) =>
                setMetadataFile(event.target.files?.[0] || null)
              }
            />
          </label>
        </div>
        {!metadataFile && (
          <div className="history-upload-grid">
            <label>
              原模型 · 可选
              <input
                value={model}
                maxLength={200}
                onChange={(event) => setModel(event.target.value)}
                placeholder="未填写时标为未知或从保存目录推断"
              />
            </label>
            <label>
              原服务商 · 可选
              <input
                value={provider}
                maxLength={120}
                onChange={(event) => setProvider(event.target.value)}
              />
            </label>
            <label>
              原测试类型
              <select
                value={testType}
                onChange={(event) => setTestType(event.target.value)}
              >
                <option value="unknown">未记录 / 从文件名推断</option>
                {[
                  "concurrency",
                  "prefill",
                  "segmented_prefill",
                  "long_context",
                  "throughput_matrix",
                  "custom_text",
                  "stability",
                ].map((value) => (
                  <option key={value} value={value}>
                    {labels[value] || value}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}
        <button
          className="button subtle"
          disabled={busy || loading || !csvFile}
          onClick={() => void upload()}
        >
          保存 CSV 并生成报告
        </button>
      </section>
      {report && overall && (
        <>
          <section className="surface history-origin">
            <span className="eyebrow">HISTORICAL OBSERVATIONS</span>
            <h2 className="history-file-name">
              {report.entry.model_id} ·{" "}
              {labels[report.entry.test_type] || "未知测试类型"}
            </h2>
            <p>历史 CSV · 原执行状态、预热与计划请求数未经控制队列核验。</p>
            <p className="note">
              {overall.unknown_outcomes || 0}{" "}
              条请求的成功状态未知；成功率仅在已知状态请求中计算。
            </p>
            <code>CSV SHA-256：{report.summary.origin.csv_sha256}</code>
            {report.summary.origin.metadata_sha256 && (
              <code>
                元数据 SHA-256：{report.summary.origin.metadata_sha256}
              </code>
            )}
          </section>
          <section className="metric-grid">
            <MetricCard
              label="请求记录"
              value={formatNumber(overall.requests, 0)}
              note={`已知成功 ${overall.successes} · 失败 ${overall.failures} · 未知 ${overall.unknown_outcomes || 0}`}
            />
            <MetricCard
              label="已知状态成功率"
              value={formatPercent(overall.success_rate)}
              note="分母仅含已知状态请求"
            />
            <MetricCard
              label="TTFT p50 · s"
              value={formatNumber(overall.metrics.ttft.median, 3)}
              note={`${overall.metrics.ttft.count} 个有效成功样本`}
            />
            <MetricCard
              label="TPS p50 · token/s"
              value={formatNumber(overall.metrics.tps.median, 2)}
              note={`${overall.metrics.tps.count} 个有效成功样本`}
            />
          </section>
          <ScenarioAnalysis
            job={report.job}
            summary={report.summary}
            token={token}
            figureEndpoint={`/api/v1/history/${report.entry.id}/figure`}
            revision={report.entry.revision}
          />
          <section className="surface">
            <div className="section-head">
              <div>
                <h2>条件统计与报告</h2>
                <p>逐请求统计口径与当前平台一致；不推断差异显著性。</p>
              </div>
              <div className="history-actions">
                {(["html", "markdown", "json", "csv"] as const).map(
                  (format) => (
                    <button
                      key={format}
                      className="text-button"
                      disabled={busy || loading}
                      onClick={() => void download(format)}
                    >
                      {format === "csv" ? "安全 CSV" : format.toUpperCase()}
                    </button>
                  ),
                )}
              </div>
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>{report.summary.group_axis}</th>
                    <th>请求</th>
                    <th>成功 / 失败 / 未知</th>
                    <th>TTFT 有效 n</th>
                    <th>TTFT p50 · s</th>
                    <th>TTFT p95 · s</th>
                    <th>TPS p50 · token/s</th>
                  </tr>
                </thead>
                <tbody>
                  {report.summary.groups.map((group) => (
                    <tr key={group.label}>
                      <td>{group.label}</td>
                      <td>{group.requests}</td>
                      <td>
                        {group.successes} / {group.failures} /{" "}
                        {group.unknown_outcomes || 0}
                      </td>
                      <td>{group.metrics.ttft.count}</td>
                      <td>{formatNumber(group.metrics.ttft.median, 3)}</td>
                      <td>{formatNumber(group.metrics.ttft.p95, 3)}</td>
                      <td>{formatNumber(group.metrics.tps.median, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <details>
              <summary>测量方法与数据质量</summary>
              <ul>
                {[
                  ...report.summary.notes,
                  ...report.summary.data_quality.warnings,
                ].map((note, index) => (
                  <li key={index}>{note}</li>
                ))}
              </ul>
            </details>
            <details>
              <summary>原配置与硬件说明 · 未核验</summary>
              <pre>
                {JSON.stringify(report.summary.origin.metadata, null, 2)}
              </pre>
            </details>
          </section>
          <section className="surface">
            <div className="section-head">
              <div>
                <h2>原始观测预览</h2>
                <p>
                  当前第 {report.preview.offset + 1}～
                  {Math.min(
                    report.preview.offset + report.preview.limit,
                    report.preview.total,
                  )}{" "}
                  条，共 {report.preview.total} 条；单格最多展示 2,000 字符。
                </p>
              </div>
              <div className="history-actions">
                <button
                  className="text-button"
                  disabled={busy || loading || offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - 50))}
                >
                  上一页
                </button>
                <button
                  className="text-button"
                  disabled={
                    busy || loading || offset + 50 >= report.preview.total
                  }
                  onClick={() => setOffset(offset + 50)}
                >
                  下一页
                </button>
              </div>
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    {report.preview.columns.map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {report.preview.items.map((row, index) => (
                    <tr key={index}>
                      {report.preview.columns.map((column) => (
                        <td key={column}>{row[column] || "—"}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
