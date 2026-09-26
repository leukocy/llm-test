import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, type WarehouseData } from "../api";
import { Empty, MetricCard } from "../components";
import { formatNumber, labels } from "../constants";
import { WarehouseHistory } from "./WarehouseHistory";
import { WarehouseTrend } from "./WarehouseTrend";
import { WarehouseCases, WarehouseCapability } from "./WarehouseCases";
import { WarehouseAdmin, WarehouseExport } from "./WarehouseAdmin";

type Tab =
  | "history"
  | "trend"
  | "matrix"
  | "scaling"
  | "inventory"
  | "cases"
  | "capability"
  | "export"
  | "admin";

const TABS: [Tab, string][] = [
  ["history", "运行历史"],
  ["trend", "趋势对比"],
  ["matrix", "硬件 × 模型"],
  ["scaling", "扩展效率"],
  ["inventory", "硬件盘点"],
  ["cases", "应用用例"],
  ["capability", "客户能力表"],
  ["export", "模板导出"],
  ["admin", "数据管理"],
];

export function Warehouse({
  token,
  jobIds,
  onOpenJob,
}: {
  token: string;
  jobIds: Set<string>;
  onOpenJob: (id: string) => void;
}) {
  const [data, setData] = useState<WarehouseData | null>(null);
  const [model, setModel] = useState("");
  const [machine, setMachine] = useState("");
  const [testType, setTestType] = useState("");
  const [status, setStatus] = useState("");
  const [external, setExternal] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [metric, setMetric] = useState("decode_tps");
  const [aggregate, setAggregate] = useState("latest");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searchParams, setSearchParams] = useSearchParams();
  const tab = (searchParams.get("tab") as Tab) || "history";

  function setTab(next: Tab) {
    setSearchParams((current) => {
      const params = new URLSearchParams(current);
      if (next === "history") params.delete("tab");
      else params.set("tab", next);
      return params;
    });
  }

  useEffect(() => {
    const timeout = window.setTimeout(() => setSearch(searchInput.trim()), 300);
    return () => window.clearTimeout(timeout);
  }, [searchInput]);

  const query = useMemo(() => {
    const params = new URLSearchParams();
    if (model) params.set("model_id", model);
    if (machine) params.set("machine_id", machine);
    if (testType) params.set("test_type", testType);
    if (status) params.set("status", status);
    if (external) params.set("external_level", external);
    if (search) params.set("search", search);
    return params;
  }, [model, machine, testType, status, external, search]);

  const reload = useCallback(() => {
    const params = new URLSearchParams(query);
    params.set("metric", metric);
    params.set("aggregate", aggregate);
    return api<WarehouseData>(token, `/api/v1/warehouse?${params}`)
      .then((result) => {
        setData(result);
        setError("");
      })
      .catch((exc) =>
        setError(exc instanceof Error ? exc.message : "仓库查询失败"),
      );
  }, [token, query, metric, aggregate]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    reload().finally(() => {
      if (active) setLoading(false);
    });
    return () => {
      active = false;
    };
  }, [reload]);

  const matrix = data?.matrix;
  const maxMatrix = Math.max(
    0,
    ...Object.values(matrix?.cells || {}).flatMap((row) =>
      Object.values(row).filter(
        (value): value is number =>
          typeof value === "number" && Number.isFinite(value),
      ),
    ),
  );
  const exportBlocked =
    !data || data.scope.truncated || data.scope.invalid_rows > 0;
  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">MEASUREMENT WAREHOUSE</span>
          <h1>数据仓库</h1>
          <p>按模型与硬件检索历史测量，查看趋势、透视、用例和可复用数据。</p>
        </div>
      </div>
      <section className="surface warehouse-filter-surface">
        <div className="section-head">
          <div>
            <span className="eyebrow">FILTERS</span>
            <h2>筛选条件</h2>
          </div>
          <button
            className="text-button"
            onClick={() => {
              setModel("");
              setMachine("");
              setTestType("");
              setStatus("");
              setExternal("");
              setSearchInput("");
            }}
          >
            清除筛选
          </button>
        </div>
        <div className="warehouse-filters">
          <label>
            模型
            <select
              aria-label="筛选模型"
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              <option value="">全部模型</option>
              {(data?.filters.model_id || []).map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            硬件
            <select
              aria-label="筛选硬件"
              value={machine}
              onChange={(event) => setMachine(event.target.value)}
            >
              <option value="">全部硬件</option>
              {(data?.filters.machine_id || []).map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            测试类型
            <select
              aria-label="筛选测试类型"
              value={testType}
              onChange={(event) => setTestType(event.target.value)}
            >
              <option value="">全部类型</option>
              {(data?.filters.test_type || []).map((value) => (
                <option key={value} value={value}>
                  {labels[value] || value}
                </option>
              ))}
            </select>
          </label>
          <label>
            状态
            <select
              aria-label="筛选状态"
              value={status}
              onChange={(event) => setStatus(event.target.value)}
            >
              <option value="">全部状态</option>
              {(data?.filters.status || []).map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            对外等级
            <select
              aria-label="筛选对外等级"
              value={external}
              onChange={(event) => setExternal(event.target.value)}
            >
              <option value="">全部等级</option>
              <option value="internal">内部</option>
              <option value="review">待审核</option>
              <option value="publishable">可对外</option>
            </select>
          </label>
          <label>
            搜索
            <input
              aria-label="搜索仓库"
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
              placeholder="模型、测试员、备注"
            />
          </label>
        </div>
      </section>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {data && (
        <>
          <div className="metric-grid warehouse-kpis">
            <MetricCard
              label="筛选记录"
              value={String(data.kpis.runs)}
              note="复测链取最新"
              accent
            />
            <MetricCard
              label="硬件"
              value={String(data.kpis.machines)}
              note="已记录机器"
            />
            <MetricCard
              label="模型"
              value={String(data.kpis.models)}
              note="目标模型"
            />
            <MetricCard
              label="可对外"
              value={String(data.kpis.publishable)}
              note="已标记可发布"
            />
          </div>
          <div
            className={`warehouse-scope ${data.scope.truncated || data.scope.invalid_rows ? "warn" : ""}`}
          >
            已扫描 {data.scope.scanned} / {data.scope.matched_total}{" "}
            条匹配记录，展示 {data.scope.shown} 条。
            {data.scope.truncated &&
              ` 查询窗口最多 ${data.scope.scan_limit} 条；请缩小筛选条件。`}
            {data.scope.invalid_rows > 0 &&
              ` ${data.scope.invalid_rows} 条元数据无法解析。`}
          </div>
          <section className="surface warehouse-results">
            <div
              className="warehouse-tabs"
              role="tablist"
              aria-label="仓库视图"
            >
              {TABS.map(([id, label]) => (
                <button
                  key={id}
                  role="tab"
                  aria-selected={tab === id}
                  className={tab === id ? "active" : ""}
                  onClick={() => setTab(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            {tab === "history" && (
              <WarehouseHistory
                token={token}
                rows={data.rows}
                jobIds={jobIds}
                onOpenJob={onOpenJob}
                onChanged={() => void reload()}
              />
            )}
            {tab === "trend" && (
              <WarehouseTrend
                token={token}
                query={query}
                candidateIds={data.rows.map((row) => ({
                  test_id: row.test_id,
                  label: `${row.date || "—"} · ${row.model_name || "—"} · ${row.machine_id || "—"}`,
                }))}
              />
            )}
            {tab === "matrix" && (
              <>
                <div className="warehouse-matrix-controls">
                  <label>
                    指标
                    <select
                      aria-label="矩阵指标"
                      value={metric}
                      onChange={(event) => setMetric(event.target.value)}
                    >
                      <option value="decode_tps">decode TPS</option>
                      <option value="effective_bandwidth_gbps">
                        等效带宽 GB/s
                      </option>
                    </select>
                  </label>
                  <label>
                    同一格取值
                    <select
                      aria-label="矩阵聚合"
                      value={aggregate}
                      onChange={(event) => setAggregate(event.target.value)}
                    >
                      <option value="latest">最新一次</option>
                      <option value="best">最大观测值</option>
                    </select>
                  </label>
                </div>
                {matrix?.row_labels.length ? (
                  <div className="table-scroll">
                    <table className="data-table stats-table warehouse-matrix">
                      <thead>
                        <tr>
                          <th>硬件 / 模型</th>
                          {matrix.col_labels.map((name) => (
                            <th key={name}>{name}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {matrix.row_labels.map((name) => (
                          <tr key={name}>
                            <td>
                              <strong>{name}</strong>
                            </td>
                            {matrix.col_labels.map((modelName) => {
                              const value = matrix.cells[name]?.[modelName];
                              return (
                                <td
                                  key={modelName}
                                  className={
                                    value == null ? "matrix-empty" : ""
                                  }
                                  style={
                                    value != null && maxMatrix > 0
                                      ? {
                                          backgroundColor: `rgba(33, 169, 141, ${0.08 + 0.3 * Math.max(0, value / maxMatrix)})`,
                                        }
                                      : undefined
                                  }
                                >
                                  {formatNumber(value, 1)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Empty
                    title="矩阵缺少硬件标识"
                    text="运行记录需要 machine_id 才能参与硬件对比。"
                  />
                )}
                <p className="chart-caption">
                  空白格表示没有该模型与硬件组合的有效观测；最大值只描述样本，不代表统计显著优势。
                </p>
              </>
            )}
            {tab === "scaling" &&
              (data.scaling.length ? (
                <div className="table-scroll">
                  <table className="data-table stats-table">
                    <thead>
                      <tr>
                        <th>模型</th>
                        <th>TP 规模</th>
                        <th>decode TPS</th>
                        <th>相对单卡加速</th>
                        <th>扩展效率</th>
                        <th>解释</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.scaling.map((row) => (
                        <tr key={`${row.model_name}-${row.tp_size}`}>
                          <td>
                            <strong>{row.model_name}</strong>
                          </td>
                          <td>TP {row.tp_size}</td>
                          <td>{formatNumber(row.decode_tps, 1)}</td>
                          <td>{formatNumber(row.speedup_vs_tp1, 2)} ×</td>
                          <td>
                            {row.efficiency == null
                              ? "—"
                              : `${formatNumber(row.efficiency * 100, 1)}%`}
                          </td>
                          <td>{row.interpretation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <Empty
                  title="暂无扩展效率数据"
                  text="需有带并行策略与 decode TPS 的有效记录。"
                />
              ))}
            {tab === "inventory" &&
              (data.inventory.length ? (
                <div className="table-scroll">
                  <table className="data-table stats-table">
                    <thead>
                      <tr>
                        <th>机器</th>
                        <th>CPU</th>
                        <th>内存</th>
                        <th>GPU</th>
                        <th>卡数</th>
                        <th>显存</th>
                        <th>带宽</th>
                        <th>系统</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.inventory.map((row) => (
                        <tr key={String(row.machine_id)}>
                          <td>
                            <strong>{String(row.machine_id)}</strong>
                          </td>
                          <td>{String(row.cpu_model || "—")}</td>
                          <td>
                            {formatNumber(
                              row.memory_capacity_gb as number | null,
                              1,
                            )}{" "}
                            GB
                          </td>
                          <td>{String(row.gpu_model || "—")}</td>
                          <td>{row.gpu_count ?? "—"}</td>
                          <td>
                            {formatNumber(row.gpu_vram_gb as number | null, 1)}{" "}
                            GB
                          </td>
                          <td>
                            {formatNumber(
                              row.gpu_bandwidth_gbps as number | null,
                              1,
                            )}{" "}
                            GB/s
                          </td>
                          <td>{String(row.os || "—")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <Empty
                  title="尚无硬件指纹"
                  text="采集机器标识后可生成硬件清单。"
                />
              ))}
            {tab === "cases" && <WarehouseCases token={token} />}
            {tab === "capability" && <WarehouseCapability token={token} />}
            {tab === "export" && (
              <WarehouseExport
                token={token}
                query={query}
                exportBlocked={exportBlocked}
              />
            )}
            {tab === "admin" && <WarehouseAdmin token={token} />}
          </section>
        </>
      )}
      {loading && !data && (
        <section className="surface">
          <Empty title="正在读取仓库" text="加载历史测量及硬件数据…" />
        </section>
      )}
    </div>
  );
}
