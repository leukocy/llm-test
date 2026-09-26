import { useEffect, useState } from "react";
import { api, type WarehouseDetail, type WarehouseRow } from "../api";
import { Empty } from "../components";
import { PlotlyFigure } from "../components/PlotlyFigure";
import type { Figure } from "plotly.js-dist-min";
import { formatNumber, labels } from "../constants";

type GateResult = {
  level: string;
  passed: boolean;
  gates: Record<string, boolean>;
  reasons: string[];
};

type RunFigures = {
  found: boolean;
  result_count: number;
  distributions: Record<string, { histogram: Figure | null; box: Figure | null }>;
  engine: {
    figure: Figure | null;
    summary: Record<string, unknown>;
  };
};

const GATE_LABELS: Record<string, string> = {
  config_complete: "闸 1 · 配置齐全",
  reproducible: "闸 2 · 可复现",
  metrics_trustworthy: "闸 3 · 指标可信",
  external_reviewed: "闸 4 · 人工复核",
};

const EDIT_FIELDS = [
  ["external_level", "可对外等级"],
  ["tester", "测试员"],
  ["comparison_group", "对比组"],
  ["bottleneck", "瓶颈归因"],
  ["status_detail", "状态明细"],
  ["next_action", "下一步"],
  ["tags", "标签（逗号分隔）"],
  ["notes", "备注"],
] as const;

export function WarehouseHistory({
  token,
  rows,
  jobIds,
  onOpenJob,
  onChanged,
}: {
  token: string;
  rows: WarehouseRow[];
  jobIds: Set<string>;
  onOpenJob: (id: string) => void;
  onChanged: () => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [msg, setMsg] = useState("");
  const [error, setError] = useState("");
  const [bulkLevel, setBulkLevel] = useState("");
  const [bulkTag, setBulkTag] = useState("");
  const [deleteConfirm, setDeleteConfirm] = useState(false);

  const [detailId, setDetailId] = useState("");
  const [detail, setDetail] = useState<WarehouseDetail | null>(null);
  const [editValues, setEditValues] = useState<Record<string, string>>({});
  const [gate, setGate] = useState<GateResult | null>(null);
  const [figures, setFigures] = useState<RunFigures | null>(null);

  useEffect(() => {
    if (!detailId) {
      setDetail(null);
      setGate(null);
      setFigures(null);
      return;
    }
    let active = true;
    setDetail(null);
    setGate(null);
    setFigures(null);
    api<WarehouseDetail>(
      token,
      `/api/v1/warehouse/runs/${encodeURIComponent(detailId)}`,
    )
      .then((result) => {
        if (!active) return;
        setDetail(result);
        const fields = result.fields;
        setEditValues(
          Object.fromEntries(
            EDIT_FIELDS.map(([key]) => [key, String(fields[key] ?? "")]),
          ),
        );
      })
      .catch((exc) => {
        if (active)
          setError(exc instanceof Error ? exc.message : "读取记录失败");
      });
    api<RunFigures>(
      token,
      `/api/v1/warehouse/runs/${encodeURIComponent(detailId)}/figures`,
    )
      .then((result) => {
        if (active) setFigures(result);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [token, detailId]);

  function toggle(testId: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(testId)) next.delete(testId);
      else next.add(testId);
      return next;
    });
  }

  async function applyLevel() {
    try {
      let updated = 0;
      for (const testId of selected) {
        await api(
          token,
          `/api/v1/warehouse/runs/${encodeURIComponent(testId)}/metadata`,
          {
            method: "PATCH",
            body: JSON.stringify({ fields: { external_level: bulkLevel } }),
          },
        );
        updated += 1;
      }
      setMsg(`已把 ${updated} 条设为 ${bulkLevel}`);
      setSelected(new Set());
      onChanged();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "设置等级失败");
    }
  }

  async function applyTag() {
    try {
      let updated = 0;
      for (const testId of selected) {
        const row = rows.find((item) => item.test_id === testId);
        const merged = row?.tags
          ? `${row.tags},${bulkTag.trim()}`
          : bulkTag.trim();
        await api(
          token,
          `/api/v1/warehouse/runs/${encodeURIComponent(testId)}/metadata`,
          {
            method: "PATCH",
            body: JSON.stringify({ fields: { tags: merged } }),
          },
        );
        updated += 1;
      }
      setMsg(`已给 ${updated} 条追加标签 ${bulkTag.trim()}`);
      setBulkTag("");
      setSelected(new Set());
      onChanged();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "追加标签失败");
    }
  }

  async function deleteSelected() {
    try {
      const result = await api<{
        deleted: number[];
        failed: Record<string, string>;
        unknown_test_ids: string[];
      }>(token, "/api/v1/warehouse/runs/delete", {
        method: "POST",
        body: JSON.stringify({ test_ids: Array.from(selected) }),
      });
      setMsg(
        `已删除 ${result.deleted.length} 条` +
          (Object.keys(result.failed).length
            ? `，失败 ${Object.keys(result.failed).length} 条`
            : "") +
          (result.unknown_test_ids.length
            ? `，${result.unknown_test_ids.length} 条不存在`
            : ""),
      );
      setSelected(new Set());
      setDeleteConfirm(false);
      onChanged();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "删除失败");
    }
  }

  async function saveMetadata() {
    try {
      await api(
        token,
        `/api/v1/warehouse/runs/${encodeURIComponent(detailId)}/metadata`,
        {
          method: "PATCH",
          body: JSON.stringify({ fields: editValues }),
        },
      );
      setMsg("元数据已保存");
      onChanged();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "保存失败");
    }
  }

  async function reviewGate() {
    try {
      const result = await api<GateResult>(
        token,
        `/api/v1/warehouse/runs/${encodeURIComponent(detailId)}/gate`,
        { method: "POST" },
      );
      setGate(result);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "门禁评估失败");
    }
  }

  return (
    <div>
      {msg && <p className="minor-tag admin-msg">{msg}</p>}
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {selected.size > 0 && (
        <div className="bulk-bar" role="region" aria-label="批量操作">
          <strong>已选 {selected.size} 条</strong>
          <select
            aria-label="批量设置等级"
            value={bulkLevel}
            onChange={(event) => setBulkLevel(event.target.value)}
          >
            <option value="">设置等级…</option>
            <option value="internal">internal</option>
            <option value="review">review</option>
            <option value="publishable">publishable</option>
          </select>
          <button
            className="button subtle"
            disabled={!bulkLevel}
            onClick={() => void applyLevel()}
          >
            应用等级
          </button>
          <input
            aria-label="批量追加标签"
            value={bulkTag}
            onChange={(event) => setBulkTag(event.target.value)}
            placeholder="追加标签"
          />
          <button
            className="button subtle"
            disabled={!bulkTag.trim()}
            onClick={() => void applyTag()}
          >
            追加标签
          </button>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={deleteConfirm}
              onChange={(event) => setDeleteConfirm(event.target.checked)}
            />
            确认删除
          </label>
          <button
            className="button subtle danger"
            disabled={!deleteConfirm}
            onClick={() => void deleteSelected()}
          >
            删除所选
          </button>
        </div>
      )}
      {rows.length ? (
        <div className="table-scroll">
          <table className="data-table stats-table">
            <thead>
              <tr>
                <th aria-label="选择" />
                <th>日期 / 类型</th>
                <th>模型</th>
                <th>硬件</th>
                <th>并发</th>
                <th>TTFT</th>
                <th>decode TPS</th>
                <th>带宽</th>
                <th>等级</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.test_id} className={selected.has(row.test_id) ? "row-selected" : ""}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`选择 ${row.test_id}`}
                      checked={selected.has(row.test_id)}
                      onChange={() => toggle(row.test_id)}
                    />
                  </td>
                  <td>
                    <strong>{row.date || "—"}</strong>
                    <small>{labels[row.test_type] || row.test_type}</small>
                  </td>
                  <td>{row.model_name || "—"}</td>
                  <td>{row.machine_id || "未记录"}</td>
                  <td>{row.concurrency ?? "—"}</td>
                  <td>{formatNumber(row.ttft_s, 3)} s</td>
                  <td>{formatNumber(row.decode_tps, 1)}</td>
                  <td>{formatNumber(row.effective_bandwidth_gbps, 1)} GB/s</td>
                  <td>{row.external_level}</td>
                  <td>
                    <button
                      className="text-button"
                      onClick={() => setDetailId(row.test_id)}
                    >
                      详情 →
                    </button>
                    {jobIds.has(row.test_id) && (
                      <button
                        className="text-button"
                        onClick={() => onOpenJob(row.test_id)}
                      >
                        运行报告 ↗
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty title="暂无匹配记录" text="调整筛选条件后重试。" />
      )}

      {detailId && (
        <div className="warehouse-detail">
          <div className="section-head">
            <div>
              <span className="eyebrow">RECORD / {detailId}</span>
              <h3>运行详情</h3>
            </div>
            <button className="text-button" onClick={() => setDetailId("")}>
              收起
            </button>
          </div>
          {detail ? (
            <>
              <div className="detail-subgrid">
                <section>
                  <h4>编辑元数据</h4>
                  <div className="schema-form">
                    {EDIT_FIELDS.map(([key, label]) => (
                      <label key={key} className="schema-field">
                        <span className="input-label">{label}</span>
                        {key === "external_level" ? (
                          <select
                            value={editValues[key] || "internal"}
                            onChange={(event) =>
                              setEditValues((current) => ({
                                ...current,
                                [key]: event.target.value,
                              }))
                            }
                          >
                            <option value="internal">internal</option>
                            <option value="review">review</option>
                            <option value="publishable">publishable</option>
                          </select>
                        ) : (
                          <input
                            value={editValues[key] || ""}
                            onChange={(event) =>
                              setEditValues((current) => ({
                                ...current,
                                [key]: event.target.value,
                              }))
                            }
                          />
                        )}
                      </label>
                    ))}
                  </div>
                  <div className="form-footer">
                    <span />
                    <button className="button primary" onClick={() => void saveMetadata()}>
                      保存元数据
                    </button>
                  </div>
                </section>
                <section>
                  <h4>发布门禁复评</h4>
                  <p className="field-help">
                    按当前记录重新评估四项门禁，不写库；通过后可在左侧调整等级。
                  </p>
                  <button className="button subtle" onClick={() => void reviewGate()}>
                    重新评估门禁
                  </button>
                  {gate && (
                    <div className="gate-result">
                      <p>
                        {gate.passed ? (
                          <strong className="text-good">门禁通过（{gate.level}）</strong>
                        ) : (
                          <strong className="text-danger">
                            门禁未通过（{gate.level}）
                          </strong>
                        )}
                      </p>
                      <ul>
                        {Object.entries(gate.gates).map(([name, ok]) => (
                          <li key={name} className={ok ? "text-good" : "text-danger"}>
                            [{ok ? "通过" : "未过"}] {GATE_LABELS[name] || name}
                          </li>
                        ))}
                      </ul>
                      {gate.reasons.map((reason) => (
                        <small key={reason}>· {reason}</small>
                      ))}
                    </div>
                  )}
                </section>
              </div>
              {figures && figures.result_count > 0 && (
                <section>
                  <h4>分布图（{figures.result_count} 条请求样本）</h4>
                  <div className="detail-subgrid">
                    {(["ttft", "tps", "tpot"] as const).map((field) => (
                      <div key={field}>
                        {figures.distributions[field]?.histogram && (
                          <PlotlyFigure
                            figure={figures.distributions[field].histogram}
                            ariaLabel={`${field} 分布直方图`}
                          />
                        )}
                        {figures.distributions[field]?.box && (
                          <PlotlyFigure
                            figure={figures.distributions[field].box}
                            ariaLabel={`${field} 按并发箱线图`}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}
              {figures?.engine.figure && (
                <section>
                  <h4>引擎指标时间线</h4>
                  <PlotlyFigure
                    figure={figures.engine.figure}
                    ariaLabel="引擎指标时间线"
                  />
                </section>
              )}
            </>
          ) : (
            <p className="muted-cell">正在读取字段…</p>
          )}
        </div>
      )}
    </div>
  );
}
