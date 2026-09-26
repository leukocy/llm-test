import { useEffect, useState } from "react";
import { api, downloadFile } from "../api";
import { Empty, MetricCard } from "../components";
import { formatNumber } from "../constants";

export function WarehouseExport({
  token,
  query,
  exportBlocked,
}: {
  token: string;
  query: URLSearchParams;
  exportBlocked: boolean;
}) {
  const [error, setError] = useState("");

  async function exportTemplate(
    template: "hwInventory" | "hmTest" | "maTest",
    format: "csv" | "json",
  ) {
    try {
      const params = new URLSearchParams(query);
      params.set("template", template);
      params.set("format", format);
      await downloadFile(
        token,
        `/api/v1/warehouse/export?${params}`,
        `llm-test-${template}.${format}`,
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "导出失败");
    }
  }

  async function exportZip(format: "csv" | "json") {
    try {
      const params = new URLSearchParams(query);
      params.set("format", format);
      await downloadFile(
        token,
        `/api/v1/warehouse/export.zip?${params}`,
        `llm-test-templates-${format}.zip`,
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "导出失败");
    }
  }

  const templates = [
    ["hwInventory", "硬件盘点（hwInventory）"],
    ["hmTest", "性能测量（hmTest）"],
    ["maTest", "应用用例（maTest）"],
  ] as const;

  return (
    <div>
      <p className="chart-caption">
        按手册三套字段模板导出仓库全集行（可筛选、可追溯、可对外口径），不是单次报告的图。
      </p>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      <div className="table-scroll">
        <table className="data-table stats-table">
          <thead>
            <tr>
              <th>模板</th>
              <th>CSV</th>
              <th>JSON</th>
            </tr>
          </thead>
          <tbody>
            {templates.map(([id, label]) => (
              <tr key={id}>
                <td>
                  <strong>{label}</strong>
                </td>
                <td>
                  <button
                    className="button subtle"
                    disabled={exportBlocked && id !== "maTest"}
                    onClick={() => void exportTemplate(id, "csv")}
                  >
                    CSV ↓
                  </button>
                </td>
                <td>
                  <button
                    className="button subtle"
                    disabled={exportBlocked && id !== "maTest"}
                    onClick={() => void exportTemplate(id, "json")}
                  >
                    JSON ↓
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="section-head split">
        <div>
          <span className="eyebrow">BUNDLE</span>
          <h3>一键打包全部模板（ZIP）</h3>
        </div>
      </div>
      <div className="warehouse-export">
        <button
          className="button subtle"
          disabled={exportBlocked}
          onClick={() => void exportZip("csv")}
        >
          全部 CSV (ZIP) ↓
        </button>
        <button
          className="button subtle"
          disabled={exportBlocked}
          onClick={() => void exportZip("json")}
        >
          全部 JSON (ZIP) ↓
        </button>
      </div>
      {exportBlocked && (
        <p className="chart-caption">
          筛选窗口过大或存在非法行时导出被禁用（maTest 不受影响）——请收窄筛选。
        </p>
      )}
    </div>
  );
}

type Backup = {
  path: string;
  name: string;
  size_mb: number;
  created_at: string;
};
type DbHealth = {
  size_bytes: number;
  schema_version: string;
  tables: Record<string, number>;
  integrity: string;
};

export function WarehouseAdmin({ token }: { token: string }) {
  const [health, setHealth] = useState<DbHealth | null>(null);
  const [backups, setBackups] = useState<Backup[]>([]);
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const [restoreTarget, setRestoreTarget] = useState("");
  const [restoreConfirm, setRestoreConfirm] = useState(false);

  function load() {
    api<DbHealth>(token, "/api/v1/admin/db/health")
      .then(setHealth)
      .catch((exc) =>
        setError(exc instanceof Error ? exc.message : "健康读取失败"),
      );
    api<{ items: Backup[] }>(token, "/api/v1/admin/backups")
      .then((data) => setBackups(data.items))
      .catch(() => setBackups([]));
  }
  useEffect(load, [token]);

  async function upload(path: string, files: FileList | null) {
    if (!files?.length) return;
    setMsg("");
    try {
      const body = new FormData();
      Array.from(files).forEach((file) => body.append("files", file));
      const response = await fetch(path, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body,
      });
      const payload = await response.json();
      if (!response.ok)
        throw new Error(payload.detail || `失败（${response.status}）`);
      setMsg(
        path.includes("hw-snapshot")
          ? `快照导入：成功 ${payload.imported}，跳过 ${payload.skipped}，失败 ${payload.failed}`
          : `CSV 导入：${payload.imported} 条${payload.errors?.length ? `，${payload.errors.join("; ")}` : ""}`,
      );
      load();
    } catch (exc) {
      setMsg(exc instanceof Error ? exc.message : "导入失败");
    }
  }

  async function createBackup() {
    setMsg("");
    try {
      const result = await api<{ path: string }>(
        token,
        "/api/v1/admin/backups",
        {
          method: "POST",
        },
      );
      setMsg(`备份已创建：${result.path}`);
      load();
    } catch (exc) {
      setMsg(exc instanceof Error ? exc.message : "备份失败");
    }
  }

  async function restoreBackup() {
    setMsg("");
    try {
      await api(token, "/api/v1/admin/backups/restore", {
        method: "POST",
        body: JSON.stringify({ path: restoreTarget }),
      });
      setMsg("恢复完成。请重启 API 服务以重建数据库连接。");
    } catch (exc) {
      setMsg(exc instanceof Error ? exc.message : "恢复失败");
    }
  }

  return (
    <div>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      <div className="section-head">
        <div>
          <span className="eyebrow">IMPORT</span>
          <h3>导入回导</h3>
        </div>
      </div>
      <div className="admin-import-grid">
        <label className="upload-box">
          <strong>benchmark CSV 回导</strong>
          <span>raw_data 格式结果 CSV，生成已完成运行 + 逐请求结果</span>
          <input
            type="file"
            accept=".csv"
            aria-label="上传结果 CSV"
            onChange={(event) => {
              const body = new FormData();
              const file = event.target.files?.[0];
              if (!file) return;
              body.append("file", file);
              setMsg("");
              fetch("/api/v1/admin/import/csv", {
                method: "POST",
                headers: { Authorization: `Bearer ${token}` },
                body,
              })
                .then(async (response) => {
                  const payload = await response.json();
                  if (!response.ok)
                    throw new Error(
                      payload.detail || `失败（${response.status}）`,
                    );
                  setMsg(`CSV 导入：${payload.imported} 条`);
                  load();
                })
                .catch((exc) =>
                  setMsg(exc instanceof Error ? exc.message : "导入失败"),
                );
              event.target.value = "";
            }}
          />
        </label>
        <label className="upload-box">
          <strong>硬件快照 JSON</strong>
          <span>hw-snapshot/v1 格式，可多选，按机器指纹去重</span>
          <input
            type="file"
            accept=".json"
            multiple
            aria-label="上传硬件快照"
            onChange={(event) => {
              void upload(
                "/api/v1/admin/import/hw-snapshot",
                event.target.files,
              );
              event.target.value = "";
            }}
          />
        </label>
      </div>
      {msg && <p className="minor-tag admin-msg">{msg}</p>}

      <div className="section-head">
        <div>
          <span className="eyebrow">BACKUP</span>
          <h3>备份与恢复</h3>
        </div>
        <button className="button subtle" onClick={() => void createBackup()}>
          立即创建备份
        </button>
      </div>
      {backups.length ? (
        <div className="table-scroll">
          <table className="data-table stats-table">
            <thead>
              <tr>
                <th>文件</th>
                <th>大小 (MB)</th>
                <th>创建时间</th>
                <th>恢复</th>
              </tr>
            </thead>
            <tbody>
              {backups.map((backup) => (
                <tr key={backup.path}>
                  <td>
                    <strong>{backup.name}</strong>
                  </td>
                  <td>{backup.size_mb}</td>
                  <td>{backup.created_at}</td>
                  <td>
                    {restoreTarget === backup.path ? (
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={restoreConfirm}
                          onChange={(event) =>
                            setRestoreConfirm(event.target.checked)
                          }
                        />
                        确认整库覆盖
                        <button
                          className="text-button danger"
                          disabled={!restoreConfirm}
                          onClick={() => void restoreBackup()}
                        >
                          执行恢复
                        </button>
                        <button
                          className="text-button"
                          onClick={() => {
                            setRestoreTarget("");
                            setRestoreConfirm(false);
                          }}
                        >
                          取消
                        </button>
                      </label>
                    ) : (
                      <button
                        className="text-button"
                        onClick={() => setRestoreTarget(backup.path)}
                      >
                        恢复…
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty title="暂无备份" text="备份保留最近 10 份，超量自动轮换。" />
      )}

      <div className="section-head">
        <div>
          <span className="eyebrow">HEALTH</span>
          <h3>数据库健康</h3>
        </div>
      </div>
      {health && (
        <>
          <div className="metric-grid">
            <MetricCard
              label="数据库大小"
              value={`${formatNumber(health.size_bytes / 1024 / 1024, 2)} MB`}
              note="data/benchmark.db"
              accent
            />
            <MetricCard
              label="schema 版本"
              value={String(health.schema_version || "—")}
              note="迁移口径"
            />
            <MetricCard
              label="完整性"
              value={health.integrity === "ok" ? "通过" : "异常"}
              note="PRAGMA integrity_check"
            />
          </div>
          <div className="table-scroll">
            <table className="data-table stats-table">
              <thead>
                <tr>
                  <th>表</th>
                  <th>行数</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(health.tables).map(([table, count]) => (
                  <tr key={table}>
                    <td>{table}</td>
                    <td>{count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
