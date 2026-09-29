import { useEffect, useState } from "react";
import { api } from "../api";
import { Empty } from "../components";

/** 环境快照的嵌套 JSON 按小节展开渲染（顶级键一节）。 */
export function Environment({ token }: { token: string }) {
  const [snapshot, setSnapshot] = useState<Record<string, unknown> | null>(
    null,
  );
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    api<Record<string, unknown>>(token, "/api/v1/environment")
      .then((data) => {
        if (active) setSnapshot(data);
      })
      .catch((exc) => {
        if (active)
          setError(exc instanceof Error ? exc.message : "环境信息读取失败");
      });
    return () => {
      active = false;
    };
  }, [token]);

  function renderValue(value: unknown, depth: number): React.ReactNode {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "object") return String(value);
    if (Array.isArray(value)) {
      if (!value.length) return "（空）";
      if (typeof value[0] !== "object")
        return value.map((item) => String(item)).join("、");
      return (
        <div className="env-nested">
          {value.map((item, index) => (
            <div className="env-card" key={index}>
              {renderValue(item, depth + 1)}
            </div>
          ))}
        </div>
      );
    }
    return (
      <dl className={depth ? "env-kv env-kv-nested" : "env-kv"}>
        {Object.entries(value as Record<string, unknown>).map(([key, item]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{renderValue(item, depth + 1)}</dd>
          </div>
        ))}
      </dl>
    );
  }

  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">EXECUTION ENVIRONMENT</span>
          <h1>环境信息</h1>
          <p>
            当前 API 服务所在机器的硬件与系统快照。远程模型服务器和独立 worker
            可能位于其他机器；每次运行的报告会分别保留执行端快照与用户填写的环境信息。
          </p>
        </div>
      </div>
      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}
      {snapshot ? (
        Object.keys(snapshot).length ? (
          Object.entries(snapshot).map(([section, value]) => (
            <section className="surface" key={section}>
              <div className="section-head">
                <div>
                  <span className="eyebrow">SECTION</span>
                  <h2>{section}</h2>
                </div>
              </div>
              {renderValue(value, 0)}
            </section>
          ))
        ) : (
          <section className="surface">
            <Empty
              title="环境快照为空"
              text="system_info 缓存尚未填充，稍后刷新。"
            />
          </section>
        )
      ) : (
        !error && (
          <section className="surface">
            <Empty title="正在读取环境信息" text="采集硬件与系统快照…" />
          </section>
        )
      )}
    </div>
  );
}
