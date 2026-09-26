import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError, type Endpoint, type Job } from "./api";
import { activeStates, type JobType, type View } from "./constants";
import { Mark, MetricCard, JobTable } from "./components";
import { Login } from "./pages/Login";
import { NewRun } from "./pages/NewRun";
import { Detail } from "./pages/Detail";
import { Warehouse } from "./pages/Warehouse";

export default function App() {
  const [token, setToken] = useState("");
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [total, setTotal] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<View>("overview");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");
  const selected = jobs.find((job) => job.job_id === selectedId) || null;

  const refresh = useCallback(async (credential: string) => {
    const data = await api<{ items: Job[]; total: number }>(
      credential,
      "/api/v1/jobs?limit=200",
    );
    setJobs(data.items);
    setTotal(data.total);
  }, []);
  useEffect(() => {
    if (!token) return;
    const interval = window.setInterval(() => {
      refresh(token).catch((exc) => {
        if (exc instanceof ApiError && exc.status === 401) setToken("");
      });
    }, 3500);
    return () => window.clearInterval(interval);
  }, [token, refresh]);
  const visible = useMemo(
    () =>
      jobs.filter((job) =>
        `${job.model_id} ${job.test_type} ${job.job_id} ${job.status}`
          .toLowerCase()
          .includes(filter.toLowerCase()),
      ),
    [jobs, filter],
  );
  const active = jobs.filter((job) => activeStates.has(job.status)).length;
  const completed = jobs.filter((job) => job.status === "completed").length;
  const failed = jobs.filter((job) => job.status === "failed").length;

  async function login(value: string) {
    const [endpointData] = await Promise.all([
      api<{ items: Endpoint[] }>(value, "/api/v1/endpoints"),
      refresh(value),
    ]);
    setEndpoints(endpointData.items);
    setToken(value);
  }
  async function submit(
    endpoint: string,
    type: JobType,
    params: Record<string, unknown>,
  ) {
    setBusy(true);
    setError("");
    try {
      const job = await api<Job>(token, "/api/v1/jobs", {
        method: "POST",
        headers: { "Idempotency-Key": crypto.randomUUID() },
        body: JSON.stringify({
          schema_version: 1,
          endpoint_id: endpoint,
          test_type: type,
          parameters: params,
        }),
      });
      await refresh(token);
      setSelectedId(job.job_id);
      setView("runs");
    } finally {
      setBusy(false);
    }
  }
  async function cancel() {
    if (!selected) return;
    try {
      await api(token, `/api/v1/jobs/${selected.job_id}/cancel`, {
        method: "POST",
      });
      await refresh(token);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "取消失败");
    }
  }

  if (!token) return <Login onLogin={login} />;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Mark />
          <span>
            <strong>LLM TEST</strong>
            <small>测量控制台</small>
          </span>
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="主导航">
          <button
            className={view === "overview" && !selected ? "active" : ""}
            onClick={() => {
              setView("overview");
              setSelectedId(null);
            }}
          >
            <span>◫</span> 总览
          </button>
          <button
            className={view === "runs" || selected ? "active" : ""}
            onClick={() => {
              setView("runs");
              setSelectedId(null);
            }}
          >
            <span>▤</span> 运行记录 <i>{total}</i>
          </button>
          <button
            className={view === "new" ? "active" : ""}
            onClick={() => {
              setView("new");
              setSelectedId(null);
            }}
          >
            <span>＋</span> 创建测量
          </button>
          <button
            className={view === "warehouse" ? "active" : ""}
            onClick={() => {
              setView("warehouse");
              setSelectedId(null);
            }}
          >
            <span>▦</span> 数据仓库
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="worker-indicator">
            <i />
            独立执行架构<small>API + 持久任务队列</small>
          </div>
          <button
            onClick={() => {
              setToken("");
              setJobs([]);
              setSelectedId(null);
            }}
          >
            退出工作台 ↗
          </button>
          <span>LLM TEST / 2026</span>
        </div>
      </aside>
      <div className="main-column">
        <header className="topbar">
          <div>
            <span className="crumb">WORKSPACE</span>
            <span className="crumb-sep">/</span>
            <strong>
              {selected
                ? "运行详情"
                : view === "new"
                  ? "创建测量"
                  : view === "warehouse"
                    ? "数据仓库"
                    : view === "runs"
                      ? "运行记录"
                      : "总览"}
            </strong>
          </div>
          <div className="topbar-right">
            <span className="live-dot" /> 数据每 3.5 秒刷新{" "}
            <span className="topbar-divider" />
            <span className="avatar">LT</span>
          </div>
        </header>
        <main className="content">
          {error && (
            <div className="alert" role="alert">
              {error}
              <button onClick={() => setError("")}>×</button>
            </div>
          )}
          {selected ? (
            <Detail
              job={selected}
              token={token}
              onBack={() => setSelectedId(null)}
              onCancel={cancel}
            />
          ) : view === "new" ? (
            <NewRun
              endpoints={endpoints}
              token={token}
              onSubmit={submit}
              busy={busy}
            />
          ) : view === "warehouse" ? (
            <Warehouse
              token={token}
              jobIds={new Set(jobs.map((job) => job.job_id))}
              onOpenJob={(id) => {
                setSelectedId(id);
                setView("runs");
              }}
            />
          ) : view === "runs" ? (
            <div className="page-grid">
              <div className="page-head">
                <div>
                  <span className="eyebrow">RUN ARCHIVE</span>
                  <h1>运行记录</h1>
                  <p>查看任务状态、逐请求样本和可导出报告。</p>
                </div>
                <button
                  className="button primary"
                  onClick={() => setView("new")}
                >
                  ＋ 创建测量
                </button>
              </div>
              <section className="surface">
                <div className="section-head">
                  <div>
                    <span className="eyebrow">ALL RUNS</span>
                    <h2>
                      全部任务 <span className="count-tag">{total}</span>
                    </h2>
                  </div>
                  <input
                    className="search"
                    value={filter}
                    onChange={(event) => setFilter(event.target.value)}
                    placeholder="搜索模型、类型或 ID"
                    aria-label="搜索运行记录"
                  />
                </div>
                <JobTable
                  jobs={visible}
                  onSelect={(job) => setSelectedId(job.job_id)}
                />
              </section>
            </div>
          ) : (
            <div className="page-grid">
              <div className="page-head">
                <div>
                  <span className="eyebrow">MEASUREMENT OVERVIEW</span>
                  <h1>
                    测量工作台<span className="title-dot">.</span>
                  </h1>
                  <p>从任务执行到统计报告，掌握每一次模型表现。</p>
                </div>
                <button
                  className="button primary"
                  onClick={() => setView("new")}
                >
                  ＋ 创建测量
                </button>
              </div>
              <div className="metric-grid">
                <MetricCard
                  label="全部任务"
                  value={String(total)}
                  note="持久化运行记录"
                  accent
                />
                <MetricCard
                  label="运行中"
                  value={String(active)}
                  note="包含排队与取消中"
                />
                <MetricCard
                  label="已完成"
                  value={String(completed)}
                  note="可查看统计报告"
                />
                <MetricCard
                  label="失败任务"
                  value={String(failed)}
                  note="需检查事件和日志"
                />
              </div>
              <div className="overview-grid">
                <section className="surface recent">
                  <div className="section-head">
                    <div>
                      <span className="eyebrow">LATEST ACTIVITY</span>
                      <h2>最近运行</h2>
                    </div>
                    <button
                      className="text-button"
                      onClick={() => setView("runs")}
                    >
                      查看全部 →
                    </button>
                  </div>
                  <JobTable
                    jobs={jobs}
                    compact
                    onSelect={(job) => {
                      setSelectedId(job.job_id);
                      setView("runs");
                    }}
                  />
                </section>
                <aside className="overview-aside">
                  <div className="aside-card dark">
                    <span className="eyebrow">HOW IT WORKS</span>
                    <h3>
                      从测量到结论，
                      <br />
                      每一步可追溯。
                    </h3>
                    <div className="step">
                      <b>01</b>
                      <span>配置受控端点与测试方案</span>
                    </div>
                    <div className="step">
                      <b>02</b>
                      <span>独立 worker 执行并持续存储</span>
                    </div>
                    <div className="step">
                      <b>03</b>
                      <span>按统一统计口径生成报告</span>
                    </div>
                    <button onClick={() => setView("new")}>开始新测量 ↗</button>
                  </div>
                  <div className="aside-tip">
                    <strong>统计说明</strong>
                    <p>
                      成功率包含所有请求；延迟分位数只统计成功且数值有效的样本。
                    </p>
                  </div>
                </aside>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
