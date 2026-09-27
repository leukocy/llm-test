import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import { api, ApiError, type Endpoint, type Job } from "./api";
import { activeStates, type JobType } from "./constants";
import { Mark, MetricCard, JobTable } from "./components";
import { Login } from "./pages/Login";
import { NewRun } from "./pages/NewRun";
import { Detail } from "./pages/Detail";
import { Warehouse } from "./pages/Warehouse";
import { Batch } from "./pages/Batch";
import { Compare } from "./pages/Compare";
import { Advanced } from "./pages/Advanced";
import { Environment } from "./pages/Environment";
import { ApiSettings } from "./pages/ApiSettings";

const TOKEN_KEY = "llm-test-token";

export default function App() {
  const [token, setToken] = useState(
    () => window.sessionStorage.getItem(TOKEN_KEY) || "",
  );
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [total, setTotal] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const location = useLocation();

  function logout() {
    window.sessionStorage.removeItem(TOKEN_KEY);
    setToken("");
    setEndpoints([]);
    setJobs([]);
  }

  const refresh = useCallback(async (credential: string) => {
    const data = await api<{ items: Job[]; total: number }>(
      credential,
      "/api/v1/jobs?limit=200",
    );
    setJobs(data.items);
    setTotal(data.total);
  }, []);

  const refreshEndpoints = useCallback(async (credential: string) => {
    const data = await api<{ items: Endpoint[] }>(
      credential,
      "/api/v1/endpoints",
    );
    setEndpoints(data.items);
  }, []);

  useEffect(() => {
    if (!token) return;
    const interval = window.setInterval(() => {
      refresh(token).catch((exc) => {
        if (exc instanceof ApiError && exc.status === 401) logout();
      });
    }, 3500);
    return () => window.clearInterval(interval);
  }, [token, refresh]);

  async function login(value: string) {
    await Promise.all([refreshEndpoints(value), refresh(value)]);
    window.sessionStorage.setItem(TOKEN_KEY, value);
    setToken(value);
  }

  // 刷新后凭 sessionStorage 的 token 恢复会话
  useEffect(() => {
    if (!token) return;
    refreshEndpoints(token).catch((exc) => {
      if (exc instanceof ApiError && exc.status === 401) logout();
    });
    refresh(token).catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, refreshEndpoints]);

  async function submit(
    endpoint: string,
    type: JobType,
    params: Record<string, unknown>,
    runConfig?: Record<string, unknown>,
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
          ...(runConfig ? { run_config: runConfig } : {}),
        }),
      });
      await refresh(token);
      navigate(`/runs/${job.job_id}`);
    } finally {
      setBusy(false);
    }
  }

  async function cancel(job: Job) {
    try {
      await api(token, `/api/v1/jobs/${job.job_id}/cancel`, { method: "POST" });
      await refresh(token);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "取消失败");
    }
  }

  if (!token) return <Login onLogin={login} />;

  const active = jobs.filter((job) => activeStates.has(job.status)).length;
  const completed = jobs.filter((job) => job.status === "completed").length;
  const failed = jobs.filter((job) => job.status === "failed").length;
  const nav = (path: string) => () => navigate(path);
  const navActive = (prefix: string) =>
    location.pathname === prefix || location.pathname.startsWith(`${prefix}/`);

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
            className={location.pathname === "/" ? "active" : ""}
            onClick={nav("/")}
          >
            <span>◫</span> 总览
          </button>
          <button
            className={navActive("/runs") ? "active" : ""}
            onClick={nav("/runs")}
          >
            <span>▤</span> 运行记录 <i>{total}</i>
          </button>
          <button
            className={navActive("/new") ? "active" : ""}
            onClick={nav("/new")}
          >
            <span>＋</span> 创建测量
          </button>
          <button
            className={navActive("/batch") ? "active" : ""}
            onClick={nav("/batch")}
          >
            <span>▣</span> 批量测量
          </button>
          <button
            className={navActive("/compare") ? "active" : ""}
            onClick={nav("/compare")}
          >
            <span>◧</span> 模型对比
          </button>
          <button
            className={navActive("/advanced") ? "active" : ""}
            onClick={nav("/advanced")}
          >
            <span>◇</span> 高级评估
          </button>
          <button
            className={navActive("/warehouse") ? "active" : ""}
            onClick={nav("/warehouse")}
          >
            <span>▦</span> 数据仓库
          </button>
          <button
            className={navActive("/environment") ? "active" : ""}
            onClick={nav("/environment")}
          >
            <span>▨</span> 环境信息
          </button>
          <button
            className={navActive("/settings/api") ? "active" : ""}
            onClick={nav("/settings/api")}
          >
            <span>⚙</span> 受测 API 设置
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="worker-indicator">
            <i />
            独立执行架构<small>API + 持久任务队列</small>
          </div>
          <button onClick={logout}>退出工作台 ↗</button>
          <span>LLM TEST / 2026</span>
        </div>
      </aside>
      <div className="main-column">
        <header className="topbar">
          <div>
            <span className="crumb">WORKSPACE</span>
            <span className="crumb-sep">/</span>
            <strong>
              {location.pathname === "/"
                ? "总览"
                : navActive("/runs")
                  ? "运行记录"
                  : navActive("/new")
                    ? "创建测量"
                    : navActive("/settings/api")
                      ? "受测 API 设置"
                      : "数据仓库"}
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
          <Routes>
            <Route
              path="/"
              element={
                <Overview
                  jobs={jobs}
                  total={total}
                  active={active}
                  completed={completed}
                  failed={failed}
                  onOpenRuns={nav("/runs")}
                  onOpenNew={nav("/new")}
                  onOpenJob={(id) => navigate(`/runs/${id}`)}
                />
              }
            />
            <Route
              path="/runs"
              element={
                <RunsList
                  jobs={jobs}
                  total={total}
                  onOpenJob={(id) => navigate(`/runs/${id}`)}
                  onOpenNew={nav("/new")}
                />
              }
            />
            <Route
              path="/runs/:jobId"
              element={
                <DetailRoute jobs={jobs} token={token} onCancel={cancel} />
              }
            />
            <Route
              path="/new"
              element={
                <NewRun
                  endpoints={endpoints}
                  token={token}
                  onSubmit={submit}
                  busy={busy}
                />
              }
            />
            <Route
              path="/batch"
              element={
                <Batch
                  endpoints={endpoints}
                  token={token}
                  onSubmitted={(batchId) => navigate(`/runs?batch=${batchId}`)}
                />
              }
            />
            <Route
              path="/warehouse/*"
              element={
                <Warehouse
                  token={token}
                  jobIds={new Set(jobs.map((job) => job.job_id))}
                  onOpenJob={(id) => navigate(`/runs/${id}`)}
                />
              }
            />
            <Route
              path="/compare"
              element={<Compare jobs={jobs} token={token} />}
            />
            <Route path="/advanced" element={<Advanced token={token} />} />
            <Route
              path="/environment"
              element={<Environment token={token} />}
            />
            <Route
              path="/settings/api"
              element={
                <ApiSettings
                  endpoints={endpoints}
                  token={token}
                  onChanged={() => refreshEndpoints(token)}
                />
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function DetailRoute({
  jobs,
  token,
  onCancel,
}: {
  jobs: Job[];
  token: string;
  onCancel: (job: Job) => Promise<void>;
}) {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const job = jobs.find((item) => item.job_id === jobId);
  if (!job) return <Navigate to="/runs" replace />;
  return (
    <Detail
      job={job}
      token={token}
      onBack={() => navigate("/runs")}
      onCancel={() => onCancel(job)}
    />
  );
}

function Overview({
  jobs,
  total,
  active,
  completed,
  failed,
  onOpenRuns,
  onOpenNew,
  onOpenJob,
}: {
  jobs: Job[];
  total: number;
  active: number;
  completed: number;
  failed: number;
  onOpenRuns: () => void;
  onOpenNew: () => void;
  onOpenJob: (id: string) => void;
}) {
  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">MEASUREMENT OVERVIEW</span>
          <h1>
            测量工作台<span className="title-dot">.</span>
          </h1>
          <p>从任务执行到统计报告，掌握每一次模型表现。</p>
        </div>
        <button className="button primary" onClick={onOpenNew}>
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
            <button className="text-button" onClick={onOpenRuns}>
              查看全部 →
            </button>
          </div>
          <JobTable
            jobs={jobs}
            compact
            onSelect={(job) => onOpenJob(job.job_id)}
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
            <button onClick={onOpenNew}>开始新测量 ↗</button>
          </div>
          <div className="aside-tip">
            <strong>统计说明</strong>
            <p>成功率包含所有请求；延迟分位数只统计成功且数值有效的样本。</p>
          </div>
        </aside>
      </div>
    </div>
  );
}

function RunsList({
  jobs,
  total,
  onOpenJob,
  onOpenNew,
}: {
  jobs: Job[];
  total: number;
  onOpenJob: (id: string) => void;
  onOpenNew: () => void;
}) {
  const [filter, setFilter] = useState("");
  const [searchParams] = useSearchParams();
  const batchId = searchParams.get("batch") || "";
  const visible = useMemo(
    () =>
      jobs.filter(
        (job) =>
          (!batchId || job.parent_job_id === batchId) &&
          `${job.model_id} ${job.test_type} ${job.job_id} ${job.status}`
            .toLowerCase()
            .includes(filter.toLowerCase()),
      ),
    [jobs, filter, batchId],
  );
  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">RUN ARCHIVE</span>
          <h1>运行记录</h1>
          <p>
            查看任务状态、逐请求样本和可导出报告。
            {batchId && `（正在按批次 ${batchId} 过滤）`}
          </p>
        </div>
        <button className="button primary" onClick={onOpenNew}>
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
        <JobTable jobs={visible} onSelect={(job) => onOpenJob(job.job_id)} />
      </section>
    </div>
  );
}
