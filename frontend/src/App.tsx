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
import {
  api,
  ApiError,
  type BatchSummary,
  type Endpoint,
  type Job,
} from "./api";
import { activeStates, date, type JobType } from "./constants";
import { Mark, MetricCard, JobTable, Empty } from "./components";
import { Login } from "./pages/Login";
import { NewRun } from "./pages/NewRun";
import { Detail } from "./pages/Detail";
import { Warehouse } from "./pages/Warehouse";
import { Batch } from "./pages/Batch";
import { Compare } from "./pages/Compare";
import { Advanced } from "./pages/Advanced";
import { Environment } from "./pages/Environment";
import { ApiSettings } from "./pages/ApiSettings";
import { Help, WelcomeGuide } from "./pages/Help";
import { HistoryCsv } from "./pages/HistoryCsv";

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

  async function cancelBatch(batchId: string) {
    await api(
      token,
      `/api/v1/jobs/batch/${encodeURIComponent(batchId)}/cancel`,
      {
        method: "POST",
      },
    );
    await refresh(token);
  }

  async function control(job: Job, action: "pause" | "resume" | "recover") {
    await api(token, `/api/v1/jobs/${job.job_id}/${action}`, {
      method: "POST",
    });
    await refresh(token);
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
            className={navActive("/history") ? "active" : ""}
            onClick={nav("/history")}
          >
            <span>▥</span> 历史 CSV
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
          <button
            className={navActive("/help") ? "active" : ""}
            onClick={nav("/help")}
          >
            <span>?</span> 帮助与引导
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
                  : navActive("/history")
                    ? "历史 CSV"
                    : navActive("/new")
                      ? "创建测量"
                      : navActive("/batch")
                        ? "批量测量"
                        : navActive("/help")
                          ? "帮助与引导"
                          : navActive("/settings/api")
                            ? "受测 API 设置"
                            : navActive("/advanced")
                              ? "高级评估"
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
                <>
                  <WelcomeGuide />
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
                </>
              }
            />
            <Route
              path="/runs"
              element={
                <RunsList
                  jobs={jobs}
                  total={total}
                  token={token}
                  onOpenJob={(id) => navigate(`/runs/${id}`)}
                  onOpenNew={nav("/new")}
                  onCancelBatch={cancelBatch}
                />
              }
            />
            <Route
              path="/runs/:jobId"
              element={
                <DetailRoute
                  jobs={jobs}
                  token={token}
                  onCancel={cancel}
                  onControl={control}
                />
              }
            />
            <Route path="/history" element={<HistoryCsv token={token} />} />
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
            <Route path="/help" element={<Help key={location.key} />} />
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
  onControl,
}: {
  jobs: Job[];
  token: string;
  onCancel: (job: Job) => Promise<void>;
  onControl: (
    job: Job,
    action: "pause" | "resume" | "recover",
  ) => Promise<void>;
}) {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [loadedJob, setLoadedJob] = useState<Job | null>(null);
  const [loadError, setLoadError] = useState("");
  const [reloadGeneration, setReloadGeneration] = useState(0);
  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    setLoadedJob((previous) => (previous?.job_id === jobId ? previous : null));
    setLoadError("");
    const load = async () => {
      try {
        const selected = await api<Job>(token, `/api/v1/jobs/${jobId}`);
        if (!alive) return;
        setLoadedJob(selected);
        setLoadError("");
        if (activeStates.has(selected.status))
          timer = window.setTimeout(load, 3500);
      } catch (exc) {
        if (alive) {
          setLoadError(exc instanceof Error ? exc.message : "任务读取失败");
          timer = window.setTimeout(load, 3500);
        }
      }
    };
    void load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [jobId, token, reloadGeneration]);
  const job =
    loadedJob?.job_id === jobId
      ? loadedJob
      : jobs.find((item) => item.job_id === jobId);
  if (!job)
    return (
      <Empty
        title={loadError ? "任务读取失败" : "正在读取任务…"}
        text={loadError || "正在加载持久化运行记录"}
      />
    );
  async function reloadSelected() {
    setLoadedJob(await api<Job>(token, `/api/v1/jobs/${jobId}`));
    setReloadGeneration((value) => value + 1);
  }
  return (
    <Detail
      job={job}
      token={token}
      onBack={() => navigate("/runs")}
      onCancel={async () => {
        await onCancel(job);
        await reloadSelected();
      }}
      onControl={async (action) => {
        await onControl(job, action);
        await reloadSelected();
      }}
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
  token,
  onOpenJob,
  onOpenNew,
  onCancelBatch,
}: {
  jobs: Job[];
  total: number;
  token: string;
  onOpenJob: (id: string) => void;
  onOpenNew: () => void;
  onCancelBatch: (batchId: string) => Promise<void>;
}) {
  const navigate = useNavigate();
  const [filter, setFilter] = useState("");
  const [pausedOnly, setPausedOnly] = useState(false);
  const [recoverableOnly, setRecoverableOnly] = useState(false);
  const [savedOnly, setSavedOnly] = useState(false);
  const [controlPage, setControlPage] = useState(0);
  const [controlLoading, setControlLoading] = useState(false);
  const [controlJobs, setControlJobs] = useState<Job[]>([]);
  const [controlTotal, setControlTotal] = useState(0);
  const [cancellingBatch, setCancellingBatch] = useState(false);
  const [batchError, setBatchError] = useState("");
  const [batchJobs, setBatchJobs] = useState<Job[]>([]);
  const [batchSummary, setBatchSummary] = useState<BatchSummary | null>(null);
  const [recentBatches, setRecentBatches] = useState<BatchSummary[]>([]);
  const [searchParams] = useSearchParams();
  const batchId = searchParams.get("batch") || "";
  useEffect(() => setControlPage(0), [batchId]);
  useEffect(() => {
    if (!pausedOnly && !recoverableOnly && !savedOnly) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    setControlJobs([]);
    setControlTotal(0);
    const query = new URLSearchParams({
      limit: "50",
      offset: String(controlPage * 50),
    });
    if (pausedOnly) query.set("status", "paused");
    if (recoverableOnly) query.set("recoverable", "true");
    if (savedOnly || recoverableOnly) query.set("saved_progress", "true");
    if (batchId) query.set("parent_job_id", batchId);
    const load = async () => {
      setControlLoading(true);
      try {
        const data = await api<{ items: Job[]; total: number }>(
          token,
          `/api/v1/jobs?${query}`,
          { signal: controller.signal },
        );
        if (alive) {
          setControlJobs(data.items);
          setControlTotal(data.total);
          setBatchError("");
          if (controlPage > 0 && controlPage * 50 >= data.total)
            setControlPage(Math.max(0, Math.ceil(data.total / 50) - 1));
        }
      } catch (exc) {
        if (alive)
          setBatchError(
            exc instanceof Error ? exc.message : "运行控制任务读取失败",
          );
      } finally {
        if (alive) {
          setControlLoading(false);
          timer = setTimeout(() => void load(), 3500);
        }
      }
    };
    void load();
    return () => {
      alive = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [token, pausedOnly, recoverableOnly, savedOnly, batchId, controlPage]);
  useEffect(() => {
    let alive = true;
    void api<{ items: BatchSummary[] }>(token, "/api/v1/batches?limit=20")
      .then((data) => {
        if (alive) setRecentBatches(data.items);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [token]);
  useEffect(() => {
    if (!batchId) return;
    let alive = true;
    setBatchJobs([]);
    setBatchSummary(null);
    void api<BatchSummary>(
      token,
      `/api/v1/jobs/batch/${encodeURIComponent(batchId)}`,
    )
      .then((data) => {
        if (alive) setBatchSummary(data);
      })
      .catch(() => {}); // Older batches have child jobs but no metadata row.
    const load = () => {
      void api<{ items: Job[] }>(
        token,
        `/api/v1/jobs?parent_job_id=${encodeURIComponent(batchId)}&limit=200`,
      )
        .then((data) => {
          if (alive) setBatchJobs(data.items);
        })
        .catch((exc) => {
          if (alive)
            setBatchError(exc instanceof Error ? exc.message : "批次读取失败");
        });
    };
    load();
    const interval = window.setInterval(load, 3500);
    return () => {
      alive = false;
      window.clearInterval(interval);
    };
  }, [batchId, token]);
  const controlFiltered = pausedOnly || recoverableOnly || savedOnly;
  const sourceJobs = controlFiltered ? controlJobs : batchId ? batchJobs : jobs;
  const visible = useMemo(
    () =>
      sourceJobs.filter(
        (job) =>
          (!batchId || job.parent_job_id === batchId) &&
          (!pausedOnly || job.status === "paused") &&
          `${job.model_id} ${job.test_type} ${job.job_id} ${job.status}`
            .toLowerCase()
            .includes(filter.toLowerCase()),
      ),
    [sourceJobs, filter, batchId, pausedOnly],
  );
  const batchHasActive = Boolean(
    batchId && batchJobs.some((job) => activeStates.has(job.status)),
  );
  return (
    <div className="page-grid">
      <div className="page-head">
        <div>
          <span className="eyebrow">RUN ARCHIVE</span>
          <h1>{batchSummary?.name || "运行记录"}</h1>
          <p>
            {batchSummary?.description ||
              "查看任务状态、逐请求样本和可导出报告。"}
            {batchId && `（正在按批次 ${batchId} 过滤）`}
          </p>
          {batchSummary && (
            <p>
              已提交 {batchSummary.submitted_items} /{" "}
              {batchSummary.requested_items} 项；其余子任务在提交前停用。
              {batchSummary.max_parallel > 1
                ? ` 最多 ${batchSummary.max_parallel} 项并行。`
                : " 串行执行。"}
              {batchSummary.stop_on_error && " 失败即停已启用。"}
            </p>
          )}
        </div>
        <div className="detail-actions">
          {batchHasActive && (
            <button
              className="button subtle danger"
              disabled={cancellingBatch}
              onClick={async () => {
                if (!window.confirm("停止此批次中所有待执行或运行中的子任务？"))
                  return;
                setCancellingBatch(true);
                setBatchError("");
                try {
                  await onCancelBatch(batchId);
                } catch (exc) {
                  setBatchError(
                    exc instanceof Error ? exc.message : "停止批次失败",
                  );
                } finally {
                  setCancellingBatch(false);
                }
              }}
            >
              {cancellingBatch ? "停止中…" : "停止此批次"}
            </button>
          )}
          <button className="button primary" onClick={onOpenNew}>
            ＋ 创建测量
          </button>
        </div>
      </div>
      {batchError && (
        <p className="form-error" role="alert">
          {batchError}
        </p>
      )}
      {!batchId && recentBatches.length > 0 && (
        <section className="surface">
          <div className="section-head">
            <div>
              <span className="eyebrow">BATCH HISTORY</span>
              <h2>最近批次</h2>
            </div>
          </div>
          <div className="batch-history-grid">
            {recentBatches.map((batch) => (
              <button
                className="batch-history-card"
                key={batch.batch_id}
                onClick={() =>
                  navigate(`/runs?batch=${encodeURIComponent(batch.batch_id)}`)
                }
              >
                <strong>{batch.name}</strong>
                <span>{batch.description || "无说明"}</span>
                <small>
                  {date(batch.created_at)} · 已提交 {batch.submitted_items} /{" "}
                  {batch.requested_items} 项
                </small>
              </button>
            ))}
          </div>
        </section>
      )}
      <section className="surface">
        <div className="section-head">
          <div>
            <span className="eyebrow">ALL RUNS</span>
            <h2>
              {pausedOnly
                ? "已暂停任务"
                : recoverableOnly
                  ? "可恢复中断任务"
                  : savedOnly
                    ? "保存进度历史"
                    : batchId
                      ? "批次任务"
                      : "全部任务"}{" "}
              <span className="count-tag">
                {controlFiltered
                  ? controlTotal
                  : batchId
                    ? sourceJobs.length
                    : total}
              </span>
            </h2>
          </div>
          <label className="batch-plan">
            <input
              type="checkbox"
              checked={pausedOnly}
              onChange={(event) => {
                setPausedOnly(event.target.checked);
                setRecoverableOnly(false);
                setSavedOnly(false);
                setControlPage(0);
              }}
            />
            只看已暂停
          </label>
          <label className="batch-plan">
            <input
              type="checkbox"
              checked={recoverableOnly}
              onChange={(event) => {
                setRecoverableOnly(event.target.checked);
                setPausedOnly(false);
                setSavedOnly(false);
                setControlPage(0);
              }}
            />
            只看可恢复中断
          </label>
          <label className="batch-plan">
            <input
              type="checkbox"
              checked={savedOnly}
              onChange={(event) => {
                setSavedOnly(event.target.checked);
                setPausedOnly(false);
                setRecoverableOnly(false);
                setControlPage(0);
              }}
            />
            只看保存进度
          </label>
          <input
            className="search"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder={
              controlFiltered
                ? "搜索当前页的模型、类型或 ID"
                : "搜索模型、类型或 ID"
            }
            aria-label="搜索运行记录"
          />
        </div>
        {pausedOnly && (
          <p className="batch-plan">
            选择任务进入详情，点击“继续运行”即可接着执行。暂停任务仍保留执行资源。
          </p>
        )}
        {recoverableOnly && (
          <p className="batch-plan">
            选择任务进入详情，从保存进度恢复。恢复时会核验模型、参数、代码与记录完整性；已提交结果复用，未提交的样本或完整测量组可能重跑。
          </p>
        )}
        {savedOnly && (
          <p className="batch-plan">
            查看保存时间与样本或测量组进度。进入任务详情可恢复符合条件的测试；已停止或结束任务的保存进度可删除，结果、报告和审计记录保留。
          </p>
        )}
        {controlFiltered && (
          <div className="detail-actions" aria-label="运行控制任务分页">
            <button
              className="button subtle"
              disabled={controlLoading || controlPage === 0}
              onClick={() => setControlPage((page) => page - 1)}
            >
              上一页
            </button>
            <span>
              第 {controlPage + 1} / {Math.max(1, Math.ceil(controlTotal / 50))}{" "}
              页 · 共 {controlTotal} 项{controlLoading ? " · 更新中…" : ""}
            </span>
            <button
              className="button subtle"
              disabled={
                controlLoading || (controlPage + 1) * 50 >= controlTotal
              }
              onClick={() => setControlPage((page) => page + 1)}
            >
              下一页
            </button>
          </div>
        )}
        {controlFiltered && visible.length === 0 ? (
          <Empty
            title={
              controlLoading
                ? "正在读取任务…"
                : filter
                  ? "本页没有匹配任务"
                  : "没有符合条件的任务"
            }
            text={
              controlLoading
                ? "正在读取持久化运行记录。"
                : filter
                  ? "搜索只匹配当前页，清空搜索或切换分页查看其他任务。"
                  : recoverableOnly
                    ? "没有保存计划且符合恢复条件的中断任务。"
                    : savedOnly
                      ? "没有保存进度记录。"
                      : "没有已暂停任务。"
            }
          />
        ) : (
          <JobTable
            jobs={visible}
            onSelect={(job) => onOpenJob(job.job_id)}
            savedProgress={savedOnly || recoverableOnly}
          />
        )}
      </section>
    </div>
  );
}
