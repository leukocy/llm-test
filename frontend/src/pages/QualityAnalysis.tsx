import { ReasoningAnalysis } from "../components/ReasoningAnalysis";
import { useMemo, useState } from "react";
import type { QualityReport } from "../api";
import { Empty, MetricCard } from "../components";
import { formatNumber, formatPercent } from "../constants";
import { PlotlyFigure } from "../components/PlotlyFigure";
import { FailureAnalysis } from "../components/FailureAnalysis";

function wilson(successes: number, trials: number): [number, number] | null {
  if (!Number.isFinite(trials) || trials <= 0) return null;
  const z = 1.96;
  const p = successes / trials;
  const denominator = 1 + (z * z) / trials;
  const center = (p + (z * z) / (2 * trials)) / denominator;
  const half =
    (z * Math.sqrt((p * (1 - p)) / trials + (z * z) / (4 * trials * trials))) /
    denominator;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

export function QualityAnalysis({
  report,
  onExportErrors,
  onExportSamples,
}: {
  report: QualityReport;
  onExportErrors: () => void;
  onExportSamples: () => void;
}) {
  const names = Object.keys(report.datasets);
  const [selectedName, setSelectedName] = useState(names[0] || "");
  const [category, setCategory] = useState("");
  const [search, setSearch] = useState("");
  const [selectedError, setSelectedError] = useState<number | null>(null);
  const [view, setView] = useState("all");
  const [sort, setSort] = useState("default");
  const [page, setPage] = useState(0);
  const name = report.datasets[selectedName] ? selectedName : names[0] || "";
  const dataset = report.datasets[name];
  const details = dataset?.details || [];
  const failures = useMemo(
    () =>
      details
        .map((item, index) => ({ item, index }))
        .filter(({ item }) => !item.is_correct || !!item.error),
    [details],
  );
  const categories = useMemo(
    () => [...new Set(details.map((item) => item.category || "未分类"))].sort(),
    [details],
  );
  const candidates =
    view === "failures"
      ? failures
      : details
          .map((item, index) => ({ item, index }))
          .filter(({ item }) =>
            view === "judge"
              ? item.is_judge_corrected
              : view === "correct"
                ? item.is_correct === true && !item.error
                : true,
          );
  const filtered = candidates.filter(
    ({ item }) =>
      (!category || (item.category || "未分类") === category) &&
      (!search ||
        `${item.question} ${item.sample_id} ${item.predicted_answer}`
          .toLowerCase()
          .includes(search.toLowerCase())),
  );
  function score(item: (typeof details)[number]): number | null {
    const recorded = item.reasoning_assessment;
    if (
      recorded?.version === "heuristic-reasoning-v1" &&
      recorded.source === "local_rule_heuristics"
    ) {
      const value = recorded.overall;
      return typeof value === "number" &&
        Number.isFinite(value) &&
        value >= 0 &&
        value <= 10
        ? value
        : null;
    }
    const value = item.reasoning_quality_overall ?? item.reasoning_quality;
    // Legacy dataclass zero is indistinguishable from an unmeasured default.
    return typeof value === "number" &&
      Number.isFinite(value) &&
      value > 0 &&
      value <= 10
      ? value
      : null;
  }
  function latency(item: (typeof details)[number]): number | null {
    return typeof item.latency_ms === "number" &&
      Number.isFinite(item.latency_ms) &&
      item.latency_ms > 0
      ? item.latency_ms
      : null;
  }
  if (sort !== "default")
    filtered.sort((a, b) => {
      const left = sort === "latency" ? latency(a.item) : score(a.item);
      const right = sort === "latency" ? latency(b.item) : score(b.item);
      return left === null
        ? right === null
          ? a.index - b.index
          : 1
        : right === null
          ? -1
          : right - left || a.index - b.index;
    });
  const selected = selectedError === null ? null : details[selectedError];
  const pageCount = Math.max(1, Math.ceil(filtered.length / 50));
  const currentPage = Math.min(page, pageCount - 1);
  if (!dataset)
    return <Empty title="没有质量评估样本" text="报告不包含可分析的数据集。" />;
  const analysis = report.analysis?.datasets[name];
  const interval = analysis
    ? analysis.ci95
    : wilson(dataset.correct_samples, dataset.total_samples);
  const provenance = (dataset.config.dataset_provenance || {}) as Record<
    string,
    unknown
  >;
  const sandbox = (dataset.config.sandbox_identity || {}) as Record<
    string,
    unknown
  >;
  const categoryRows = Object.entries(dataset.by_category || {}).sort(
    (a, b) => (b[1].count || 0) - (a[1].count || 0),
  );
  const methodCounts = details.reduce<Record<string, number>>(
    (counts, item) => {
      const method = item.evaluation_method || "未记录";
      counts[method] = (counts[method] || 0) + 1;
      return counts;
    },
    {},
  );
  const failureCounts = failures.reduce<Record<string, number>>(
    (counts, { item }) => {
      const reason = item.execution_error
        ? "代码测试未通过"
        : item.failure_category || (item.error ? "请求错误" : "未分类");
      counts[reason] = (counts[reason] || 0) + 1;
      return counts;
    },
    {},
  );
  return (
    <div className="quality-analysis">
      <div className="quality-analysis-picker">
        <label htmlFor="quality-dataset">分析数据集</label>
        <select
          id="quality-dataset"
          value={name}
          onChange={(event) => {
            setSelectedName(event.target.value);
            setCategory("");
            setSearch("");
            setSelectedError(null);
            setPage(0);
          }}
        >
          {names.map((name) => (
            <option key={name}>{name}</option>
          ))}
        </select>
        <span>来源：{String(provenance.source || "未记录")}</span>
      </div>
      <div className="metric-grid quality-metrics">
        <MetricCard
          label="准确率"
          value={formatPercent(
            analysis
              ? analysis.accuracy
              : dataset.total_samples
                ? dataset.correct_samples / dataset.total_samples
                : null,
          )}
          note={
            interval
              ? `95% Wilson ${formatPercent(interval[0])}–${formatPercent(interval[1])}`
              : "未通过完整评分核验，不计算区间"
          }
          accent
        />
        <MetricCard
          label="记录样本"
          value={String(dataset.total_samples)}
          note={`${dataset.correct_samples} 条正确`}
        />
        <MetricCard
          label="首字延迟"
          value={`${formatNumber(analysis ? analysis.metrics.ttft_ms.mean : dataset.performance_stats?.avg_ttft_ms, 1)} ms`}
          note="有正值观测的样本均值"
        />
        <MetricCard
          label="生成速度"
          value={formatNumber(
            analysis
              ? analysis.metrics.tps.mean
              : dataset.performance_stats?.avg_tps,
            1,
          )}
          note="tokens/s · 单次响应均值"
        />
      </div>
      <div className="quality-analysis-grid">
        <section className="surface">
          <div className="section-head">
            <div>
              <span className="eyebrow">CATEGORY BREAKDOWN</span>
              <h2>类别表现</h2>
            </div>
          </div>
          {categoryRows.length ? (
            <div className="quality-category-list">
              {categoryRows.map(([name, stats]) => (
                <div className="quality-category-row" key={name}>
                  <div>
                    <strong>{name}</strong>
                    <small>n = {stats.count}</small>
                  </div>
                  <div className="quality-category-track">
                    <i
                      style={{
                        width: `${Math.max(0, Math.min(100, (stats.accuracy || 0) * 100))}%`,
                      }}
                    />
                  </div>
                  <b>{formatPercent(stats.accuracy)}</b>
                </div>
              ))}
            </div>
          ) : (
            <Empty title="暂无类别维度" text="当前评测器未提供分类结果。" />
          )}
          <p className="chart-caption">
            每一类独立计分；请结合样本数判断波动。
          </p>
          {analysis && (
            <PlotlyFigure
              figure={analysis.category_figure}
              ariaLabel="质量类别准确率图"
            />
          )}
        </section>
        <section className="surface quality-provenance">
          <div className="section-head">
            <div>
              <span className="eyebrow">DATA LINEAGE</span>
              <h2>数据来源</h2>
            </div>
          </div>
          <dl>
            <dt>数据集来源</dt>
            <dd>{String(provenance.source || "未记录")}</dd>
            <dt>评分 / few-shot 分区</dt>
            <dd>
              {String(provenance.evaluation_split || "未记录")} /{" "}
              {String(provenance.few_shot_split || "未使用或未记录")}
            </dd>
            {provenance.few_shot_policy === "same_subject_dev" && (
              <>
                <dt>示例策略</dt>
                <dd>
                  同科目 dev；每题{" "}
                  {String(provenance.few_shot_per_sample ?? "未记录")}{" "}
                  个示例，冻结示例池{" "}
                  {String(provenance.few_shot_count ?? "未记录")} 条。
                </dd>
                <dt>答案评分口径</dt>
                <dd>生成文本的答案字母，不是选项概率评分。</dd>
              </>
            )}
            {dataset.config.requires_code_execution === true && (
              <>
                <dt>代码评分环境</dt>
                <dd>{String(sandbox.version || "旧报告未记录")}</dd>
                <dt>实际执行镜像</dt>
                <dd className="mono">{String(sandbox.image_id || "未记录")}</dd>
                <dt>环境 SHA-256</dt>
                <dd className="mono">
                  {String(dataset.config.sandbox_contract || "未记录")}
                </dd>
              </>
            )}
            <dt>样本 SHA-256</dt>
            <dd className="mono">
              {String(provenance.sample_sha256 || "未记录")}
            </dd>
            <dt>Few-shot SHA-256</dt>
            <dd className="mono">
              {String(provenance.few_shot_sha256 || "未记录")}
            </dd>
            <dt>输入 token 总量</dt>
            <dd>
              {formatNumber(dataset.performance_stats?.total_input_tokens, 0)}
            </dd>
            <dt>输出 token 总量</dt>
            <dd>
              {formatNumber(dataset.performance_stats?.total_output_tokens, 0)}
            </dd>
          </dl>
        </section>
      </div>
      <div className="quality-analysis-grid">
        <section className="surface">
          <div className="section-head">
            <div>
              <span className="eyebrow">SCORING METHODS</span>
              <h2>评分方式</h2>
            </div>
          </div>
          {Object.entries(methodCounts).length ? (
            <div className="quality-count-list">
              {Object.entries(methodCounts)
                .sort((a, b) => b[1] - a[1])
                .map(([method, count]) => (
                  <div key={method}>
                    <span>{method}</span>
                    <strong>{count} 条</strong>
                  </div>
                ))}
            </div>
          ) : (
            <Empty title="尚无评分方式记录" text="逐样本元数据缺失。" />
          )}
          <p className="chart-caption">只统计当前数据集的逐样本评分记录。</p>
          {analysis && (
            <>
              <h3>答案解析方式</h3>
              <div className="quality-count-list">
                {Object.entries(analysis.parsers).map(([method, count]) => (
                  <div key={method}>
                    <span>{method}</span>
                    <strong>{count} 条</strong>
                  </div>
                ))}
              </div>
              <p className="chart-caption">
                正值解析置信度：{analysis.confidence.count} 条，均值{" "}
                {formatPercent(analysis.confidence.mean)}；≥80%{" "}
                {analysis.confidence.high} 条，&lt;50% {analysis.confidence.low}{" "}
                条。未记录的默认零不推算置信度；解析置信度不是模型校准概率。
              </p>
            </>
          )}
        </section>
        <section className="surface">
          <div className="section-head">
            <div>
              <span className="eyebrow">FAILURE CAUSES</span>
              <h2>失败归因</h2>
            </div>
          </div>
          {Object.entries(failureCounts).length ? (
            <div className="quality-count-list">
              {Object.entries(failureCounts)
                .sort((a, b) => b[1] - a[1])
                .map(([reason, count]) => (
                  <div key={reason}>
                    <span>{reason}</span>
                    <strong>{count} 条</strong>
                  </div>
                ))}
            </div>
          ) : (
            <Empty title="没有错误样本" text="当前逐样本结果全部正确。" />
          )}
          <p className="chart-caption">
            归因来自评测器元数据；缺失时标记为未分类。
          </p>
        </section>
      </div>
      {analysis?.reasoning_analysis && (
        <ReasoningAnalysis summary={analysis.reasoning_analysis} />
      )}
      {analysis && <FailureAnalysis summary={analysis.failure_analysis} />}
      <section className="surface">
        <div className="section-head">
          <div>
            <span className="eyebrow">ERROR REVIEW</span>
            <h2>
              逐样本诊断 <span className="count-tag">{details.length}</span>
            </h2>
          </div>
          <button className="button subtle" onClick={onExportErrors}>
            导出全部错误 CSV ↓
          </button>
          <button className="button subtle" onClick={onExportSamples}>
            导出全部样本 CSV ↓
          </button>
        </div>
        <div className="quality-error-controls">
          <select
            aria-label="选择样本 ID"
            value={selectedError === null ? "" : String(selectedError)}
            onChange={(event) =>
              setSelectedError(
                event.target.value ? Number(event.target.value) : null,
              )
            }
          >
            <option value="">选择本页样本 ID</option>
            {filtered
              .slice(currentPage * 50, (currentPage + 1) * 50)
              .map(({ item, index }) => (
                <option key={index} value={index}>
                  {item.sample_id}
                </option>
              ))}
          </select>
          <select
            aria-label="样本查看范围"
            value={view}
            onChange={(event) => {
              setView(event.target.value);
              setPage(0);
              setSelectedError(null);
            }}
          >
            <option value="failures">错误样本</option>
            <option value="all">全部样本</option>
            <option value="correct">仅正确样本</option>
            <option value="judge">Judge 改判样本</option>
          </select>
          <select
            aria-label="样本排序"
            value={sort}
            onChange={(event) => {
              setSort(event.target.value);
              setPage(0);
              setSelectedError(null);
            }}
          >
            <option value="default">默认顺序</option>
            <option value="latency">按耗时（从高到低）</option>
            <option value="reasoning">按推理质量（从高到低）</option>
          </select>
          <select
            aria-label="筛选错误类别"
            value={category}
            onChange={(event) => {
              setCategory(event.target.value);
              setPage(0);
              setSelectedError(null);
            }}
          >
            <option value="">全部类别</option>
            {categories.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
          <input
            aria-label="搜索错误样本"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(0);
              setSelectedError(null);
            }}
            placeholder="搜索题目、预测或样本 ID"
          />
          <span>
            第 {currentPage + 1} / {pageCount} 页 · {filtered.length} 条
          </span>
        </div>
        <p className="chart-caption">
          耗时与推理评分来自已保存样本，缺失值排在最后；历史默认零无法证明已测量，不当作有效评分。推理评分为启发式分数，不代表经校准的推理能力。导出按钮下载全部样本，不受当前筛选和排序影响。
        </p>
        {filtered.length ? (
          <div className="table-scroll">
            <table className="data-table stats-table">
              <thead>
                <tr>
                  <th>样本</th>
                  <th>类别</th>
                  <th>题目摘要</th>
                  <th>标准答案</th>
                  <th>模型预测</th>
                  <th>诊断</th>
                </tr>
              </thead>
              <tbody>
                {filtered
                  .slice(currentPage * 50, (currentPage + 1) * 50)
                  .map(({ item, index }) => (
                    <tr key={`${item.sample_id}-${index}`}>
                      <td>
                        <strong>{item.sample_id}</strong>
                      </td>
                      <td>{item.category || "未分类"}</td>
                      <td className="quality-question">
                        {item.question.slice(0, 110)}
                        {item.question.length > 110 ? "…" : ""}
                      </td>
                      <td>{item.correct_answer || "—"}</td>
                      <td>{item.predicted_answer || "—"}</td>
                      <td>
                        <button
                          className="text-button"
                          onClick={() => setSelectedError(index)}
                        >
                          查看 →
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty title="当前筛选下没有样本" text="可调整类别或搜索条件。" />
        )}
        <div className="api-form-actions">
          <button
            className="button subtle"
            disabled={currentPage === 0}
            onClick={() => {
              setPage(currentPage - 1);
              setSelectedError(null);
            }}
          >
            上一页样本
          </button>
          <button
            className="button subtle"
            disabled={currentPage + 1 >= pageCount}
            onClick={() => {
              setPage(currentPage + 1);
              setSelectedError(null);
            }}
          >
            下一页样本
          </button>
        </div>
        {selected && (
          <div className="quality-sample">
            <div className="section-head">
              <div>
                <span className="eyebrow">SAMPLE / {selected.sample_id}</span>
                <h3>逐样本诊断</h3>
              </div>
              <button
                className="text-button"
                onClick={() => setSelectedError(null)}
              >
                收起
              </button>
            </div>
            <div className="metric-grid">
              <MetricCard
                label="样本调用耗时"
                value={formatNumber(latency(selected), 1)}
                note="ms · 已保存测量"
              />
              <MetricCard
                label="推理质量评分"
                value={
                  score(selected) === null
                    ? "未记录"
                    : formatNumber(score(selected), 1)
                }
                note="/ 10 · 启发式评分"
              />
            </div>
            {selected.reasoning_content && (
              <details>
                <summary>推理过程</summary>
                <pre>{selected.reasoning_content}</pre>
              </details>
            )}
            <div className="quality-sample-columns">
              <div>
                <strong>输入提示词</strong>
                <pre>{selected.prompt || selected.question}</pre>
              </div>
              <div>
                <strong>模型完整响应</strong>
                <pre>{selected.model_response || "未记录"}</pre>
              </div>
            </div>
            <p>
              评分方式：{selected.evaluation_method || "未记录"} · 解析方式：
              {selected.answer_parse_method || "未记录"} · 解析置信度：
              {formatPercent(selected.answer_parse_confidence)}
            </p>
            {selected.measurement_provenance?.output_token_scope ===
              "response_candidates_excluding_thoughts" && (
              <section className="gemini-sample-usage">
                <h3>Gemini 本次用量</h3>
                <p className="chart-caption">
                  首内容延迟计至首个答案文本；输出 token 和速度仅含答案。
                  思考摘要不参与答案评分，缺失用量保持“未记录”。
                </p>
                <dl>
                  {[
                    ["promptTokenCount", "输入 token"],
                    ["candidatesTokenCount", "答案 token"],
                    ["thoughtsTokenCount", "思考 token"],
                    ["cachedContentTokenCount", "服务端缓存 token"],
                    ["totalTokenCount", "接口报告总 token"],
                  ].map(([field, label]) => {
                    const count =
                      selected.measurement_provenance?.provider_usage?.[field];
                    return (
                      <div key={field}>
                        <dt>{label}</dt>
                        <dd>
                          {typeof count === "number"
                            ? formatNumber(count, 0)
                            : "未记录"}
                        </dd>
                      </div>
                    );
                  })}
                  <div>
                    <dt>完成原因</dt>
                    <dd>
                      {selected.measurement_provenance.finish_reason ||
                        "未记录"}
                    </dd>
                  </div>
                </dl>
              </section>
            )}
            <p>
              最终判定：{selected.is_correct ? "正确" : "错误"} · Judge 改判：
              {selected.is_judge_corrected ? "是" : "否"} · 复核原始判定：
              {selected.judge_verdict || "未记录"}。自评复核不构成独立验证。
            </p>
            {selected.failure_root_cause && (
              <p>规则推测的根因：{selected.failure_root_cause}（待核验）</p>
            )}
            {typeof selected.failure_confidence === "number" && (
              <p>
                规则分类标记：{formatPercent(selected.failure_confidence)}
                （规则给定，未校准；不是模型置信概率）
              </p>
            )}
            {selected.failure_suggestions?.length ? (
              <ul>
                {selected.failure_suggestions.map((suggestion, index) => (
                  <li key={index}>{suggestion}</li>
                ))}
              </ul>
            ) : null}
            {(selected.failure_category ||
              selected.failure_analysis ||
              selected.execution_error ||
              selected.error) && (
              <p>
                失败归因：{selected.failure_category || "未分类"} ·{" "}
                {selected.execution_error ||
                  selected.failure_analysis ||
                  selected.error}
              </p>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
