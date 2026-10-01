import { useMemo, useState } from "react";
import type { QualityReport } from "../api";
import { Empty, MetricCard } from "../components";
import { formatNumber, formatPercent } from "../constants";

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
}: {
  report: QualityReport;
  onExportErrors: () => void;
}) {
  const names = Object.keys(report.datasets);
  const [selectedName, setSelectedName] = useState(names[0] || "");
  const [category, setCategory] = useState("");
  const [search, setSearch] = useState("");
  const [selectedError, setSelectedError] = useState<number | null>(null);
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
    () =>
      [
        ...new Set(failures.map(({ item }) => item.category || "未分类")),
      ].sort(),
    [failures],
  );
  const filtered = failures.filter(
    ({ item }) =>
      (!category || (item.category || "未分类") === category) &&
      (!search ||
        `${item.question} ${item.sample_id} ${item.predicted_answer}`
          .toLowerCase()
          .includes(search.toLowerCase())),
  );
  const selected = failures.find(({ index }) => index === selectedError)?.item;
  if (!dataset)
    return <Empty title="没有质量评估样本" text="报告不包含可分析的数据集。" />;
  const interval = wilson(dataset.correct_samples, dataset.total_samples);
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
            dataset.total_samples
              ? dataset.correct_samples / dataset.total_samples
              : null,
          )}
          note={
            interval
              ? `95% Wilson ${formatPercent(interval[0])}–${formatPercent(interval[1])}`
              : "无有效样本"
          }
          accent
        />
        <MetricCard
          label="有效样本"
          value={String(dataset.total_samples)}
          note={`${dataset.correct_samples} 条正确`}
        />
        <MetricCard
          label="首字延迟"
          value={`${formatNumber(dataset.performance_stats?.avg_ttft_ms, 1)} ms`}
          note="有效样本均值"
        />
        <MetricCard
          label="生成速度"
          value={formatNumber(dataset.performance_stats?.avg_tps, 1)}
          note="tokens/s · 有效样本均值"
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
      <section className="surface">
        <div className="section-head">
          <div>
            <span className="eyebrow">ERROR REVIEW</span>
            <h2>
              错误样本诊断 <span className="count-tag">{failures.length}</span>
            </h2>
          </div>
          <button className="button subtle" onClick={onExportErrors}>
            导出全部错误 CSV ↓
          </button>
        </div>
        <div className="quality-error-controls">
          <select
            aria-label="筛选错误类别"
            value={category}
            onChange={(event) => {
              setCategory(event.target.value);
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
            onChange={(event) => setSearch(event.target.value)}
            placeholder="搜索题目、预测或样本 ID"
          />
          <span>
            显示 {Math.min(filtered.length, 50)} / {filtered.length}
          </span>
        </div>
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
                {filtered.slice(0, 50).map(({ item, index }) => (
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
          <Empty title="当前筛选下没有错误样本" text="可调整类别或搜索条件。" />
        )}
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
