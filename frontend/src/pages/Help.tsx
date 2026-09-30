import { useState } from "react";
import { Link } from "react-router-dom";

const COMPLETED_KEY = "llm-test-guide-completed-v1";
const SKIPPED_KEY = "llm-test-guide-skipped-v1";

function guideHidden() {
  try {
    return (
      localStorage.getItem(COMPLETED_KEY) === "true" ||
      sessionStorage.getItem(SKIPPED_KEY) === "true"
    );
  } catch {
    return false;
  }
}

function rememberGuide(complete: boolean) {
  try {
    if (complete) localStorage.setItem(COMPLETED_KEY, "true");
    else sessionStorage.setItem(SKIPPED_KEY, "true");
  } catch {
    // Storage restrictions must not prevent using the platform.
  }
}

export function WelcomeGuide() {
  const [hidden, setHidden] = useState(guideHidden);
  if (hidden) return null;
  return (
    <section className="welcome-guide surface" aria-label="首次使用提示">
      <div>
        <span className="eyebrow">GET STARTED</span>
        <strong>从第一项可重复测量开始</strong>
        <p>
          配置受测
          API，核对工作负载，再查看结果与报告。操作引导随时可从帮助入口重新打开。
        </p>
      </div>
      <div className="guide-actions">
        <Link className="button primary" to="/help">
          快速引导 →
        </Link>
        <button
          className="button subtle"
          onClick={() => {
            rememberGuide(true);
            setHidden(true);
          }}
        >
          知道了
        </button>
      </div>
    </section>
  );
}

const steps = [
  {
    title: "认识工作台",
    subtitle: "性能、质量与鲁棒性，分别选定测量目标",
    text: "性能测试记录逐请求延迟、吞吐和失败率；质量评估验证答案；鲁棒性评估观察扰动后的答案保持度。先确定目标，再选择对应的工作负载。",
    path: "/new",
    action: "查看测量类型",
  },
  {
    title: "配置受测 API",
    subtitle: "明确端点、模型和凭证",
    text: "选择服务商模板，保存 API 地址、模型与凭证。可刷新模型列表、检测连接、测量 /models 首部参考耗时。参考耗时包含服务商处理时间，不能当作纯网络 RTT。凭证由后端加密保存，列表不回传明文。",
    path: "/settings/api",
    action: "配置受测 API",
  },
  {
    title: "选择测试与参数",
    subtitle: "一键设置，再按需要细调",
    text: "创建测量可选快速检查、标准测量、深入测量或初版参数。支持自定义提示词与 TXT 导入，保存方案后可复用；报告环境可填写处理器、主板、内存、GPU、系统和引擎。基础三项快捷批次对应并发、Prefill 和长上下文。",
    path: "/new",
    action: "创建测量",
  },
  {
    title: "核对计划并运行",
    subtitle: "区分正式请求与预热，关注执行条件",
    text: "提交前核对请求预算、输入长度与样本量。批次默认串行，也可设置并行和失败即停。后台执行不会因关闭页面而中断；性能任务在请求组排空后暂停，质量和鲁棒性任务在当前样本保存后暂停。质量/鲁棒性支持 worker 中断或手动停止后的显式恢复，会核验模型、参数与代码，复用已提交样本；尚未提交的样本可能重跑。性能测量的 worker 重启恢复仍待补齐。",
    path: "/batch",
    action: "查看批量配置",
  },
  {
    title: "读懂结果与报告",
    subtitle: "先核验完整性，再看分布与来源",
    text: "运行详情显示进度、条件切片、逐请求数据和日志。先检查完整性与数据质量提示，核对 token 来源、样本量、预热和暂停条件；导出 CSV、JSON、Markdown、HTML 报告或性能 PNG。模型对比前核对相同工作负载和环境。",
    path: "/runs",
    action: "查看运行记录",
  },
  {
    title: "准备就绪",
    subtitle: "保存可复用的测量方案",
    text: "先用快速检查确认连接，再为正式对比准备固定参数、充足样本和明确的测试环境。帮助入口随时可以重新打开，完成状态仅保存在当前浏览器。",
    path: "/new",
    action: "开始创建测量",
  },
];

const faqs = [
  [
    "如何选择并发和样本量？",
    "先用较低并发确认服务可用，再逐级增加。正式比较时固定输入、输出预算和预热设置；初版默认的单样本参数只能用于复现操作。小样本 p95 / p99 分辨率有限，应结合完整分布和失败率。",
  ],
  [
    "请求失败时如何排查？",
    "从运行详情的日志和错误分类开始：核对 API 地址、凭证、模型名称、限流和上下文上限。保留失败记录，减少并发或缩短输入后以新任务复测，不把失败结果从比较中隐去。",
  ],
  [
    "如何公平地比较模型？",
    "保持数据集版本与样本指纹、输入规模、生成参数、token 算法和执行环境一致。分别核对成功率、延迟和吞吐；暂停、并行批次、缓存与估算 token 来源都可能影响可比性。报告不自动推断统计显著性。",
  ],
  [
    "测试数据存在哪里，如何下载？",
    "队列和逐请求观测保存在服务端数据库，报告、预热记录和执行日志保存在服务端工件目录。运行详情可下载报告与 CSV，数据仓库支持历史查询和导出；导出报告可能包含业务文本或用户填写的环境信息。",
  ],
];

export function Help() {
  const [step, setStep] = useState(0);
  const [visible, setVisible] = useState(true);
  const [notice, setNotice] = useState("");
  const current = steps[step];
  return (
    <div className="page-grid help-page">
      <div className="page-head">
        <div>
          <span className="eyebrow">GUIDE & REFERENCE</span>
          <h1>帮助与使用引导</h1>
          <p>从受测 API 到可追溯报告，六步了解完整测量流程。</p>
        </div>
        <button
          className="button subtle"
          onClick={() => {
            setStep(0);
            setVisible(true);
            setNotice("");
          }}
        >
          重新开始引导
        </button>
      </div>
      {notice && (
        <p className="form-success" role="status">
          {notice}
        </p>
      )}
      {visible && (
        <section className="surface guide-panel" aria-label="操作引导">
          <ol className="guide-progress" aria-label="引导进度">
            {steps.map((item, index) => (
              <li
                key={item.title}
                aria-current={index === step ? "step" : undefined}
                className={index <= step ? "reached" : ""}
              >
                <span>{index + 1}</span>
                {item.title}
              </li>
            ))}
          </ol>
          <div className="guide-content">
            <span className="eyebrow">
              STEP {step + 1} / {steps.length}
            </span>
            <h2>{current.title}</h2>
            <h3>{current.subtitle}</h3>
            <p>{current.text}</p>
            <Link className="text-action" to={current.path}>
              {current.action} →
            </Link>
          </div>
          <div className="guide-footer">
            <button
              className="button subtle"
              disabled={step === 0}
              onClick={() => setStep((value) => value - 1)}
            >
              上一步
            </button>
            <button
              className="button primary"
              onClick={() => {
                if (step < steps.length - 1) setStep((value) => value + 1);
                else {
                  rememberGuide(true);
                  setVisible(false);
                  setNotice("引导已完成。可以随时重新开始。");
                }
              }}
            >
              {step < steps.length - 1 ? "下一步" : "完成引导"}
            </button>
            <button
              className="text-action"
              onClick={() => {
                rememberGuide(false);
                setVisible(false);
                setNotice("已跳过本次引导，仍可随时重新打开。");
              }}
            >
              跳过引导
            </button>
          </div>
        </section>
      )}
      <section className="surface">
        <div className="section-head">
          <h2>快速参考</h2>
        </div>
        <div className="guide-reference">
          {[
            ["01", "配置端点", "/settings/api", "选择模型并确认连接可用"],
            ["02", "核对参数", "/new", "固定工作负载、样本量与环境"],
            ["03", "执行测量", "/batch", "明确串行、并行与停止策略"],
            ["04", "分析与导出", "/runs", "核验完整性与统计口径"],
          ].map(([index, title, path, text]) => (
            <Link key={index} to={path}>
              <span>{index}</span>
              <strong>{title}</strong>
              <small>{text}</small>
            </Link>
          ))}
        </div>
      </section>
      <details className="surface help-details">
        <summary>功能亮点</summary>
        <ul>
          <li>多种性能工作负载，质量数据集和输入扰动评估。</li>
          <li>参数表单、JSON、初版参数快选、保存方案与批量配置文件。</li>
          <li>任务状态、进度、暂停/继续、取消，以及可筛选的执行日志。</li>
          <li>逐请求统计、完整性核验、token 来源、历史图表与模型对比。</li>
          <li>CSV / JSON / Markdown / HTML 报告，性能和历史图表 PNG 下载。</li>
        </ul>
      </details>
      <details className="surface help-details">
        <summary>常见问题（FAQ）</summary>
        {faqs.map(([question, answer]) => (
          <details className="help-faq" key={question}>
            <summary>{question}</summary>
            <p>{answer}</p>
          </details>
        ))}
      </details>
    </div>
  );
}
