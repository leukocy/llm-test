import { useEffect, useState } from "react";
import { api } from "../api";
import { TABS, type Catalog } from "./advanced/Shared";
import { ParserDemo } from "./advanced/Parser";
import { ConsistencyDemo } from "./advanced/Consistency";
import { PerturbDemo } from "./advanced/Perturb";
import { ReasoningDemo } from "./advanced/Reasoning";

export function Advanced({ token }: { token: string }) {
  const [tab, setTab] = useState(0);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setCatalog(null);
    setCatalogError("");
    void api<Catalog>(token, "/api/v1/advanced/catalog", {
      signal: controller.signal,
    })
      .then((data) => {
        if (!controller.signal.aborted) setCatalog(data);
      })
      .catch((exc) => {
        if (!controller.signal.aborted)
          setCatalogError(
            exc instanceof Error ? exc.message : "工具清单加载失败",
          );
      });
    return () => controller.abort();
  }, [token, retry]);
  return (
    <div className="page-grid advanced-tools">
      <div className="page-head">
        <div>
          <span className="eyebrow">ADVANCED EVALUATION</span>
          <h1>高级评估</h1>
          <p>
            解析答案、检查重复回答、生成文本扰动与评分推理过程。分析已有文本，不调用模型。
          </p>
        </div>
        <span className="minor-tag">初版工具 · 可复现记录</span>
      </div>
      <div className="warehouse-tabs" role="tablist" aria-label="高级评估工具">
        {TABS.map((label, index) => (
          <button
            key={label}
            id={`advanced-tab-${index}`}
            role="tab"
            aria-selected={tab === index}
            aria-controls={`advanced-panel-${index}`}
            tabIndex={tab === index ? 0 : -1}
            className={tab === index ? "active" : ""}
            onClick={() => setTab(index)}
            onKeyDown={(event) => {
              const next =
                event.key === "ArrowRight"
                  ? (index + 1) % TABS.length
                  : event.key === "ArrowLeft"
                    ? (index + TABS.length - 1) % TABS.length
                    : event.key === "Home"
                      ? 0
                      : event.key === "End"
                        ? TABS.length - 1
                        : null;
              if (next != null) {
                event.preventDefault();
                setTab(next);
                document.getElementById(`advanced-tab-${next}`)?.focus();
              }
            }}
          >
            {label}
          </button>
        ))}
      </div>
      {[
        <ParserDemo token={token} />,
        <ConsistencyDemo token={token} />,
        <PerturbDemo
          token={token}
          catalog={catalog}
          catalogError={catalogError}
          onRetry={() => setRetry((value) => value + 1)}
        />,
        <ReasoningDemo token={token} />,
      ].map((panel, index) => (
        <div
          key={index}
          role="tabpanel"
          id={`advanced-panel-${index}`}
          aria-labelledby={`advanced-tab-${index}`}
          hidden={tab !== index}
        >
          {panel}
        </div>
      ))}
    </div>
  );
}
