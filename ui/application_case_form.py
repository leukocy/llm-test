"""
模型×应用用例 录入与管理 UI（手册 maTest 模板的手动采集层）。

evaluator（quality_evaluator 自动采集）覆盖代码/长文/检索/对答四类场景，但
citation_score / quality_score / tool_success_rate / retrieval_latency_s（真实
RAG 引用 / Agent 工具链）evaluator 不产出。本面板让 tester 手动录入这些用例，
与自动采集行写同一张 application_cases 表，统一导出 maTest。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.application_scenario import APPLICATION_SCENARIOS
from core.database import db_manager
from core.models import ApplicationCase
from core.warehouse import distinct_values

# 列表展示列
_LIST_COLUMNS = [
    "date",
    "scenario",
    "model_name",
    "machine_id",
    "success",
    "quality_score",
    "citation_score",
    "tool_success_rate",
    "retrieval_latency_s",
    "decode_tps",
    "external_level",
    "tester",
    "source",
    "evaluator_name",
]

# 全部场景选项（应用四类 + 能力基准 + 其它）
_SCENARIO_OPTIONS = [
    "coding",
    "long_doc",
    "retrieval",
    "dialogue",
    "agent",
    "knowledge_qa",
    "other",
]


def render_application_case_manager() -> None:
    """渲染应用用例录入 + 列表 + 删除。"""
    st.subheader("模型 × 应用用例")
    st.caption(
        "手动录入 evaluator 覆盖不到的应用场景（真实 RAG 引用 / Agent 工具链），补 "
        "quality_score / citation_score / tool_success_rate / retrieval_latency_s。"
        "自动采集行（source=auto）也在此列表。"
    )

    tab_form, tab_list = st.tabs(["录入用例", "用例列表"])

    with tab_form:
        _render_form()
    with tab_list:
        _render_list()


# ---------------------------------------------------------------------------
# 录入表单
# ---------------------------------------------------------------------------


def _render_form() -> None:
    db = db_manager

    with st.form("app_case_form", clear_on_submit=True):
        st.markdown("**对象**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            model_opts = [""] + distinct_values(db, "model_name")
            model_name = st.selectbox("模型 model_name *", model_opts, key="ac_model")
            if not model_name:
                model_name = st.text_input("或手填模型名", key="ac_model_manual")
        with c2:
            machine_opts = [""] + distinct_values(db, "machine_id")
            machine_id = st.selectbox("machine_id", machine_opts, key="ac_machine")
        with c3:
            engine_opts = [""] + distinct_values(db, "engine")
            engine = st.selectbox("引擎 engine", engine_opts, key="ac_engine")
        with c4:
            tester = st.text_input(
                "测试员 tester *",
                value=st.session_state.get("tester", ""),
                key="ac_tester",
            )

        st.markdown("**场景**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            scenario = st.selectbox(
                "scenario *",
                _SCENARIO_OPTIONS,
                index=0,
                key="ac_scenario",
                help="coding/long_doc/retrieval/dialogue/agent 为手册应用四类",
            )
        with c2:
            task_name = st.text_input("task_name", key="ac_task")
        with c3:
            customer_type = st.text_input("customer_type", key="ac_customer")
        with c4:
            usecase_set_version = st.text_input("usecase_set_version", key="ac_usecase")

        st.markdown("**质量评分（evaluator 不产出，手动补）**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            success = st.selectbox(
                "success", ["未定", "成功", "失败"], key="ac_success"
            )
        with c2:
            quality_score = st.number_input(
                "quality_score",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                format="%.1f",
                key="ac_quality",
            )
        with c3:
            citation_score = st.number_input(
                "citation_score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="ac_citation",
            )
        with c4:
            tool_success_rate = st.number_input(
                "tool_success_rate",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="ac_tool",
            )

        st.markdown("**性能（可选）**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            retrieval_latency_s = st.number_input(
                "retrieval_latency_s",
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="ac_retrieval",
            )
        with c2:
            ttft_s = st.number_input(
                "ttft_s",
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="ac_ttft",
            )
        with c3:
            total_latency_s = st.number_input(
                "total_latency_s",
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="ac_total",
            )
        with c4:
            decode_tps = st.number_input(
                "decode_tps",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.1f",
                key="ac_decode",
            )

        st.markdown("**元数据**")
        c1, c2, c3 = st.columns(3)
        with c1:
            external_level = st.selectbox(
                "external_level", ["internal", "review", "publishable"], key="ac_level"
            )
        with c2:
            privacy_requirement = st.text_input("privacy_requirement", key="ac_privacy")
        with c3:
            evidence_path = st.text_input("evidence_path", key="ac_evidence")
        failure_reason = st.text_input("failure_reason（失败时填）", key="ac_failure")
        next_action = st.text_input("next_action", key="ac_next")
        sales_summary = st.text_area(
            "sales_summary（对外口径一句话）", key="ac_sales", height=68
        )

        submitted = st.form_submit_button("保存用例")
        if submitted:
            if not model_name or not tester:
                st.error("model_name 和 tester 必填")
                return
            success_val = {"未定": None, "成功": True, "失败": False}[success]
            case = ApplicationCase(
                source="manual",
                date=datetime.now().strftime("%Y-%m-%d"),
                tester=tester,
                scenario=scenario,
                task_name=task_name,
                customer_type=customer_type,
                model_name=model_name,
                machine_id=machine_id,
                engine=engine,
                usecase_set_version=usecase_set_version,
                quality_score=quality_score or None,
                success=success_val,
                citation_score=citation_score or None,
                tool_success_rate=tool_success_rate or None,
                retrieval_latency_s=retrieval_latency_s or None,
                ttft_s=ttft_s or None,
                total_latency_s=total_latency_s or None,
                decode_tps=decode_tps or None,
                privacy_requirement=privacy_requirement,
                external_level=external_level,
                evidence_path=evidence_path,
                failure_reason=failure_reason,
                next_action=next_action,
                sales_summary=sales_summary,
            )
            try:
                db.save_application_case(case)
                st.success(f"已保存用例（scenario={scenario}, model={model_name}）")
            except Exception as e:  # noqa: BLE001
                st.error(f"保存失败: {e}")


# ---------------------------------------------------------------------------
# 用例列表 + 筛选 + 删除
# ---------------------------------------------------------------------------


def _render_list() -> None:
    db = db_manager
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        scenario = st.selectbox(
            "scenario", ["全部"] + _SCENARIO_OPTIONS, key="acl_scenario"
        )
    with f2:
        model = st.selectbox(
            "model", ["全部"] + distinct_values(db, "model_name"), key="acl_model"
        )
    with f3:
        level = st.selectbox(
            "external_level",
            ["全部", "internal", "review", "publishable"],
            key="acl_level",
        )
    with f4:
        source = st.selectbox("source", ["全部", "auto", "manual"], key="acl_source")

    cases = db.list_application_cases(
        scenario=None if scenario == "全部" else scenario,
        model_name=None if model == "全部" else model,
        external_level=None if level == "全部" else level,
        source=None if source == "全部" else source,
        limit=500,
    )

    if not cases:
        st.info("无应用用例。自动采集（跑 Model Quality Test）或手动录入后会出现在此。")
        return

    st.caption(
        f"共 {len(cases)} 条；其中应用场景 "
        f"{sum(1 for c in cases if c.scenario in APPLICATION_SCENARIOS)} 条"
    )

    rows = [{k: getattr(c, k) for k in _LIST_COLUMNS} for c in cases]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 行内删除
    st.markdown("**删除用例**")
    opts = [f"{c.case_id[:12]}… {c.scenario} {c.model_name} {c.date}" for c in cases]
    if opts:
        pick = st.selectbox("选择要删除的用例", opts, key="acl_delete_pick")
        if st.button("删除", key="acl_delete_btn"):
            idx = opts.index(pick)
            db.delete_application_case(cases[idx].case_id)
            st.success("已删除")
            st.rerun()
