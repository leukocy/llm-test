"""数据管理页（仓库浏览器「数据管理」标签）。

三块能力，全部复用既有后端：
1. 导入回导 —— benchmark CSV / 硬件快照 JSON（DataImportService / hw_inventory_import）
2. 备份与恢复 —— DatabaseBackup（含恢复前自动快照）
3. 数据库健康 —— 大小 / 表行数 / schema 版本 / 完整性检查
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core.database import db_manager

_TABLES = [
    "test_runs",
    "test_results",
    "api_logs",
    "execution_logs",
    "reports",
    "application_cases",
    "control_jobs",
    "job_events",
    "control_presets",
]


def render_warehouse_admin() -> None:
    st.subheader("数据管理")
    st.caption("导入历史数据、备份/恢复数据库、检查数据库健康。操作直接作用于 data/benchmark.db。")

    sec_import, sec_backup, sec_health = st.tabs(["导入", "备份与恢复", "数据库健康"])
    with sec_import:
        _render_import_section()
    with sec_backup:
        _render_backup_section()
    with sec_health:
        _render_health_section()


# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------


def _render_import_section() -> None:
    st.markdown("#### 导入 benchmark CSV")
    st.caption(
        "把 raw_data 格式的结果 CSV 回导入库（生成已完成运行 + 逐请求结果），"
        "适合补录离线跑的历史数据。"
    )
    csv_files = st.file_uploader(
        "选择 CSV 文件（可多选）",
        type=["csv"],
        accept_multiple_files=True,
        key="wh_admin_csv",
    )
    c1, c2 = st.columns(2)
    model_override = c1.text_input("模型名覆盖（可选）", key="wh_admin_model")
    type_override = c2.text_input("测试类型覆盖（可选）", key="wh_admin_type")

    if st.button("开始导入 CSV", key="wh_admin_csv_btn", disabled=not csv_files):
        rows = []
        for f in csv_files:
            # import_csv_file 接受文件路径：先把上传内容落到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{f.name}", mode="wb") as tmp:
                tmp.write(f.getbuffer())
                tmp_path = tmp.name
            try:
                imported, errors = db_manager.import_csv(
                    tmp_path,
                    model_id=model_override or None,
                    test_type=type_override or None,
                )
                if errors:
                    rows.append({"文件": f.name, "结果": "失败", "明细": "; ".join(errors)})
                else:
                    rows.append({"文件": f.name, "结果": f"成功 {imported} 条", "明细": ""})
            except Exception as e:  # noqa: BLE001  单文件失败不阻断其余
                rows.append({"文件": f.name, "结果": "失败", "明细": str(e)})
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 导入硬件快照 JSON")
    st.caption("hw-snapshot/v1 格式的硬件快照，导入后在硬件盘点中可见（按机器指纹去重）。")
    hw_files = st.file_uploader(
        "选择快照 JSON（可多选）",
        type=["json"],
        accept_multiple_files=True,
        key="wh_admin_hw",
    )
    dedupe = st.checkbox("同机器同指纹跳过", value=True, key="wh_admin_dedupe")
    if st.button("开始导入快照", key="wh_admin_hw_btn", disabled=not hw_files):
        from core.hw_inventory_import import import_snapshots

        tmp_paths: list[str | Path] = []
        try:
            for f in hw_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="wb") as tmp:
                    tmp.write(f.getbuffer())
                    tmp_paths.append(tmp.name)
            summary = import_snapshots(db_manager, tmp_paths, dedupe=dedupe)
            st.success(
                f"导入完成：成功 {summary['imported']}，跳过 {summary['skipped']}，"
                f"失败 {summary['failed']}（共 {summary['total']} 份）"
            )
            failures = [
                r for r in summary.get("results", []) if not r.get("ok") and not r.get("skipped")
            ]
            if failures:
                st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"快照导入失败：{e}")
        finally:
            for p in tmp_paths:
                Path(p).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 备份与恢复
# ---------------------------------------------------------------------------


def _render_backup_section() -> None:
    st.markdown("#### 备份与恢复")

    if st.button("立即创建备份", key="wh_admin_backup_btn"):
        path = db_manager.create_backup("manual-ui")
        if path:
            st.success(f"备份已创建：{path}")
        else:
            st.error("备份失败，详见日志。")

    backups = db_manager.list_backups()
    if not backups:
        st.info("暂无备份。备份保留最近 10 份，超量自动轮换。")
        return

    st.dataframe(
        pd.DataFrame(backups).rename(
            columns={"name": "文件", "size_mb": "大小(MB)", "created_at": "创建时间"}
        )[["文件", "大小(MB)", "创建时间"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("##### 恢复备份")
    st.warning(
        "恢复会**整库替换**当前 data/benchmark.db（恢复前自动再打一份快照）。"
        "恢复后请重启应用以重建数据库连接。"
    )
    options = {b["path"]: f"{b['created_at']} · {b['size_mb']} MB · {b['name']}" for b in backups}
    picked = st.selectbox("选择要恢复的备份", list(options), format_func=lambda p: options[p])
    confirm = st.checkbox("确认整库恢复（当前库将被覆盖）", key="wh_admin_restore_confirm")
    if st.button(
        "执行恢复",
        key="wh_admin_restore_btn",
        disabled=not confirm,
        type="primary",
    ):
        if db_manager.restore_backup(picked):
            st.success("恢复完成。请重启应用（streamlit run app.py）。")
        else:
            st.error("恢复失败，详见日志（原库已尝试自动回滚）。")


# ---------------------------------------------------------------------------
# 数据库健康
# ---------------------------------------------------------------------------


def _render_health_section() -> None:
    st.markdown("#### 数据库健康")

    db = db_manager.db
    size_mb = round(db.get_database_size() / 1024 / 1024, 2)
    version = db.fetch_value("SELECT value FROM db_meta WHERE key = 'schema_version'") or "—"
    c1, c2 = st.columns(2)
    c1.metric("数据库大小", f"{size_mb} MB")
    c2.metric("schema 版本", version)

    st.markdown("##### 各表行数")
    counts = []
    for table in _TABLES:
        if db.table_exists(table):
            counts.append({"表": table, "行数": db.count(table)})
    st.dataframe(pd.DataFrame(counts), use_container_width=True, hide_index=True)

    if st.button("运行完整性检查", key="wh_admin_health_btn"):
        from core.database.migrations import check_database_health

        with db.get_connection() as conn:
            health = check_database_health(conn)
        if health.get("integrity") == "ok":
            st.success("完整性检查通过（PRAGMA integrity_check = ok）")
        else:
            st.warning(f"完整性检查发现异常：{health.get('integrity')}")
        st.json(health)
