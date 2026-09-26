"""Authenticated, process-independent control and read API."""

from __future__ import annotations

import dataclasses
import hmac
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from core.run_lifecycle import InvalidRunTransition, RunStatus
from server.analytics import run_results, run_results_csv, run_summary
from server.figures import compare_figures, run_detail_figures, trend_figure
from server.quality_export import quality_errors_csv
from server.reports import render_html, render_quality_html
from server.settings import Settings
from server.specs import (
    JobSubmission,
    PresetSubmission,
    StrictSpec,
    expected_requests,
    spec_catalog,
)
from server.store import IdempotencyConflict, JobNotFound, JobStore, PresetConflict, PresetNotFound
from server.warehouse import (
    ExportScopeTooLarge,
    WarehouseReader,
    WarehouseRunNotFound,
    WarehouseSelection,
)

_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


# ---------------------------------------------------------------------------
# 请求体模型（模块级定义：api.py 使用 __future__ annotations,
# 函数内嵌套类无法被 FastAPI 的前向引用解析, 会被误判为 query 参数）
# ---------------------------------------------------------------------------


class CompareBody(StrictSpec):
    test_ids: list[str] = Field(min_length=2, max_length=8)


class DeleteRunsBody(StrictSpec):
    test_ids: list[str] = Field(min_length=1, max_length=200)


class MetadataBody(StrictSpec):
    fields: dict[str, Any] = Field(min_length=1)


class CaseBody(StrictSpec):
    scenario: str = Field(min_length=1, max_length=200)
    model_name: str = Field(min_length=1, max_length=200)
    task_name: str = ""
    customer_type: str = ""
    machine_id: str = ""
    engine: str = ""
    tester: str = ""
    usecase_set_version: str = ""
    quality_score: float | None = Field(default=None, ge=0, le=100)
    citation_score: float | None = Field(default=None, ge=0, le=100)
    tool_success_rate: float | None = Field(default=None, ge=0, le=100)
    success: bool | None = None
    external_level: Literal["internal", "review", "publishable"] = "internal"
    privacy_requirement: str = ""
    evidence_path: str = ""
    failure_reason: str = ""
    next_action: str = ""
    sales_summary: str = ""


class RestoreBody(StrictSpec):
    path: str = Field(min_length=1, max_length=500)


def create_app(settings: Settings | None = None, store: JobStore | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or JobStore(settings.db_path)
    warehouse_reader = WarehouseReader(settings.db_path)
    # 数据管理/用例/图表路径复用 core 的 manager（生产进程内单例同路径；
    # 测试通过重置 Database/DatabaseManager 单例隔离）
    from core.database.manager import DatabaseManager

    data_manager = DatabaseManager(str(settings.db_path))
    app = FastAPI(title="LLM Test Control API", version="1.0.0", docs_url=None, redoc_url=None)

    def authenticate(authorization: Annotated[str | None, Header()] = None) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, settings.api_token):
            raise HTTPException(
                status_code=401,
                detail="Valid bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    auth = Depends(authenticate)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = (
            "no-store" if request.url.path.startswith("/api/") else "no-cache"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/health/live")
    def live():
        return {"status": "live"}

    @app.get("/health/ready")
    def ready():
        try:
            store.list(limit=1)
        except Exception as exc:
            raise HTTPException(503, "Database unavailable") from exc
        return {"status": "ready"}

    @app.get("/api/v1/endpoints", dependencies=[auth])
    def endpoints():
        return {"items": [endpoint.public() for endpoint in settings.endpoints.values()]}

    @app.get("/api/v1/specs", dependencies=[auth])
    def specs():
        """8 个测试类型的展示名 + JSON Schema + 运行配置旋钮 schema（驱动前端表单）。"""
        from server.specs import RunConfig

        return {"items": spec_catalog(), "run_config_schema": RunConfig.model_json_schema()}

    @app.get("/api/v1/presets", dependencies=[auth])
    def presets():
        return {"items": store.list_presets()}

    def validate_preset_endpoint(body: PresetSubmission) -> None:
        if body.endpoint_id not in settings.endpoints:
            raise HTTPException(422, "Unknown endpoint ID")

    @app.post("/api/v1/presets", dependencies=[auth], status_code=201)
    def create_preset(body: PresetSubmission):
        validate_preset_endpoint(body)
        try:
            return store.save_preset(
                name=body.name,
                endpoint_id=body.endpoint_id,
                test_type=body.test_type,
                parameters=body.parameters,
            )
        except PresetConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.put("/api/v1/presets/{preset_id}", dependencies=[auth])
    def update_preset(preset_id: str, body: PresetSubmission):
        validate_preset_endpoint(body)
        try:
            return store.save_preset(
                preset_id=preset_id,
                name=body.name,
                endpoint_id=body.endpoint_id,
                test_type=body.test_type,
                parameters=body.parameters,
            )
        except PresetNotFound as exc:
            raise HTTPException(404, "Preset not found") from exc
        except PresetConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/v1/presets/{preset_id}", dependencies=[auth], status_code=204)
    def delete_preset(preset_id: str):
        try:
            store.delete_preset(preset_id)
        except PresetNotFound as exc:
            raise HTTPException(404, "Preset not found") from exc
        return Response(status_code=204)

    def warehouse_selection(
        model_id: Annotated[str | None, Query(max_length=200)] = None,
        machine_id: Annotated[str | None, Query(max_length=120)] = None,
        test_type: Annotated[str | None, Query(max_length=80)] = None,
        status: Annotated[str | None, Query(max_length=80)] = None,
        external_level: Literal["internal", "review", "publishable"] | None = None,
        search: Annotated[str | None, Query(max_length=120)] = None,
    ) -> WarehouseSelection:
        return WarehouseSelection(
            model_id=model_id,
            machine_id=machine_id,
            test_type=test_type,
            status=status,
            external_level=external_level,
            search=search.strip() or None if search else None,
        )

    @app.get("/api/v1/warehouse", dependencies=[auth])
    def warehouse(
        selection: WarehouseSelection = Depends(warehouse_selection),
        metric: Literal["decode_tps", "effective_bandwidth_gbps"] = "decode_tps",
        aggregate: Literal["latest", "best"] = "latest",
    ):
        return warehouse_reader.dashboard(selection, metric=metric, aggregate=aggregate)

    @app.get("/api/v1/warehouse/runs/{test_id}", dependencies=[auth])
    def warehouse_run(test_id: str):
        try:
            return warehouse_reader.detail(test_id)
        except WarehouseRunNotFound as exc:
            raise HTTPException(404, "Warehouse run not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/warehouse/export", dependencies=[auth])
    def warehouse_export(
        template: Literal["hwInventory", "hmTest", "maTest"],
        selection: WarehouseSelection = Depends(warehouse_selection),
        format: Literal["csv", "json"] = "csv",
    ):
        # maTest 的真源是 application_cases（不受运行筛选影响）
        if template == "maTest":
            from core.warehouse import (
                build_ma_test_rows_from_cases,
                export_template_csv,
                export_template_json,
            )

            rows = build_ma_test_rows_from_cases(data_manager)
            content = (
                export_template_csv(template, rows)
                if format == "csv"
                else export_template_json(template, rows)
            )
            media_type = "text/csv; charset=utf-8" if format == "csv" else "application/json"
            return Response(
                content,
                media_type=media_type,
                headers={"Content-Disposition": f'attachment; filename="llm-test-maTest.{format}"'},
            )
        try:
            content = warehouse_reader.export(selection, template, format)
        except ExportScopeTooLarge as exc:
            raise HTTPException(409, str(exc)) from exc
        media_type = "text/csv; charset=utf-8" if format == "csv" else "application/json"
        return Response(
            content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="llm-test-{template}.{format}"'},
        )

    @app.get("/api/v1/warehouse/export.zip", dependencies=[auth])
    def warehouse_export_zip(
        selection: WarehouseSelection = Depends(warehouse_selection),
        format: Literal["csv", "json"] = "csv",
    ):
        """三套模板打包 ZIP（hw/hm 受筛选、maTest 全量用例）。"""
        from core.warehouse import (
            build_hardware_inventory_rows,
            build_hm_test_rows,
            build_ma_test_rows_from_cases,
            export_all_templates_zip,
        )

        try:
            window = warehouse_reader.window(selection)
        except ExportScopeTooLarge as exc:
            raise HTTPException(409, str(exc)) from exc
        if window.truncated or window.invalid_rows:
            raise HTTPException(409, "筛选窗口过大或存在非法行, 收窄筛选后再导出")
        bundles = {
            "hwInventory": build_hardware_inventory_rows(window.runs),
            "hmTest": build_hm_test_rows(window.runs),
            "maTest": build_ma_test_rows_from_cases(data_manager),
        }
        payload = export_all_templates_zip(bundles, fmt=format)
        return Response(
            payload,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="llm-test-templates-{format}.zip"'
            },
        )

    # ------------------------------------------------------------------
    # 图表 JSON（plotly.js 复用后端纯构建器）
    # ------------------------------------------------------------------

    @app.get("/api/v1/warehouse/figures/trend", dependencies=[auth])
    def warehouse_figure_trend(
        selection: WarehouseSelection = Depends(warehouse_selection),
        metric: str = "decode_tps",
        group_dim: str = "model_name",
        publishable_only: bool = False,
    ):
        try:
            return trend_figure(data_manager, selection, metric, group_dim, publishable_only)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/warehouse/figures/compare", dependencies=[auth])
    def warehouse_figure_compare(body: CompareBody):
        result = compare_figures(data_manager, body.test_ids)
        if not result["table"]:
            raise HTTPException(404, "所选运行均不存在或无指标数据")
        return result

    @app.get("/api/v1/warehouse/runs/{test_id}/figures", dependencies=[auth])
    def warehouse_run_figures(test_id: str):
        result = run_detail_figures(data_manager, test_id)
        if not result.get("found"):
            raise HTTPException(404, "Warehouse run not found")
        return result

    # ------------------------------------------------------------------
    # 数据管理写操作（与 Streamlit 仓库页同源的后端能力）
    # ------------------------------------------------------------------

    def _run_id_or_404(test_id: str) -> int:
        row = data_manager.db.fetch_one("SELECT id FROM test_runs WHERE test_id = ?", (test_id,))
        if row is None:
            raise HTTPException(404, "Warehouse run not found")
        return int(row["id"])

    @app.post("/api/v1/warehouse/runs/delete", dependencies=[auth])
    def warehouse_runs_delete(body: DeleteRunsBody):
        """按 test_id 批量删除运行（级联；部分失败不中断）。"""
        ids: list[int] = []
        unknown: list[str] = []
        for test_id in body.test_ids:
            row = data_manager.db.fetch_one(
                "SELECT id FROM test_runs WHERE test_id = ?", (test_id,)
            )
            (ids.append(int(row["id"])) if row else unknown.append(test_id))
        result = data_manager.delete_runs(ids)
        return {**result, "unknown_test_ids": unknown}

    @app.patch("/api/v1/warehouse/runs/{test_id}/metadata", dependencies=[auth])
    def warehouse_run_metadata(test_id: str, body: MetadataBody):
        """写回可对外元数据（字段走 update_publish_metadata 白名单）。"""
        run_id = _run_id_or_404(test_id)
        if not data_manager.update_publish_metadata(run_id, body.fields):
            raise HTTPException(422, "无有效元数据字段或更新失败")
        return {"updated": True}

    @app.post("/api/v1/warehouse/runs/{test_id}/gate", dependencies=[auth])
    def warehouse_run_gate(test_id: str):
        """对存量运行复评发布门禁（不写库）。"""
        row = data_manager.db.fetch_one("SELECT * FROM test_runs WHERE test_id = ?", (test_id,))
        if row is None:
            raise HTTPException(404, "Warehouse run not found")
        from core.publish_gate import gate_from_run

        success_rate = None
        if row.get("total_requests"):
            success_rate = (row.get("completed_requests") or 0) / row["total_requests"]
        result = gate_from_run(
            {
                **row,
                "system_info": row.get("system_info_json"),
                "config": row.get("config_json"),
            },
            success_rate=success_rate,
        )
        return dataclasses.asdict(result)

    # ------------------------------------------------------------------
    # 应用用例（application_cases）
    # ------------------------------------------------------------------

    @app.get("/api/v1/cases", dependencies=[auth])
    def list_cases(
        scenario: Annotated[str | None, Query(max_length=200)] = None,
        model_name: Annotated[str | None, Query(max_length=200)] = None,
        machine_id: Annotated[str | None, Query(max_length=120)] = None,
        external_level: Literal["internal", "review", "publishable"] | None = None,
        source: Annotated[str | None, Query(max_length=40)] = None,
        limit: Annotated[int, Query(ge=1, le=2000)] = 500,
    ):
        cases = data_manager.list_application_cases(
            scenario=scenario,
            model_name=model_name,
            machine_id=machine_id,
            external_level=external_level,
            source=source,
            limit=limit,
        )
        return {"items": [c.to_dict() for c in cases]}

    @app.post("/api/v1/cases", dependencies=[auth], status_code=201)
    def create_case(body: CaseBody):
        from datetime import datetime

        from core.models.application_case import ApplicationCase

        case = ApplicationCase(source="manual", date=datetime.now().strftime("%Y-%m-%d"))
        for key, value in body.model_dump().items():
            setattr(case, key, value)
        data_manager.save_application_case(case)
        return {"case_id": case.case_id}

    @app.delete("/api/v1/cases/{case_id}", dependencies=[auth], status_code=204)
    def delete_case(case_id: str):
        if not data_manager.delete_application_case(case_id):
            raise HTTPException(404, "Case not found")
        return Response(status_code=204)

    @app.get("/api/v1/cases/capability", dependencies=[auth])
    def cases_capability(
        min_level: Literal["internal", "review", "publishable"] = "review",
        group_by: Annotated[str, Query(max_length=120)] = "customer_type,scenario,model_name",
    ):
        from core.warehouse import build_capability_markdown, build_capability_sheet

        dims = tuple(d.strip() for d in group_by.split(",") if d.strip())
        if not dims or any(d not in {"customer_type", "scenario", "model_name"} for d in dims):
            raise HTTPException(422, "group_by 仅支持 customer_type/scenario/model_name")
        cases = data_manager.list_application_cases(limit=2000)
        sheet = build_capability_sheet(cases, group_by=dims, min_external_level=min_level)
        return {"items": sheet, "markdown": build_capability_markdown(sheet)}

    # ------------------------------------------------------------------
    # 数据管理（导入 / 备份 / 健康）
    # ------------------------------------------------------------------

    @app.post("/api/v1/admin/import/csv", dependencies=[auth])
    async def admin_import_csv(
        file: UploadFile,
        model_id: Annotated[str | None, Query(max_length=200)] = None,
        test_type: Annotated[str | None, Query(max_length=80)] = None,
    ):
        """上传 benchmark CSV 回导入库（生成已完成运行 + 逐请求结果）。"""
        import tempfile

        if not (file.filename or "").lower().endswith(".csv"):
            raise HTTPException(422, "仅支持 .csv 文件")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            imported, errors = data_manager.import_csv(
                tmp_path, model_id=model_id, test_type=test_type
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        status = 422 if errors else 200
        return JSONResponse({"imported": imported, "errors": errors}, status_code=status)

    @app.post("/api/v1/admin/import/hw-snapshot", dependencies=[auth])
    async def admin_import_hw_snapshot(files: list[UploadFile], dedupe: bool = True):
        """上传硬件快照 JSON（hw-snapshot/v1）批量导入。"""
        import tempfile

        from core.hw_inventory_import import import_snapshots

        if not files or len(files) > 20:
            raise HTTPException(422, "一次上传 1~20 份快照")
        tmp_paths: list[str | Path] = []
        try:
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="wb") as tmp:
                    tmp.write(await f.read())
                    tmp_paths.append(tmp.name)
            return import_snapshots(data_manager, tmp_paths, dedupe=dedupe)
        finally:
            for p in tmp_paths:
                Path(p).unlink(missing_ok=True)

    @app.get("/api/v1/admin/db/health", dependencies=[auth])
    def admin_db_health():
        from core.database.migrations import check_database_health

        db = data_manager.db
        tables = {}
        for table in (
            "test_runs",
            "test_results",
            "api_logs",
            "execution_logs",
            "reports",
            "application_cases",
            "control_jobs",
            "job_events",
            "control_presets",
        ):
            if db.table_exists(table):
                tables[table] = db.count(table)
        with db.get_connection() as conn:
            health = check_database_health(conn)
        return {
            "size_bytes": db.get_database_size(),
            "schema_version": db.fetch_value(
                "SELECT value FROM db_meta WHERE key = 'schema_version'"
            ),
            "tables": tables,
            "integrity": health.get("integrity"),
        }

    @app.get("/api/v1/admin/backups", dependencies=[auth])
    def admin_list_backups():
        return {"items": data_manager.list_backups()}

    @app.post("/api/v1/admin/backups", dependencies=[auth], status_code=201)
    def admin_create_backup():
        path = data_manager.create_backup("api")
        if not path:
            raise HTTPException(500, "备份失败")
        return {"path": str(path)}

    @app.post("/api/v1/admin/backups/restore", dependencies=[auth])
    def admin_restore_backup(body: RestoreBody):
        """整库恢复（仅限备份目录内的文件；恢复后服务需重启）。"""
        backup_dir = (Path(str(settings.db_path)).parent / "backups").resolve()
        target = Path(body.path).resolve()
        if not target.is_relative_to(backup_dir) or not target.is_file():
            raise HTTPException(422, "备份文件不存在或不在备份目录内")
        if not data_manager.restore_backup(target):
            raise HTTPException(500, "恢复失败")
        return {"restored": True, "restart_required": True}

    @app.post("/api/v1/jobs", dependencies=[auth], status_code=201)
    def submit(
        body: JobSubmission,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ):
        if body.endpoint_id not in settings.endpoints:
            raise HTTPException(422, "Unknown endpoint ID")
        if idempotency_key is not None and not _KEY.fullmatch(idempotency_key):
            raise HTTPException(422, "Invalid idempotency key")
        endpoint = settings.endpoints[body.endpoint_id]
        try:
            endpoint.api_key()
        except RuntimeError as exc:
            raise HTTPException(503, "Endpoint credential unavailable") from exc
        try:
            # run_config 以 `_run_config` 保留键并入 parameters_json 持久化,
            # worker 执行前由 runner_adapter 拆出透传（spec 校验不受影响）
            parameters = dict(body.parameters)
            if body.run_config:
                parameters["_run_config"] = body.run_config.model_dump(exclude_none=True)
            return store.submit(
                test_type=body.test_type,
                endpoint_id=endpoint.id,
                model_id=endpoint.model_id,
                parameters=parameters,
                progress_total=expected_requests(body.test_type, body.parameters),
                idempotency_key=idempotency_key,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/jobs", dependencies=[auth])
    def list_jobs(
        status: RunStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        items, total = store.list(status=status, limit=limit, offset=offset)
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def job_or_404(job_id: str) -> dict:
        try:
            return store.get(job_id)
        except JobNotFound as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.get("/api/v1/jobs/{job_id}", dependencies=[auth])
    def get_job(job_id: str):
        return job_or_404(job_id)

    @app.get("/api/v1/jobs/{job_id}/events", dependencies=[auth])
    def events(job_id: str, limit: Annotated[int, Query(ge=1, le=500)] = 100):
        job_or_404(job_id)
        return {"items": store.events(job_id, limit=limit)}

    @app.post("/api/v1/jobs/{job_id}/cancel", dependencies=[auth])
    def cancel(job_id: str):
        try:
            return store.request_cancel(job_id)
        except JobNotFound as exc:
            raise HTTPException(404, "Job not found") from exc
        except InvalidRunTransition as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/jobs/{job_id}/summary", dependencies=[auth])
    def summary(job_id: str):
        job = job_or_404(job_id)
        if job["result_run_id"] is None:
            raise HTTPException(409, "No persisted performance run for this job")
        return run_summary(str(settings.db_path), job["result_run_id"])

    @app.get("/api/v1/jobs/{job_id}/results", dependencies=[auth])
    def results(
        job_id: str,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
        since_id: Annotated[int, Query(ge=0)] = 0,
    ):
        job = job_or_404(job_id)
        if job["result_run_id"] is None:
            raise HTTPException(409, "No persisted performance run for this job")
        return run_results(
            str(settings.db_path),
            job["result_run_id"],
            limit=limit,
            offset=offset,
            since_id=since_id,
        )

    @app.get("/api/v1/jobs/{job_id}/export.csv", dependencies=[auth])
    def export_csv(job_id: str):
        job = job_or_404(job_id)
        if job["result_run_id"] is None:
            raise HTTPException(409, "No persisted performance run for this job")
        return Response(
            run_results_csv(str(settings.db_path), job["result_run_id"]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="llm-test-{job_id}.csv"'},
        )

    @app.get("/api/v1/jobs/{job_id}/report", dependencies=[auth])
    def report(job_id: str, format: str = "json"):
        job = job_or_404(job_id)
        if job["result_run_id"] is not None:
            data = run_summary(str(settings.db_path), job["result_run_id"])
            if format == "html":
                return HTMLResponse(render_html(job, data))
            if format == "json":
                return {"job": job, "summary": data}
            raise HTTPException(422, "Format must be json or html")
        if not job["result_artifact"]:
            raise HTTPException(409, "No report is available for this job")
        if format not in {"json", "html"}:
            raise HTTPException(422, "Format must be json or html")
        payload = quality_payload(job)
        if format == "html":
            return HTMLResponse(render_quality_html(job, payload))
        return JSONResponse(payload)

    def quality_payload(job: dict[str, Any]) -> dict[str, Any]:
        if job["result_run_id"] is not None or not job["result_artifact"]:
            raise HTTPException(409, "No quality report is available for this job")
        artifact = (settings.artifact_root / job["result_artifact"]).resolve()
        if not artifact.is_relative_to(settings.artifact_root.resolve()) or not artifact.is_file():
            raise HTTPException(404, "Report artifact missing")
        try:
            payload: object = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(422, "Quality report is invalid") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
            raise HTTPException(422, "Quality report is invalid")
        return cast(dict[str, Any], payload)

    @app.get("/api/v1/jobs/{job_id}/report/errors.csv", dependencies=[auth])
    def quality_errors(job_id: str):
        payload = quality_payload(job_or_404(job_id))
        try:
            csv_content = quality_errors_csv(payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return Response(
            csv_content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="llm-test-{job_id}-errors.csv"'},
        )

    frontend = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend.is_dir():
        assets = frontend / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            if path.startswith(("api/", "health/")):
                raise HTTPException(404, "Not found")
            return FileResponse(frontend / "index.html")

    return app
