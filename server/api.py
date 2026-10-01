"""Authenticated, process-independent control and read API."""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import hashlib
import hmac
import io
import json
import re
import time
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import Field, model_validator

from core.providers.factory import get_provider
from core.run_lifecycle import InvalidRunTransition, RunStatus
from core.url_validator import SSRFError
from server.advanced_api import advanced_router
from server.analytics import (
    MetricContractConflict,
    latest_output,
    run_results,
    run_results_csv,
    run_summary,
)
from server.checkpoints import CheckpointConflict, checkpoint_info, delete_checkpoint, recover_job
from server.control import PAUSABLE_TEST_TYPES
from server.endpoints import (
    EndpointConflict,
    EndpointCredentialUnavailable,
    EndpointNotFound,
    EndpointRegistry,
)
from server.figures import (
    compare_figures,
    performance_report_figure,
    run_detail_figures,
    trend_figure,
)
from server.history import MAX_CSV_BYTES
from server.history_api import history_router
from server.model_discovery import ModelDiscoveryError, discover_models, measure_reference_latency
from server.preset_conversion import LegacyPresetInput, convert_legacy_preset
from server.quality_export import quality_errors_csv
from server.reports import (
    checkpoint_html,
    render_html,
    render_markdown,
    render_quality_html,
    render_quality_markdown,
    render_robustness_markdown,
    report_environment_html,
)
from server.robustness_figures import sensitivity_svg
from server.settings import Endpoint, Settings
from server.specs import (
    JobSubmission,
    PresetImport,
    PresetSubmission,
    QualitySpec,
    RunConfig,
    StrictSpec,
    expected_requests,
    measurement_plan,
    spec_catalog,
)
from server.store import (
    BatchNotFound,
    IdempotencyConflict,
    JobNotFound,
    JobStore,
    LeaseLost,
    PresetConflict,
    PresetNotFound,
)
from server.tokenizer_queue import InstallNotFound, TokenizerInstallQueue
from server.tokenizer_tools import count_text, tokenizer_catalog
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


class CompareQualityBody(StrictSpec):
    job_id_a: str = Field(min_length=1, max_length=64)
    job_id_b: str = Field(min_length=1, max_length=64)
    score_basis: Literal["standard", "final"] = "standard"


class QualityMatrixBody(StrictSpec):
    job_ids: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        min_length=2, max_length=8
    )
    score_basis: Literal["standard", "final"] = "final"


class OnlineComparisonBody(StrictSpec):
    endpoint_id_a: str = Field(min_length=1, max_length=64)
    endpoint_id_b: str = Field(min_length=1, max_length=64)
    parameters: QualitySpec
    run_config: RunConfig | None = None


class BatchItem(StrictSpec):
    enabled: bool = True
    test_type: Literal[
        "concurrency",
        "prefill",
        "segmented_prefill",
        "long_context",
        "matrix",
        "stability",
        "custom_text",
        "dataset",
        "quality",
        "robustness",
    ]
    parameters: dict[str, Any]
    run_config: RunConfig | None = None
    endpoint_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_parameters(self) -> BatchItem:
        if not self.enabled:
            try:
                encoded = json.dumps(self.parameters, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("Disabled item parameters must be finite JSON") from exc
            if len(encoded) > 100_000:
                raise ValueError("Disabled item parameters are too large")
            return self
        from server.specs import SPEC_MODELS, TypeAdapter

        parsed = TypeAdapter(SPEC_MODELS[self.test_type]).validate_python(self.parameters)
        self.parameters = parsed.model_dump()
        return self


class BatchSubmission(StrictSpec):
    name: str = Field(default="批量测量", min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    max_parallel: int = Field(default=1, ge=1, le=8)
    stop_on_error: bool = False
    endpoint_id: str = Field(min_length=1, max_length=64)
    items: list[BatchItem] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_enabled(self) -> BatchSubmission:
        self.name = self.name.strip()
        self.description = self.description.strip()
        if not self.name:
            raise ValueError("Batch name cannot be blank")
        if not any(item.enabled for item in self.items):
            raise ValueError("Enable at least one batch item")
        return self


class EndpointConfigBody(StrictSpec):
    label: str = Field(min_length=1, max_length=80)
    provider: Literal["OpenAI", "Gemini", "Anthropic"] = "OpenAI"
    api_base_url: str = Field(min_length=1, max_length=500)
    model_id: str = Field(min_length=1, max_length=200)
    tokenizer_option: str = Field(default="auto", min_length=1, max_length=120)
    api_key: str | None = Field(default=None, max_length=4096)


class TokenCountBody(StrictSpec):
    text: str = Field(min_length=1, max_length=10000)
    mode: Literal["local", "tiktoken", "characters"]
    name: str | None = Field(default=None, max_length=120)


class TokenizerInstallBody(StrictSpec):
    names: list[str] | None = Field(default=None, min_length=1, max_length=32)


class DatasetPreparationBody(StrictSpec):
    names: list[str] | None = Field(default=None, min_length=1, max_length=32)


def create_app(settings: Settings | None = None, store: JobStore | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or JobStore(settings.db_path)
    tokenizer_queue = TokenizerInstallQueue(store)
    endpoint_registry = EndpointRegistry(settings, store)
    warehouse_reader = WarehouseReader(settings.db_path)
    # 数据管理/用例/图表路径复用 core 的 manager（生产进程内单例同路径；
    # 测试通过重置 Database/DatabaseManager 单例隔离）
    from core.database.manager import DatabaseManager

    data_manager = DatabaseManager(str(settings.db_path))
    app = FastAPI(title="LLM Test Control API", version="1.0.0", docs_url=None, redoc_url=None)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_request: Request, exc: RequestValidationError):
        # FastAPI's default error includes submitted input, which may contain an API key.
        detail = [
            {name: error[name] for name in ("loc", "msg", "type") if name in error}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    def authenticate(authorization: Annotated[str | None, Header()] = None) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, settings.api_token):
            raise HTTPException(
                status_code=401,
                detail="Valid bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    auth = Depends(authenticate)
    app.include_router(history_router(settings), dependencies=[auth])
    app.include_router(advanced_router(), dependencies=[auth])

    def resolve_endpoint(endpoint_id: str) -> Endpoint:
        try:
            return endpoint_registry.get(endpoint_id)
        except EndpointNotFound as exc:
            raise HTTPException(422, "Unknown endpoint ID") from exc
        except EndpointCredentialUnavailable as exc:
            raise HTTPException(503, "Endpoint credential unavailable") from exc

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = None
        if request.method == "POST" and request.url.path == "/api/v1/history/uploads":
            try:
                authenticate(request.headers.get("authorization"))
            except HTTPException as exc:
                response = JSONResponse(
                    {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
                )
            if response is None:
                try:
                    length = int(request.headers.get("content-length", "-1"))
                except ValueError:
                    length = -1
                if length < 0 or request.headers.get("transfer-encoding"):
                    response = JSONResponse(
                        {"detail": "上传需提供有效 Content-Length"}, status_code=411
                    )
                elif length > MAX_CSV_BYTES + 1024 * 1024:
                    response = JSONResponse({"detail": "上传请求超过 11 MiB 上限"}, status_code=413)
        if response is None:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = (
            "no-store" if request.url.path.startswith("/api/") else "no-cache"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'"
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
        return {"items": endpoint_registry.list_public()}

    @app.post("/api/v1/endpoints", dependencies=[auth], status_code=201)
    def create_endpoint(body: EndpointConfigBody):
        try:
            return endpoint_registry.save(**body.model_dump())
        except (ValueError, SSRFError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.put("/api/v1/endpoints/{endpoint_id}", dependencies=[auth])
    def update_endpoint(endpoint_id: str, body: EndpointConfigBody):
        try:
            return endpoint_registry.save(endpoint_id=endpoint_id, **body.model_dump())
        except EndpointNotFound as exc:
            raise HTTPException(404, "Endpoint not found") from exc
        except EndpointConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, SSRFError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/v1/endpoints/{endpoint_id}", dependencies=[auth], status_code=204)
    def delete_endpoint(endpoint_id: str):
        try:
            endpoint_registry.delete(endpoint_id)
        except EndpointNotFound as exc:
            raise HTTPException(404, "Endpoint not found") from exc
        except EndpointConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204)

    @app.post("/api/v1/endpoints/{endpoint_id}/probe", dependencies=[auth])
    async def probe_endpoint(endpoint_id: str):
        """Issue one bounded, low-output request without returning generated content."""
        endpoint = resolve_endpoint(endpoint_id)
        try:
            api_key = endpoint.api_key()
        except RuntimeError as exc:
            raise HTTPException(503, "Endpoint credential unavailable") from exc
        try:
            provider = get_provider(
                endpoint.provider, endpoint.api_base_url, api_key, endpoint.model_id
            )
        except SSRFError as exc:
            raise HTTPException(422, "Endpoint URL is no longer trusted") from exc
        try:
            start = time.monotonic()
            result = await asyncio.wait_for(
                provider.get_completion(
                    client=None,
                    session_id=-1,
                    prompt="Reply with OK.",
                    max_tokens=8,
                    request_timeout=15,
                ),
                timeout=20,
            )
        except TimeoutError:
            return {"ok": False, "message": "连接超时，请检查网络与 API 地址"}
        except Exception:
            return {"ok": False, "message": "连接失败，请检查网络与 API 地址"}
        if result.get("error"):
            error = str(result["error"])
            status = re.search(r"HTTP\s+(\d{3})", error)
            code = int(status.group(1)) if status else None
            if code in {401, 403}:
                message = "认证失败，请检查 API key"
            elif code == 404:
                message = "端点或模型不存在，请检查地址与模型 ID"
            elif code == 429:
                message = "API 已限流，请稍后重试"
            else:
                message = "API 未返回有效内容，请检查模型能力与地址"
            return {"ok": False, "message": message, "status_code": code}
        return {
            "ok": True,
            "message": "连接成功",
            "latency_ms": round((time.monotonic() - start) * 1000),
        }

    @app.get("/api/v1/endpoints/{endpoint_id}/models", dependencies=[auth])
    async def endpoint_models(endpoint_id: str):
        endpoint = resolve_endpoint(endpoint_id)
        try:
            return await discover_models(endpoint)
        except SSRFError as exc:
            raise HTTPException(422, "Endpoint URL is no longer trusted") from exc
        except ModelDiscoveryError as exc:
            raise HTTPException(502, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, "Endpoint credential unavailable") from exc

    @app.post("/api/v1/endpoints/{endpoint_id}/reference-latency", dependencies=[auth])
    async def endpoint_reference_latency(endpoint_id: str):
        endpoint = resolve_endpoint(endpoint_id)
        try:
            return await measure_reference_latency(endpoint)
        except SSRFError as exc:
            raise HTTPException(422, "Endpoint URL is no longer trusted") from exc
        except ModelDiscoveryError as exc:
            raise HTTPException(502, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, "Endpoint credential unavailable") from exc

    @app.get("/api/v1/specs", dependencies=[auth])
    def specs():
        """9 个测试类型的展示名 + JSON Schema + 运行配置旋钮 schema（驱动前端表单）。"""
        from server.specs import RunConfig

        return {"items": spec_catalog(), "run_config_schema": RunConfig.model_json_schema()}

    @app.get("/api/v1/tokenizers", dependencies=[auth])
    def tokenizers(model_id: Annotated[str, Query(max_length=200)] = ""):
        catalog = tokenizer_catalog(model_id)
        latest = tokenizer_queue.latest()
        for item in catalog["items"]:
            item["installation"] = latest.get(item["name"])
        return catalog

    @app.post("/api/v1/tokenizers/installations", dependencies=[auth], status_code=202)
    def install_tokenizers(body: TokenizerInstallBody):
        names = body.names
        if names is None:
            names = [
                item["name"] for item in tokenizer_catalog("")["items"] if not item["available"]
            ]
        if not names:
            return {"items": [], "skipped_names": []}
        try:
            return tokenizer_queue.enqueue(names)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/tokenizers/installations/{install_id}", dependencies=[auth])
    def tokenizer_installation(install_id: str):
        try:
            return tokenizer_queue.get(install_id)
        except InstallNotFound as exc:
            raise HTTPException(404, "Installation not found") from exc

    @app.post("/api/v1/tokenizers/installations/{install_id}/cancel", dependencies=[auth])
    def cancel_tokenizer_installation(install_id: str):
        try:
            return tokenizer_queue.cancel(install_id)
        except InstallNotFound as exc:
            raise HTTPException(404, "Installation not found") from exc

    @app.post("/api/v1/tokenizers/count", dependencies=[auth])
    async def tokenizer_count(body: TokenCountBody):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(count_text, body.text, body.mode, body.name), timeout=10
            )
        except TimeoutError as exc:
            raise HTTPException(504, "Tokenizer count timed out") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/environment", dependencies=[auth])
    def environment():
        """执行环境快照（硬件指纹 / 系统 / 版本，env_overview 的 API 版）。"""
        from core.system_info import get_cached_system_info

        return get_cached_system_info(wait=False) or {}

    @app.get("/api/v1/quality/datasets", dependencies=[auth])
    def quality_dataset_catalog():
        from core.dataset_manager import get_manager
        from server.dataset_preparation import downloadable_names
        from server.quality_catalog import quality_catalog

        manager = get_manager()
        allowed = downloadable_names()
        with store._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM control_jobs WHERE test_type='dataset_prepare' ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
        latest: dict[str, dict | None] = {}
        for row in rows:
            latest.setdefault(row["model_id"], store._as_job(row))
        items = quality_catalog()
        for item in items:
            name = item["id"]
            files = (
                list(manager.get_local_path(name).glob("*.json")) if name in manager.configs else []
            )
            item.update(
                local_available=manager.is_available(name) if name in manager.configs else None,
                bytes=sum(path.stat().st_size for path in files if path.is_file()),
                downloadable=name in allowed,
                preparation=latest.get(name),
            )
            preparation = item["preparation"]
            if preparation:
                events = store.events(preparation["job_id"], limit=5)
                stage = next(
                    (
                        json.loads(event["detail_json"])
                        for event in reversed(events)
                        if event["event"] == "DATASET_PREPARATION" and event["detail_json"]
                    ),
                    None,
                )
                item["preparation_stage"] = stage
        return {"items": items}

    @app.post("/api/v1/quality/datasets/preparations", dependencies=[auth], status_code=202)
    def prepare_quality_datasets(body: DatasetPreparationBody):
        from core.dataset_manager import get_manager
        from server.dataset_preparation import downloadable_names

        allowed = downloadable_names()
        names = sorted(allowed) if body.names is None else body.names
        if len(set(names)) != len(names) or any(name not in allowed for name in names):
            raise HTTPException(422, "Select unique registered public datasets")
        manager = get_manager()
        skipped = [name for name in names if manager.is_available(name)]
        tasks = [
            store.submit(
                test_type="dataset_prepare",
                endpoint_id="dataset-manager",
                model_id=name,
                parameters={"name": name},
                progress_total=1,
            )
            for name in names
            if name not in skipped
        ]
        return {"items": tasks, "skipped_names": skipped}

    @app.get("/api/v1/datasets", dependencies=[auth])
    def datasets():
        """可用数据集目录：质量评估注册表 + dataset_loader 自定义集。"""
        from core.dataset_loader import DatasetLoader
        from core.dataset_manager import list_available_datasets

        try:
            builtin = list_available_datasets()
        except Exception:  # noqa: BLE001  注册表读取失败不阻断自定义集
            builtin = []
        try:
            custom = DatasetLoader().list_datasets()
        except Exception:  # noqa: BLE001
            custom = []
        return {"builtin": builtin, "custom": custom}

    @app.get("/api/v1/presets/templates", dependencies=[auth])
    def preset_templates():
        from server.preset_templates import builtin_templates

        return {"items": builtin_templates()}

    @app.get("/api/v1/presets", dependencies=[auth])
    def presets():
        return {"items": store.list_presets()}

    def validate_preset_endpoint(body: PresetSubmission) -> None:
        resolve_endpoint(body.endpoint_id)

    @app.post("/api/v1/presets", dependencies=[auth], status_code=201)
    def create_preset(body: PresetSubmission):
        validate_preset_endpoint(body)
        try:
            return store.save_preset(
                name=body.name,
                description=body.description,
                tags=body.tags,
                source_metadata=body.source_metadata.model_dump() if body.source_metadata else None,
                endpoint_id=body.endpoint_id,
                test_type=body.test_type,
                parameters=body.parameters,
                run_config=body.run_config.model_dump(exclude_none=True)
                if body.run_config
                else None,
            )
        except PresetConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/v1/presets/convert", dependencies=[auth])
    def convert_preset(body: LegacyPresetInput):
        try:
            return convert_legacy_preset(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/presets/import", dependencies=[auth], status_code=201)
    def import_preset(body: PresetImport):
        return create_preset(body.preset)

    @app.get("/api/v1/presets/{preset_id}/export", dependencies=[auth])
    def export_preset(preset_id: str):
        try:
            saved = store.get_preset(preset_id)
        except PresetNotFound as exc:
            raise HTTPException(404, "Preset not found") from exc
        fields = (
            "name",
            "description",
            "tags",
            "endpoint_id",
            "test_type",
            "parameters",
            "run_config",
            "source_metadata",
        )
        try:
            preset = PresetSubmission.model_validate({key: saved[key] for key in fields})
        except ValueError as exc:
            raise HTTPException(
                422, "Saved preset is incompatible with the current schema"
            ) from exc
        return {
            "format": "llm-test-preset",
            "version": 1,
            "preset": preset.model_dump(mode="json", exclude_none=True),
        }

    @app.put("/api/v1/presets/{preset_id}", dependencies=[auth])
    def update_preset(preset_id: str, body: PresetSubmission):
        validate_preset_endpoint(body)
        try:
            return store.save_preset(
                preset_id=preset_id,
                name=body.name,
                description=body.description,
                tags=body.tags,
                source_metadata=body.source_metadata.model_dump() if body.source_metadata else None,
                endpoint_id=body.endpoint_id,
                test_type=body.test_type,
                parameters=body.parameters,
                run_config=body.run_config.model_dump(exclude_none=True)
                if body.run_config
                else None,
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
    # A/B 对比与高级评估（离线分析）
    # ------------------------------------------------------------------

    @app.post("/api/v1/comparisons", dependencies=[auth], status_code=201)
    def submit_comparison(
        body: OnlineComparisonBody,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ):
        import uuid

        from server.shared_quality import PROTOCOL

        if body.endpoint_id_a == body.endpoint_id_b:
            raise HTTPException(422, "Choose two independently configured model endpoints")
        if idempotency_key is not None and (
            not _KEY.fullmatch(idempotency_key) or len(idempotency_key) > 64
        ):
            raise HTTPException(422, "Invalid idempotency key")
        endpoints = [resolve_endpoint(body.endpoint_id_a), resolve_endpoint(body.endpoint_id_b)]
        for endpoint in endpoints:
            try:
                endpoint.api_key()
            except RuntimeError as exc:
                raise HTTPException(503, "Endpoint credential unavailable") from exc
        batch_id = idempotency_key or uuid.uuid4().hex[:16]
        request_hash = hashlib.sha256(
            json.dumps(
                body.model_dump(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        prepared = []
        for role, endpoint in zip(("A", "B"), endpoints, strict=True):
            parameters = body.parameters.model_dump()
            parameters["_comparison"] = {
                "protocol": PROTOCOL,
                "role": role,
                "endpoint_snapshot": {
                    field: getattr(endpoint, field)
                    for field in ("model_id", "provider", "api_base_url", "tokenizer_option")
                },
            }
            if body.run_config:
                parameters["_run_config"] = body.run_config.model_dump(exclude_none=True)
            prepared.append(
                {
                    "test_type": "quality",
                    "endpoint_id": endpoint.id,
                    "model_id": endpoint.model_id,
                    "parameters": parameters,
                    "progress_total": expected_requests("quality", body.parameters.model_dump()),
                }
            )
        try:
            jobs = store.submit_batch(
                batch_id=batch_id,
                name="在线双模型对比",
                description="A/B reuse a shared frozen sample and few-shot plan",
                default_endpoint_id=body.endpoint_id_a,
                request_hash=request_hash,
                requested_items=2,
                items=prepared,
                max_parallel=1,
                stop_on_error=False,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"batch_id": batch_id, "protocol": PROTOCOL, "jobs": jobs}

    @app.post("/api/v1/compare", dependencies=[auth])
    def compare_quality_jobs(body: CompareQualityBody):
        """两个已完成质量作业的 McNemar 显著性对比（按 sample_id 对齐逐样本）。"""
        if body.job_id_a == body.job_id_b:
            raise HTTPException(422, "两个作业不能相同")
        job_a = job_or_404(body.job_id_a)
        job_b = job_or_404(body.job_id_b)
        if job_a["status"] != "completed" or job_b["status"] != "completed":
            raise HTTPException(409, "对比需要两个已完成质量作业")
        payload_a = quality_payload(job_a)
        payload_b = quality_payload(job_b)

        from server.paired_quality import PairingConflict, adjust_family, compare_dataset

        names_a = set(payload_a["datasets"])
        names_b = set(payload_b["datasets"])
        common = sorted(names_a & names_b)
        if not common:
            raise HTTPException(422, "两个作业没有共同数据集，无法对比")

        datasets: dict[str, Any] = {}
        for name in common:
            da = payload_a["datasets"][name]
            db_ = payload_b["datasets"][name]
            try:
                datasets[name] = compare_dataset(da, db_, body.score_basis, name)
            except PairingConflict as exc:
                raise HTTPException(409, f"{name}: {exc}") from exc
        if not datasets:
            raise HTTPException(422, "共同数据集的样本无法按 sample_id 对齐")
        adjust_family(datasets)

        return {
            "job_a": {
                "job_id": job_a["job_id"],
                "model_id": job_a["model_id"],
                "endpoint_id": job_a["endpoint_id"],
            },
            "job_b": {
                "job_id": job_b["job_id"],
                "model_id": job_b["model_id"],
                "endpoint_id": job_b["endpoint_id"],
            },
            "datasets": datasets,
            "skipped_datasets": sorted((names_a | names_b) - set(common)),
        }

    @app.post("/api/v1/compare/matrix", dependencies=[auth])
    def compare_quality_matrix(body: QualityMatrixBody):
        from server.quality_matrix import quality_matrix

        if len(set(body.job_ids)) != len(body.job_ids):
            raise HTTPException(422, "不能重复选择同一作业")
        entries = []
        for identity in body.job_ids:
            job = job_or_404(identity)
            if job["test_type"] != "quality" or job["status"] != "completed":
                raise HTTPException(409, "多模型对照需要已完成质量作业")
            entries.append((job, quality_payload(job)))
        try:
            return quality_matrix(entries, body.score_basis)
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise HTTPException(422, "Quality comparison observations are invalid") from exc

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
            "tokenizer_installs",
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

    @app.post("/api/v1/jobs/plan", dependencies=[auth])
    def preview_measurement(body: JobSubmission):
        """Validate a workload and show its per-condition sample and warmup budget."""
        resolve_endpoint(body.endpoint_id)
        return measurement_plan(body.test_type, body.parameters)

    @app.post("/api/v1/jobs", dependencies=[auth], status_code=201)
    def submit(
        body: JobSubmission,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ):
        endpoint = resolve_endpoint(body.endpoint_id)
        if idempotency_key is not None and not _KEY.fullmatch(idempotency_key):
            raise HTTPException(422, "Invalid idempotency key")
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
        recoverable: bool = False,
        saved_progress: bool = False,
        parent_job_id: Annotated[str | None, Query(max_length=64)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        items, total = store.list(
            status=status,
            recoverable=recoverable,
            saved_progress=saved_progress,
            limit=limit,
            offset=offset,
            parent_job_id=parent_job_id,
        )
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.post("/api/v1/jobs/batch", dependencies=[auth], status_code=201)
    def submit_batch(
        body: BatchSubmission,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ):
        """Validate targets, then atomically submit enabled children and metadata."""
        if idempotency_key is not None and (
            not _KEY.fullmatch(idempotency_key) or len(idempotency_key) > 64
        ):
            raise HTTPException(422, "Invalid idempotency key")
        # Resolve every target before writing any child so a bad later item
        # cannot leave a partially submitted batch behind.
        enabled = [item for item in body.items if item.enabled]
        targets = [resolve_endpoint(item.endpoint_id or body.endpoint_id) for item in enabled]
        for endpoint in targets:
            try:
                endpoint.api_key()
            except RuntimeError as exc:
                raise HTTPException(503, "Endpoint credential unavailable") from exc

        import uuid as _uuid

        batch_id = idempotency_key or _uuid.uuid4().hex[:16]
        request_json = json.dumps(
            body.model_dump(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        prepared = []
        for item, endpoint in zip(enabled, targets, strict=True):
            parameters = dict(item.parameters)
            if item.run_config:
                parameters["_run_config"] = item.run_config.model_dump(exclude_none=True)
            prepared.append(
                {
                    "test_type": item.test_type,
                    "endpoint_id": endpoint.id,
                    "model_id": endpoint.model_id,
                    "parameters": parameters,
                    "progress_total": expected_requests(item.test_type, item.parameters),
                }
            )
        try:
            jobs = store.submit_batch(
                batch_id=batch_id,
                name=body.name,
                description=body.description,
                default_endpoint_id=body.endpoint_id,
                request_hash=request_hash,
                requested_items=len(body.items),
                items=prepared,
                max_parallel=body.max_parallel,
                stop_on_error=body.stop_on_error,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"batch_id": batch_id, "items": jobs}

    @app.get("/api/v1/jobs/batch/{batch_id}", dependencies=[auth])
    def get_batch(batch_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", batch_id):
            raise HTTPException(422, "Invalid batch ID")
        try:
            return store.get_batch(batch_id)
        except BatchNotFound as exc:
            raise HTTPException(404, "Batch not found") from exc

    @app.get("/api/v1/batches", dependencies=[auth])
    def list_batches(limit: Annotated[int, Query(ge=1, le=100)] = 50):
        return {"items": store.list_batches(limit=limit)}

    @app.post("/api/v1/jobs/batch/{batch_id}/cancel", dependencies=[auth])
    def cancel_batch(batch_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", batch_id):
            raise HTTPException(422, "Invalid batch ID")
        items, _ = store.list(parent_job_id=batch_id, limit=200)
        if not items:
            raise HTTPException(404, "Batch not found")
        cancelled = []
        for item in items:
            if item["status"] in {RunStatus.COMPLETED.value, RunStatus.FAILED.value}:
                continue
            try:
                cancelled.append(store.request_cancel(item["job_id"], actor="api:batch"))
            except InvalidRunTransition:
                # A worker may have completed the child between listing and cancellation.
                continue
        return {"batch_id": batch_id, "requested": len(cancelled), "items": cancelled}

    @app.get("/api/v1/jobs/{job_id}/logs", dependencies=[auth])
    def job_logs(
        job_id: str,
        since_id: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ):
        """作业执行日志（worker 落盘的 logs.jsonl，按行号游标增量读取）。"""
        job_or_404(job_id)
        path = settings.artifact_root / job_id / "logs.jsonl"
        items: list[dict[str, Any]] = []
        if path.is_file():
            try:
                with path.open(encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle, start=1):
                        if line_no <= since_id:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(entry, dict):
                            continue
                        entry["level"] = str(entry.get("level", "INFO")).removeprefix("LogLevel.")
                        entry["id"] = line_no
                        items.append(entry)
                        if len(items) >= limit:
                            break
            except OSError:
                pass
        return {"items": items, "since_id": since_id}

    @app.get("/api/v1/jobs/{job_id}/logs/export", dependencies=[auth])
    def export_job_logs(job_id: str, format: Literal["json", "txt", "csv"] = "json"):
        """Export the complete bounded job log, independent of the UI polling cursor."""
        job_or_404(job_id)
        path = settings.artifact_root / job_id / "logs.jsonl"
        if not path.is_file():
            raise HTTPException(404, "No execution log for this job")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise HTTPException(413, "Log exceeds the 20 MiB export limit")
        rows: list[dict[str, Any]] = []
        bytes_read = 0
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                bytes_read += len(line.encode("utf-8"))
                if bytes_read > 20 * 1024 * 1024:
                    raise HTTPException(413, "Log exceeds the 20 MiB export limit")
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                entry["level"] = str(entry.get("level", "INFO")).removeprefix("LogLevel.")
                entry["id"] = line_no
                rows.append(entry)
        if format == "json":
            return Response(
                json.dumps(rows, ensure_ascii=False, indent=2),
                media_type="application/json; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="llm-test-{job_id}-logs.json"'
                },
            )
        if format == "txt":
            content = "\n".join(
                f"{row.get('timestamp') or '—'} [{row['level']}] {row.get('message', '')}"
                + (
                    f" {json.dumps(row['metrics'], ensure_ascii=False)}"
                    if row.get("metrics")
                    else ""
                )
                for row in rows
            )
            return Response(
                content + ("\n" if content else ""),
                media_type="text/plain; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="llm-test-{job_id}-logs.txt"'
                },
            )
        output = io.StringIO()
        writer = csv.writer(output)
        columns = (
            "id",
            "timestamp",
            "level",
            "session_id",
            "test_type",
            "message",
            "metrics",
            "error",
        )
        writer.writerow(columns)
        for row in rows:
            cells = [
                json.dumps(row[key], ensure_ascii=False)
                if key == "metrics" and row.get(key)
                else str(row.get(key) or "")
                for key in columns
            ]
            writer.writerow(
                ["'" + cell if cell.startswith(("=", "+", "-", "@")) else cell for cell in cells]
            )
        return Response(
            "\ufeff" + output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="llm-test-{job_id}-logs.csv"'},
        )

    def job_or_404(job_id: str) -> dict:
        try:
            return store.get(job_id)
        except JobNotFound as exc:
            raise HTTPException(404, "Job not found") from exc

    def warmup_artifact(job):
        path = settings.artifact_root / job["job_id"] / f"attempt-{job['attempts']}" / "warmup.csv"
        return path if path.is_file() else settings.artifact_root / job["job_id"] / "warmup.csv"

    @app.get("/api/v1/jobs/{job_id}/warmup.csv", dependencies=[auth])
    def warmup_csv(job_id: str):
        """Download warmup observations, which are excluded from result statistics."""
        path = warmup_artifact(job_or_404(job_id))
        if not path.is_file():
            raise HTTPException(404, "Warmup observations not available")
        return FileResponse(path, media_type="text/csv", filename=f"{job_id}-warmup.csv")

    def performance_summary(job: dict) -> dict:
        try:
            return run_summary(
                str(settings.db_path),
                job["result_run_id"],
                job=job,
                warmup_path=warmup_artifact(job),
            )
        except MetricContractConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/jobs/{job_id}/figure", dependencies=[auth])
    def job_figure(
        job_id: str,
        metric: str = Query(
            "ttft",
            pattern="^(ttft|tpot|tps|prefill_speed|total_time|system_input_wall|system_output_wall|system_total_wall|system_qpm|phase_input|phase_input_uncached_api|phase_input_uncached_inferred|phase_output|phase_total|phase_qpm|cache_tokens_api|cache_rate_api|cache_tokens_inferred|cache_rate_inferred|ttft_zero_cache_api|ttft_cache_api)$",
        ),
        view: str = Query("comparison", pattern="^(comparison|profile|heatmap|timeline)$"),
        statistic: str = Query("median", pattern="^(median|mean|p95|p99|min|max)$"),
        attempt: Annotated[int | None, Query(ge=1)] = None,
    ):
        job = job_or_404(job_id)
        if job["result_run_id"] is None:
            raise HTTPException(409, "尚无性能观测可供绘图")
        summary = performance_summary(job)
        try:
            return performance_report_figure(
                job, summary, metric, view=view, statistic=statistic, attempt=attempt
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

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

    @app.post("/api/v1/jobs/{job_id}/pause", dependencies=[auth])
    def pause(job_id: str):
        job = job_or_404(job_id)
        if job["test_type"] not in PAUSABLE_TEST_TYPES:
            raise HTTPException(409, "此测试按连续时间或评测样本执行，暂不支持暂停")
        try:
            return store.request_pause(job_id)
        except (InvalidRunTransition, LeaseLost) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/v1/jobs/{job_id}/resume", dependencies=[auth])
    def resume(job_id: str):
        job_or_404(job_id)
        try:
            return store.resume(job_id)
        except (InvalidRunTransition, LeaseLost) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/jobs/{job_id}/checkpoint", dependencies=[auth])
    def evaluation_checkpoint(job_id: str):
        info = checkpoint_info(store, job_or_404(job_id))
        headers = {"ETag": f'"{info["revision"]}"'} if info["revision"] else {}
        return JSONResponse(info, headers=headers)

    @app.delete("/api/v1/jobs/{job_id}/checkpoint", dependencies=[auth])
    def remove_saved_progress(
        job_id: str, if_match: Annotated[str | None, Header(alias="If-Match", max_length=68)] = None
    ):
        job = job_or_404(job_id)
        if if_match is None:
            raise HTTPException(428, "删除保存进度前请先读取并提供版本")
        try:
            return delete_checkpoint(store, job, if_match.strip('"'))
        except CheckpointConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/v1/jobs/{job_id}/recover", dependencies=[auth])
    def recover_evaluation(job_id: str):
        job = job_or_404(job_id)
        try:
            endpoint = endpoint_registry.get(job["endpoint_id"])
            return recover_job(store, job, endpoint)
        except (CheckpointConflict, LeaseLost, InvalidRunTransition) as exc:
            raise HTTPException(409, str(exc)) from exc
        except (EndpointNotFound, EndpointCredentialUnavailable) as exc:
            raise HTTPException(409, "恢复所需端点不可用") from exc

    @app.get("/api/v1/jobs/{job_id}/summary", dependencies=[auth])
    def summary(job_id: str):
        job = job_or_404(job_id)
        if job["result_run_id"] is None:
            raise HTTPException(409, "No persisted performance run for this job")
        return performance_summary(job)

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

    @app.get("/api/v1/jobs/{job_id}/latest-output", dependencies=[auth])
    def latest_job_output(job_id: str):
        job = job_or_404(job_id)
        if job["result_run_id"] is None:
            raise HTTPException(409, "No persisted performance run for this job")
        return latest_output(str(settings.db_path), job["result_run_id"])

    @app.get("/api/v1/jobs/{job_id}/export.csv", dependencies=[auth])
    def export_csv(job_id: str):
        job = job_or_404(job_id)
        if job["result_run_id"] is None:
            raise HTTPException(409, "No persisted performance run for this job")
        try:
            content = run_results_csv(str(settings.db_path), job["result_run_id"])
        except MetricContractConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(
            content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="llm-test-{job_id}.csv"'},
        )

    @app.get("/api/v1/jobs/{job_id}/report", dependencies=[auth])
    def report(job_id: str, format: str = "json"):
        job = job_or_404(job_id)
        if job["result_run_id"] is not None:
            data = performance_summary(job)
            if format == "html":
                return HTMLResponse(render_html(job, data))
            if format == "markdown":
                return Response(
                    render_markdown(job, data),
                    media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="llm-test-{job_id}.md"'},
                )
            if format == "json":
                return {"job": job, "summary": data}
            raise HTTPException(422, "Format must be json, html or markdown")
        if not job["result_artifact"]:
            raise HTTPException(409, "No report is available for this job")
        if format not in {"json", "html", "markdown"}:
            raise HTTPException(422, "Format must be json, html or markdown")
        payload = artifact_payload(job)
        if isinstance(payload.get("dataset_preparation"), dict):
            if format != "json":
                raise HTTPException(422, "Dataset preparation receipts use JSON")
            return JSONResponse(payload)
        if isinstance(payload.get("datasets"), dict):
            from server.quality_analysis import quality_analysis

            try:
                payload["analysis"] = quality_analysis(payload)
            except (ValueError, TypeError, AttributeError) as exc:
                raise HTTPException(422, "Quality report observations are invalid") from exc
            if format == "html":
                return HTMLResponse(render_quality_html(job, payload))
            if format == "markdown":
                return Response(
                    render_quality_markdown(job, payload),
                    media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="llm-test-{job_id}.md"'},
                )
            return JSONResponse(payload)
        if isinstance(payload.get("robustness"), dict):
            if format == "html":
                return HTMLResponse(_render_robustness_html(job, payload))
            if format == "markdown":
                return Response(
                    render_robustness_markdown(job, payload),
                    media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="llm-test-{job_id}.md"'},
                )
            return JSONResponse(payload)
        raise HTTPException(422, "Report artifact is invalid")

    def _render_robustness_html(job: dict[str, Any], payload: dict[str, Any]) -> str:
        """鲁棒性报告的极简 HTML（分数卡片 + 扰动类型敏感性 + 逐样本表）。"""
        import html as _html

        rob = payload["robustness"]
        esc = _html.escape

        def percent(value):
            return "未评分" if value is None else f"{value:.1%}"

        def score(value):
            return "未评分" if value is None else f"{value:.2f}"

        rows = "".join(
            f"<tr><td>{esc(name)}</td><td>{value:.2f}</td></tr>"
            for name, value in (rob.get("sensitivity_by_type") or {}).items()
        )
        sample_rows = "".join(
            "<tr>"
            f"<td>{esc(str(r.get('sample_id', '')))}</td>"
            f"<td>{'未评分' if r.get('original_correct') is None else ('✓' if r.get('original_correct') else '✗')}</td>"
            f"<td>{score(r.get('robustness_score'))}</td>"
            f"<td>{r.get('consistency_score', 0):.2f}</td>"
            "</tr>"
            for r in (rob.get("results") or [])
        )
        return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Robustness · {esc(job["job_id"][:12])}</title>
<style>body{{font-family:system-ui;margin:32px;color:#1c2b33}}table{{border-collapse:collapse;margin:12px 0}}td,th{{border:1px solid #d7e0e5;padding:6px 14px;font-size:13px}}.card{{display:inline-block;border:1px solid #d7e0e5;border-radius:10px;padding:12px 20px;margin-right:12px}}.card strong{{font-size:22px}}</style>
</head><body>
<h1>鲁棒性报告 · {esc(job["model_id"])}</h1>
{checkpoint_html(payload)}
{report_environment_html(payload.get("report_environment"))}
<p>作业 <code>{esc(job["job_id"])}</code> · {rob.get("total_samples", 0)} 样本 × {rob.get("perturbations_per_sample", 0)} 扰动</p>
<div class="card">原始准确率<br><strong>{percent(rob.get("original_accuracy"))}</strong></div>
<div class="card">扰动后准确率<br><strong>{percent(rob.get("perturbed_accuracy"))}</strong></div>
<div class="card">准确率落差<br><strong>{percent(rob.get("accuracy_drop"))}</strong></div>
<div class="card">鲁棒性<br><strong>{percent(rob.get("overall_robustness"))}</strong></div>
<div class="card">一致性<br><strong>{rob.get("overall_consistency", 0):.1%}</strong></div>
<p>指标契约：{esc(str(rob.get("metric_contract_version") or "未记录，可能使用旧公式；对比前请重新评测"))}</p>
<p>评分样本：{rob.get("scored_samples", "未记录")} · 未评分：{rob.get("unscored_samples", "未记录")} · 评分扰动：{rob.get("scored_perturbations", "未记录")} · 文本未改变：{rob.get("unchanged_perturbations", "未记录")}</p>
<p>缺少标准答案不计入准确率；一致性包含全部完成样本。未改变文本的扰动保留在统计中，不能证明抗扰动能力；同一样本的扰动并非独立观测。</p>
<h2>按扰动类型错误率（越高越敏感）</h2>
{sensitivity_svg(rob.get("sensitivity_by_type") or {})}
<table><tr><th>扰动类型</th><th>错误率</th></tr>{rows}</table>
<h2>逐样本</h2>
<table><tr><th>样本</th><th>原始正确</th><th>鲁棒性</th><th>一致性</th></tr>{sample_rows}</table>
</body></html>"""

    def artifact_payload(job: dict[str, Any]) -> dict[str, Any]:
        if job["result_run_id"] is not None or not job["result_artifact"]:
            raise HTTPException(409, "No report is available for this job")
        artifact = (settings.artifact_root / job["result_artifact"]).resolve()
        if not artifact.is_relative_to(settings.artifact_root.resolve()) or not artifact.is_file():
            raise HTTPException(404, "Report artifact missing")
        try:
            payload: object = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(422, "Report artifact is invalid") from exc
        if not isinstance(payload, dict):
            raise HTTPException(422, "Report artifact is invalid")
        return cast(dict[str, Any], payload)

    def quality_payload(job: dict[str, Any]) -> dict[str, Any]:
        payload = artifact_payload(job)
        if not isinstance(payload.get("datasets"), dict):
            raise HTTPException(422, "Quality report is invalid")
        return payload

    @app.get("/api/v1/jobs/{job_id}/report/sample-visuals", dependencies=[auth])
    def quality_sample_visuals(
        job_id: str,
        dataset: Annotated[str, Query(max_length=120)],
        index: Annotated[int, Query(ge=0, le=100000)],
        format: Literal["json", "html"] = "json",
    ):
        from server.sample_visuals import sample_visuals, sample_visuals_html

        payload = quality_payload(job_or_404(job_id))
        result = payload["datasets"].get(dataset)
        if not isinstance(result, dict) or not isinstance(result.get("details"), list):
            raise HTTPException(404, "Sample dataset is unavailable")
        rows = result["details"]
        if index >= len(rows):
            raise HTTPException(404, "Sample index is unavailable")
        if not isinstance(rows[index], dict):
            raise HTTPException(422, "Sample record is invalid")
        visuals = {**sample_visuals(rows[index]), "dataset": dataset, "index": index}
        return HTMLResponse(sample_visuals_html(visuals)) if format == "html" else visuals

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

    @app.get("/api/v1/jobs/{job_id}/report/summary.csv", dependencies=[auth])
    def quality_summary_export(job_id: str):
        from server.quality_export import quality_summary_csv

        try:
            content = quality_summary_csv(quality_payload(job_or_404(job_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(422, "Quality report observations are invalid") from exc
        return Response(
            content,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="llm-test-{job_id}-summary.csv"'
            },
        )

    @app.get("/api/v1/jobs/{job_id}/report/samples.csv", dependencies=[auth])
    def quality_samples_export(job_id: str):
        from server.quality_export import quality_samples_csv

        try:
            content = quality_samples_csv(quality_payload(job_or_404(job_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(422, "Quality report observations are invalid") from exc
        return Response(
            content,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="llm-test-{job_id}-samples.csv"'
            },
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
