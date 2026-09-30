"""Authenticated saved-file browsing, report regeneration and recoverable deletion."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import Field

from server.figures import performance_report_figure
from server.history import (
    MAX_CSV_BYTES,
    MAX_METADATA_BYTES,
    HistoryConflict,
    HistoryError,
    HistoryNotFound,
    SavedCsvHistory,
)
from server.reports import render_html, render_markdown
from server.scenario_reports import METRICS
from server.settings import Settings
from server.specs import StrictSpec


class DeleteCsvBody(StrictSpec):
    confirmed: bool
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")


def history_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/v1/history")
    history = SavedCsvHistory(settings.history_root or settings.artifact_root.parent / "history")

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except HistoryNotFound as exc:
            raise HTTPException(404, "历史 CSV 不存在，请刷新列表") from exc
        except HistoryConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except HistoryError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                503, "历史目录不可读写或文件正在变更，请检查目录权限后重试"
            ) from exc

    @router.get("")
    def catalog():
        return call(history.catalog)

    @router.post("/uploads", status_code=201)
    async def upload(file: UploadFile, metadata: UploadFile | None = None):
        try:
            raw = await file.read(MAX_CSV_BYTES + 1)
            meta_raw = await metadata.read(MAX_METADATA_BYTES + 1) if metadata else b""
            if len(raw) > MAX_CSV_BYTES or len(meta_raw) > MAX_METADATA_BYTES:
                raise HTTPException(413, "上传超过 CSV 10 MiB / 元数据 64 KiB 上限")
            return await asyncio.to_thread(call, history.upload, raw, file.filename or "", meta_raw)
        finally:
            await file.close()
            if metadata:
                await metadata.close()

    @router.get("/{identifier}")
    def load(
        identifier: str,
        revision: str | None = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ):
        return call(history.load, identifier, revision=revision, offset=offset, limit=limit)

    @router.get("/{identifier}/report")
    def report(
        identifier: str,
        format: Literal["json", "html", "markdown", "csv"] = "json",
        revision: str | None = None,
    ):
        filename = f"llm-test-history-{identifier[:12]}"
        if format == "csv":
            return Response(
                call(history.export_csv, identifier, revision),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
            )
        data = call(history.load, identifier, revision=revision)
        if format == "html":
            return HTMLResponse(render_html(data["job"], data["summary"]))
        if format == "markdown":
            return Response(
                render_markdown(data["job"], data["summary"]),
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}.md"'},
            )
        return {key: data[key] for key in ("entry", "job", "summary")}

    @router.get("/{identifier}/figure")
    def figure(
        identifier: str,
        revision: str | None = None,
        metric: str = Query("ttft", pattern="^(" + "|".join(METRICS) + ")$"),
        view: str = Query("comparison", pattern="^(comparison|profile|heatmap)$"),
        statistic: str = Query("median", pattern="^(median|mean|p95|p99|min|max)$"),
    ):
        data = call(history.load, identifier, revision=revision)
        try:
            return performance_report_figure(
                data["job"], data["summary"], metric, view=view, statistic=statistic
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/{identifier}/delete")
    def delete(identifier: str, body: DeleteCsvBody):
        if not body.confirmed:
            raise HTTPException(422, "必须明确确认移除所选 CSV 和元数据")
        return call(history.delete, identifier, body.revision)

    return router
