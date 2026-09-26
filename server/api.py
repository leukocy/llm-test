"""Authenticated, process-independent control and read API."""

from __future__ import annotations

import hmac
import json
import re
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from core.run_lifecycle import InvalidRunTransition, RunStatus
from server.analytics import run_results, run_results_csv, run_summary
from server.reports import render_html, render_quality_html
from server.settings import Settings
from server.specs import JobSubmission, expected_requests
from server.store import IdempotencyConflict, JobNotFound, JobStore

_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def create_app(settings: Settings | None = None, store: JobStore | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or JobStore(settings.db_path)
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
            return store.submit(
                test_type=body.test_type,
                endpoint_id=endpoint.id,
                model_id=endpoint.model_id,
                parameters=body.parameters,
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
    ):
        job = job_or_404(job_id)
        if job["result_run_id"] is None:
            raise HTTPException(409, "No persisted performance run for this job")
        return run_results(str(settings.db_path), job["result_run_id"], limit=limit, offset=offset)

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
        artifact = (settings.artifact_root / job["result_artifact"]).resolve()
        if not artifact.is_relative_to(settings.artifact_root.resolve()) or not artifact.is_file():
            raise HTTPException(404, "Report artifact missing")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if format == "html":
            return HTMLResponse(render_quality_html(job, payload))
        return JSONResponse(payload)

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
