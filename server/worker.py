"""One-job-at-a-time worker; run separately from the HTTP process."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import threading
import uuid

from core.cancel_state import request_stop, reset_all
from core.run_lifecycle import RunStatus
from server.runner_adapter import execute_job
from server.settings import Settings
from server.store import JobStore, LeaseLost

logger = logging.getLogger(__name__)


def _preserve_partial_run(store: JobStore, job_id: str, worker_id: str) -> None:
    try:
        store.sync_result_run(job_id, worker_id)
    except Exception:
        logger.exception("Could not link partial measurement to job: %s", job_id)


def _monitor(store: JobStore, job_id: str, worker_id: str, stop: threading.Event) -> None:
    while not stop.wait(2.0):
        if not store.heartbeat(job_id, worker_id):
            request_stop()
            return
        store.sync_result_run(job_id, worker_id)
        state = store.get(job_id)["status"]
        if state == RunStatus.CANCELLING.value:
            request_stop()


async def run_claimed_job(job: dict, settings: Settings, store: JobStore, worker_id: str) -> None:
    reset_all()
    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor, args=(store, job["job_id"], worker_id, stop), daemon=True
    )
    monitor.start()
    try:
        endpoint = settings.endpoints[job["endpoint_id"]]
        output = await execute_job(job, endpoint, settings, store, worker_id)
        current = store.get(job["job_id"])["status"]
        outcome = (
            RunStatus.CANCELLED if current == RunStatus.CANCELLING.value else RunStatus.COMPLETED
        )
        if outcome == RunStatus.COMPLETED and output.completed != output.total:
            raise RuntimeError("Measurement request count differs from the scheduled count")
        if outcome == RunStatus.COMPLETED and job["test_type"] not in {"quality", "robustness"}:
            if output.result_run_id is None:
                raise RuntimeError("Measurement run was not persisted")
            planned = store.get(job["job_id"])["progress_total"]
            if planned > 0 and planned != output.completed:
                raise RuntimeError("Measurement request count differs from the submitted plan")
            store.verify_persisted_run(job["job_id"], output.result_run_id, output.completed)
        store.update_progress(
            job["job_id"], worker_id, completed=output.completed, total=output.total
        )
        store.finish(
            job["job_id"],
            worker_id,
            outcome=outcome,
            result_run_id=output.result_run_id,
            result_artifact=output.result_artifact,
        )
    except asyncio.CancelledError:
        logger.info("Job execution interrupted: %s", job["job_id"])
        try:
            _preserve_partial_run(store, job["job_id"], worker_id)
            cancelling = store.get(job["job_id"])["status"] == RunStatus.CANCELLING.value
            store.finish(
                job["job_id"],
                worker_id,
                outcome=RunStatus.CANCELLED if cancelling else RunStatus.FAILED,
                error_code=None if cancelling else "INTERRUPTED",
                error_message=None if cancelling else "Measurement was interrupted",
            )
        except LeaseLost:
            logger.warning("Job lease lost after interruption: %s", job["job_id"])
    except LeaseLost:
        logger.warning("Job lease lost: %s", job["job_id"])
    except Exception:
        logger.exception("Job execution failed: %s", job["job_id"])
        try:
            _preserve_partial_run(store, job["job_id"], worker_id)
            cancelling = store.get(job["job_id"])["status"] == RunStatus.CANCELLING.value
            store.finish(
                job["job_id"],
                worker_id,
                outcome=RunStatus.CANCELLED if cancelling else RunStatus.FAILED,
                error_code=None if cancelling else "EXECUTION_FAILED",
                error_message=None if cancelling else "Execution failed; inspect worker logs",
            )
        except LeaseLost:
            logger.warning("Job lease lost before failure could be recorded: %s", job["job_id"])
    finally:
        stop.set()
        monitor.join(timeout=3)
        reset_all()


async def serve(settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    store = JobStore(settings.db_path)
    worker_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        shutdown.set()
        request_stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_shutdown)
        except NotImplementedError:
            pass
    logger.info("Worker started: %s", worker_id)
    while not shutdown.is_set():
        store.reap_expired()
        job = store.claim(worker_id)
        if job is None:
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=settings.worker_poll_seconds)
            except asyncio.TimeoutError:
                pass
            continue
        await run_claimed_job(job, settings, store, worker_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(serve())


if __name__ == "__main__":
    main()
