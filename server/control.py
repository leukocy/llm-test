"""Cooperative pause at drained measurement boundaries, retaining the worker lease."""

from __future__ import annotations

import asyncio
import time

from core.cancel_state import is_stop_requested
from core.run_lifecycle import RunStatus
from server.store import JobStore, LeaseLost

PAUSABLE_TEST_TYPES = frozenset(
    {
        "concurrency",
        "prefill",
        "segmented_prefill",
        "long_context",
        "matrix",
        "custom_text",
        "dataset",
        "stability",
        "quality",
        "robustness",
    }
)


class JobControl:
    def __init__(self, store: JobStore, job_id: str, worker_id: str) -> None:
        self.store = store
        self.job_id = job_id
        self.worker_id = worker_id

    async def checkpoint(self) -> None:
        """Called only when the current request group has drained and been persisted."""
        while True:
            job = self.store.get(self.job_id)
            if (
                job["lease_owner"] != self.worker_id
                or job["lease_until"] is None
                or job["lease_until"] < time.time()
            ):
                raise LeaseLost(self.job_id)
            if is_stop_requested() or job["status"] == RunStatus.CANCELLING.value:
                raise asyncio.CancelledError
            if job["status"] == RunStatus.PAUSING.value:
                self.store.acknowledge_pause(self.job_id, self.worker_id)
            elif job["status"] == RunStatus.RUNNING.value:
                return
            elif job["status"] != RunStatus.PAUSED.value:
                raise LeaseLost(self.job_id)
            await asyncio.sleep(0.2)

    def pause_requested(self) -> bool:
        """Stop continuous admission before acknowledging a drained pause."""
        job = self.store.get(self.job_id)
        if (
            job["lease_owner"] != self.worker_id
            or job["lease_until"] is None
            or job["lease_until"] < time.time()
        ):
            raise LeaseLost(self.job_id)
        if is_stop_requested() or job["status"] == RunStatus.CANCELLING.value:
            raise asyncio.CancelledError
        if job["status"] not in {"running", "pausing", "paused"}:
            raise LeaseLost(self.job_id)
        return job["status"] in {"pausing", "paused"}
