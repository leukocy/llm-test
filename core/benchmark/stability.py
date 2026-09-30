"""Continuous load with bounded live flushes and pause only after in-flight work drains."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any


async def continuous_load(
    request: Callable[[int], Awaitable[dict[str, Any]]],
    *,
    concurrency: int,
    duration: float,
    session_id_start: int = 0,
    stopped: Callable[[], bool],
    pause_requested: Callable[[], bool] | None = None,
    checkpoint: Callable[[], Awaitable[None]] | None = None,
    observe: Callable[[list[dict[str, Any]], dict[str, Any]], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """Keep the configured load until active scheduling time expires.

    Completion positions use wrapper time, independently of provider metrics. A pause
    stops admission, drains requests, then waits at the control checkpoint. Paused time
    extends the scheduling deadline; the wall timeline retains the resulting gap.
    """
    origin = clock()
    window_id = uuid.uuid4().hex
    paused_seconds = 0.0
    scheduled = 0
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    active: set[asyncio.Task] = set()
    last_flush = origin
    state = "running"

    def flush(*, force: bool = False) -> None:
        nonlocal last_flush
        now = clock()
        if not force and len(pending) < 50 and now - last_flush < 0.5:
            return
        snapshot = {
            "version": "stability-clock-v1",
            "clock": "monotonic",
            "anchor": "scheduler_start",
            "id": window_id,
            "window_seconds": max(0.000001, now - origin),
            "planned_seconds": duration,
            "expected_requests": scheduled,
            "recorded_requests": len(results),
            "paused_seconds": paused_seconds,
            "scheduling_seconds": max(0, now - origin - paused_seconds),
            "state": state,
        }
        if observe:
            observe(pending, snapshot)
        pending.clear()
        last_flush = now

    async def measured(index: int) -> dict[str, Any]:
        start = clock()
        row = await request(session_id_start + index)
        end = clock()
        row.setdefault("extra_metrics", {})["timing_observation"] = {
            "version": "stability-clock-v1",
            "clock": "monotonic",
            "anchor": "scheduler_start",
            "id": window_id,
            "index": index,
            "start_seconds": start - origin,
            "end_seconds": end - origin,
        }
        row["batch_id"] = window_id
        row["request_index"] = index
        return row

    def collect(done: set[asyncio.Task]) -> None:
        for task in done:
            if task.cancelled():
                continue
            row = task.result()
            if row.get("error") != "UserCancelled":
                results.append(row)
                pending.append(row)

    try:
        flush(force=True)
        while True:
            if stopped():
                raise asyncio.CancelledError
            pausing = bool(pause_requested and pause_requested())
            elapsed = clock() - origin - paused_seconds
            if pausing and not active:
                state = "paused"
                flush(force=True)
                pause_start = clock()
                if checkpoint is None:
                    raise RuntimeError("Pause requested without a drained checkpoint")
                try:
                    await checkpoint()
                finally:
                    paused_seconds += clock() - pause_start
                state = "running"
                flush(force=True)
                continue
            if not pausing and elapsed < duration:
                while len(active) < concurrency:
                    active.add(asyncio.create_task(measured(scheduled)))
                    scheduled += 1
            if not active:
                break
            done, active = await asyncio.wait(
                active, timeout=0.2, return_when=asyncio.FIRST_COMPLETED
            )
            collect(done)
            flush()
        state = "completed"
    finally:
        # Preserve responses that finished before cancellation reached this coordinator.
        finished = {task for task in active if task.done()}
        collect(finished)
        active -= finished
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        if state != "completed":
            state = "interrupted"
        flush(force=True)
    return results
