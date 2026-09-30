"""Bounded sample admission, drained pause boundaries and durable completion hooks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class EvaluationJournal(Protocol):
    def prepare_scope(
        self, key: str, factory: Callable[[], tuple[dict, list[dict]]]
    ) -> tuple[dict, list[dict]]: ...
    def results(self, key: str) -> dict[int, dict[str, Any]]: ...
    def start(self, key: str, index: int) -> None: ...
    def commit(self, key: str, index: int, result: dict[str, Any]) -> None: ...
    def describe(self) -> dict[str, Any]: ...


async def run_samples(
    evaluate: Callable[[int], Awaitable[T]],
    *,
    total: int,
    concurrency: int,
    restored: dict[int, T] | None = None,
    checkpoint: Callable[[], Awaitable[None]] | None = None,
    pause_requested: Callable[[], bool] | None = None,
    start: Callable[[int], None] | None = None,
    commit: Callable[[int, T], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[T]:
    """Preserve input order while keeping only the configured concurrency in flight.

    Committed samples are never called again. Persistence callbacks are critical and
    must propagate failures; UI callbacks belong outside this function. On interruption
    keep completed responses, cancel in-flight work, and leave uncommitted units pending.
    """
    if total < 0 or concurrency < 1:
        raise ValueError("Invalid sample schedule")
    results = dict(restored or {})
    if any(type(i) is not int or not 0 <= i < total for i in results):
        raise ValueError("Checkpoint sample position is outside the frozen plan")
    remaining = iter(i for i in range(total) if i not in results)
    active: dict[asyncio.Task[T], int] = {}
    exhausted = False

    def collect(done: set[asyncio.Task[T]], *, draining: bool = False) -> None:
        for task in sorted(done, key=lambda task: active[task]):
            index = active.pop(task)
            if task.cancelled():
                if draining:
                    continue
                raise asyncio.CancelledError("Sample execution was interrupted")
            value = task.result()
            if commit:
                commit(index, value)
            results[index] = value
            if progress:
                progress(len(results), total)

    async def invoke(index: int) -> T:
        return await evaluate(index)

    try:
        if checkpoint:
            await checkpoint()
        if progress and results:
            progress(len(results), total)
        while active or not exhausted:
            pausing = bool(pause_requested and pause_requested())
            if pausing and not active:
                if checkpoint is None:
                    raise RuntimeError("Sample pause requires a drained checkpoint")
                await checkpoint()
                continue
            if not pausing:
                while len(active) < concurrency and not exhausted:
                    index = next(remaining, None)
                    if index is None:
                        exhausted = True
                        break
                    if start:
                        start(index)
                    active[asyncio.create_task(invoke(index))] = index
            if active:
                done, _ = await asyncio.wait(
                    active, timeout=0.2, return_when=asyncio.FIRST_COMPLETED
                )
                collect(done)
        return [results[index] for index in range(total)]
    finally:
        try:
            collect({task for task in active if task.done()}, draining=True)
        finally:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
