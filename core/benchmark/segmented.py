"""Ordered prefix plans and complete sequence observations, independent of UI state."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from core.benchmark.batch_observations import tag_batch_window
from core.cancel_state import is_stop_requested


def encode_prefix(tokenizer: Any, prompt: str) -> list[int]:
    """HF tokenizers accept add_special_tokens; reference encodings do not."""
    try:
        return list(tokenizer.encode(prompt, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(prompt))


def build_round_prompts(
    *,
    levels: list[int],
    requests_per_segment: int,
    concurrency: int,
    tokenizer: Any,
    target: Callable[[int], int],
    generate: Callable[[int], str],
    bases: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Materialize the complete ordered sequence before any model request is admitted."""
    encoded = [encode_prefix(tokenizer, base) for base in bases] if bases else []
    prompts, sources = [], []
    for length in levels:
        segment_prompts, segment_sources = [], []
        for index in range(concurrency):
            if bases is None:
                prompt, source = generate(length), "segmented:independent"
            elif target(length) >= len(encoded[index]):
                prompt, source = bases[index], "segmented:whole_base"
            elif hasattr(tokenizer, "decode"):
                prompt = tokenizer.decode(encoded[index][: target(length)])
                source = "segmented:tokenizer_prefix"
            else:
                ratio = len(bases[index]) / len(encoded[index])
                prompt = bases[index][: int(target(length) * ratio)]
                source = "segmented:character_estimate"
            segment_prompts.append(prompt)
            segment_sources.append(source)
        for _ in range(requests_per_segment):
            prompts.extend(segment_prompts)
            sources.extend(segment_sources)
    return prompts, sources


async def observe_round(
    runner: Any,
    prompts: list[str],
    *,
    levels: list[int],
    requests_per_segment: int,
    concurrency: int,
    max_tokens: int,
    round_index: int,
    session_start: int,
    cumulative_mode: bool,
    before_batch: Callable[[], Awaitable[None]],
) -> list[dict]:
    """Preserve segment/repeat order; only each same-condition batch shares a clock."""
    observations = []
    offset = 0
    for segment_index, length in enumerate(levels):
        for repeat_index in range(requests_per_segment):
            await before_batch()
            if is_stop_requested():
                raise asyncio.CancelledError
            runner.status_text.info(
                f"轮次 {round_index + 1} · 分段 {segment_index + 1}/{len(levels)} · "
                f"{length} tokens · 并发 {concurrency}"
            )
            started = time.monotonic()
            tasks = [
                asyncio.create_task(
                    runner._run_segmented_request(
                        prompts[offset + index],
                        max_tokens,
                        session_start + offset + index,
                        length,
                        concurrency,
                        round_index,
                        cumulative_mode,
                        seg_idx=segment_index,
                        c_idx=index,
                    )
                )
                for index in range(concurrency)
            ]
            try:
                results = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            tag_batch_window(
                results, elapsed_seconds=time.monotonic() - started, expected_requests=concurrency
            )
            for index, result in enumerate(results):
                if not isinstance(result, dict):
                    raise RuntimeError("Segmented request returned no observation")
                result.setdefault("extra_metrics", {})["segmented_request"] = {
                    "round": round_index + 1,
                    "segment_index": segment_index,
                    "repeat_index": repeat_index,
                    "concurrency_index": index,
                    "label": f"R{round_index + 1}_S{segment_index + 1}_C{index + 1}_R{repeat_index + 1}",
                }
            observations.extend(results)
            offset += concurrency
            runner._update_log(
                f"分段轮次 {round_index + 1} 已返回 {offset}/{len(prompts)} 次请求；整轮完成后保存测量组",
                returned_requests=offset,
                expected_requests=len(prompts),
            )
    return observations
