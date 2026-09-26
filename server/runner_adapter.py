"""Run the existing measurement engine without a browser or Streamlit session."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from server.settings import Endpoint, Settings
from server.store import JobStore


class HeadlessOutput:
    def progress(self, _: float) -> None:
        return None

    def info(self, _: str) -> None:
        return None

    def warning(self, _: str) -> None:
        return None

    def error(self, _: str) -> None:
        return None

    def success(self, _: str) -> None:
        return None


@dataclass
class RunOutput:
    result_run_id: int | None = None
    result_artifact: str | None = None
    completed: int = 0
    total: int = 0


async def execute_job(
    job: dict[str, Any], endpoint: Endpoint, settings: Settings, store: JobStore, worker_id: str
) -> RunOutput:
    job_id = job["job_id"]
    job_dir = settings.artifact_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if job["test_type"] == "quality":
        from core.quality_evaluator import QualityEvaluator, QualityTestConfig

        config = QualityTestConfig(**job["parameters"])
        evaluator = QualityEvaluator(
            api_base_url=endpoint.api_base_url,
            model_id=endpoint.model_id,
            api_key=endpoint.api_key(),
            provider=endpoint.provider,
            output_dir=str(job_dir),
            enable_cache=config.use_cache,
        )

        def progress(completed: int, total: int, _: str) -> None:
            store.update_progress(job_id, worker_id, completed=completed, total=total)

        result = await evaluator.run_evaluation(config, progress_callback=progress)
        if not result or any(item.total_samples == 0 for item in result.values()):
            raise RuntimeError("Quality evaluation produced no complete samples")
        artifact = job_dir / "report.json"
        artifact.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "model_id": endpoint.model_id,
                    "datasets": {name: item.to_dict() for name, item in result.items()},
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return RunOutput(
            result_artifact=f"{job_id}/report.json",
            completed=sum(item.total_samples for item in result.values()),
            total=sum(item.total_samples for item in result.values()),
        )

    from core.benchmark_runner import BenchmarkRunner

    output = HeadlessOutput()
    runner = BenchmarkRunner(
        placeholder=output,
        progress_bar=output,
        status_text=output,
        api_base_url=endpoint.api_base_url,
        model_id=endpoint.model_id,
        tokenizer_option=endpoint.tokenizer_option,
        csv_filename=str(job_dir / "requests.csv"),
        api_key=endpoint.api_key(),
        log_placeholder=None,
        provider=endpoint.provider,
        external_test_id=job_id,
        enable_live_log_server=False,
    )
    methods: dict[str, Callable[..., Awaitable[Any]]] = {
        "concurrency": runner.run_concurrency_test,
        "prefill": runner.run_prefill_test,
        "segmented_prefill": runner.run_segmented_prefill_test,
        "long_context": runner.run_long_context_test,
        "matrix": runner.run_throughput_matrix_test,
        "stability": runner.run_stability_test,
        "custom_text": runner.run_custom_text_test,
    }
    try:
        await methods[job["test_type"]](**job["parameters"])
    finally:
        if runner._db_run is not None:
            runner._complete_db_run(success=False)
    if runner.last_run_id is None:
        raise RuntimeError("Measurement was not persisted")
    recorded = len(runner.results_list)
    if recorded == 0:
        raise RuntimeError("Measurement produced no request observations")
    return RunOutput(
        result_run_id=runner.last_run_id,
        completed=recorded,
        total=max(runner.total_requests, recorded),
    )
