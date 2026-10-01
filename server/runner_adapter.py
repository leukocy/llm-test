"""Run the existing measurement engine without a browser or Streamlit session."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from server.checkpoints import MEASUREMENT_TYPES, JobJournal
from server.control import PAUSABLE_TEST_TYPES, JobControl
from server.measurement_checkpoints import MeasurementJournal
from server.settings import Endpoint, Settings
from server.specs import describe_report_environment
from server.stability_checkpoints import StabilityJournal
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


class _LogTee:
    """runner render_log 回调：把 BenchmarkLogger 的增量条目追加到 logs.jsonl。

    BenchmarkLogger.entries 有容量上限(500), 通过 stats.total 跟踪累计条数，
    从窗口末尾取新增项，避免运行超过 500 条日志后停止落盘。
    落盘文件才是完整轨迹——GET /api/v1/jobs/{id}/logs 据此读取。
    """

    def __init__(self, path) -> None:
        self._path = path
        self._seen = 0

    def __call__(self, logger) -> None:
        entries = getattr(logger, "entries", None) or []
        total = int(getattr(logger, "stats", {}).get("total", len(entries)))
        fresh_count = max(0, total - self._seen)
        fresh = entries[-fresh_count:] if fresh_count else []
        if not fresh:
            return
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                for entry in fresh:
                    handle.write(
                        json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n"
                    )
        except OSError:
            return
        self._seen = total


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

    # `_run_config` 保留键：提交时并入 parameters_json, 执行前拆出
    # （spec 校验在此前已完成, 方法实参不受污染）
    params = dict(job["parameters"])
    run_config = params.pop("_run_config", {}) or {}
    comparison = params.pop("_comparison", None)
    report_environment = describe_report_environment(run_config.get("report_environment"))

    control = JobControl(store, job_id, worker_id)
    journal = (
        JobJournal(store, job, worker_id, endpoint)
        if job["test_type"] in {"quality", "robustness"}
        else StabilityJournal(store, job, worker_id, endpoint)
        if job["test_type"] == "stability"
        else MeasurementJournal(store, job, worker_id, endpoint)
        if job["test_type"] in MEASUREMENT_TYPES
        else None
    )
    if comparison:
        from server.shared_quality import SharedQualityJournal

        journal = SharedQualityJournal(store, job, worker_id, endpoint)
    artifact_name = (
        "report.json" if job["attempts"] == 1 else f"report-attempt-{job['attempts']}.json"
    )

    def save_report(payload: dict) -> str:
        current = store.get(job_id)
        payload["execution_control"].update(
            pause_count=current["pause_count"],
            paused_seconds=current["paused_seconds"],
        )
        artifact = job_dir / artifact_name
        temporary = artifact.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(artifact)
        return f"{job_id}/{artifact_name}"

    if job["test_type"] == "robustness":
        import dataclasses

        from core.providers.factory import get_provider
        from core.robustness_tester import PerturbationType, RobustnessTester

        types = (
            [PerturbationType(t) for t in params["perturbation_types"]]
            if params.get("perturbation_types")
            else None
        )
        provider = get_provider(
            endpoint.provider, endpoint.api_base_url, endpoint.api_key(), endpoint.model_id
        )

        async def get_response(prompt: str) -> str:
            result = await provider.get_completion(
                client=None,
                session_id=-1,
                prompt=prompt,
                max_tokens=params.get("max_tokens", 256),
            )
            if result.get("error"):
                raise RuntimeError(str(result["error"]))
            return str(result.get("full_response_content") or "")

        def robustness_progress(completed: int, total: int) -> None:
            factor = 1 + len(tester.perturbation_types)
            store.update_progress(
                job_id, worker_id, completed=completed * factor, total=total * factor
            )

        tester = RobustnessTester(perturbation_types=types)
        report = await tester.test_batch(
            params["samples"],
            get_response,
            progress_callback=robustness_progress,
            control_checkpoint=control.checkpoint,
            control_poll=control.pause_requested,
            journal=journal,
        )
        report.model_id = endpoint.model_id
        artifact_path = save_report(
            {
                "job_id": job_id,
                "model_id": endpoint.model_id,
                "robustness": dataclasses.asdict(report),
                "report_environment": report_environment,
                "checkpoint": journal.describe() if journal else None,
                "execution_control": {
                    "attempts": job["attempts"],
                    "duration_scope": "sample_observations_across_attempts",
                },
            }
        )
        total_requests = sum(1 + len(r.perturbed_results) for r in report.results)
        return RunOutput(
            result_artifact=artifact_path,
            completed=total_requests,
            total=total_requests,
        )

    if job["test_type"] == "quality":
        from core.quality_evaluator import QualityEvaluator, QualityTestConfig

        config = QualityTestConfig(**params)
        evaluator = QualityEvaluator(
            api_base_url=endpoint.api_base_url,
            model_id=endpoint.model_id,
            api_key=endpoint.api_key(),
            provider=endpoint.provider,
            output_dir=str(job_dir),
            enable_cache=config.use_cache,
            control_checkpoint=control.checkpoint,
            control_poll=control.pause_requested,
            journal=journal,
        )

        def progress(completed: int, total: int, _: str) -> None:
            store.update_progress(job_id, worker_id, completed=completed, total=total)

        result = await evaluator.run_evaluation(config, progress_callback=progress)
        if not result or any(item.total_samples == 0 for item in result.values()):
            raise RuntimeError("Quality evaluation produced no complete samples")
        artifact_path = save_report(
            {
                "job_id": job_id,
                "model_id": endpoint.model_id,
                "datasets": {name: item.to_dict() for name, item in result.items()},
                "comparison": comparison,
                "report_environment": report_environment,
                "checkpoint": journal.describe() if journal else None,
                "execution_control": {
                    "attempts": job["attempts"],
                    "duration_scope": "current_attempt_dataset_phase",
                },
            }
        )
        return RunOutput(
            result_artifact=artifact_path,
            completed=sum(item.total_samples for item in result.values()),
            total=sum(item.total_samples for item in result.values()),
        )

    from core.benchmark_runner import BenchmarkRunner

    output = HeadlessOutput()
    log_tee = _LogTee(job_dir / "logs.jsonl")
    measurement_dir = job_dir / f"attempt-{job['attempts']}" if journal else job_dir
    measurement_dir.mkdir(parents=True, exist_ok=True)
    control = JobControl(store, job_id, worker_id)
    runner = BenchmarkRunner(
        placeholder=output,
        progress_bar=output,
        status_text=output,
        api_base_url=endpoint.api_base_url,
        model_id=endpoint.model_id,
        tokenizer_option=run_config.get("tokenizer_option") or endpoint.tokenizer_option,
        csv_filename=str(measurement_dir / "requests.csv"),
        api_key=endpoint.api_key(),
        log_placeholder=None,
        provider=endpoint.provider,
        external_test_id=job_id,
        enable_live_log_server=False,
        render_log=log_tee,
        hf_tokenizer_model_id=run_config.get("hf_tokenizer_model_id"),
        latency_offset=run_config.get("latency_offset", 0.0),
        thinking_enabled=run_config.get("thinking_enabled"),
        thinking_budget=run_config.get("thinking_budget"),
        reasoning_effort=run_config.get("reasoning_effort"),
        random_seed=run_config.get("random_seed"),
        skip_first_token_for_tps=run_config.get("skip_first_token_for_tps", False),
        template_tokens=run_config.get("template_tokens", 0),
        temperature=run_config.get("temperature"),
        custom_params=run_config.get("custom_params"),
        report_environment=run_config.get("report_environment"),
        control_checkpoint=(
            control.checkpoint if job["test_type"] in PAUSABLE_TEST_TYPES else None
        ),
        control_poll=control.pause_requested if job["test_type"] == "stability" else None,
        persistence_owner=worker_id,
        measurement_journal=journal,
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
        if job["test_type"] == "dataset":
            # 行源: 内联 rows 或 dataset_loader 已存数据集; 预算按加载后实数复核
            def load_rows():
                if params.get("rows"):
                    rows = params["rows"]
                else:
                    from core.dataset_loader import DatasetLoader

                    frame = DatasetLoader().load_dataset(params["dataset"])
                    if frame is None:
                        raise RuntimeError(f"数据集不存在或无法读取: {params['dataset']}")
                    rows = frame.to_dict("records")
                return {"rows": rows}, []

            if journal:
                metadata, _ = journal.prepare_scope(
                    "measurement:dataset", load_rows, allow_empty=True
                )
                dataset_rows = metadata["rows"]
            else:
                dataset_rows = load_rows()[0]["rows"]
            rounds = params.get("rounds", 1)
            if len(dataset_rows) * rounds > 1000:
                raise RuntimeError(
                    f"数据集任务超出 1000 请求预算（{len(dataset_rows)} 行 × {rounds} 轮）"
                )
            await runner.run_dataset_test(
                dataset_rows,
                params["concurrency"],
                params["max_tokens"],
                rounds=rounds,
                dataset_filename=params.get("dataset") or "inline_rows",
            )
        else:
            await methods[job["test_type"]](**params)
    finally:
        if runner._db_run is not None:
            runner._complete_db_run(success=False)
        # 兜底冲刷: 最后一次 throttled render_log 可能漏掉尾部条目
        log_tee(runner.logger)
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
