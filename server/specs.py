"""Bounded, versioned run specifications accepted by the public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from core.measurement_protocol import measurement_plan as _measurement_plan


class StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ConcurrencySpec(StrictSpec):
    selected_concurrencies: list[int] = Field(min_length=1, max_length=8)
    rounds_per_level: int = Field(ge=1, le=20)
    max_tokens: int = Field(ge=1, le=8192)
    input_tokens_target: int = Field(default=0, ge=0, le=131072)
    warmup_rounds_per_level: int = Field(default=0, ge=0, le=3)

    @field_validator("selected_concurrencies")
    @classmethod
    def concurrency_levels(cls, value: list[int]) -> list[int]:
        if any(level < 1 or level > 128 for level in value) or len(set(value)) != len(value):
            raise ValueError("Concurrency levels must be unique and between 1 and 128")
        return value

    @model_validator(mode="after")
    def request_budget(self) -> ConcurrencySpec:
        if (
            sum(self.selected_concurrencies)
            * (self.rounds_per_level + self.warmup_rounds_per_level)
            > 1000
        ):
            raise ValueError("A job may issue at most 1000 requests")
        return self


class PrefillSpec(StrictSpec):
    token_levels: list[int] = Field(min_length=1, max_length=12)
    requests_per_level: int = Field(ge=1, le=50)
    max_tokens: int = Field(ge=1, le=8192)
    warmup_requests_per_level: int = Field(default=0, ge=0, le=5)

    @field_validator("token_levels")
    @classmethod
    def tokens(cls, value: list[int]) -> list[int]:
        if any(level < 1 or level > 131072 for level in value) or len(set(value)) != len(value):
            raise ValueError("Token levels must be unique and between 1 and 131072")
        return value

    @model_validator(mode="after")
    def request_budget(self) -> PrefillSpec:
        if (
            len(self.token_levels) * (self.requests_per_level + self.warmup_requests_per_level)
            > 1000
        ):
            raise ValueError("A job may issue at most 1000 requests")
        return self


class SegmentedPrefillSpec(StrictSpec):
    segment_levels: list[int] = Field(min_length=1, max_length=12)
    requests_per_segment: int = Field(ge=1, le=20)
    max_tokens: int = Field(ge=1, le=8192)
    cumulative_mode: bool = True
    total_rounds: int = Field(default=1, ge=1, le=20)
    per_round_unique: bool = False
    concurrency: int = Field(default=1, ge=1, le=128)

    @field_validator("segment_levels")
    @classmethod
    def tokens(cls, value: list[int]) -> list[int]:
        if any(level < 1 or level > 131072 for level in value):
            raise ValueError("Segment levels must be between 1 and 131072")
        return value

    @model_validator(mode="after")
    def request_budget(self) -> SegmentedPrefillSpec:
        if len(self.segment_levels) * self.requests_per_segment * self.total_rounds > 1000:
            raise ValueError("A job may issue at most 1000 requests")
        return self


class LongContextSpec(StrictSpec):
    context_lengths: list[int] = Field(min_length=1, max_length=12)
    rounds_per_level: int = Field(ge=1, le=20)
    max_tokens: int = Field(ge=1, le=8192)

    @field_validator("context_lengths")
    @classmethod
    def lengths(cls, value: list[int]) -> list[int]:
        if any(level < 1 or level > 131072 for level in value):
            raise ValueError("Context length must be between 1 and 131072")
        return value


class MatrixSpec(StrictSpec):
    concurrencies: list[int] = Field(min_length=1, max_length=8)
    context_lengths: list[int] = Field(min_length=1, max_length=8)
    rounds: int = Field(ge=1, le=20)
    max_tokens: int = Field(ge=1, le=8192)
    enable_warmup: bool = False

    @model_validator(mode="after")
    def request_budget(self) -> MatrixSpec:
        if any(c < 1 or c > 128 for c in self.concurrencies) or len(set(self.concurrencies)) != len(
            self.concurrencies
        ):
            raise ValueError("Concurrency levels must be unique and between 1 and 128")
        if any(length < 1 or length > 131072 for length in self.context_lengths) or len(
            set(self.context_lengths)
        ) != len(self.context_lengths):
            raise ValueError("Context lengths must be unique and between 1 and 131072")
        if (
            sum(self.concurrencies)
            * len(self.context_lengths)
            * (self.rounds + int(self.enable_warmup))
            > 1000
        ):
            raise ValueError("A job may issue at most 1000 requests")
        return self


class StabilitySpec(StrictSpec):
    concurrency: int = Field(ge=1, le=128)
    duration_seconds: int = Field(ge=5, le=3600)
    max_tokens: int = Field(ge=1, le=8192)
    input_tokens_target: int = Field(default=0, ge=0, le=131072)


class CustomTextSpec(StrictSpec):
    selected_concurrencies: list[int] = Field(min_length=1, max_length=8)
    rounds_per_level: int = Field(ge=1, le=20)
    base_prompt: str = Field(min_length=1, max_length=50000)
    suffix_instruction: str = Field(default="", max_length=5000)
    max_tokens: int = Field(ge=1, le=8192)
    avoid_cache: bool = True

    @model_validator(mode="after")
    def request_budget(self) -> CustomTextSpec:
        if any(c < 1 or c > 128 for c in self.selected_concurrencies):
            raise ValueError("Concurrency must be between 1 and 128")
        if sum(self.selected_concurrencies) * self.rounds_per_level > 1000:
            raise ValueError("A job may issue at most 1000 requests")
        return self


class DatasetPerfSpec(StrictSpec):
    """数据集性能测试：内联行或 dataset_loader 已存数据集（二选一）。"""

    dataset: str | None = Field(default=None, max_length=120)
    rows: list[dict[str, str]] | None = Field(default=None, max_length=1000)
    concurrency: int = Field(ge=1, le=128)
    max_tokens: int = Field(ge=1, le=8192)
    rounds: int = Field(default=1, ge=1, le=20)

    @model_validator(mode="after")
    def one_source(self) -> DatasetPerfSpec:
        if bool(self.dataset) == bool(self.rows):
            raise ValueError("dataset 与 rows 必须且只能提供一个")
        if self.rows:
            for i, row in enumerate(self.rows):
                if not (row.get("prompt") or "").strip():
                    raise ValueError(f"第 {i + 1} 行缺少非空 prompt")
            if len(self.rows) * self.rounds > 1000:
                raise ValueError("A job may issue at most 1000 requests")
        return self


class RobustnessSpec(StrictSpec):
    """鲁棒性测试：样本经扰动后答案保持度（每样本 1 次原始 + N 次扰动请求）。"""

    samples: list[dict[str, str]] = Field(min_length=1, max_length=50)
    perturbation_types: list[str] | None = Field(default=None, max_length=10)
    max_tokens: int = Field(default=256, ge=1, le=8192)

    @model_validator(mode="after")
    def validate_samples(self) -> RobustnessSpec:
        for i, row in enumerate(self.samples):
            if not (row.get("question") or "").strip():
                raise ValueError(f"第 {i + 1} 个样本缺少非空 question")
        n_types = 5
        if self.perturbation_types:
            from core.robustness_tester import PerturbationType

            valid = {p.value for p in PerturbationType}
            unknown = [t for t in self.perturbation_types if t not in valid]
            if unknown:
                raise ValueError(f"未知扰动类型: {unknown}; 可选: {sorted(valid)}")
            n_types = len(self.perturbation_types)
        if len(self.samples) * (1 + n_types) > 500:
            raise ValueError("A job may issue at most 500 requests")
        return self


class QualitySpec(StrictSpec):
    datasets: list[str] = Field(min_length=1, max_length=8)
    max_samples: int = Field(ge=1, le=1000)
    num_shots: int = Field(default=0, ge=0, le=10)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    concurrency: int = Field(default=4, ge=1, le=32)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    use_cache: bool = False

    @field_validator("datasets")
    @classmethod
    def registered_datasets(cls, value: list[str]) -> list[str]:
        from evaluators import list_available_datasets

        available = set(list_available_datasets())
        if any(name not in available for name in value) or len(set(value)) != len(value):
            raise ValueError("Datasets must be unique registered evaluators")
        return value

    @model_validator(mode="after")
    def request_budget(self) -> QualitySpec:
        if len(self.datasets) * self.max_samples > 1000:
            raise ValueError("A job may evaluate at most 1000 samples")
        return self


SPEC_MODELS: dict[str, type[StrictSpec]] = {
    "concurrency": ConcurrencySpec,
    "prefill": PrefillSpec,
    "segmented_prefill": SegmentedPrefillSpec,
    "long_context": LongContextSpec,
    "matrix": MatrixSpec,
    "stability": StabilitySpec,
    "custom_text": CustomTextSpec,
    "dataset": DatasetPerfSpec,
    "quality": QualitySpec,
    "robustness": RobustnessSpec,
}


class RunConfig(StrictSpec):
    """BenchmarkRunner 构造器级旋钮（全部可选；未设走端点/引擎默认）。

    随作业提交存入 parameters_json 的 `_run_config` 保留键，worker 执行时
    由 runner_adapter 拆出并透传。仅对性能类测试生效（quality 作业参数
    已含 temperature 等自有字段）。
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    thinking_enabled: bool | None = None
    thinking_budget: int | None = Field(default=None, ge=0, le=131072)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    random_seed: int | None = Field(default=None, ge=0)
    skip_first_token_for_tps: bool = False
    template_tokens: int = Field(default=0, ge=0, le=131072)
    latency_offset: float = Field(default=0.0, ge=-10.0, le=10.0)
    tokenizer_option: str | None = Field(default=None, max_length=120)
    hf_tokenizer_model_id: str | None = Field(default=None, max_length=200)
    custom_params: list[dict[str, str]] | None = Field(default=None, max_length=20)


class JobSubmission(StrictSpec):
    schema_version: Literal[1] = 1
    endpoint_id: str = Field(min_length=1, max_length=64)
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
    parameters: dict
    run_config: RunConfig | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> JobSubmission:
        parsed = TypeAdapter(SPEC_MODELS[self.test_type]).validate_python(self.parameters)
        self.parameters = parsed.model_dump()
        return self


class PresetSubmission(JobSubmission):
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("Preset name cannot be blank")
        return name


def expected_requests(test_type: str, parameters: dict) -> int:
    """Estimate a finite workload for queue display; stability runs are time bounded."""
    if test_type in {"concurrency", "custom_text"}:
        return int(sum(parameters["selected_concurrencies"]) * parameters["rounds_per_level"])
    if test_type == "prefill":
        return int(len(parameters["token_levels"]) * parameters["requests_per_level"])
    if test_type == "segmented_prefill":
        return int(
            len(parameters["segment_levels"])
            * parameters["requests_per_segment"]
            * parameters["total_rounds"]
        )
    if test_type == "long_context":
        return int(len(parameters["context_lengths"]) * parameters["rounds_per_level"])
    if test_type == "matrix":
        return int(
            sum(parameters["concurrencies"])
            * len(parameters["context_lengths"])
            * parameters["rounds"]
        )
    if test_type == "quality":
        return int(len(parameters["datasets"]) * parameters["max_samples"])
    if test_type == "dataset" and parameters.get("rows"):
        return int(len(parameters["rows"]) * parameters.get("rounds", 1))
    if test_type == "robustness":
        n_types = len(parameters.get("perturbation_types") or [0] * 5)
        return int(len(parameters["samples"]) * (1 + n_types))
    return 0


def measurement_plan(test_type: str, parameters: dict) -> dict[str, Any]:
    return _measurement_plan(
        test_type, parameters, fallback_requests=expected_requests(test_type, parameters)
    )


def spec_catalog() -> dict[str, dict[str, Any]]:
    """全部测试类型的展示元数据 + JSON Schema（驱动前端表单）。

    label 优先取 config/test_types 的中文展示名，取不到回退原始 id。
    schema 来自 pydantic model_json_schema（含字段默认值/取值范围/约束）。
    """
    try:
        from config.test_types import test_type_label
    except Exception:  # noqa: BLE001  配置模块缺失不阻断 API
        test_type_label = None  # type: ignore[assignment]
    items: dict[str, dict[str, Any]] = {}
    for test_type, model in SPEC_MODELS.items():
        label = test_type
        if test_type_label is not None:
            try:
                label = test_type_label(test_type)
            except Exception:  # noqa: BLE001
                pass
        items[test_type] = {"label": label, "schema": model.model_json_schema()}
    return items
