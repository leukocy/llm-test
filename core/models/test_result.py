"""
Test ResultsModel
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TestResult:
    """Test ResultsModel"""

    # 主键
    id: int | None = None

    # 关联Test运行
    run_id: int = 0

    # 请求标识
    session_id: int | None = None
    request_index: int | None = None
    round: int | None = None
    concurrency_level: int | None = None
    batch_id: str | None = None

    # onunder文/输入
    input_tokens_target: int | None = None
    context_length_target: int | None = None

    # Performance Metrics - Latency
    ttft: float | None = None  # Time to First Token (seconds)
    tpot: float | None = None  # Time Per Output Token (seconds)
    tpot_p95: float | None = None
    tpot_p99: float | None = None
    total_time: float | None = None
    decode_time: float | None = None
    prefill_speed: float | None = None  # tokens/s

    # Performance Metrics - 吞吐
    tps: float | None = None  # Tokens Per Second
    system_throughput: float | None = None
    system_input_throughput: float | None = None
    system_output_throughput: float | None = None
    system_total_throughput: float | None = None
    rps: float | None = None  # Requests Per Second

    # Token Statistics
    prefill_tokens: int | None = None
    decode_tokens: int | None = None
    cache_hit_tokens: int | None = None
    api_prefill: int | None = None
    api_decode: int | None = None
    effective_prefill_tokens: int | None = None
    effective_decode_tokens: int | None = None
    token_source: str | None = None
    token_calc_method: str | None = None
    cache_hit_source: str | None = None

    # 时间戳
    start_time: float | None = None
    end_time: float | None = None
    created_at: datetime | None = None

    # Error信息
    error: str | None = None
    error_type: str | None = None

    # Prompt and输出（用于可复现性）
    prompt_text: str | None = None
    output_text: str | None = None

    # 扩展字段
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_result(cls, run_id: int, result: dict[str, Any]) -> "TestResult":
        """从 API Result字典Create"""
        return cls(
            run_id=run_id,
            session_id=result.get("session_id"),
            round=result.get("round"),
            concurrency_level=result.get("concurrency"),
            input_tokens_target=result.get("input_tokens_target"),
            context_length_target=result.get("context_length_target"),
            ttft=result.get("ttft"),
            tpot=result.get("tpot"),
            tpot_p95=result.get("tpot_p95"),
            tpot_p99=result.get("tpot_p99"),
            total_time=result.get("total_time"),
            decode_time=result.get("decode_time"),
            prefill_speed=result.get("prefill_speed"),
            tps=result.get("tps"),
            system_throughput=result.get("system_throughput"),
            system_input_throughput=result.get("system_input_throughput"),
            system_output_throughput=result.get("system_output_throughput"),
            system_total_throughput=result.get("system_total_throughput"),
            rps=result.get("rps"),
            prefill_tokens=result.get("prefill_tokens"),
            decode_tokens=result.get("decode_tokens"),
            cache_hit_tokens=result.get("cache_hit_tokens"),
            api_prefill=result.get("api_prefill"),
            api_decode=result.get("api_decode"),
            effective_prefill_tokens=result.get("effective_prefill_tokens"),
            effective_decode_tokens=result.get("effective_decode_tokens"),
            token_source=result.get("token_source"),
            token_calc_method=result.get("token_calc_method"),
            cache_hit_source=result.get("cache_hit_source"),
            start_time=result.get("start_time"),
            end_time=result.get("end_time"),
            error=result.get("error"),
            prompt_text=result.get("prompt_text"),
            output_text=result.get("output_text"),
            extra_metrics=result.get("extra_metrics", {}),
            created_at=datetime.now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convertis字典"""
        result = {}
        for k, v in asdict(self).items():
            if k == "extra_metrics":
                result["extra_metrics"] = json.dumps(v, ensure_ascii=False) if v else None
            elif k == "created_at":
                result["created_at"] = v.isoformat() if v else None
            else:
                result[k] = v
        return result

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TestResult":
        """从Database行Create"""
        return cls(
            id=row.get("id"),
            run_id=row.get("run_id", 0),
            session_id=row.get("session_id"),
            request_index=row.get("request_index"),
            round=row.get("round"),
            concurrency_level=row.get("concurrency_level"),
            batch_id=row.get("batch_id"),
            input_tokens_target=row.get("input_tokens_target"),
            context_length_target=row.get("context_length_target"),
            ttft=row.get("ttft"),
            tpot=row.get("tpot"),
            tpot_p95=row.get("tpot_p95"),
            tpot_p99=row.get("tpot_p99"),
            total_time=row.get("total_time"),
            decode_time=row.get("decode_time"),
            prefill_speed=row.get("prefill_speed"),
            tps=row.get("tps"),
            system_throughput=row.get("system_throughput"),
            system_input_throughput=row.get("system_input_throughput"),
            system_output_throughput=row.get("system_output_throughput"),
            system_total_throughput=row.get("system_total_throughput"),
            rps=row.get("rps"),
            prefill_tokens=row.get("prefill_tokens"),
            decode_tokens=row.get("decode_tokens"),
            cache_hit_tokens=row.get("cache_hit_tokens"),
            api_prefill=row.get("api_prefill"),
            api_decode=row.get("api_decode"),
            effective_prefill_tokens=row.get("effective_prefill_tokens"),
            effective_decode_tokens=row.get("effective_decode_tokens"),
            token_source=row.get("token_source"),
            token_calc_method=row.get("token_calc_method"),
            cache_hit_source=row.get("cache_hit_source"),
            start_time=row.get("start_time"),
            end_time=row.get("end_time"),
            created_at=cls._parse_datetime(row.get("created_at")),
            error=row.get("error"),
            error_type=row.get("error_type"),
            prompt_text=row.get("prompt_text"),
            output_text=row.get("output_text"),
            extra_metrics=json.loads(row.get("extra_metrics", "{}") or "{}"),
        )

    @staticmethod
    def _parse_datetime(value) -> datetime | None:
        """Parse日期时间"""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    @property
    def is_success(self) -> bool:
        """is否succeeded"""
        return self.error is None

    @property
    def total_tokens(self) -> int:
        """总 Token 数"""
        prefill = self.prefill_tokens or 0
        decode = self.decode_tokens or 0
        return prefill + decode
