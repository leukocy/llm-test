"""
Test ResultsModel
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class TestResult:
    """Test ResultsModel"""

    # 主键
    id: Optional[int] = None

    # 关联Test运行
    run_id: int = 0

    # 请求标识
    session_id: Optional[int] = None
    request_index: Optional[int] = None
    round: Optional[int] = None
    concurrency_level: Optional[int] = None
    batch_id: Optional[str] = None

    # onunder文/输入
    input_tokens_target: Optional[int] = None
    context_length_target: Optional[int] = None

    # Performance Metrics - Latency
    ttft: Optional[float] = None  # Time to First Token (seconds)
    tpot: Optional[float] = None  # Time Per Output Token (seconds)
    tpot_p95: Optional[float] = None
    tpot_p99: Optional[float] = None
    total_time: Optional[float] = None
    decode_time: Optional[float] = None
    prefill_speed: Optional[float] = None  # tokens/s

    # Performance Metrics - 吞吐
    tps: Optional[float] = None  # Tokens Per Second
    system_throughput: Optional[float] = None
    system_input_throughput: Optional[float] = None
    system_output_throughput: Optional[float] = None
    system_total_throughput: Optional[float] = None
    rps: Optional[float] = None  # Requests Per Second

    # Token Statistics
    prefill_tokens: Optional[int] = None
    decode_tokens: Optional[int] = None
    cache_hit_tokens: Optional[int] = None
    api_prefill: Optional[int] = None
    api_decode: Optional[int] = None
    effective_prefill_tokens: Optional[int] = None
    effective_decode_tokens: Optional[int] = None
    token_source: Optional[str] = None
    token_calc_method: Optional[str] = None
    cache_hit_source: Optional[str] = None

    # 时间戳
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    created_at: Optional[datetime] = None

    # Error信息
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Prompt and输出（用于可复现性）
    prompt_text: Optional[str] = None
    output_text: Optional[str] = None

    # 扩展字段
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_result(cls, run_id: int, result: Dict[str, Any]) -> "TestResult":
        """从 API Result字典Create"""
        return cls(
            run_id=run_id,
            session_id=result.get('session_id'),
            round=result.get('round'),
            concurrency_level=result.get('concurrency'),
            input_tokens_target=result.get('input_tokens_target'),
            context_length_target=result.get('context_length_target'),
            ttft=result.get('ttft'),
            tpot=result.get('tpot'),
            tpot_p95=result.get('tpot_p95'),
            tpot_p99=result.get('tpot_p99'),
            total_time=result.get('total_time'),
            decode_time=result.get('decode_time'),
            prefill_speed=result.get('prefill_speed'),
            tps=result.get('tps'),
            system_throughput=result.get('system_throughput'),
            system_input_throughput=result.get('system_input_throughput'),
            system_output_throughput=result.get('system_output_throughput'),
            system_total_throughput=result.get('system_total_throughput'),
            rps=result.get('rps'),
            prefill_tokens=result.get('prefill_tokens'),
            decode_tokens=result.get('decode_tokens'),
            cache_hit_tokens=result.get('cache_hit_tokens'),
            api_prefill=result.get('api_prefill'),
            api_decode=result.get('api_decode'),
            effective_prefill_tokens=result.get('effective_prefill_tokens'),
            effective_decode_tokens=result.get('effective_decode_tokens'),
            token_source=result.get('token_source'),
            token_calc_method=result.get('token_calc_method'),
            cache_hit_source=result.get('cache_hit_source'),
            start_time=result.get('start_time'),
            end_time=result.get('end_time'),
            error=result.get('error'),
            prompt_text=result.get('prompt_text'),
            output_text=result.get('output_text'),
            extra_metrics=result.get('extra_metrics', {}),
            created_at=datetime.now(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convertis字典"""
        result = {}
        for k, v in asdict(self).items():
            if k == 'extra_metrics':
                result['extra_metrics'] = json.dumps(v, ensure_ascii=False) if v else None
            elif k == 'created_at':
                result['created_at'] = v.isoformat() if v else None
            else:
                result[k] = v
        return result

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "TestResult":
        """从Database行Create"""
        return cls(
            id=row.get('id'),
            run_id=row.get('run_id', 0),
            session_id=row.get('session_id'),
            request_index=row.get('request_index'),
            round=row.get('round'),
            concurrency_level=row.get('concurrency_level'),
            batch_id=row.get('batch_id'),
            input_tokens_target=row.get('input_tokens_target'),
            context_length_target=row.get('context_length_target'),
            ttft=row.get('ttft'),
            tpot=row.get('tpot'),
            tpot_p95=row.get('tpot_p95'),
            tpot_p99=row.get('tpot_p99'),
            total_time=row.get('total_time'),
            decode_time=row.get('decode_time'),
            prefill_speed=row.get('prefill_speed'),
            tps=row.get('tps'),
            system_throughput=row.get('system_throughput'),
            system_input_throughput=row.get('system_input_throughput'),
            system_output_throughput=row.get('system_output_throughput'),
            system_total_throughput=row.get('system_total_throughput'),
            rps=row.get('rps'),
            prefill_tokens=row.get('prefill_tokens'),
            decode_tokens=row.get('decode_tokens'),
            cache_hit_tokens=row.get('cache_hit_tokens'),
            api_prefill=row.get('api_prefill'),
            api_decode=row.get('api_decode'),
            effective_prefill_tokens=row.get('effective_prefill_tokens'),
            effective_decode_tokens=row.get('effective_decode_tokens'),
            token_source=row.get('token_source'),
            token_calc_method=row.get('token_calc_method'),
            cache_hit_source=row.get('cache_hit_source'),
            start_time=row.get('start_time'),
            end_time=row.get('end_time'),
            created_at=cls._parse_datetime(row.get('created_at')),
            error=row.get('error'),
            error_type=row.get('error_type'),
            prompt_text=row.get('prompt_text'),
            output_text=row.get('output_text'),
            extra_metrics=json.loads(row.get('extra_metrics', '{}') or '{}'),
        )

    @staticmethod
    def _parse_datetime(value) -> Optional[datetime]:
        """Parse日期时间"""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except:
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
