"""
应用用例 Model（application_cases 表）。

每个评估样本（quality_evaluator 的 SampleResult）/ 手动录入的应用用例 = 一行。
列名逐字对齐手册 maTest 模板（core/warehouse/templates.py MA_TEST_FIELDS），
导出时直接 SELECT 列即可。元数据列（source/evaluator_name/sample_id/run_id/extra_json）
供追溯与去重。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ApplicationCase:
    """模型×应用用例（maTest 模板的一行）。"""

    __test__ = False  # 防止 pytest 把本 dataclass 误当作测试类收集

    # 主键 / 唯一键
    id: int | None = None
    case_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # 关联（quality_evaluator 不开 test_run，故可空）
    run_id: int | None = None

    # 来源
    source: str = "manual"  # 'auto'（quality_evaluator 采集）/ 'manual'（UI 录入）
    evaluator_name: str = ""  # 自动采集时的 dataset/evaluator 名
    sample_id: str = ""  # 自动采集时的 SampleResult.sample_id

    # ---- maTest 对齐字段（顺序与 MA_TEST_FIELDS 一致）----
    date: str = ""  # YYYY-MM-DD
    tester: str = ""
    scenario: str = ""  # coding/long_doc/retrieval/dialogue/agent/knowledge_qa/other
    task_name: str = ""
    customer_type: str = ""
    model_name: str = ""
    machine_id: str = ""
    engine: str = ""
    usecase_set_version: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    context_length: int | None = None
    concurrency: int | None = None
    ttft_s: float | None = None
    retrieval_latency_s: float | None = None  # 自动采集留空（evaluator 不分阶段）
    prefill_latency_s: float | None = None
    total_latency_s: float | None = None
    decode_tps: float | None = None
    quality_score: float | None = None  # 自动采集留空（需人评/judge）
    success: bool | None = None  # ← SampleResult.is_correct
    citation_score: float | None = None  # 自动采集留空（RAG 引用分）
    tool_success_rate: float | None = None  # 自动采集留空（工具调用成功率）
    privacy_requirement: str = ""
    cost_note: str = ""
    recommended_config: str = ""
    sales_summary: str = ""
    external_level: str = "internal"  # internal / review / publishable
    failure_reason: str = ""
    evidence_path: str = ""
    next_action: str = ""

    # 元数据
    created_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # reasoning_quality/failure_category/...

    def to_dict(self) -> dict[str, Any]:
        """序列化为 application_cases 行（extra → extra_json）。"""
        data = asdict(self)
        extra = data.pop("extra")  # 总是移除 extra，转成 extra_json
        data["extra_json"] = json.dumps(extra, ensure_ascii=False) if extra else None
        if isinstance(self.success, bool):
            data["success"] = int(self.success)
        data["created_at"] = self.created_at.isoformat() if self.created_at else None
        if data.get("external_level") is None:
            data["external_level"] = "internal"
        return data

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ApplicationCase:
        """从数据库行构造。"""
        extra_raw = row.get("extra_json")
        return cls(
            id=row.get("id"),
            case_id=row.get("case_id") or "",
            run_id=row.get("run_id"),
            source=row.get("source") or "manual",
            evaluator_name=row.get("evaluator_name") or "",
            sample_id=row.get("sample_id") or "",
            date=row.get("date") or "",
            tester=row.get("tester") or "",
            scenario=row.get("scenario") or "",
            task_name=row.get("task_name") or "",
            customer_type=row.get("customer_type") or "",
            model_name=row.get("model_name") or "",
            machine_id=row.get("machine_id") or "",
            engine=row.get("engine") or "",
            usecase_set_version=row.get("usecase_set_version") or "",
            input_tokens=row.get("input_tokens"),
            output_tokens=row.get("output_tokens"),
            context_length=row.get("context_length"),
            concurrency=row.get("concurrency"),
            ttft_s=row.get("ttft_s"),
            retrieval_latency_s=row.get("retrieval_latency_s"),
            prefill_latency_s=row.get("prefill_latency_s"),
            total_latency_s=row.get("total_latency_s"),
            decode_tps=row.get("decode_tps"),
            quality_score=row.get("quality_score"),
            success=_parse_bool(row.get("success")),
            citation_score=row.get("citation_score"),
            tool_success_rate=row.get("tool_success_rate"),
            privacy_requirement=row.get("privacy_requirement") or "",
            cost_note=row.get("cost_note") or "",
            recommended_config=row.get("recommended_config") or "",
            sales_summary=row.get("sales_summary") or "",
            external_level=row.get("external_level") or "internal",
            failure_reason=row.get("failure_reason") or "",
            evidence_path=row.get("evidence_path") or "",
            next_action=row.get("next_action") or "",
            created_at=_parse_datetime(row.get("created_at")),
            extra=json.loads(extra_raw) if extra_raw else {},
        )


def _parse_bool(value) -> bool | None:
    """解析布尔（DB 存 0/1/None）。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
