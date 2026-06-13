"""
客户能力表（Capability Sheet）—— 手册核心产出层。

手册 #examples 例子4 + 角色表：「销售可以自动生成客户能力表」「产品牵头交付客户
能力表」。「能在 10 分钟内从仓库抽出客户可用材料，才算体系成立。」

把 application_cases（模型×应用用例）按 customer_type × scenario × model_name
聚合成一行 = 一个客户场景的能力切片：成功率、质量分、性能、对外口径、sales_summary。
纯函数（无 Streamlit），可测、可导出。
"""

from __future__ import annotations

from typing import Any

from core.models import ApplicationCase

# 对外口径等级排序（数值越大越可对外）
_EXTERNAL_LEVEL_RANK = {"internal": 0, "review": 1, "publishable": 2}

# 客户能力表列（导出口径）
CAPABILITY_COLUMNS: list[str] = [
    "customer_type", "scenario", "model_name", "case_count",
    "success_rate", "avg_quality_score", "avg_decode_tps",
    "external_level", "sales_summary", "evidence_count", "tasks",
]


def _max_external_level(levels: list[str]) -> str:
    """取一组 external_level 里最可对外的一个（publishable > review > internal）。"""
    if not levels:
        return "internal"
    return max(levels, key=lambda lv: _EXTERNAL_LEVEL_RANK.get(lv, 0))


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 3) if nums else None


def build_capability_sheet(
    cases: list[ApplicationCase],
    group_by: tuple[str, ...] = ("customer_type", "scenario", "model_name"),
    min_external_level: str = "internal",
) -> list[dict[str, Any]]:
    """把应用用例聚合成客户能力表行。

    Args:
        cases: ApplicationCase 列表（通常来自 db.list_application_cases）。
        group_by: 聚合维度（默认客户类型 × 场景 × 模型）。
        min_external_level: 只保留对外口径 ≥ 此等级的组（internal=全部）。

    Returns:
        每组一行 dict（键见 CAPABILITY_COLUMNS）；按 customer_type / scenario 排序。
    """
    min_rank = _EXTERNAL_LEVEL_RANK.get(min_external_level, 0)

    # 分组
    groups: dict[tuple, list[ApplicationCase]] = {}
    for c in cases:
        key = tuple(getattr(c, dim, "") or "" for dim in group_by)
        groups.setdefault(key, []).append(c)

    rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        levels = [(c.external_level or "internal") for c in group]
        best_level = _max_external_level(levels)
        if _EXTERNAL_LEVEL_RANK.get(best_level, 0) < min_rank:
            continue  # 该组对外口径不够，跳过

        successes = [c.success for c in group if c.success is not None]
        success_rate = (
            round(sum(1 for s in successes if s) / len(successes), 3) if successes else None
        )
        sales = next((c.sales_summary for c in group if c.sales_summary), "")
        tasks = sorted({c.task_name for c in group if c.task_name})

        row = dict(zip(group_by, key, strict=False))
        row.update({
            "case_count": len(group),
            "success_rate": success_rate,
            "avg_quality_score": _mean([c.quality_score for c in group]),
            "avg_decode_tps": _mean([c.decode_tps for c in group]),
            "external_level": best_level,
            "sales_summary": sales,
            "evidence_count": sum(1 for c in group if c.evidence_path),
            "tasks": ", ".join(tasks) if tasks else "",
        })
        rows.append(row)

    # 排序：customer_type → scenario → model_name → external_level 降序
    rows.sort(key=lambda r: (
        str(r.get("customer_type", "")),
        str(r.get("scenario", "")),
        str(r.get("model_name", "")),
        -_EXTERNAL_LEVEL_RANK.get(r.get("external_level", "internal"), 0),
    ))
    return rows


def build_capability_markdown(sheet: list[dict[str, Any]]) -> str:
    """把客户能力表渲染成对外 markdown（按 customer_type 分节）。"""
    if not sheet:
        return "# 客户能力表\n\n（暂无可用数据——跑应用用例测试或手动录入后会生成。）\n"

    lines = ["# 客户能力表", ""]
    lines.append("> 数据源：数据仓库 application_cases；只含 external_level 达标的切片。")
    lines.append("> 缺测项记为 —（手册：缺测本身就是决策信息）。")
    lines.append("")

    # 按 customer_type 分节
    by_customer: dict[str, list[dict[str, Any]]] = {}
    for row in sheet:
        by_customer.setdefault(str(row.get("customer_type") or "未分类"), []).append(row)

    for customer, group in by_customer.items():
        lines.append(f"## {customer}")
        lines.append("")
        lines.append("| 场景 | 模型 | 用例数 | 成功率 | 质量分 | decode TPS | 对外 | sales_summary |")
        lines.append("|------|------|-------:|-------:|-------:|----------:|------|---------------|")
        for r in group:
            def _cell(v, pct=False):
                if v is None:
                    return "—"
                return f"{v:.1%}" if pct else str(v)
            lines.append(
                f"| {r.get('scenario', '—')} | {r.get('model_name', '—')} | "
                f"{r.get('case_count', 0)} | {_cell(r.get('success_rate'), pct=True)} | "
                f"{_cell(r.get('avg_quality_score'))} | {_cell(r.get('avg_decode_tps'))} | "
                f"{r.get('external_level', '—')} | {r.get('sales_summary', '') or '—'} |"
            )
        lines.append("")

    return "\n".join(lines)
