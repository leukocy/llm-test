"""
应用场景映射（手册"模型×应用矩阵"维度）。

手册"测试方法"节要求同时覆盖 **硬件×模型** 与 **模型×应用** 两个矩阵。模型×应用的
四类场景：RAG（引用/检索）、代码（读仓库/改 bug/工具调用）、文档（结构化抽取）、
Agent（多轮/工具链/失败恢复）。本模块把 evaluator 名映射到这些场景，供
application_cases.scenario 字段使用。

映射用包含匹配（鲁棒于注册键拼写差异，如 humaneval vs human_eval）。能力基准类
（MMLU/GSM8K/数学/选择题）不算"应用场景"，但仍入库便于对照，用 is_application_scenario
供 UI 过滤。
"""

from __future__ import annotations

# 手册四类应用场景
APPLICATION_SCENARIOS: frozenset[str] = frozenset(
    {"coding", "long_doc", "retrieval", "dialogue", "agent"}
)

# 包含匹配规则（顺序敏感：先命中先用）
_SCENARIO_RULES: list[tuple[str, str]] = [
    # 代码：执行/工具/仓库
    ("humaneval", "coding"),
    ("human_eval", "coding"),
    ("mbpp", "coding"),
    ("swebench", "coding"),
    ("sw_bench", "coding"),
    # 长文档：QA/摘要/结构化抽取
    ("longbench", "long_doc"),
    ("long_bench", "long_doc"),
    # 检索：Needle Haystack / Custom Needle
    ("needle", "retrieval"),
    # 对答：Arena Hard
    ("arena", "dialogue"),
    # Agent / 工具链（预留）
    ("agent", "agent"),
    ("tool", "agent"),
    # 能力基准（非应用，但记录便于对照）
    ("mmlu", "knowledge_qa"),
    ("ceval", "knowledge_qa"),
    ("cmmlu", "knowledge_qa"),
    ("gsm", "knowledge_qa"),
    ("math", "knowledge_qa"),
    ("aime", "knowledge_qa"),
    ("arc", "knowledge_qa"),
    ("hellaswag", "knowledge_qa"),
    ("wino", "knowledge_qa"),
    ("truthful", "knowledge_qa"),
    ("gpqa", "knowledge_qa"),
    ("piqa", "knowledge_qa"),
]

DEFAULT_SCENARIO = "other"


def scenario_from_dataset(name: str | None) -> str:
    """把 evaluator/dataset 名映射到应用场景。

    用包含匹配（小写）。未命中 → 'other'。
    """
    if not name:
        return DEFAULT_SCENARIO
    key = name.lower()
    for needle, scenario in _SCENARIO_RULES:
        if needle in key:
            return scenario
    return DEFAULT_SCENARIO


def is_application_scenario(scenario: str | None) -> bool:
    """该场景是否属于手册"模型×应用"四类（vs 能力基准/其它）。"""
    return bool(scenario) and scenario in APPLICATION_SCENARIOS
