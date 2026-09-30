"""Quality selection groups retained from the first application revision."""

from evaluators import list_available_datasets

GROUPS = {
    "基础": ["mmlu", "gsm8k", "math500", "humaneval", "ceval"],
    "推理": ["gpqa", "arc", "truthfulqa", "winogrande", "hellaswag"],
    "代码与扩展": ["mbpp", "aime2025", "longbench", "arena_hard", "global_piqa"],
    "特殊": ["swebench_lite", "needle_haystack", "custom_needle"],
}


def quality_catalog() -> list[dict]:
    registered = set(list_available_datasets())
    items = [
        {"id": name, "group": group, "available": name in registered}
        for group, names in GROUPS.items()
        for name in names
    ]
    initial = {item["id"] for item in items}
    items.extend(
        {"id": name, "group": "其他已注册", "available": True}
        for name in sorted(registered - initial)
    )
    return items
