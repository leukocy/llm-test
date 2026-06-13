"""
配置指纹（config_hash）—— 手册红线 CASE 02 / #testing“同配置才能承诺”。

手册原话：「销售取数前必须按 machine_id/config_hash 过滤」「用 test_id 和配置 hash
管理，复测要标明 supersede 关系」。config_hash 把定义“同配置”的关键字段压成稳定
短串，用于：判断两次 run 是否同配置、对照分组（同模型不同并发/不同 tp 是否同配置）、
去重与“同配置实测通过”查询。

哪些字段算“同配置”：模型 / 引擎 + 版本 / 并行策略（tp-dp-ep-pp）/ 量化 / 精度 /
max_context。concurrency 不算（同配置可在不同并发下测）。空值统一成空串，保证
None 与 "" 同 hash。
"""

from __future__ import annotations

import hashlib
from typing import Any

# 定义“同配置”的字段集（顺序固定以保证稳定）
CONFIG_HASH_FIELDS: tuple[str, ...] = (
    "model_name",
    "engine",
    "engine_version",
    "parallel_strategy",
    "quantization",
    "dtype",
    "max_context",
)


def _norm(value: Any) -> str:
    """规范化：None → ''，其余 str(value)。保证 None 与 '' 同 hash。"""
    return "" if value is None else str(value)


def compute_config_hash(**fields: Any) -> str:
    """对配置关键字段取 sha1 前 16 位。

    Args:
        任意 CONFIG_HASH_FIELDS 的子集；未传字段视为 ''。

    Returns:
        16 字符 hex 串。
    """
    parts = [f"{name}={_norm(fields.get(name))}" for name in CONFIG_HASH_FIELDS]
    canon = "|".join(parts)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]
