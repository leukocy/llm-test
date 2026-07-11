"""
可对外闸门（Publish Gate）—— 手册 #redlines 的“四道闸”机制。

一条数据能不能对外，先过四道闸：
1. 配置完整：tester 已填 + machine_id 存在（硬件指纹已冻结）+ CASE 03 强制字段
   （PCIe Gen/宽度、内存通道数、内存频率）非空（仅当传入硬件指纹时校验）。
2. 可复现：machine_id + 硬件指纹 + 随机种子已记录（或明确标注随机）。
3. 指标可信：无 ❌ 关键问题 + 成功率达标 + 资源监控已记录。
4. 人工复核：external_level 已被人工置为 'publishable'（永不自动提升）。

level 推导：publishable 需四闸全过；1-3 过 → review；否则 internal。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.test_attribution import _has_critical


def missing_required_hw_fields(fp: dict[str, Any] | None) -> list[str]:
    """检查 CASE 03 红线强制字段，返回缺失项的中文名列表。

    fp 为 None/空 → 返回 []（视为无指纹，不校验；由闸2 has_hardware_fingerprint 兜底）。
    """
    if not fp:
        return []
    gpus = fp.get("gpus") or []
    gpu0 = gpus[0] if gpus else {}
    mem = fp.get("memory") or {}

    checks = {
        "PCIe Gen": gpu0.get("pcie_gen"),
        "PCIe 宽度": gpu0.get("pcie_width"),
        "内存通道数": mem.get("channels"),
        "内存频率": mem.get("speed_mt_s"),
    }
    return [name for name, val in checks.items() if val in (None, "", 0)]


@dataclass
class GateResult:
    """四道闸评估结果。"""

    level: str  # "internal" | "review" | "publishable"
    passed: bool  # 是否达到 publishable
    gates: dict[str, bool] = field(default_factory=dict)  # 每道闸的通过情况
    reasons: list[str] = field(default_factory=list)  # 未通过的原因（人话）

    @property
    def gate_all(self) -> bool:
        return all(self.gates.values())


def evaluate_publish_gate(
    *,
    tester: str | None,
    machine_id: str | None,
    has_hardware_fingerprint: bool,
    seed_recorded: bool,
    insights: list[str] | None,
    success_rate: float | None,
    has_monitor: bool,
    requested_external_level: str = "internal",
    success_rate_threshold: float = 0.95,
    hardware_fingerprint: dict[str, Any] | None = None,
    sample_size: int | None = None,
    min_sample_size: int = 30,
    success_count: int | None = None,
    use_wilson: bool = False,
) -> GateResult:
    """评估四道闸。requested_external_level 为当前（用户设定的）external_level。

    hardware_fingerprint 非空时，闸1 额外校验 CASE 03 强制字段（PCIe/通道/频率）；
    为 None（如远程 API 压测机或旧调用）则仅校验 tester + machine_id，保持向后兼容。

    统计严谨性（补手册漏洞：小样本高成功率不可信）：
    - sample_size 传入时，闸3 要求样本量 ≥ min_sample_size（默认 30）。
    - use_wilson=True 且提供 success_count + sample_size 时，用 Wilson 95% 置信下界
      替代裸 success_rate 判阈值——小样本下界自动降低，更诚实。
    - sample_size=None（旧调用）不触发样本量校验，保持向后兼容。
    """
    reasons: list[str] = []

    # 闸 1：配置完整（含 CASE 03 强制字段）
    g1_config = bool(tester and tester.strip()) and bool(machine_id)
    if not (tester and tester.strip()):
        reasons.append("缺 tester（测试人）")
    if not machine_id:
        reasons.append("缺 machine_id（硬件指纹未冻结）")
    if hardware_fingerprint:  # CASE 03：有指纹才校验强制字段
        missing = missing_required_hw_fields(hardware_fingerprint)
        if missing:
            g1_config = False
            reasons.append(f"缺 CASE 03 强制字段: {', '.join(missing)}")

    # 闸 2：可复现
    g2_repro = bool(machine_id) and has_hardware_fingerprint and seed_recorded
    if not has_hardware_fingerprint:
        reasons.append("缺结构化硬件指纹")
    if not seed_recorded:
        reasons.append("未记录随机种子（不可复现）")

    # 闸 3：指标可信（含统计严谨性）
    no_critical = not _has_critical(insights or [])

    # 成功率可信度：Wilson 下界（可选）或裸 success_rate
    wilson_lower: float | None = None
    if use_wilson and success_count is not None and sample_size:
        try:
            from core.metrics import wilson_score_interval

            wilson_lower, _ = wilson_score_interval(success_count, sample_size, 0.95)
        except Exception:  # noqa: BLE001
            wilson_lower = None

    if wilson_lower is not None:
        sr_ok = wilson_lower >= success_rate_threshold
        sr_label = f"Wilson 95% 下界 {wilson_lower:.0%}"
    else:
        sr_ok = success_rate is None or success_rate >= success_rate_threshold
        sr_label = (
            f"成功率 {success_rate:.0%}" if success_rate is not None else "成功率"
        )

    # 样本量门槛（仅当 sample_size 传入）
    enough_samples = sample_size is None or sample_size >= min_sample_size

    g3_metrics = no_critical and sr_ok and has_monitor and enough_samples
    if not no_critical:
        reasons.append("存在 ❌ 关键问题（指标不可信）")
    if not sr_ok:
        reasons.append(f"{sr_label} 低于阈值 {success_rate_threshold:.0%}")
    if not enough_samples and sample_size is not None:
        reasons.append(
            f"样本量 {sample_size} 不足（建议≥{min_sample_size}，成功率不可信）"
        )
    if not has_monitor:
        reasons.append("缺资源监控数据")

    # 闸 4：人工复核（永不自动提升）
    g4_reviewed = requested_external_level == "publishable"
    if not g4_reviewed:
        reasons.append("未经人工复核（external_level 非 publishable）")

    gates = {
        "config_complete": g1_config,
        "reproducible": g2_repro,
        "metrics_trustworthy": g3_metrics,
        "external_reviewed": g4_reviewed,
    }

    g123 = g1_config and g2_repro and g3_metrics
    if g123 and g4_reviewed:
        level = "publishable"
    elif g123:
        level = "review"
    else:
        level = "internal"

    return GateResult(
        level=level,
        passed=(level == "publishable"),
        gates=gates,
        reasons=reasons,
    )


def gate_from_run(run: dict[str, Any], **extra) -> GateResult:
    """从 TestRun 的 to_dict / DB row 便捷评估。

    extra 可覆盖 insights / success_rate / has_monitor 等运行期信号。
    """
    sys_info = run.get("system_info") or {}
    if isinstance(sys_info, str):
        import json as _json

        try:
            sys_info = _json.loads(sys_info or "{}")
        except (ValueError, TypeError):
            sys_info = {}
    config = run.get("config") or {}
    if isinstance(config, str):
        import json as _json

        try:
            config = _json.loads(config or "{}")
        except (ValueError, TypeError):
            config = {}

    return evaluate_publish_gate(
        tester=run.get("tester") or extra.get("tester"),
        machine_id=run.get("machine_id") or sys_info.get("machine_id"),
        has_hardware_fingerprint=bool(sys_info.get("hardware_fingerprint")),
        seed_recorded=extra.get("seed_recorded", config.get("random_seed") is not None),
        insights=extra.get("insights"),
        success_rate=extra.get("success_rate", run.get("success_rate")),
        has_monitor=extra.get("has_monitor", bool(run.get("resource_monitor_json"))),
        requested_external_level=run.get("external_level") or "internal",
        hardware_fingerprint=sys_info.get("hardware_fingerprint"),
        sample_size=extra.get(
            "sample_size",
            run.get("completed_requests") or run.get("total_requests"),
        ),
        success_count=extra.get("success_count"),
        use_wilson=extra.get("use_wilson", False),
    )


GATE_LABELS = {
    "config_complete": "配置完整",
    "reproducible": "可复现",
    "metrics_trustworthy": "指标可信",
    "external_reviewed": "人工复核",
}

LEVEL_BADGE = {
    "publishable": ("✅ 可对外", "green"),
    "review": ("🟡 待复核", "orange"),
    "internal": ("⚪ 内部", "gray"),
}
