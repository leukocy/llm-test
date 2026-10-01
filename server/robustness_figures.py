"""Offline sensitivity bars from the persisted report, without remote assets."""

import math
from html import escape
from typing import Any


def sensitivity_svg(sensitivity: dict[str, Any]) -> str:
    rows = sorted(
        (
            (name, value)
            for name, value in sensitivity.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        ),
        key=lambda row: (-row[1], row[0]),
    )
    if not rows:
        return "<p>没有有效的按类型评分数据，不绘制零值图。</p>"
    height = 70 + len(rows) * 38
    bars = []
    for index, (name, value) in enumerate(rows):
        y = 25 + index * 38
        color = "#ef4444" if value > 0.3 else "#eab308" if value > 0.1 else "#22c55e"
        label = escape(name)
        bars.append(
            f"<g><title>{label}：{value:.1%}</title>"
            f'<text x="170" y="{y + 17}" text-anchor="end">{label}</text>'
            f'<rect x="185" y="{y}" width="{value * 400:.2f}" height="24" fill="{color}"/>'
            f'<text x="{195 + value * 400:.2f}" y="{y + 17}">{value:.1%}</text></g>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="鲁棒性扰动敏感性" viewBox="0 0 680 {height}" '
        'style="width:100%;max-width:900px;font:13px system-ui;fill:#334155">'
        "<title>扰动敏感性分析：按类型错误率</title>"
        + "".join(bars)
        + f'<text x="185" y="{height - 14}">0%</text>'
        + f'<text x="585" y="{height - 14}" text-anchor="end">100% · 错误率</text></svg>'
        + "<p>越高越敏感。颜色沿用原版展示阈值：≤10% 绿、10%–30% 黄、&gt;30% 红；"
        "不是显著性或合格标准。</p>"
    )
