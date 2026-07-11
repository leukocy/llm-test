#!/usr/bin/env python3
"""Final sweep: fix all remaining Chinese characters and punctuation in all UI files."""
import os
import re

ui_dir = "ui"

# Chinese punctuation replacements
PUNCT_MAP = {
    "。": ".",
    "，": ",",
    "！": "!",
    "？": "?",
    "：": ":",
    "；": ";",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
    '"': '"',
    '"': '"',
    """: "'",
    """: "'",
    "、": ",",
}

# Additional word-level fixes
WORD_FIXES = {
    "输入长度 (tokens)": "Input Length (tokens)",
    "# Blue - 缓存": "# Blue - Cache",
    "# Teal - Input/Prefill 相关": "# Teal - Input/Prefill related",
    "# Orange - 输出/Decode 相关": "# Orange - Output/Decode related",
    "# 新增列": "# New column",
    "总Cache Hit Rate": "Overall Cache Hit Rate",
    "# LogLevel分布": "# Log Level Distribution",
    "# TTFT 图表": "# TTFT chart",
    "# TPS 图表": "# TPS chart",
    "# AI Judge 修正记录": "# AI Judge correction records",
}

# Skip files that use Chinese intentionally for backward compat
SKIP_FILES = {"formatters.py", "page_layout.py"}

total_fixes = 0

for fn in sorted(os.listdir(ui_dir)):
    if not fn.endswith(".py") or fn in SKIP_FILES:
        continue

    fp = os.path.join(ui_dir, fn)
    with open(fp, encoding="utf-8") as f:
        content = f.read()

    original = content

    # Apply word-level fixes first
    for old, new in WORD_FIXES.items():
        content = content.replace(old, new)

    # Apply punctuation fixes
    for cn_p, en_p in PUNCT_MAP.items():
        content = content.replace(cn_p, en_p)

    if content != original:
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)

        # Count remaining Chinese chars
        remaining = re.findall(r"[\u4e00-\u9fff]", content)
        cn_segs = re.findall(r"[\u4e00-\u9fff]+", content)
        fixes = len(original) - len(content) + sum(1 for c in original if c != content[0])  # rough
        print(f"📝 {fn}: fixed punctuation/words, {len(cn_segs)} Chinese segments remaining")
        total_fixes += 1
    else:
        cn_segs = re.findall(r"[\u4e00-\u9fff]+", content)
        if cn_segs:
            print(f"⚠️  {fn}: {len(cn_segs)} Chinese segments (no changes applied)")

print("\n--- Final verification ---")
for fn in sorted(os.listdir(ui_dir)):
    if not fn.endswith(".py") or fn in SKIP_FILES:
        continue
    fp = os.path.join(ui_dir, fn)
    with open(fp, encoding="utf-8") as f:
        content = f.read()
    cn_segs = re.findall(r"[\u4e00-\u9fff]+", content)
    cn_punct = re.findall(r"[\u3000-\u303f\uff01-\uff5e]", content)
    status = (
        "✅" if not cn_segs and not cn_punct else f"⚠️  chars:{len(cn_segs)} punct:{len(cn_punct)}"
    )
    print(f"  {status} {fn}")
