"""Final comprehensive verification."""

import os
import re

compat = {"formatters.py", "page_layout.py"}
ui_dir = "ui"

clean = 0
total = 0
issues = []

for fn in sorted(os.listdir(ui_dir)):
    if not fn.endswith(".py"):
        continue
    total += 1
    fp = os.path.join(ui_dir, fn)
    with open(fp, encoding="utf-8") as f:
        content = f.read()
    cn_segs = re.findall(r"[\u4e00-\u9fff]+", content)
    cn_punct = re.findall(r"[\u3000-\u303f\uff01-\uff5e]", content)

    if fn in compat:
        print(f"  {fn}: {len(cn_segs)} Chinese segments (backward-compat, intentional)")
    elif cn_segs or cn_punct:
        print(f"   {fn}: chars={len(cn_segs)}, punct={len(cn_punct)}")
        issues.append(fn)
    else:
        print(f"  {fn}")
        clean += 1

print(f"\n{'='*50}")
print(f"Results: {clean}/{total-len(compat)} UI files fully clean")
print(f"Backward-compat files: {len(compat)} (intentional Chinese)")
print(f"Issues remaining: {len(issues)}")
if issues:
    print(f"  Files with issues: {issues}")
else:
    print("ALL UI FILES FULLY TRANSLATED!")
