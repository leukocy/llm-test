"""Check backward compat keys in formatters and page_layout."""
import re

# Check formatters.py for backward-compat Chinese keys
with open('ui/formatters.py', encoding='utf-8') as f:
    content = f.read()

# Find Chinese test type keys (they should still be there as dict keys for backward compat)
cn_keys = re.findall(r'[\u4e00-\u9fff]+', content)
print(f"formatters.py: {len(cn_keys)} Chinese key fragments found")
if cn_keys:
    print(f"  Fragments: {cn_keys[:10]}")

# Check page_layout.py
with open('ui/page_layout.py', encoding='utf-8') as f:
    content = f.read()

cn_keys = re.findall(r'[\u4e00-\u9fff]+', content)
print(f"page_layout.py: {len(cn_keys)} Chinese key fragments found")
if cn_keys:
    print(f"  Fragments: {cn_keys[:10]}")
