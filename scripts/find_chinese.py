import re, os

results = []
for root, _, files in os.walk('.'):
    if '.git' in root or '__pycache__' in root or '.pytest_cache' in root or 'node_modules' in root:
        continue
    for f in files:
        if f.endswith('.py') and f not in ('find_chinese.py', 'translate_reports.py'):
            path = os.path.join(root, f)
            try:
                with open(path, encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
                if re.search(r'[\u4e00-\u9fff]', content):
                    count = len(re.findall(r'[\u4e00-\u9fff]+', content))
                    results.append((path, count))
            except:
                pass

results.sort(key=lambda x: -x[1])
with open('chinese_files.txt', 'w', encoding='utf-8') as f:
    for path, count in results:
        f.write(f'{count:4d} | {path}\n')
    f.write(f'\nTotal: {len(results)} files\n')

print(f'Found {len(results)} files with Chinese text. Written to chinese_files.txt')
