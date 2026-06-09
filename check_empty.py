import re

with open(r'D:\a-test\cutQP\ocr_results_rapid.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 按 ## 记录分割
records = re.split(r'\n(?=## )', content)

empty = []
for rec in records:
    m = re.match(r'## (\S+)\s+\[', rec)
    if not m:
        continue
    fname = m.group(1)
    # 提取 ``` 之间的内容
    parts = rec.split('```')
    if len(parts) >= 3:
        body = parts[1].strip()
        if body == '':
            empty.append(fname)
    elif '```' not in rec:
        empty.append(fname)

print(f"空结果图片数: {len(empty)}")
for f in empty:
    print(f"  {f}")
