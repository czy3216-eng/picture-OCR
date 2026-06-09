import re

with open(r'C:\Users\Administrator\WorkBuddy\2026-06-08-19-55-08\ocr_results_rapid.txt', 'r', encoding='utf-8') as f:
    content = f.read()

records = re.split(r'\n(?=## )', content)
print(f'总记录: {len(records)}')

colon_prefix = []
xia_xiang = []
short_results = []

for rec in records:
    m = re.match(r'## (\S+)\s+\[', rec)
    if not m:
        continue
    fname = m.group(1)
    parts = rec.split('```')
    if len(parts) < 3:
        continue
    text = parts[1].strip()
    if not text:
        continue
    
    if re.match(r'^[:：]', text):
        colon_prefix.append((fname, text[:40]))
    
    if '霞' in text:
        xia_xiang.append((fname, text[:60]))
    
    if len(text) < 3:
        short_results.append((fname, text))

print(f'冒号前缀问题: {len(colon_prefix)}')
for f, t in colon_prefix[:8]:
    print(f'  {f}: [{t}]')

print(f'\n含"霞"字(可能应是"霰"): {len(xia_xiang)}')
for f, t in xia_xiang[:15]:
    print(f'  {f}: {t}')

print(f'\n结果过短(<3字符): {len(short_results)}')
for f, t in short_results[:10]:
    print(f'  {f}: [{t}]')
