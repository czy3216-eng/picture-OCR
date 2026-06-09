"""
RapidOCR 结果后处理脚本
修正常见识别错误，输出带时间戳的结果文件
"""

import re
import datetime

INPUT_FILE = r'C:\Users\Administrator\WorkBuddy\2026-06-08-19-55-08\ocr_results_rapid.txt'
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_FILE = rf'C:\Users\Administrator\WorkBuddy\2026-06-08-19-55-08\ocr_results_rapid_{TIMESTAMP}.txt'

# 修正规则

# 1. 冒号前缀
COLON_PREFIX_RE = re.compile(r'^[:：]\s*')

# 2. 字符替换表
CHAR_REPLACE = {
    '霞': '霰',
}

# 3. 常见武器名拼写修正
WEAPON_FIXES = {
    '突击步枪': '突击步枪',
    '精确射手步枪': '精确射手步枪',
    '冲锋枪': '冲锋枪',
    '狙击枪': '狙击枪',
    '射手步枪': '射手步枪',
}


def fix_text(text):
    if not text:
        return text
    original = text
    
    # 规则1：去掉冒号前缀
    text = COLON_PREFIX_RE.sub('', text)
    
    # 规则2：字符替换
    for bad, good in CHAR_REPLACE.items():
        text = text.replace(bad, good)
    
    # 规则3：修复常见武器名错误
    for bad, good in WEAPON_FIXES.items():
        text = text.replace(bad, good)
    
    return text.strip()


def process_file():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按记录分割（## 开头）
    records = re.split(r'(?=^## )', content, flags=re.MULTILINE)
    
    fix_count_colon = 0
    fix_count_char = 0
    total_fixed = 0
    
    result_records = []
    for rec in records:
        rec = rec.strip()
        if not rec:
            continue
        # 保留头部注释
        if rec.startswith('#') and not rec.startswith('## '):
            result_records.append(rec)
            continue
        
        # 处理 ## 开头的记录
        if rec.startswith('## '):
            # 提取 ``` 之间的内容
            parts = rec.split('```')
            if len(parts) >= 3:
                header = parts[0].rstrip('\n')
                ocr_text = parts[1].strip()
                footer = '```'.join(parts[2:]).lstrip('\n')
                
                fixed_text = fix_text(ocr_text)
                
                if ocr_text != fixed_text:
                    total_fixed += 1
                    if COLON_PREFIX_RE.search(ocr_text):
                        fix_count_colon += 1
                    if any(c in ocr_text for c in CHAR_REPLACE):
                        fix_count_char += 1
                
                new_rec = header + '\n```\n' + fixed_text + '\n```'
                if footer.strip():
                    new_rec += '\n' + footer
                result_records.append(new_rec)
            else:
                result_records.append(rec)
        else:
            result_records.append(rec)
    
    # 写回
    output = '\n\n'.join(result_records)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f'处理完成！')
    print(f'输入: {INPUT_FILE}')
    print(f'输出: {OUTPUT_FILE}')
    print(f'修正统计:')
    print(f'  冒号前缀修正: {fix_count_colon} 条')
    print(f'  字符替换修正: {fix_count_char} 条')
    print(f'  总修正条数: {total_fixed}')


if __name__ == '__main__':
    process_file()
