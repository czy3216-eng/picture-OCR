"""
Test enhanced OCR prompt for accuracy with case/symbols.
"""
import base64, json, time, os, requests

# Enhanced prompt with very explicit rules
PROMPT = (
    "请提取图片中的白色字符部分。"
    "字符提取必须与图片中完全一致，严格遵循以下规则：\n\n"
    "1. 英文大小写必须与图片完全一致，不要转换大小写。\n"
    "   例如图片中为 Vector-冲锋枪，必须输出 Vector-冲锋枪，\n"
    "   不能输出 vector-冲锋枪 或 VECTOR-冲锋枪。\n\n"
    "2. 保留所有符号，特别是连字符-和间隔号·，不能省略或替换。\n"
    "   例如 SR-3M紧凑突击步枪-变色龙 不能输出 sr-3m紧凑突击步枪-变色龙 或 sr3m紧凑突击步枪变色龙。\n"
    "   勇士冲锋枪-能天使·午夜邮差 不能输出 勇士冲锋枪能天使午夜邮差 或 勇士冲锋枪-能天使-午夜邮差。\n\n"
    "3. 不要添加额外的符号前缀（如箭头、冒号）或后缀（如破折号）。\n"
    "4. 中文和数字之间不要添加或删除空格。\n\n"
    "输出格式：每张图片一行，不要额外解释。\n"
    "图N: 文字内容\n"
    "如果某张图片没有文字，输出：图N: [无文字]"
)

# Test images with known formatting challenges
TEST_IMAGES = [
    (r"D:\a-test\cutQP\枪皮_52.jpg", "勇士冲锋枪-能天使\u00b7午夜邮差"),  # middle dot
    (r"D:\a-test\cutQP\枪皮_55.jpg", "勇士冲锋枪-能天使\u00b7午夜邮差"),  # middle dot (duplicate)
    (r"D:\a-test\cutQP\枪皮_11.jpg", "AUG突击步枪-西装暴徒"),            # - with Chinese
    (r"D:\a-test\cutQP\枪皮_12.jpg", "MK47突击步枪-暗星"),               # letters+numbers
    (r"D:\a-test\cutQP\枪皮_437.jpg", "M1014霰弹枪-变色龙"),             # colon prefix issue
]

print("=" * 60)
print("  Testing Enhanced OCR Prompt (qwen3-vl:8b)")
print("=" * 60)

# Encode all test images
images_b64 = []
filenames = []
for path, expected in TEST_IMAGES:
    with open(path, "rb") as f:
        images_b64.append(base64.b64encode(f.read()).decode())
    filenames.append(os.path.basename(path))

print(f"\nTest images: {len(TEST_IMAGES)}")
for fn in filenames:
    print(f"  - {fn}")

print(f"\nSending to Ollama...")
t0 = time.time()
resp = requests.post(
    "http://127.0.0.1:11434/api/chat",
    json={
        "model": "qwen3-vl:8b",
        "messages": [{"role": "user", "content": PROMPT, "images": images_b64}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1024},
    },
    timeout=300,
)
elapsed = time.time() - t0
raw_text = resp.json()["message"]["content"]

# Save raw output
with open(r"C:\Users\Administrator\WorkBuddy\2026-06-08-19-55-08\test_prompt_output.txt",
          "w", encoding="utf-8") as f:
    f.write(f"# Test Output ({elapsed:.1f}s)\n")
    f.write(f"# Prompt length: {len(PROMPT)} chars\n\n")
    f.write(raw_text)

# Parse
import re
parsed = {}
for m in re.finditer(r"图(\d+):\s*(.+)", raw_text):
    idx = int(m.group(1)) - 1
    text = m.group(2).strip()
    parsed[idx] = text

print(f"\n{'='*60}")
print(f"Results ({elapsed:.1f}s total):")
print(f"{'='*60}")

all_ok = True
for i, (path, expected) in enumerate(TEST_IMAGES):
    fn = os.path.basename(path)
    actual = parsed.get(i, "[NOT FOUND]")
    actual = actual.replace("[无文字]", "")
    match = "CORRECT" if actual == expected else "WRONG"
    if actual != expected:
        all_ok = False
    print(f"\n  [{match}] {fn}")
    print(f"    Expected: {expected}")
    print(f"    Actual:   {actual}")

print(f"\n{'='*60}")
print(f"  Overall: {'ALL OK' if all_ok else 'SOME ERRORS'} | Time: {elapsed:.1f}s")
print(f"{'='*60}")
