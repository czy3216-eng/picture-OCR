"""
Test enhanced OCR prompt - round 2 with correct expected values.
"""
import base64, json, time, os, requests, re

# Enhanced prompt with very explicit rules
PROMPT = (
    "请提取图片中的白色字符部分，必须与图片完全一致。\n\n"
    "规则1 - 英文大小写严格保持：图片中的大写必须输出大写，小写必须输出小写。\n"
    "   Vector-冲锋枪 不能输出为 vector-冲锋枪 或 VECTOR-冲锋枪。\n"
    "   SR-3M紧凑突击步枪 不能输出为 sr-3m紧凑突击步枪。\n\n"
    "规则2 - 保留所有符号，不能省略：连字符-、间隔号·、数字、英文全部保留。\n"
    "   SR-3M紧凑突击步枪-变色龙 不能输出为 sr3m紧凑突击步枪变色龙。\n\n"
    "规则3 - 不要添加图片中不存在的前缀符号（如箭头、冒号）或后缀符号。\n\n"
    "输出格式（每张图片一行）：\n"
    "图N: 文字内容\n"
    "无文字时输出：图N: [无文字]"
)

# Correctly matched test images (verified by reading actual images)
TEST_IMAGES = [
    (r"D:\a-test\cutQP\枪皮_11.jpg", "SR-3M紧凑突击步枪-暗星"),
    (r"D:\a-test\cutQP\枪皮_12.jpg", "SKS射手步枪-骑士"),
    (r"D:\a-test\cutQP\枪皮_52.jpg", "勇士冲锋枪-能天使-午夜邮差"),
    (r"D:\a-test\cutQP\枪皮_437.jpg", "M1014霰弹枪-变色龙"),
    (r"D:\a-test\cutQP\枪皮_14.jpg", None),  # unknown - let's discover
]

print("=" * 60)
print("  Enhanced Prompt Test Round 2")
print("=" * 60)

images_b64 = []
for path, expected in TEST_IMAGES:
    with open(path, "rb") as f:
        images_b64.append(base64.b64encode(f.read()).decode())
    fn = os.path.basename(path)
    print(f"  {fn} -> expected: {expected}")

print(f"\nSending {len(TEST_IMAGES)} images to qwen3-vl:8b...")
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

# Save
out_file = r"C:\Users\Administrator\WorkBuddy\2026-06-08-19-55-08\test_prompt_output2.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(f"# Test Output ({elapsed:.1f}s)\n\n")
    f.write(raw_text)

# Parse
parsed = {}
for m in re.finditer(r"图(\d+):\s*(.+)", raw_text):
    idx = int(m.group(1)) - 1
    text = m.group(2).strip().replace("[无文字]", "")
    parsed[idx] = text

print(f"\n{'='*60}")
print(f"Results ({elapsed:.1f}s):")
print(f"{'='*60}")

all_ok = True
for i, (path, expected) in enumerate(TEST_IMAGES):
    fn = os.path.basename(path)
    actual = parsed.get(i, "[MISSING]")
    if expected and actual == expected:
        print(f"  [OK]   {fn}: {actual}")
    elif expected:
        all_ok = False
        print(f"  [FAIL] {fn}: expected={repr(expected)}, actual={repr(actual)}")
    else:
        print(f"  [NEW]  {fn}: {actual}")

print(f"\nOverall: {'ALL OK' if all_ok else 'SOME FAILURES'} | {elapsed:.1f}s")
print(f"Per image avg: {elapsed/len(TEST_IMAGES):.1f}s")

# Compare with RapidOCR
print("\n--- RapidOCR Comparison ---")
import importlib.util
spec = importlib.util.spec_from_file_location("rapid_ocr",
    r"C:\Users\Administrator\WorkBuddy\2026-06-08-19-55-08\rapid_ocr.py")
if spec:
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    engine = mod.init_engine()
    for i, (path, expected) in enumerate(TEST_IMAGES):
        fn = os.path.basename(path)
        t0 = time.time()
        text = mod.ocr_image(engine, path)
        t = time.time() - t0
        text = text.replace(" | ", "")
        print(f"  RapidOCR [{t:.2f}s] {fn}: {text}")
