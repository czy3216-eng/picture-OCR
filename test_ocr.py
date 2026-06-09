import sys
sys.stdout.reconfigure(encoding='utf-8')

"""Ollama qwen3-vl:8b OCR test - single image (streaming)"""
import base64, json, requests, time

img_path = r"D:\a-test\cutQP\1.jpg"

print(f"[Test] Image: {img_path}")
img_b64 = base64.b64encode(open(img_path, "rb").read()).decode()

payload = {
    "model": "qwen3-vl:8b",
    "prompt": "请识别图片中的所有文字，只输出文字内容，不要解释。",
    "images": [img_b64],
    "stream": True,
}

t0 = time.time()
print("[Test] Calling OCR (streaming, first load may take 2-5min)...")
resp = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json=payload,
    stream=True,
    timeout=(30, 600)  # 30s connect, 600s read
)

full_text = ""
for line in resp.iter_lines(decode_unicode=True):
    if line:
        data = json.loads(line)
        chunk = data.get("response", "")
        full_text += chunk
        if data.get("done", False):
            break

elapsed = time.time() - t0
print(f"[OK] ({elapsed:.1f}s) Result: {full_text.strip()}")
