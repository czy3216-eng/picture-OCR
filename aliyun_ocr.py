# -*- coding: utf-8 -*-
r"""
Aliyun OCR Batch Image Text Recognition
=========================================
Uses Aliyun Cloud Market OCR API (AppCode authentication).
Concurrent processing with ThreadPoolExecutor.

Speed: ~0.2-0.5s/image, 703 images in ~2-3 minutes.

Usage:
    python aliyun_ocr.py              # Process all images
    python aliyun_ocr.py --limit 5    # Test with first 5 images
    python aliyun_ocr.py --batch 50   # Concurrent requests (default: 50)

Credentials:
    AppCode from Aliyun Cloud Market OCR product.
"""

import sys
import os
import time
import base64
import json
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


# ============== Config ==============
IMAGE_DIR = r"D:\a-test\cutQP"
OUTPUT_FILE = os.path.join(IMAGE_DIR, "ocr_results.txt")
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Aliyun Cloud Market OCR credentials
APP_CODE = "30eab2e9abba4316b226ad6112c326448"

# API endpoint (may need environment-specific override)
API_URL = "https://textanalysis.market.alicloudapi.com/api/ocrapi"

RETRY_COUNT = 3
RETRY_DELAY = 1.0
DEFAULT_BATCH_SIZE = 50
DEFAULT_LIMIT = None
# =====================================


def encode_image_base64(image_path):
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ocr_image(img_path, api_url=None):
    """
    Call Aliyun Market OCR API.
    Sends base64-encoded image, returns recognized text.
    """
    if api_url is None:
        api_url = API_URL

    img_b64 = encode_image_base64(img_path)

    headers = {
        "Authorization": f"APPCODE {APP_CODE}",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }

    data = {"image": img_b64}

    resp = requests.post(
        api_url,
        headers=headers,
        data=data,
        timeout=30,
    )

    if resp.status_code == 403:
        # Detailed auth/quot error
        raise Exception(f"Auth failed or quota exceeded: {resp.text[:200]}")
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")

    result = resp.json()

    # Check error
    error_code = result.get("error_code", -1)
    if error_code != 0:
        msg = result.get("error_msg") or result.get("msg") or str(result)
        raise Exception(f"API error: {msg}")

    # Extract text from various possible response formats
    content = result.get("content") or result.get("retData") or {}

    if isinstance(content, dict):
        # Format 1: prism_wordsList (market API typical response)
        words_list = (
            content.get("prism_wordsList")
            or content.get("wordList")
            or content.get("words_result")
            or []
        )
        if words_list:
            texts = []
            for item in words_list:
                if isinstance(item, dict):
                    t = item.get("word") or item.get("text") or item.get("words") or ""
                    if t.strip():
                        texts.append(t.strip())
                elif isinstance(item, str) and item.strip():
                    texts.append(item.strip())
            if texts:
                return "\n".join(texts)

        # Format 2: Direct text fields
        for key in ("result", "text", "output", "content"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            elif isinstance(val, list) and val:
                texts = []
                for v in val:
                    if isinstance(v, str):
                        texts.append(v.strip())
                    elif isinstance(v, dict):
                        t = v.get("word") or v.get("text") or ""
                        if t.strip():
                            texts.append(t.strip())
                if texts:
                    return "\n".join(texts)

    elif isinstance(content, str) and content.strip():
        return content.strip()

    elif isinstance(content, list) and content:
        texts = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("word") or item.get("text") or ""
                if t.strip():
                    texts.append(t.strip())
            elif isinstance(item, str) and item.strip():
                texts.append(item.strip())
        if texts:
            return "\n".join(texts)

    # Fallback: dump raw
    return json.dumps(result, ensure_ascii=False)[:500]


def process_single(args_tuple):
    """Worker: process one image with retries."""
    idx, total, img_path, api_url = args_tuple
    filename = os.path.basename(img_path)
    t0 = time.time()

    for attempt in range(RETRY_COUNT + 1):
        try:
            text = ocr_image(img_path, api_url=api_url)
            elapsed = time.time() - t0
            return (filename, text, elapsed, True, "")
        except Exception as e:
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
            else:
                elapsed = time.time() - t0
                return (filename, f"[ERROR] {e}", elapsed, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Aliyun OCR Batch Processing")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--url", type=str, default=None,
                        help="Override API endpoint URL")
    args = parser.parse_args()

    api_url = args.url or API_URL

    print(f"\n{'='*55}")
    print(f"  Aliyun OCR Batch (Market API)")
    print(f"  Endpoint: {api_url}")
    print(f"  Source: {IMAGE_DIR}")
    print(f"{'='*55}\n")

    all_images = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ], key=lambda x: os.path.basename(x))

    if args.limit:
        all_images = all_images[:args.limit]

    total = len(all_images)
    print(f"Found {total} images\n")

    if total == 0:
        print("No images found.")
        return

    batch_size = min(args.batch, total)
    print(f"[Config] Concurrency: {batch_size} | Retries: {RETRY_COUNT}")
    print(f"[Config] Output: {OUTPUT_FILE}")
    print("-" * 55)

    results = []
    success_count = 0
    fail_count = 0
    total_start = time.time()
    work_items = [(i + 1, total, path, api_url) for i, path in enumerate(all_images)]

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(process_single, item): item[0] for item in work_items}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            filename, text, elapsed, success, error = future.result()
            results.append((filename, text, elapsed, success))

            if success:
                success_count += 1
                disp = text.replace("\n", "\\n")[:60]
                print(f"[{completed}/{total}] {filename} [{elapsed:.2f}s] OK: {disp}")
            else:
                fail_count += 1
                print(f"[{completed}/{total}] {filename} [{elapsed:.2f}s] FAIL: {error[:100]}")

    total_elapsed = time.time() - total_start

    # Sort by original filename order
    name_order = {os.path.basename(p): i for i, p in enumerate(all_images)}
    results.sort(key=lambda x: name_order.get(x[0], 9999))

    success_times = [t for _, _, t, s in results if s and t > 0]

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# Aliyun OCR Results\n")
        f.write(f"# Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"# Total: {len(results)} | Success: {success_count} | Fail: {fail_count}\n")
        f.write(f"# Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n")
        if success_times:
            f.write(f"# Per image: avg {sum(success_times)/len(success_times):.2f}s | "
                    f"min {min(success_times):.2f}s | max {max(success_times):.2f}s\n")
        f.write("# " + "=" * 54 + "\n\n")
        for fn, txt, t, _ in results:
            f.write(f"## [{fn}] ({t:.2f}s)\n```\n{txt}\n```\n\n")

    # Summary
    print("\n" + "=" * 55)
    print(f"  Done! Success: {success_count} | Failed: {fail_count}")
    print(f"  Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    if success_times:
        print(f"  Per image: avg {sum(success_times)/len(success_times):.2f}s | "
              f"min {min(success_times):.2f}s | max {max(success_times):.2f}s")
        print(f"  Throughput: {total/total_elapsed:.1f} images/sec")
    print("=" * 55)

    ok_results = [r for r in results if r[3]]
    if ok_results:
        print("\n--- Preview (first 10) ---")
        for i, (fn, txt, t, _) in enumerate(ok_results[:10], 1):
            preview = txt.replace("\n", " ")[:60]
            print(f"  {i}. {fn} [{t:.2f}s]: {preview}")


if __name__ == "__main__":
    main()
