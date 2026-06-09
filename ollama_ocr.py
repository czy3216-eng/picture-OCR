r"""
Ollama Qwen3-VL 批量图片OCR Demo (批次模式)
===========================================
使用 ollama 本地部署的 qwen3-vl:8b 模型，
批量识别 D:\a-test\cutQP 文件夹中的图片文字。

核心优化：每批次发送多张图片到一个 API 请求，
         Qwen3-VL 一次性处理，无需 OLLAMA_NUM_PARALLEL。
         703 张图 → 每批 5 张 → 仅 141 次 API 调用。

用法：
    python ollama_ocr.py                    # 每批5张，处理全部
    python ollama_ocr.py --batch 8          # 每批8张（更快但可能超出上下文）
    python ollama_ocr.py --batch 1          # 单张模式（调试用）
    python ollama_ocr.py --limit 10         # 仅处理前10张
    python ollama_ocr.py --limit 20 --batch 4  # 组合使用
"""

import base64
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import requests

# ======================== 配置 ========================
OLLAMA_API = "http://127.0.0.1:11434"
MODEL = "qwen3-vl:8b"
IMAGE_DIR = r"D:\a-test\cutQP"
OUTPUT_FILE = r"D:\a-test\cutQP\ocr_results.txt"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# API 参数
TIMEOUT_PER_BATCH = 300    # 每批次超时（多张图推理更久）
RETRY_COUNT = 2
RETRY_DELAY = 3
DEFAULT_BATCH_SIZE = 5     # 每批图片数量

# 批次 OCR 提示词
BATCH_OCR_PROMPT = (
    "请提取图片中的白色字符部分。"
    "字符提取必须与图片中完全一致，严格遵循以下规则：\n"
    "1. 英文大小写必须与图片一致，不要转换大小写。\n"
    "   例如图片中为'Vector-冲锋枪'，必须输出'Vector-冲锋枪'，\n"
    "   不能输出'vector-冲锋枪'或'VECTOR-冲锋枪'。\n"
    "2. 保留所有符号，特别是连字符'-'和间隔号'·'，不能省略或替换。\n"
    "   例如'SR-3M紧凑突击步枪-变色龙'不能输出'sr-3m紧凑突击步枪-变色龙'或'sr3m紧凑突击步枪变色龙'。\n"
    "   '勇士冲锋枪-能天使·午夜邮差'不能输出'勇士冲锋枪能天使午夜邮差'。\n"
    "3. 中文和数字之间不要添加或删除空格。\n"
    "4. 严格按格式输出，每张图片一行：\n"
    "   图N: 文字内容\n"
    "5. 如果某张图片没有文字，输出：图N: [无文字]\n"
    "6. 不要输出任何额外解释。"
)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m{s:.0f}s"


def check_ollama_service():
    try:
        resp = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        if resp.status_code == 200:
            return True, resp.json().get("models", [])
        return False, []
    except requests.ConnectionError:
        return False, []


def check_model_installed(model_name):
    ok, models = check_ollama_service()
    if not ok:
        return False
    model_names = [m.get("name", "") for m in models]
    return any(m.startswith(model_name.split(":")[0]) for m in model_names)


def pull_model(model_name):
    print(f"\n   Model {model_name} not found, pulling...")
    try:
        with requests.post(
            f"{OLLAMA_API}/api/pull",
            json={"name": model_name, "stream": True},
            stream=True,
            timeout=None,
        ) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    total = data.get("total", "")
                    completed = data.get("completed", "")
                    if total and completed.isdigit() and total.isdigit():
                        pct = int(completed) / int(total) * 100
                        print(f"\r   {status}: {pct:.1f}%", end="", flush=True)
                    else:
                        print(f"\r   {status}          ", end="", flush=True)
        print("\n   Model pulled successfully!")
        return True
    except Exception as e:
        print(f"\n   Pull failed: {e}")
        print(f"   Please run manually: ollama pull {model_name}")
        return False


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ocr_batch(image_paths: list[str], filenames: list[str], model: str, batch_no: int, total_batches: int) -> list[dict]:
    """
    一次 API 请求处理多张图片。
    返回 [{filename, text, elapsed}, ...]
    """
    # 编码所有图片
    images_b64 = [encode_image_base64(p) for p in image_paths]

    # 构建提示词（告诉模型有几张图）
    prompt = BATCH_OCR_PROMPT

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256 * len(image_paths),  # 每张图预留256 token
        },
    }

    batch_start = time.time()
    resp = requests.post(
        f"{OLLAMA_API}/api/chat",
        json=payload,
        timeout=TIMEOUT_PER_BATCH,
    )
    resp.raise_for_status()
    result = resp.json()
    raw_text = result["message"]["content"].strip()
    batch_elapsed = time.time() - batch_start

    # 解析模型输出：图1: xxx \n 图2: xxx
    parsed = parse_batch_output(raw_text, filenames)

    results = []
    # 拆分为每张图的结果
    for i, fname in enumerate(filenames):
        text = parsed.get(i, f"[PARSE ERROR] {raw_text[:80]}")
        results.append({
            "filename": fname,
            "text": text,
            "elapsed": batch_elapsed / len(filenames),  # 平均分配到每张
        })

    return results, raw_text, batch_elapsed


def parse_batch_output(raw_text: str, filenames: list[str]) -> dict[int, str]:
    """
    解析模型批量输出，返回 {图片索引(0-based): 文字内容}
    支持多种输出格式：
      图1: xxx
      图1：xxx
      1: xxx
      [1] xxx
    """
    result = {}

    # 尝试按 "图N:" 或 "图N：" 分割
    pattern = r'(?:图\s*)?(\d+)\s*[:：]\s*(.*?)(?=(?:图\s*\d+\s*[:：]|\Z))'
    matches = re.findall(pattern, raw_text + "\n图999:", re.DOTALL)

    for num_str, text in matches:
        idx = int(num_str) - 1  # 转为0-based
        if 0 <= idx < len(filenames):
            result[idx] = text.strip().rstrip("|").strip()

    # 如果解析失败，尝试按行分割作为后备
    if not result:
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        # 如果行数匹配图片数，直接映射
        if len(lines) == len(filenames):
            for i, line in enumerate(lines):
                # 去掉可能的序号前缀
                cleaned = re.sub(r'^[\d]+[\.\、\s:：]+', '', line).strip()
                result[i] = cleaned
        elif len(lines) > 0:
            # 全量输出放在第一张图上
            result[0] = raw_text
            for i in range(1, len(filenames)):
                result[i] = "[见上图]"

    return result


def scan_images(directory: str) -> list[str]:
    dir_path = Path(directory)
    files = []
    for f in sorted(dir_path.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(f))
    return files


def main():
    parser = argparse.ArgumentParser(description="Ollama Qwen3-VL 批量图片OCR (批次模式)")
    parser.add_argument("--model", default=MODEL, help="Ollama 模型名称")
    parser.add_argument("--dir", default=IMAGE_DIR, help="图片目录路径")
    parser.add_argument("--output", default=OUTPUT_FILE, help="结果输出文件路径")
    parser.add_argument("--limit", type=int, default=0, help="限制处理的图片数量 (0=全部)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"每批图片数量 (默认: {DEFAULT_BATCH_SIZE}，设为1=单张模式)")
    args = parser.parse_args()

    model = args.model
    image_dir = args.dir
    output_file = args.output
    batch_size = max(1, args.batch)

    # ========== Step 1: 环境检查 ==========
    print("=" * 60)
    print(f"  Ollama Qwen3-VL 批量图片OCR (批次 x{batch_size})")
    print("=" * 60)

    print("\n[1/4] 检查 Ollama 服务...")
    ok, models = check_ollama_service()
    if not ok:
        print("ERROR: Cannot connect to Ollama. Please run: ollama serve")
        sys.exit(1)
    print(f"OK - Ollama running, {len(models)} models installed")

    print(f"\n[2/4] 检查模型 [{model}]...")
    if not check_model_installed(model):
        if not pull_model(model):
            sys.exit(1)
    else:
        print("OK - Model ready")

    # 预热模型（首次加载模型到内存需要时间）
    print("   Warming up model (first load may take a while)...")
    try:
        warm_start = time.time()
        requests.post(
            f"{OLLAMA_API}/api/generate",
            json={"model": model, "prompt": "test", "stream": False},
            timeout=120,
        )
        print(f"   Model loaded in {time.time() - warm_start:.1f}s")
    except Exception:
        print("   Warmup skipped (timeout, model may load on first request)")

    print(f"\n[3/4] 扫描图片目录: {image_dir}")
    images = scan_images(image_dir)
    if not images:
        print("ERROR: No supported image files found")
        sys.exit(1)

    if args.limit > 0:
        images = images[:args.limit]
        print(f"Found {len(images)} images (limited to {args.limit})")
    else:
        print(f"Found {len(images)} images")

    # ========== Step 2: 批量 OCR (批次模式) ==========
    total_count = len(images)

    # 将图片分组为批次
    batches = []
    for i in range(0, total_count, batch_size):
        batch_images = images[i:i + batch_size]
        batch_names = [os.path.basename(p) for p in batch_images]
        batches.append((batch_images, batch_names))

    n_batches = len(batches)
    print(f"\n[4/4] 开始批量OCR ({n_batches} 批, 每批 ≤{batch_size}张)...")
    print("-" * 65)

    all_results = []
    success_count = 0
    fail_count = 0
    total_start = time.time()

    for batch_no, (batch_imgs, batch_names) in enumerate(batches, 1):
        batch_label = f"Batch {batch_no}/{n_batches}"
        names_preview = ", ".join(batch_names[:3])
        if len(batch_names) > 3:
            names_preview += f" ... +{len(batch_names)-3}"
        elapsed_sofar = time.time() - total_start

        print(f"[{batch_label}] {names_preview}")
        print(f"  Elapsed: {format_duration(elapsed_sofar)} | Sending...", end=" ", flush=True)

        batch_start = time.time()
        success = False
        for attempt in range(RETRY_COUNT + 1):
            try:
                batch_results, raw_text, batch_elapsed = ocr_batch(
                    batch_imgs, batch_names, model, batch_no, n_batches
                )
                success = True
                break
            except Exception as e:
                if attempt < RETRY_COUNT:
                    print(f"\n  Retry {attempt+1}/{RETRY_COUNT} ({e})...", end=" ", flush=True)
                    time.sleep(RETRY_DELAY)
                else:
                    batch_elapsed = time.time() - batch_start
                    batch_results = []
                    for fname in batch_names:
                        batch_results.append({
                            "filename": fname,
                            "text": f"[ERROR] {e}",
                            "elapsed": batch_elapsed / len(batch_names),
                        })

        if success:
            print(f"{batch_elapsed:.1f}s | {len(batch_results)}/{len(batch_names)} ok")
        else:
            print(f"FAILED in {batch_elapsed:.1f}s")

        # 显示本批结果
        for r in batch_results:
            status = "  OK" if not r["text"].startswith("[ERROR") else "FAIL"
            preview = r["text"].replace("\n", " ")[:50]
            print(f"    {status} [{r['elapsed']:.1f}s] {r['filename']}: {preview}")

        for r in batch_results:
            if r["text"].startswith("[ERROR"):
                fail_count += 1
            else:
                success_count += 1

        all_results.extend(batch_results)

        # ETA
        avg_batch_time = (time.time() - total_start) / batch_no
        eta = avg_batch_time * (n_batches - batch_no)
        print(f"  ETA: {format_duration(eta)} remaining\n")

    total_elapsed = time.time() - total_start

    # 统计
    success_times = [r["elapsed"] for r in all_results if not r["text"].startswith("[ERROR")]
    avg_time = sum(success_times) / len(success_times) if success_times else 0
    max_time = max(success_times) if success_times else 0
    min_time = min(success_times) if success_times else 0

    # ========== Step 3: 输出结果 ==========
    print("=" * 65)
    print("  COMPLETE!")
    print(f"  Images: {total_count} | OK: {success_count} | Fail: {fail_count}")
    print(f"  Batch size: {batch_size} | Batches: {n_batches}")
    print(f"  Wall time: {format_duration(total_elapsed)} ({total_elapsed:.1f}s)")
    if success_times:
        real_throughput = success_count / total_elapsed
        print(f"  Per image (avg): {avg_time:.1f}s | Range: {min_time:.1f}s - {max_time:.1f}s")
        print(f"  Throughput: {real_throughput:.1f} img/s | {real_throughput*60:.0f} img/min")
    print("=" * 65)
    print(f"  Output: {output_file}")

    # 写入结果文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Ollama Qwen3-VL OCR Results (batch mode)\n")
        f.write(f"# Model: {model} | Batch size: {batch_size} | Batches: {n_batches}\n")
        f.write(f"# Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"# Total: {total_count} | OK: {success_count} | Fail: {fail_count}\n")
        f.write(f"# Wall time: {format_duration(total_elapsed)}\n")
        if success_times:
            f.write(f"# Throughput: {success_count/total_elapsed:.1f} img/s\n")
        f.write("# " + "=" * 54 + "\n\n")

        for r in all_results:
            tag = "ERROR" if r["text"].startswith("[ERROR") else "OK"
            f.write(f"## [{tag}] {r['filename']}  [{r['elapsed']:.1f}s]\n")
            f.write(f"```\n{r['text']}\n```\n\n")

    # 打印前10条预览
    print("\n--- Preview (first 10) ---")
    for i, r in enumerate(all_results[:10], 1):
        preview = r["text"].replace("\n", " ")[:60]
        print(f"  {i}. [{r['elapsed']:.1f}s] {r['filename']}: {preview}")
    if len(all_results) > 10:
        print(f"  ... +{len(all_results)-10} more in output file")


if __name__ == "__main__":
    main()
