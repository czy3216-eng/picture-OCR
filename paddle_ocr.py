r"""
PaddleOCR 批量图片OCR (RapidOCR 引擎版)
==========================================
使用 RapidOCR (ONNX Runtime)，无需 paddlepaddle，秒级启动

用法：
    python paddle_ocr.py                 # 处理全部图片
    python paddle_ocr.py --limit 10      # 测试前10张
    python paddle_ocr.py --workers 4     # 指定线程数 (默认4)
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================== 配置 ========================
IMAGE_DIR = r"D:\a-test\cutQP"
OUTPUT_DIR = r"D:\a-test\result"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
DEFAULT_WORKERS = 4


# ======================== 颜色 & 后处理 (与 ocr_color.py 一致) ========================

def color_to_grade(hex_color):
    hc = hex_color.upper().lstrip('#')
    if hc in ("FFFFFF", "000000"):
        return "white"
    r = int(hc[0:2], 16) / 255.0
    g = int(hc[2:4], 16) / 255.0
    b = int(hc[4:6], 16) / 255.0
    maxc, minc = max(r, g, b), min(r, g, b)
    delta = maxc - minc
    if delta < 0.1:
        return "white"
    if delta == 0:
        h = 0
    elif maxc == r:
        h = 60 * (((g - b) / delta) % 6)
    elif maxc == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)
    if h < 0:
        h += 360
    if h < 15 or h >= 345:
        return "red"
    elif 15 <= h < 45:
        return "orange"
    elif 75 <= h < 165:
        return "green"
    elif 165 <= h < 270:
        return "blue"
    elif 270 <= h < 345:
        return "purple"
    return "orange"


def extract_icon_color(img_path):
    from PIL import Image
    from collections import Counter
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        icon_w = int(w * 0.30)
        region = img.crop((0, 0, icon_w, h))

        saturated = []
        for r, g, b in region.get_flattened_data():
            maxc, minc = max(r, g, b), min(r, g, b)
            saturation = maxc - minc
            if maxc > 50 and saturation > 15 and maxc < 240:
                qr = (r // 8) * 8 + 4
                qg = (g // 8) * 8 + 4
                qb = (b // 8) * 8 + 4
                saturated.append((qr, qg, qb))

        if saturated:
            counter = Counter(saturated)
            top_color = counter.most_common(1)[0][0]
            r, g, b = top_color
            hex_result = f"#{r:02X}{g:02X}{b:02X}"
            if max(r, g, b) < 80:
                quantized = []
                for r2, g2, b2 in region.get_flattened_data():
                    qr = (r2 // 32) * 32 + 16
                    qg = (g2 // 32) * 32 + 16
                    qb = (b2 // 32) * 32 + 16
                    quantized.append((qr, qg, qb))
                counter2 = Counter(quantized)
                for (cr, cg, cb), _ in counter2.most_common(3):
                    if cr > 192 and cg > 192 and cb > 192 and max(cr,cg,cb)-min(cr,cg,cb) < 32:
                        return "#FFFFFF"
            return hex_result

        quantized = []
        for r, g, b in region.get_flattened_data():
            qr = (r // 32) * 32 + 16
            qg = (g // 32) * 32 + 16
            qb = (b // 32) * 32 + 16
            quantized.append((qr, qg, qb))
        counter = Counter(quantized)
        for (cr, cg, cb), _ in counter.most_common(3):
            if cr > 192 and cg > 192 and cb > 192 and max(cr,cg,cb)-min(cr,cg,cb) < 32:
                return "#FFFFFF"

        all_colors = []
        for r, g, b in region.get_flattened_data():
            maxc = max(r, g, b)
            if maxc > 15 and not (r > 230 and g > 230 and b > 230):
                all_colors.append(((r // 32) * 32, (g // 32) * 32, (b // 32) * 32))
        if all_colors:
            counter = Counter(all_colors)
            top_color = counter.most_common(1)[0][0]
            r, g, b = top_color
            return f"#{r:02X}{g:02X}{b:02X}"
        return "#000000"
    except Exception:
        return "#000000"


def postprocess_text(text):
    text = text.replace("￥", "").replace("$", "")
    text = text.replace("绒花", "缄花")
    text = text.replace("今", "")
    text = text.replace("→", "")
    text = text.strip("—")
    text = text.strip(":：")
    return text.strip()


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m"


def scan_images(directory: str) -> list:
    dir_path = Path(directory)
    return sorted([str(f) for f in dir_path.iterdir()
                   if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS])


def ocr_single(engine, img_path):
    """RapidOCR 识别一张图片，返回文本"""
    result = engine(img_path)
    if result is None:
        return ""
    boxes_texts = result[0]
    if not boxes_texts:
        return ""
    texts = [str(item[1]) for item in boxes_texts if len(item) >= 2]
    return postprocess_text(" | ".join(texts).strip())


def process_one(idx, img_path, engine):
    """处理一张图片: 取色 + OCR"""
    filename = os.path.basename(img_path)
    t0 = time.time()
    try:
        color = extract_icon_color(img_path)
        text = ocr_single(engine, img_path)
        grade = color_to_grade(color)
        elapsed = time.time() - t0
        return {"grade": grade, "name": text, "filename": filename,
                "idx": idx, "elapsed": elapsed, "success": True}
    except Exception as e:
        elapsed = time.time() - t0
        return {"grade": "white", "name": f"[ERROR] {e}", "filename": filename,
                "idx": idx, "elapsed": elapsed, "success": False}


# ======================== 主流程 ========================

def main():
    parser = argparse.ArgumentParser(description="RapidOCR 批量OCR+取色 (PaddleOCR文件名兼容)")
    parser.add_argument("--dir", default=IMAGE_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"线程数 (默认={DEFAULT_WORKERS})")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    workers = max(1, min(args.workers, 8))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(out_dir, f"ocr_color_paddle_{timestamp}.json")

    print("=" * 60)
    print(f"  RapidOCR 批量识别 ({workers} threads) + Color Picker")
    print("=" * 60)

    # Step 1
    print(f"\n[1/3] 扫描图片: {args.dir}")
    images = scan_images(args.dir)
    if not images:
        print("  ❌ 未找到图片")
        sys.exit(1)
    if args.limit > 0:
        images = images[:args.limit]
    total = len(images)
    print(f"  找到 {total} 张")

    # Step 2
    print("\n[2/3] 初始化 RapidOCR 引擎...")
    t_init = time.time()
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    print(f"  ✅ 初始化完成 ({time.time() - t_init:.1f}s)")

    # Step 3
    print(f"\n[3/3] 开始处理 {total} 张 ({workers} threads)...")
    print("-" * 60)

    total_start = time.time()
    all_results = []
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, i, img, engine): i
                   for i, img in enumerate(images, 1)}
        for future in as_completed(futures):
            r = future.result()
            all_results.append(r)
            if r["success"]:
                success_count += 1
            else:
                fail_count += 1

            idx = r["idx"]
            preview = r["name"].replace("\n", " ")[:45]
            elapsed_sofar = time.time() - total_start
            avg_per = elapsed_sofar / len(all_results)
            eta = avg_per * (total - len(all_results))

            print(
                f"[{len(all_results):>3d}/{total}] {r['filename']:<25s} "
                f"{'✅' if r['success'] else '❌'} {r['elapsed']:.2f}s "
                f"| {r['grade']:6s} | 总 {format_duration(elapsed_sofar):>6s} "
                f"| ETA {format_duration(eta):>6s}"
            )
            print(f"         {preview}")

    total_elapsed = time.time() - total_start

    # 排序
    all_results.sort(key=lambda r: r["idx"])

    # 统计
    success_times = [r["elapsed"] for r in all_results if r["success"]]
    avg_time = sum(success_times) / len(success_times) if success_times else 0
    max_time = max(success_times) if success_times else 0
    min_time = min(success_times) if success_times else 0

    print("\n" + "=" * 60)
    print("  ✅ 完成！")
    print(f"  总数: {total} | 成功: {success_count} | 失败: {fail_count}")
    print(f"  总耗时: {format_duration(total_elapsed)} ({total_elapsed:.1f}s)")
    if success_times:
        print(f"  单张: 平均 {avg_time:.2f}s | 最快 {min_time:.2f}s | 最慢 {max_time:.2f}s")
        throughput = success_count / total_elapsed if total_elapsed > 0 else 0
        print(f"  吞吐: {throughput:.1f} 张/s | {throughput*60:.0f} 张/min")
    print("=" * 60)
    print(f"  📄 结果: {output_file}")

    # 输出 JSON
    output_data = {
        "_meta": {
            "total": total,
            "success": success_count,
            "failed": fail_count,
            "total_time_sec": round(total_elapsed, 1),
            "avg_time_per_image_ms": round(avg_time * 1000, 1),
            "throughput_per_min": round(success_count / total_elapsed * 60, 1) if total_elapsed > 0 else 0,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_dir": args.dir,
            "engine": "RapidOCR (ONNX Runtime)",
            "workers": workers,
        },
        "results": [{"grade": r["grade"], "name": r["name"]} for r in all_results]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n--- 预览 (前10条) ---")
    for i, r in enumerate(all_results[:10], 1):
        print(f"  {i}. [{r['grade']:6s}] {r['name'][:50]}")
    if len(all_results) > 10:
        print(f"  ... +{len(all_results)-10} more in output file")


if __name__ == "__main__":
    main()
