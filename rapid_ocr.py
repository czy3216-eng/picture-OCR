"""
RapidOCR ONNX - 批量图片OCR
=================================
使用 RapidOCR + ONNX Runtime（CPU），速度 0.05-0.3s/张
703 张图片预计 2-5 分钟完成

用法：
    python rapid_ocr.py              # 全部 703 张
    python rapid_ocr.py --limit 5    # 测试前 5 张
    python rapid_ocr.py --workers 4  # 多进程加速
"""
import os
import sys
import time
import argparse
from pathlib import Path

# ========== 配置 ==========
IMAGES_DIR    = r"D:\a-test\cutQP"
DEFAULT_OUTPUT_DIR = r"D:\a-test\result"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
DEFAULT_LIMIT    = None   # None = 全部
DEFAULT_WORKERS  = 0      # 0=单进程（最快，引擎复用）
# ==============================


def get_image_list(img_dir, limit=None):
    ext_set = SUPPORTED_EXT
    images = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in ext_set
    ])
    if limit:
        images = images[:limit]
    return images


def init_engine():
    """初始化 RapidOCR 引擎"""
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(use_cuda=False, use_dml=False)
    return engine


def ocr_image(engine, img_path):
    """
    单张图片 OCR，返回识别文本（拼接为单行）
    RapidOCR 返回格式: (boxes_texts, scores)
      boxes_texts[i] = [box_coords, text_string, score_float]
    """
    result = engine(img_path)
    if result is None:
        return ""
    boxes_texts = result[0]  # list of [box, text, score]
    if not boxes_texts:
        return ""
    texts = []
    for item in boxes_texts:
        if len(item) >= 2:
            texts.append(str(item[1]))
    return " | ".join(texts)


def process_single(args_tuple):
    """Worker：处理单张图片（用于多进程）"""
    idx, total, img_path, start_time = args_tuple
    filename = os.path.basename(img_path)
    elapsed_sofar = time.time() - start_time
    print(f"[{idx}/{total}] {filename} | 已用 {elapsed_sofar:.0f}s ... ", end="", flush=True)
    t0 = time.time()
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR(use_cuda=False, use_dml=False)
        text = ocr_image(engine, img_path)
        t = time.time() - t0
        preview = text.replace("\n", " ")[:60]
        print(f"OK {t:.2f}s  {preview}")
        return filename, text, t, None
    except Exception as e:
        t = time.time() - t0
        print(f"FAIL {t:.2f}s  {e}")
        return filename, f"[ERROR] {e}", t, str(e)


def main():
    parser = argparse.ArgumentParser(description="RapidOCR Batch Processing")
    parser.add_argument("--limit",       type=int,  default=DEFAULT_LIMIT,      help="Limit number of images")
    parser.add_argument("--workers",     type=int,  default=DEFAULT_WORKERS,  help="Multi-process workers (0=single)")
    parser.add_argument("--dir",         type=str,  default=None,             help="Override images directory")
    parser.add_argument("--output-dir",  type=str,  default=DEFAULT_OUTPUT_DIR, help="Output directory for results")
    args = parser.parse_args()

    img_dir = args.dir or IMAGES_DIR
    out_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 带时间戳的输出文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(out_dir, f"ocr_results_rapid_{timestamp}.txt")
    if not os.path.isdir(img_dir):
        print(f"[ERROR] 目录不存在: {img_dir}")
        sys.exit(1)

    all_images = get_image_list(img_dir, args.limit)
    total = len(all_images)
    if total == 0:
        print("[ERROR] 未找到图片文件")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  RapidOCR Batch (ONNX Runtime CPU)")
    print(f"  Image dir: {img_dir}")
    print(f"  Total: {total} images")
    print(f"  Mode: {'multi-process workers=' + str(args.workers) if args.workers > 0 else 'single process (engine reuse)'}")
    print(f"{'='*55}\n")

    start_time   = time.time()
    results      = []
    success_count = 0
    fail_count    = 0
    times         = []

    if args.workers <= 0:
        # 单进程：引擎只加载一次（最快）
        print("[Init] Loading RapidOCR engine (first run downloads ONNX models)...")
        engine = init_engine()
        print("[OK] Engine loaded\n")

        for idx, img_path in enumerate(all_images, 1):
            filename = os.path.basename(img_path)
            elapsed_sofar = time.time() - start_time
            print(f"[{idx}/{total}] {filename} | elapsed {elapsed_sofar:.0f}s ... ", end="", flush=True)
            t0 = time.time()
            try:
                text = ocr_image(engine, img_path)
                t = time.time() - t0
                times.append(t)
                success_count += 1
                preview = text.replace("\n", " ")[:60]
                print(f"OK {t:.2f}s  {preview}")
                results.append((filename, text, t))
            except Exception as e:
                t = time.time() - t0
                times.append(t)
                fail_count += 1
                print(f"FAIL {t:.2f}s  {e}")
                results.append((filename, f"[ERROR] {e}", t))
    else:
        # 多进程模式
        from multiprocessing import Pool
        work_items = [(i + 1, total, p, start_time) for i, p in enumerate(all_images)]
        with Pool(processes=args.workers) as pool:
            raw = pool.map(process_single, work_items)
        for r in raw:
            results.append(r)
            if r[3] is None:
                success_count += 1
                times.append(r[2])
            else:
                fail_count += 1
                times.append(r[2])

    total_elapsed = time.time() - start_time
    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0
    min_time = min(times) if times else 0
    throughput = total / (total_elapsed / 60) if total_elapsed > 0 else 0

    # ========== 输出结果文件 ==========
    print(f"\n{'='*55}")
    print(f"  DONE! Success: {success_count} | Failed: {fail_count}")
    print(f"  Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    if times:
        print(f"  Per image: avg {avg_time:.2f}s | min {min_time:.2f}s | max {max_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} images/min ({1/avg_time:.1f} images/sec)")
    print(f"  Output: {output_file}")
    print(f"{'='*55}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# RapidOCR ONNX Runtime OCR Results\n")
        f.write(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total: {total} | Success: {success_count} | Failed: {fail_count}\n")
        f.write(f"# Total time: {total_elapsed:.1f}s | Avg: {avg_time:.2f}s/image\n")
        f.write(f"# {'='*54}\n\n")
        for filename, text, t in results:
            f.write(f"## {filename}  [{t:.2f}s]\n```\n{text}\n```\n\n")

    print(f"[Output] Results saved to: {output_file}")
    if fail_count > 0:
        print(f"[Warn] {fail_count} images failed, check log above")


if __name__ == "__main__":
    main()
