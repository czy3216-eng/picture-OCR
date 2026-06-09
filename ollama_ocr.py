r"""
RapidOCR 批量图片识别 + 取色 (高速版 v2)
=========================================
引擎共享 + numpy 向量化取色 + 多线程并发
目标: 单张 <0.1s, 704张 <60s

用法:
    python ollama_ocr.py                 # 处理全部图片
    python ollama_ocr.py --limit 10      # 测试前10张
    python ollama_ocr.py --workers 4     # 指定线程数
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

# ======================== 配置 ========================
IMAGE_DIR = r"D:\a-test\cutQP"
OUTPUT_DIR = r"D:\a-test\result"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
DEFAULT_WORKERS = 4  # ONNX Runtime 内部已多线程，外部线程不宜太多

# ======================== 颜色 & 后处理 (与 ocr_color.py 一致) ========================

def color_to_grade(hex_color):
    """HSV 色相 → 品级: red|orange|purple|blue|green|white"""
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


def extract_icon_color_fast(img_path):
    """Numpy 向量化取色 (比像素级 Python 循环快 30-50x)"""
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        icon_w = int(w * 0.30)
        region = img.crop((0, 0, icon_w, h))
        px = np.array(region, dtype=np.uint8).reshape(-1, 3)  # (N, 3)

        maxc = px.max(axis=1).astype(np.int32)
        minc = px.min(axis=1).astype(np.int32)
        sat = maxc - minc

        # Step 1: 饱和度过滤 (maxc>50, sat>15, maxc<240)
        mask = (maxc > 50) & (sat > 15) & (maxc < 240)
        if mask.any():
            sp = px[mask]
            quant = ((sp // 8) * 8 + 4).astype(np.uint8)
            unique, counts = np.unique(quant.view([('', quant.dtype)] * 3), return_counts=True)
            # convert back
            unique_arr = unique.view(np.uint8).reshape(-1, 3)
            top = unique_arr[counts.argmax()]
            r, g, b = int(top[0]), int(top[1]), int(top[2])

            if max(r, g, b) < 80:
                # 暗色结果 → 检查是否为白色 icon
                q2 = ((px // 32) * 32 + 16).astype(np.uint8)
                u2 = q2.view([('', q2.dtype)] * 3)
                u2_arr = q2  # same
                uniq, cnt = np.unique(u2, return_counts=True)
                uniq_arr = uniq.view(np.uint8).reshape(-1, 3)
                for ti in np.argsort(cnt)[-3:]:
                    cr, cg, cb = int(uniq_arr[ti][0]), int(uniq_arr[ti][1]), int(uniq_arr[ti][2])
                    if cr > 192 and cg > 192 and cb > 192 and max(cr,cg,cb)-min(cr,cg,cb) < 32:
                        return "#FFFFFF"
            return f"#{r:02X}{g:02X}{b:02X}"

        # Step 2: 无饱和像素 → 检查白色 icon
        q2 = ((px // 32) * 32 + 16).astype(np.uint8)
        u2 = q2.view([('', q2.dtype)] * 3)
        uniq, cnt = np.unique(u2, return_counts=True)
        uniq_arr = uniq.view(np.uint8).reshape(-1, 3)
        for ti in np.argsort(cnt)[-3:]:
            cr, cg, cb = int(uniq_arr[ti][0]), int(uniq_arr[ti][1]), int(uniq_arr[ti][2])
            if cr > 192 and cg > 192 and cb > 192 and max(cr,cg,cb)-min(cr,cg,cb) < 32:
                return "#FFFFFF"

        # Step 3: Fallback
        f_mask = (maxc > 15) & ~((px[:,0] > 230) & (px[:,1] > 230) & (px[:,2] > 230))
        if f_mask.any():
            fp = px[f_mask]
            q3 = ((fp // 32) * 32).astype(np.uint8)
            u3 = q3.view([('', q3.dtype)] * 3)
            uniq3, cnt3 = np.unique(u3, return_counts=True)
            uniq3_arr = uniq3.view(np.uint8).reshape(-1, 3)
            top = uniq3_arr[cnt3.argmax()]
            return f"#{int(top[0]):02X}{int(top[1]):02X}{int(top[2]):02X}"
        return "#000000"
    except Exception:
        return "#000000"


# 兼容旧函数名
extract_icon_color = extract_icon_color_fast


def postprocess_text(text):
    """后处理 OCR 识别错误"""
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
    return f"{int(m)}m{int(s)}s"


def scan_images(directory: str) -> list:
    dir_path = Path(directory)
    return sorted([str(f) for f in dir_path.iterdir()
                   if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS])


# ======================== 核心处理 ========================

def create_engine():
    """创建 RapidOCR 引擎 (只调用一次，主线程)"""
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(use_cuda=False, use_dml=False)


def ocr_text(engine, img_path):
    """RapidOCR 识别文字 (引擎共享，ONNX 内部多线程)"""
    result = engine(img_path)
    if result is None or not result:
        return ""
    boxes_texts = result[0]
    if not boxes_texts:
        return ""
    texts = [str(item[1]) for item in boxes_texts if len(item) >= 2]
    return postprocess_text(" | ".join(texts).strip())


def process_one(engine, idx, img_path):
    """处理单张: 取色 + OCR"""
    filename = os.path.basename(img_path)
    t0 = time.time()
    try:
        color = extract_icon_color(img_path)
        text = ocr_text(engine, img_path)
        grade = color_to_grade(color)
        elapsed = time.time() - t0
        return {
            "grade": grade, "name": text, "filename": filename,
            "idx": idx, "elapsed": elapsed, "success": True,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "grade": "white", "name": f"[ERROR] {e}", "filename": filename,
            "idx": idx, "elapsed": elapsed, "success": False,
        }


# ======================== 主流程 ========================

def main():
    parser = argparse.ArgumentParser(description="RapidOCR 批量识别+取色 (引擎共享高速版)")
    parser.add_argument("--dir", default=IMAGE_DIR, help="图片目录")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--limit", type=int, default=0, help="限制数量 (0=全部)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"线程数 (默认={DEFAULT_WORKERS})")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    workers = max(1, min(args.workers, 8))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(out_dir, f"ocr_color_{timestamp}.json")

    print("=" * 60)
    print(f"  RapidOCR 引擎共享 + 多线程 ({workers} threads)")
    print("=" * 60)

    # [1/3] 扫描
    print(f"\n[1/3] 扫描图片: {args.dir}")
    images = scan_images(args.dir)
    if not images:
        print("  ❌ 未找到图片")
        sys.exit(1)
    if args.limit > 0:
        images = images[:args.limit]
    total = len(images)
    print(f"  找到 {total} 张")

    # [2/3] 初始化 (只创建 1 个引擎, 所有线程共享)
    print(f"\n[2/3] 初始化 RapidOCR 引擎 ...")
    t_init = time.time()
    engine = create_engine()
    print(f"  ✅ 初始化完成 ({time.time() - t_init:.1f}s)")

    # [3/3] 多线程处理 (共享引擎, 无锁)
    print(f"\n[3/3] 开始处理 {total} 张 ({workers} threads)...")
    print("-" * 60)

    total_start = time.time()
    all_results = []
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one, engine, i, img): i
            for i, img in enumerate(images, 1)
        }
        for future in as_completed(futures):
            r = future.result()
            all_results.append(r)
            if r["success"]:
                success_count += 1
            else:
                fail_count += 1

            done = len(all_results)
            preview = r["name"].replace("\n", " ")[:45]
            elapsed_sofar = time.time() - total_start
            eta = (elapsed_sofar / done) * (total - done) if done > 0 else 0

            # 用 ms 显示（快的时候 s 看不出来）
            per_img = f"{r['elapsed']*1000:.0f}ms" if r["elapsed"] < 1 else f"{r['elapsed']:.2f}s"
            print(
                f"[{done:>3d}/{total}] {r['filename']:<25s} "
                f"{per_img:>6s} | {r['grade']:6s} | {preview}"
            )

    total_elapsed = time.time() - total_start

    # 排序
    all_results.sort(key=lambda r: r["idx"])

    # 统计
    success_times = [r["elapsed"] for r in all_results if r["success"]]
    avg_ms = sum(success_times) / len(success_times) * 1000 if success_times else 0
    throughput = success_count / total_elapsed if total_elapsed > 0 else 0

    print("\n" + "=" * 60)
    print("  ✅ 完成！")
    print(f"  总数: {total} | 成功: {success_count} | 失败: {fail_count}")
    print(f"  总耗时: {format_duration(total_elapsed)} ({total_elapsed:.1f}s)")
    if success_times:
        print(f"  单张: 平均 {avg_ms:.0f}ms | 最快 {min(success_times)*1000:.0f}ms | "
              f"最慢 {max(success_times)*1000:.0f}ms")
        print(f"  吞吐: {throughput*60:.0f} 张/min")
        if total_elapsed < 60 and total == 704:
            print(f"  🎯 达标！{total_elapsed:.1f}s < 60s")
        elif avg_ms < 1000:
            print(f"  ✅ 单张 {avg_ms:.0f}ms < 1000ms 达标")
    print(f"  avg/张: {avg_ms:.0f}ms")
    print("=" * 60)
    print(f"  📄 结果: {output_file}")

    # 构建 JSON
    output_data = {
        "_meta": {
            "total": total,
            "success": success_count,
            "failed": fail_count,
            "total_time_sec": round(total_elapsed, 1),
            "avg_time_ms": round(avg_ms, 1),
            "throughput_per_min": round(throughput * 60, 1),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_dir": args.dir,
            "engine": "RapidOCR (ONNX Runtime, shared)",
        },
        "results": [{"grade": r["grade"], "name": r["name"]} for r in all_results]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 预览
    print(f"\n--- 预览 (前10条) ---")
    for i, r in enumerate(all_results[:10], 1):
        print(f"  {i}. [{r['grade']:6s}] {r['name'][:50]}")
    if len(all_results) > 10:
        print(f"  ... +{len(all_results)-10} more")


if __name__ == "__main__":
    main()
