r"""
RapidOCR 批量识别 + numpy 取色 (极致速度版)
================================================
- 单线程顺序处理 (RapidOCR 内部已多线程，外层线程反而踩踏)
- cls=False (跳过无用分类器)
- 单次 Image.open 复用取色+OCR

用法:
    python paddle_ocr.py --limit 10
    python paddle_ocr.py             # 全部
"""

import os, sys, re, time, argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

# ======================== 配置 ========================
IMAGE_DIR = r"D:\a-test\cutQP"
OUTPUT_DIR = r"D:\a-test\result"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# ======================== 颜色 ========================

def color_to_grade(hex_color):
    hc = hex_color.upper().lstrip('#')
    if hc in ("FFFFFF", "000000"):
        return "white"
    r = int(hc[0:2], 16) / 255.0
    g = int(hc[2:4], 16) / 255.0
    b = int(hc[4:6], 16) / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    if delta < 0.1:
        return "white"
    if delta == 0: h = 0
    elif mx == r:  h = 60 * (((g - b) / delta) % 6)
    elif mx == g:  h = 60 * (((b - r) / delta) + 2)
    else:          h = 60 * (((r - g) / delta) + 4)
    if h < 0: h += 360
    if h < 15 or h >= 345: return "red"
    if 15 <= h < 45: return "orange"
    if 75 <= h < 165: return "green"
    if 165 <= h < 270: return "blue"
    if 270 <= h < 345: return "purple"
    return "orange"


def extract_icon_color(pil_img):
    """numpy 向量化取色: 左30%区域"""
    try:
        w, h = pil_img.size
        icon_w = int(w * 0.30)
        left = pil_img.crop((0, 0, icon_w, h))
        px = np.asarray(left, dtype=np.uint8)
        maxc = px.max(axis=2)
        minc = px.min(axis=2)
        sat = maxc.astype(np.int16) - minc.astype(np.int16)

        mask = (maxc > 50) & (sat > 15) & (maxc < 240)
        if np.count_nonzero(mask) > 20:
            sp = px[mask]
            q8 = (sp // 8).astype(np.uint16) * 8 + 4
            vals, cnts = np.unique(q8, axis=0, return_counts=True)
            top = vals[cnts.argmax()]
            r, g, b = int(top[0]), int(top[1]), int(top[2])
            if max(r, g, b) < 80:
                q32 = (px // 32).astype(np.uint16) * 32 + 16
                v32, c32 = np.unique(q32.reshape(-1, 3), axis=0, return_counts=True)
                for ix in np.argsort(c32)[-3:][::-1]:
                    cr, cg, cb = int(v32[ix][0]), int(v32[ix][1]), int(v32[ix][2])
                    if cr > 192 and cg > 192 and cb > 192 and max(cr,cg,cb)-min(cr,cg,cb) < 32:
                        return "#FFFFFF"
            return f"#{r:02X}{g:02X}{b:02X}"

        q32 = (px // 32).astype(np.uint16) * 32 + 16
        v32, c32 = np.unique(q32.reshape(-1, 3), axis=0, return_counts=True)
        for ix in np.argsort(c32)[-3:][::-1]:
            cr, cg, cb = int(v32[ix][0]), int(v32[ix][1]), int(v32[ix][2])
            if cr > 192 and cg > 192 and cb > 192 and max(cr,cg,cb)-min(cr,cg,cb) < 32:
                return "#FFFFFF"

        bm = (maxc > 15) & ~((px[:,:,0] > 230) & (px[:,:,1] > 230) & (px[:,:,2] > 230))
        if np.count_nonzero(bm) > 5:
            bp = px[bm]
            q32b = (bp // 32).astype(np.uint16) * 32
            vb, cb = np.unique(q32b, axis=0, return_counts=True)
            top_b = vb[cb.argmax()]
            return f"#{int(top_b[0]):02X}{int(top_b[1]):02X}{int(top_b[2]):02X}"
        return "#000000"
    except Exception:
        return "#000000"


# ======================== 文本后处理 ========================

def postprocess_text(text):
    text = text.replace("￥", "").replace("$", "")
    text = text.replace("今", "").replace("→", "")
    text = text.replace(".", "").replace("·", "").replace(",,", "")
    text = text.strip("》>《<》>「」『』\"'\"—")
    text = re.sub(r'^[A-Za-z](?=[\u4e00-\u9fff])', '', text)
    fixes = [
        ("紧凌", "紧凑"), ("复合引", "复合弓"), ("十字引", "十字弓"),
        ("复合写", "复合弓"), ("绳索弓1", "绳索弓"), ("绒花", "缄花"),
        ("黄窖", "黄莺"), ("黄窜", "黄莺"), ("徒|", "徒"),
    ]
    for wrong, right in fixes:
        text = text.replace(wrong, right)
    text = text.replace("——", "-").replace("—", "-")
    text = re.sub(r'\s*-\s*', '-', text)
    return text.strip("—").strip(":：").strip()


# ======================== 扫描 ========================

def scan_images(directory):
    dir_path = Path(directory)
    return sorted([str(f) for f in dir_path.iterdir()
                   if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS])


# ======================== 主流程 ========================

def main():
    parser = argparse.ArgumentParser(description="RapidOCR 批量识别 (速度优化版)")
    parser.add_argument("--dir", default=IMAGE_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"ocr_color_paddle_{timestamp}.json")

    print("=" * 60)
    print("  RapidOCR 批量识别 (单线程, cls=False)")
    print("=" * 60)

    # Step 1
    print(f"\n[1/3] 扫描图片: {args.dir}")
    images = scan_images(args.dir)
    if not images: print("  ❌ 未找到图片"); sys.exit(1)
    if args.limit > 0: images = images[:args.limit]
    total = len(images)
    print(f"  找到 {total} 张")

    # Step 2
    print("\n[2/3] 初始化 RapidOCR (cls=False)...")
    t_init = time.time()
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(cls=False)
    print(f"  ✅ {time.time()-t_init:.1f}s")

    # Step 3: 顺序处理
    print(f"\n[3/3] 处理 {total} 张...")
    print("-" * 60)

    total_start = time.time()
    results = []
    success_count = fail_count = 0
    times = []

    for idx, img_path in enumerate(images, 1):
        filename = os.path.basename(img_path)
        t0 = time.time()
        try:
            pil = Image.open(img_path).convert("RGB")
            color = extract_icon_color(pil)
            img_np = np.asarray(pil)
            ocr_result = engine(img_np)

            if ocr_result is None or not ocr_result[0]:
                text = ""
            else:
                texts = [str(item[1]) for item in ocr_result[0] if len(item) >= 2]
                text = postprocess_text(" | ".join(texts).strip())

            grade = color_to_grade(color)
            elapsed = time.time() - t0
            times.append(elapsed)
            success_count += 1
            results.append({"grade": grade, "name": text})

            preview = text[:50].replace("\n", " ")
            elapsed_sofar = time.time() - total_start
            avg = elapsed_sofar / idx
            eta = avg * (total - idx)

            print(f"[{idx:>3d}/{total}] {filename:<25s} ✅ {elapsed:.3f}s | {grade:6s} | 总 {elapsed_sofar:.0f}s ETA {eta:.0f}s")
            print(f"         {preview}")

        except Exception as e:
            elapsed = time.time() - t0
            fail_count += 1
            results.append({"grade": "white", "name": f"[ERROR] {e}"})
            print(f"[{idx:>3d}/{total}] {filename:<25s} ❌ {elapsed:.3f}s | {e}")

    total_elapsed = time.time() - total_start
    avg_time = sum(times) / len(times) if times else 0

    print("\n" + "=" * 60)
    print("  ✅ 完成！")
    print(f"  总数: {total} | 成功: {success_count} | 失败: {fail_count}")
    print(f"  总耗时: {total_elapsed:.1f}s")
    if times:
        print(f"  单张: 平均 {avg_time:.3f}s | 最快 {min(times):.3f}s | 最慢 {max(times):.3f}s")
        print(f"  吞吐: {success_count/total_elapsed:.1f} 张/s = {success_count/total_elapsed*60:.0f} 张/min")
    print("=" * 60)
    print(f"  📄 {output_file}")

    output_data = {
        "_meta": {
            "total": total, "success": success_count, "failed": fail_count,
            "total_time_sec": round(total_elapsed, 1),
            "avg_time_per_image_ms": round(avg_time * 1000, 1),
            "throughput_per_min": round(success_count / total_elapsed * 60, 1) if total_elapsed > 0 else 0,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_dir": args.dir, "engine": "RapidOCR cls=False (sequential)",
        },
        "results": results
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n--- 预览 (前10条) ---")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i}. [{r['grade']:6s}] {r['name'][:50]}")
    if len(results) > 10:
        print(f"  ... +{len(results)-10} more")


if __name__ == "__main__":
    main()
