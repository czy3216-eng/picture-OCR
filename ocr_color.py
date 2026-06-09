"""
RapidOCR ONNX + Color Picker - 批量图片OCR+取色
=================================================
对每张图片：
1. 提取左侧icon区域的主色调 -> 十六进制颜色值
2. RapidOCR识别文字内容
3. 输出格式: {"color": "#XXXXXX", "text": "..."}

用法：
    python ocr_color.py              # 全部图片
    python ocr_color.py --limit 5    # 测试前5张
"""
import os
import sys
import time
import argparse
import json
from collections import Counter
from pathlib import Path

# ========== 配置 ==========
IMAGES_DIR       = r"D:\a-test\cutQP"
DEFAULT_OUTPUT_DIR = r"D:\a-test\result"
SUPPORTED_EXT    = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
DEFAULT_LIMIT    = None   # None = 全部
# ==============================


def get_image_list(img_dir, limit=None):
    images = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])
    if limit:
        images = images[:limit]
    return images


def color_to_grade(hex_color):
    """
    将十六进制颜色映射为品级: red | orange | purple | blue | green | white
    使用 HSV 色相(Hue)分类:
      red:    H ∈ [345,360) ∪ [0,15)
      orange: H ∈ [15,45)
      green:  H ∈ [75,165)
      blue:   H ∈ [165,270)
      purple: H ∈ [270,345)
      white:  低饱和度 (< 0.1) 或 #FFFFFF / #000000
    """
    hc = hex_color.upper().lstrip('#')
    if hc in ("FFFFFF", "000000"):
        return "white"

    r = int(hc[0:2], 16) / 255.0
    g = int(hc[2:4], 16) / 255.0
    b = int(hc[4:6], 16) / 255.0

    maxc = max(r, g, b)
    minc = min(r, g, b)
    delta = maxc - minc

    # 低饱和度 → 白色/灰色
    if delta < 0.1:
        return "white"

    # 计算 Hue (0-360)
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
    # 45-75 黄/黄绿区间归入 orange
    return "orange"


def extract_icon_color(img_path):
    """
    从图片左侧区域提取icon主色，返回十六进制颜色字符串如 '#CC5C5C'

    策略：
    1. 截取图片左侧30%区域
    2. 用饱和度过滤 (max(RGB)-min(RGB) > 15) 排除灰色背景
    3. 亮度过滤 (max(RGB) > 30) 排除纯黑
    4. 排除纯白像素(文字)
    5. 8级量化聚类后取众数
    """
    from PIL import Image
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        icon_w = int(w * 0.30)

        region = img.crop((0, 0, icon_w, h))
        region_px = icon_w * h

        # 第1步：饱和度过滤 — 提取彩色像素
        # maxc > 50 排除暗色背景 (如 (4,4,36) maxc=36)
        # maxc-minc > 15 排除灰色/白色
        from collections import Counter
        saturated = []
        for r, g, b in region.get_flattened_data():
            maxc = max(r, g, b)
            minc = min(r, g, b)
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
            # 如果饱和检测结果是暗色（max<80），说明没有真正的彩色像素，
            # 得到的可能是边缘伪影 → 检查是否为白色 icon
            if max(r, g, b) < 80:
                # 量化整个区域，检查是否有近白色主色
                quantized = []
                for r2, g2, b2 in region.get_flattened_data():
                    qr = (r2 // 32) * 32 + 16
                    qg = (g2 // 32) * 32 + 16
                    qb = (b2 // 32) * 32 + 16
                    quantized.append((qr, qg, qb))
                counter2 = Counter(quantized)
                # 检查前3主色
                for (cr, cg, cb), cnt in counter2.most_common(3):
                    if cr > 192 and cg > 192 and cb > 192 and max(cr,cg,cb)-min(cr,cg,cb) < 32:
                        return "#FFFFFF"
            return hex_result

        # 第2步：无饱和像素 → 检查是否为白色 icon
        # 量化整个区域，看前3主色是否含近白色
        quantized = []
        for r, g, b in region.get_flattened_data():
            qr = (r // 32) * 32 + 16
            qg = (g // 32) * 32 + 16
            qb = (b // 32) * 32 + 16
            quantized.append((qr, qg, qb))
        counter = Counter(quantized)
        for (cr, cg, cb), cnt in counter.most_common(3):
            if cr > 192 and cg > 192 and cb > 192 and max(cr,cg,cb)-min(cr,cg,cb) < 32:
                return "#FFFFFF"

        # 第3步：Fallback - 放宽亮度要求，排除纯黑纯白
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

    except Exception as e:
        print(f"[Color Error] {img_path}: {e}")
        return "#000000"


def init_engine():
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(use_cuda=False, use_dml=False)
    return engine


def postprocess_text(text):
    """后处理修正 RapidOCR 常见识别错误"""
    # 移除多余货币符号（OCR误识）
    text = text.replace("￥", "").replace("$", "")
    # 绒花 -> 缄花 (枪皮系列皮肤名)
    text = text.replace("绒花", "缄花")
    # 移除 icon 被误识别成的字符
    text = text.replace("今", "")
    text = text.replace("→", "")
    # 去除首尾的 — (icon边缘误识别为破折号)
    text = text.strip("—")
    # 去除冒号前缀 (常见误识)
    text = text.strip(":：")
    return text.strip()


def ocr_image(engine, img_path):
    """RapidOCR 单张识别，返回文本"""
    result = engine(img_path)
    if result is None:
        return ""
    boxes_texts = result[0]
    if not boxes_texts:
        return ""
    texts = []
    for item in boxes_texts:
        if len(item) >= 2:
            texts.append(str(item[1]))
    result_text = " | ".join(texts).strip()
    return postprocess_text(result_text)


def main():
    parser = argparse.ArgumentParser(description="RapidOCR + Color Picker")
    parser.add_argument("--limit",      type=int, default=DEFAULT_LIMIT, help="Limit number of images")
    parser.add_argument("--dir",        type=str, default=None,          help="Override images directory")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    img_dir = args.dir or IMAGES_DIR
    out_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(out_dir, f"ocr_color_{timestamp}.json")

    if not os.path.isdir(img_dir):
        print(f"[ERROR] Directory not found: {img_dir}")
        sys.exit(1)

    all_images = get_image_list(img_dir, args.limit)
    total = len(all_images)
    if total == 0:
        print("[ERROR] No image files found")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  RapidOCR + Color Picker")
    print(f"  Dir:  {img_dir}")
    print(f"  Total: {total} images")
    print(f"{'='*60}\n")

    start_time = time.time()
    results = []
    success_count = 0
    fail_count = 0
    times = []

    # Init engines
    print("[Init] Loading PIL & RapidOCR...")
    engine = init_engine()
    print("[OK] Engines loaded\n")

    for idx, img_path in enumerate(all_images, 1):
        filename = os.path.basename(img_path)
        elapsed_sofar = time.time() - start_time
        print(f"[{idx}/{total}] {filename} | elapsed {elapsed_sofar:.0f}s ... ", end="", flush=True)
        t0 = time.time()
        try:
            # Step 1: Extract icon color
            color = extract_icon_color(img_path)
            t_color = time.time() - t0

            # Step 2: OCR text
            text = ocr_image(engine, img_path)
            t_total = time.time() - t0
            times.append(t_total)
            success_count += 1

            grade = color_to_grade(color)

            preview = text.replace("\n", " ")[:50]
            print(f"OK {t_total:.2f}s | {color} → {grade} | {preview}")
            results.append({
                "grade": grade,
                "name": text
            })
        except Exception as e:
            t_total = time.time() - t0
            times.append(t_total)
            fail_count += 1
            print(f"FAIL {t_total:.2f}s | {e}")
            results.append({
                "grade": "white",
                "name": f"[ERROR] {e}"
            })

    total_elapsed = time.time() - start_time
    avg_time = sum(times) / len(times) if times else 0
    throughput = total / (total_elapsed / 60) if total_elapsed > 0 else 0

    # ========== 输出JSON结果文件 ==========
    output_data = {
        "_meta": {
            "total": total,
            "success": success_count,
            "failed": fail_count,
            "total_time_sec": round(total_elapsed, 1),
            "avg_time_per_image_ms": round(avg_time * 1000, 1),
            "throughput_per_min": round(throughput, 1),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_dir": img_dir
        },
        "results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE! Success: {success_count} | Failed: {fail_count}")
    print(f"  Total: {total_elapsed:.1f}s | Avg: {avg_time:.3f}s/image")
    print(f"  Throughput: {throughput:.1f} images/min")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")

    # 打印前5条预览
    print(f"\n--- Preview (first 5) ---")
    for r in results[:5]:
        print(json.dumps(r, ensure_ascii=False))

    return output_file


if __name__ == "__main__":
    main()
