r"""
Qwen3-VL:8b 批量图片识别 + 取色 (Ollama 极速版)
================================================
核心优化:
  1. 图片缩小到 336px 再发送 (Qwen3-VL 原生分辨率, 大幅减少 prompt 处理时间)
  2. JPEG 压缩 quality=65 → base64 体积减 80%+
  3. num_predict=8 (icon 文字最多 4-5 汉字)
  4. keep_alive 让模型常驻内存
  5. 单次打开图片, 取色+编码复用 PIL Image

前置:
    docker exec <容器名> ollama pull qwen3-vl:8b
    Docker 推荐启动参数: -e OLLAMA_NUM_PARALLEL=4 -e OLLAMA_FLASH_ATTENTION=1

用法:
    python ollama_ocr.py --limit 10
    python ollama_ocr.py --workers 4 --limit 10   # 如果 GPU 够大
    python ollama_ocr.py                           # 全量 704 张
"""

import os, sys, re, time, argparse, json, base64, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
import requests

# ======================== 配置 ========================
IMAGE_DIR = r"D:\a-test\cutQP"
OUTPUT_DIR = r"D:\a-test\result"
SUPPORTED_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".gif",".webp"}

OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen3-vl:8b"

# 并发: Ollama GPU 串行 → 2-3 最佳
DEFAULT_WORKERS = 2
# OCR 超时
OCR_TIMEOUT = 60
# 最小 token (icon ≤ 5 个汉字)
MAX_TOKENS = 8
# 重试
RETRIES = 1

# VL 图片预处理: 缩小到 336px + JPEG 压缩 → base64 体积减 80%+
VL_MAX_SIZE = 336
VL_JPEG_QUALITY = 65

# Prompt — 极简到极致
PROMPT = "OUTPUT ONLY the Chinese text visible in this picture. Do NOT say anything else."

# ======================== 颜色 & 后处理 ========================

def color_to_grade(hex_color):
    hc = hex_color.upper().lstrip('#')
    if hc in ("FFFFFF","000000"): return "white"
    r,g,b = int(hc[0:2],16)/255,int(hc[2:4],16)/255,int(hc[4:6],16)/255
    mx,mn = max(r,g,b), min(r,g,b)
    d = mx - mn
    if d < 0.1: return "white"
    if d==0: h=0
    elif mx==r: h=60*(((g-b)/d)%6)
    elif mx==g: h=60*(((b-r)/d)+2)
    else: h=60*(((r-g)/d)+4)
    if h<0: h+=360
    if h<15 or h>=345: return "red"
    if h<45 or h<75: return "orange"
    if h<165: return "green"
    if h<270: return "blue"
    return "purple"

def extract_icon_color_from_array(px):
    """从 numpy 像素数组提取颜色 (px shape: N×3 uint8)"""
    maxc = px.max(axis=1).astype(np.int32)
    minc = px.min(axis=1).astype(np.int32)
    sat = maxc - minc
    # Step 1: 饱和度过滤
    mask = (maxc>50)&(sat>15)&(maxc<240)
    if mask.any():
        sp = px[mask]
        quant = ((sp//8)*8+4).astype(np.uint8)
        qv = quant.view([('',quant.dtype)]*3)
        uniq,cnt = np.unique(qv,return_counts=True)
        top = uniq.view(np.uint8).reshape(-1,3)[cnt.argmax()]
        r,g,b = int(top[0]),int(top[1]),int(top[2])
        if max(r,g,b)<80:
            q2=((px//32)*32+16).astype(np.uint8)
            u2=q2.view([('',q2.dtype)]*3)
            uu,cc=np.unique(u2,return_counts=True)
            ua=uu.view(np.uint8).reshape(-1,3)
            for ti in np.argsort(cc)[-3:]:
                cr,cg,cb=int(ua[ti][0]),int(ua[ti][1]),int(ua[ti][2])
                if cr>192 and cg>192 and cb>192 and max(cr,cg,cb)-min(cr,cg,cb)<32:
                    return "#FFFFFF"
        return f"#{r:02X}{g:02X}{b:02X}"
    # Step 2: 白色
    q2=((px//32)*32+16).astype(np.uint8)
    u2=q2.view([('',q2.dtype)]*3)
    uu,cc=np.unique(u2,return_counts=True)
    ua=uu.view(np.uint8).reshape(-1,3)
    for ti in np.argsort(cc)[-3:]:
        cr,cg,cb=int(ua[ti][0]),int(ua[ti][1]),int(ua[ti][2])
        if cr>192 and cg>192 and cb>192 and max(cr,cg,cb)-min(cr,cg,cb)<32:
            return "#FFFFFF"
    # Step 3: fallback
    f_mask=(maxc>15)&~(px[:,0]>230)&~(px[:,1]>230)&~(px[:,2]>230)
    if f_mask.any():
        fp=px[f_mask]; q3=((fp//32)*32).astype(np.uint8)
        u3=q3.view([('',q3.dtype)]*3)
        uu3,cc3=np.unique(u3,return_counts=True)
        top=uu3.view(np.uint8).reshape(-1,3)[cc3.argmax()]
        return f"#{int(top[0]):02X}{int(top[1]):02X}{int(top[2]):02X}"
    return "#000000"

def postprocess_text(text):
    """后处理: 修正 OCR/VL 常见误识别 + 清洗标点"""

    # === 符号清除 ===
    text = text.replace("￥", "").replace("$", "")
    text = text.replace("今", "").replace("→", "")
    text = text.replace(".", "").replace("·", "").replace(",,", "")

    # === 去除首尾垃圾标点 ===
    text = text.strip("》>《<》>「」『』\"'\"—")

    # === 去除行首孤立字母前缀 (如 I旧梦 → 旧梦, X新春 → 新春) ===
    text = re.sub(r'^[A-Za-z](?=[\u4e00-\u9fff])', '', text)

    # === OCR 误识别修正 (武器/装备 场景) ===
    fixes = [
        ("紧凌", "紧凑"),
        ("复合引", "复合弓"),
        ("十字引", "十字弓"),
        ("复合写", "复合弓"),
        ("绳索弓1", "绳索弓"),
        ("绒花", "缄花"),
        ("黄窖", "黄莺"),
        ("黄窜", "黄莺"),
        ("徒|", "徒"),
        ("·", ""),
    ]
    for wrong, right in fixes:
        text = text.replace(wrong, right)

    # === 破折号规范化 ===
    text = text.replace("——", "-")
    text = text.replace("—", "-")

    # === 去除分隔符两侧多余空格/引号 ===
    text = re.sub(r'\s*-\s*', '-', text)

    # === 末尾收拾 ===
    text = text.strip("—")
    text = text.strip(":：").strip("。").strip("，")
    return text.strip()

def scan_images(directory):
    dp = Path(directory)
    return sorted([str(f) for f in dp.iterdir()
                   if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS])

# ======================== Ollama ========================

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    except requests.ConnectionError:
        print("\n  ❌ Ollama 未连接! 请启动服务"); sys.exit(1)
    except requests.Timeout:
        print("\n  ❌ Ollama 超时!"); sys.exit(1)
    if r.status_code != 200:
        print(f"\n  ❌ Ollama HTTP {r.status_code}"); sys.exit(1)
    data = r.json()
    models = [m.get("name","") for m in data.get("models",[])]
    base = MODEL.split(":")[0]
    for m in models:
        if m == MODEL or m.lower().startswith(base.lower()):
            print(f"  ✅ {m} 已就绪"); return
    # 真实探测
    try:
        tr = requests.post(f"{OLLAMA_BASE}/api/generate",
            json={"model":MODEL,"prompt":"ok","stream":False}, timeout=10)
        if tr.status_code == 200:
            print(f"  ✅ 模型可用"); return
    except: pass
    print(f"\n  ❌ '{MODEL}' 不可用!")
    if models: print(f"     已有: {', '.join(models[:10])}")
    sys.exit(1)

def warmup_model():
    print("  预热...", end=" ", flush=True)
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate",
            json={"model":MODEL,"prompt":"hello","stream":False,
                  "options":{"num_predict":2,"temperature":0}},
            timeout=120)
        e = time.time()-t0
        print(f"{'✅' if r.status_code==200 else '⚠️'} ({e:.1f}s)")
        return r.status_code == 200
    except requests.Timeout:
        print(f"❌ 超时 ({time.time()-t0:.0f}s)")
    except Exception as ex:
        print(f"❌ {ex}")
    return False

def prepare_vl_b64(pil_img):
    """缩小 + JPEG 压缩 → base64 (大幅减少 Ollama 处理时间)"""
    w, h = pil_img.size
    max_dim = max(w, h)
    if max_dim > VL_MAX_SIZE:
        scale = VL_MAX_SIZE / max_dim
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=VL_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()

def ocr_one(img_b64, max_retries=RETRIES):
    """VL 识别 (支持重试)"""
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
        "keep_alive": 300,   # 模型常驻 5 分钟
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0,
        },
    }
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(f"{OLLAMA_BASE}/api/generate",
                              json=payload, timeout=OCR_TIMEOUT)
            if r.status_code == 200:
                text = r.json().get("response","").strip()
                return postprocess_text(text)
            last_err = f"HTTP {r.status_code}"
            if attempt < max_retries: time.sleep(2)
        except requests.Timeout:
            last_err = "TIMEOUT"
            if attempt < max_retries: time.sleep(3)
        except Exception as e:
            last_err = str(e)[:40]
            if attempt < max_retries: time.sleep(1)
    return f"[{last_err}]"

# ======================== 单张 (一次打开, 取色+编码复用) ========================

def process_one(idx, img_path, max_retries):
    filename = os.path.basename(img_path)
    t0 = time.time()
    try:
        # 一次打开, 两步复用
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        # 取色 (左侧 30%)
        icon_w = int(w * 0.30)
        region = np.array(img.crop((0, 0, icon_w, h)), dtype=np.uint8).reshape(-1, 3)
        color = extract_icon_color_from_array(region)
        grade = color_to_grade(color)
        # VL 识别 (缩小 + JPEG 压缩)
        img_b64 = prepare_vl_b64(img)
        text = ocr_one(img_b64, max_retries)
        elapsed = time.time() - t0
        ok = not text.startswith("[")
        return {"grade":grade,"name":text,"filename":filename,
                "idx":idx,"elapsed":elapsed,"success":ok}
    except Exception as e:
        return {"grade":"white","name":f"[ERR]{e}","filename":filename,
                "idx":idx,"elapsed":time.time()-t0,"success":False}

# ======================== Main ========================

def main():
    p = argparse.ArgumentParser(description="Qwen3-VL:8b OCR + 取色 (极速版)")
    p.add_argument("--dir", default=IMAGE_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"并发数 (默认={DEFAULT_WORKERS}, 建议 2-4)")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--no-retry", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    workers = max(1, min(args.workers, 6))
    max_retries = 0 if args.no_retry else RETRIES

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.output_dir, f"ocr_color_{ts}.json")

    print("="*60)
    print(f"  Qwen3-VL:8b | {workers}w | resize→{VL_MAX_SIZE}px "
          f"| JPEG q={VL_JPEG_QUALITY} | tokens={MAX_TOKENS} | timeout={OCR_TIMEOUT}s")
    print("="*60)

    # [1] 扫描
    print(f"\n[1/4] 扫描: {args.dir}")
    images = scan_images(args.dir)
    if not images: print("  ❌ 无图片"); sys.exit(1)
    if args.limit>0: images=images[:args.limit]
    total=len(images)
    print(f"  找到 {total} 张")

    # [2] Ollama
    print(f"\n[2/4] 检测 Ollama")
    check_ollama()

    # [3] 预热
    if not args.no_warmup:
        print(f"\n[3/4] 预热")
        warmup_model()
    else:
        print(f"\n[3/4] 跳过预热")

    # [4] 处理
    print(f"\n[4/4] 处理 {total} 张 ({workers} workers, retries={max_retries})...")
    print("-"*60)

    t_all = time.time()
    results = []; ok=0; ng=0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one,i,img,max_retries):i for i,img in enumerate(images,1)}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r["success"]: ok+=1
            else: ng+=1
            done=len(results)
            preview=r["name"].replace("\n"," ")[:40]
            per=f"{r['elapsed']*1000:.0f}ms" if r["elapsed"]<1 else f"{r['elapsed']:.1f}s"
            eta_s=(time.time()-t_all)/done*(total-done) if done>0 else 0
            mm,ss=divmod(int(eta_s),60)
            icon="✅" if r["success"] else "❌"
            print(f"[{done:3d}/{total}] {r['filename']:<22s} {per:>7s} "
                  f"{icon} {r['grade']:6s} {preview}  ETA {mm}m{ss}s")

    total_t=time.time()-t_all
    results.sort(key=lambda x:x["idx"])
    stimes=[x["elapsed"] for x in results if x["success"]]
    avg_ms=sum(stimes)/len(stimes)*1000 if stimes else 0
    tp=ok/total_t*60 if total_t>0 else 0

    print(f"\n{'='*60}")
    print(f"  ✅ {ok}/{total}  |  {ng} fail")
    print(f"  {total_t:.1f}s total  |  avg {avg_ms:.0f}ms/img  |  {tp:.0f} img/min")
    if stimes:
        print(f"  最快 {min(stimes)*1000:.0f}ms  |  最慢 {max(stimes)*1000:.0f}ms")
    print(f"  📄 {out}")

    j = {"_meta":{
        "total":total,"success":ok,"failed":ng,
        "total_time_sec":round(total_t,1),"avg_time_ms":round(avg_ms,1),
        "throughput_per_min":round(tp,1),
        "generated_at":time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine":f"Qwen3-VL:8b ({workers}w, resize{VL_MAX_SIZE}px, JPEG q={VL_JPEG_QUALITY}, tokens={MAX_TOKENS})"},
        "results":[{"grade":x["grade"],"name":x["name"]} for x in results]}
    with open(out,"w",encoding="utf-8") as f:
        json.dump(j,f,ensure_ascii=False,indent=2)

    print(f"\n--- 预览 ---")
    for i,x in enumerate(results[:10],1):
        print(f"  {i}. [{x['grade']:6s}] {x['name'][:50]}")
    if len(results)>10: print(f"  ... +{len(results)-10} more")

if __name__=="__main__":
    main()
