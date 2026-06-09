# Ollama Qwen3-VL 批量OCR

## 环境要求

- Python 3.8+
- Ollama 已安装并运行
- qwen3-vl:8b 模型（脚本会自动拉取）

## 安装依赖

```bash
pip install requests
```

## 使用方式

```bash
# 处理全部 703 张图片
python ollama_ocr.py

# 先测试前 5 张（确认效果）
python ollama_ocr.py --limit 5
```

## 输出

识别结果会保存到 `D:\a-test\cutQP\ocr_results.txt`，格式如下：

```
# Ollama Qwen3-VL OCR 结果
# 模型: qwen3-vl:8b
# 时间: 2026-06-08 19:55:00
# 总数: 703 | 成功: 703 | 失败: 0

## 📷 枪皮_1.jpg
AUG突击步枪-西装暴徒

## 📷 枪皮_2.jpg
M416突击步枪-金色传说
...
```
