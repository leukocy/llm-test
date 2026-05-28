# 数据集下载指南

本文档说明如何下载和配置各个测试集数据。

---

## 🆕 新增测试集

### 1. GPQA (Graduate-Level Google-Proof Q&A)

**难度**: ⭐⭐⭐⭐⭐ (研究生级别)  
**领域**: 物理、化学、生物  
**数据量**: ~450 题 (Diamond 子集)

**下载方式**:
```bash
# 方式1: 使用 HuggingFace datasets
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
ds['train'].to_json('datasets/gpqa/gpqa_diamond.json')
"

# 方式2: 手动下载
# 访问 https://huggingface.co/datasets/Idavidrein/gpqa
# 下载 gpqa_diamond.json 到 datasets/gpqa/ 目录
```

**数据格式**:
```json
{
    "question": "In quantum mechanics...",
    "choices": ["Option A", "Option B", "Option C", "Option D"],
    "answer": 2,
    "domain": "physics"
}
```

---

### 2. ARC-Challenge (AI2 Reasoning Challenge)

**难度**: ⭐⭐⭐⭐ (中学科学)  
**领域**: 科学常识推理  
**数据量**: ~2,500 题 (Challenge 子集)

**下载方式**:
```bash
# 方式1: 使用 HuggingFace datasets
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge')
ds['test'].to_json('datasets/arc/ARC-Challenge-Test.jsonl', orient='records', lines=True)
"

# 方式2: 直接下载
# 访问 https://huggingface.co/datasets/allenai/ai2_arc
# 或 https://allenai.org/data/arc
```

**数据格式** (ARC 原始格式):
```json
{
    "id": "Mercury_7220990",
    "question": {
        "stem": "Which property of a mineral...",
        "choices": [
            {"text": "hardness", "label": "A"},
            {"text": "color", "label": "B"}
        ]
    },
    "answerKey": "B"
}
```

---

### 3. TruthfulQA (真实性测试)

**难度**: ⭐⭐⭐⭐ (常见误解)  
**领域**: 迷信、阴谋论、健康、法律等  
**数据量**: 817 题

**下载方式**:
```bash
# 方式1: 使用 HuggingFace datasets
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')
ds['validation'].to_json('datasets/truthfulqa/truthfulqa_mc.json')
"

# 方式2: 从 GitHub 下载
# 访问 https://github.com/sylinrl/TruthfulQA/tree/main/data
# 下载 mc_task.json 到 datasets/truthfulqa/ 目录
```

**数据格式**:
```json
{
    "question": "What happens if you eat watermelon seeds?",
    "mc1_targets": {
        "choices": ["Nothing happens", "You will grow a watermelon in your stomach"],
        "labels": [1, 0]
    },
    "category": "Misconceptions"
}
```

---

## 📁 目录结构

```
datasets/
├── mmlu/           # MMLU 通用知识
├── gsm8k/          # GSM8K 小学数学
├── math500/        # MATH-500 高等数学
├── humaneval/      # HumanEval 代码生成
├── gpqa/           # 🆕 GPQA 高级推理
│   └── gpqa_diamond.json
├── arc/            # 🆕 ARC 科学推理
│   └── ARC-Challenge-Test.jsonl
└── truthfulqa/     # 🆕 TruthfulQA 真实性
    └── truthfulqa_mc.json
```

---

## 🔧 一键下载脚本

将以下脚本保存为 `download_datasets.py` 并运行:

```python
import os
import json

try:
    from datasets import load_dataset
except ImportError:
    print("正在安装 datasets 库...")
    os.system("pip install datasets")
    from datasets import load_dataset

def download_gpqa():
    print("📥 下载 GPQA Diamond...")
    try:
        ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', trust_remote_code=True)
        os.makedirs('datasets/gpqa', exist_ok=True)
        ds['train'].to_json('datasets/gpqa/gpqa_diamond.json')
        print("✅ GPQA 下载完成")
    except Exception as e:
        print(f"❌ GPQA 下载失败: {e}")

def download_arc():
    print("📥 下载 ARC-Challenge...")
    try:
        ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge')
        os.makedirs('datasets/arc', exist_ok=True)
        ds['test'].to_json('datasets/arc/ARC-Challenge-Test.jsonl', orient='records', lines=True)
        print("✅ ARC 下载完成")
    except Exception as e:
        print(f"❌ ARC 下载失败: {e}")

def download_truthfulqa():
    print("📥 下载 TruthfulQA...")
    try:
        ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')
        os.makedirs('datasets/truthfulqa', exist_ok=True)
        ds['validation'].to_json('datasets/truthfulqa/truthfulqa_mc.json')
        print("✅ TruthfulQA 下载完成")
    except Exception as e:
        print(f"❌ TruthfulQA 下载失败: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("开始下载新增测试集...")
    print("=" * 50)
    
    download_gpqa()
    download_arc()
    download_truthfulqa()
    
    print("\n✨ 下载完成!")
```

---

## 📊 测试集对比

| 数据集 | 难度 | 题量 | 类型 | 用途 |
|--------|------|------|------|------|
| MMLU | ⭐⭐⭐ | 14,042 | 选择题 | 通用知识 |
| GSM8K | ⭐⭐⭐ | 8,500 | 数学题 | 数学推理 |
| MATH-500 | ⭐⭐⭐⭐ | 500 | 数学题 | 竞赛数学 |
| HumanEval | ⭐⭐⭐ | 164 | 代码 | 代码生成 |
| **GPQA** 🆕 | ⭐⭐⭐⭐⭐ | ~450 | 选择题 | 高级推理 |
| **ARC** 🆕 | ⭐⭐⭐⭐ | ~2,500 | 选择题 | 科学常识 |
| **TruthfulQA** 🆕 | ⭐⭐⭐⭐ | 817 | 选择题 | 真实性 |

---

## ⚠️ 注意事项

1. **网络要求**: 下载需要访问 HuggingFace，如有网络问题可使用代理
2. **API 消耗**: 完整测试会消耗大量 API 调用，建议先使用快速抽样 (100题)
3. **思考模型**: 对于 DeepSeek-R1, o1 等思考模型，建议:
   - 温度设置为 0.6
   - max_tokens 设置为 1024+
   - 并发数适当降低
