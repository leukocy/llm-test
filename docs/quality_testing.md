# 模型质量测试模块

## 概述

模型质量测试模块用于评估 LLM 在标准公开测试集上的表现，涵盖通用知识、数学推理、代码生成等多个维度。

## 支持的测试集

| 测试集 | 类型 | 样本数 | 指标 | 状态 |
|--------|------|--------|------|------|
| **MMLU** | 多项选择题 (57学科) | 14,042 | Accuracy | ✅ 可用 |
| **GSM8K** | 小学数学应用题 | 8,792 | Accuracy | ✅ 可用 |
| **MATH-500** | 高等数学 (竞赛级) | 500 | Accuracy | ✅ 可用 |
| **HumanEval** | Python 代码生成 | 164 | pass@1 | ✅ 可用 |
| C-Eval | 中文知识测试 | 13,948 | Accuracy | 🚧 待实现 |

## 模块结构

```
test/
├── core/
│   └── quality_evaluator.py     # 主评估引擎
├── evaluators/
│   ├── __init__.py              # 评估器注册表
│   ├── base_evaluator.py        # 基础评估器类
│   ├── mmlu_evaluator.py        # MMLU 评估器
│   ├── gsm8k_evaluator.py       # GSM8K 评估器
│   ├── math500_evaluator.py     # MATH-500 评估器
│   └── humaneval_evaluator.py   # HumanEval 评估器
├── datasets/
│   ├── mmlu/                    # MMLU 数据集
│   ├── gsm8k/                   # GSM8K 数据集
│   ├── math500/                 # MATH-500 数据集
│   └── humaneval/               # HumanEval 数据集
├── utils/
│   └── dataset_downloader.py    # 数据集下载工具
├── ui/
│   └── quality_reports.py       # 质量报告可视化
└── quality_results/             # 评估结果输出目录
```

## 使用方法

### 1. 通过 UI 使用

1. 启动应用: `streamlit run app.py`
2. 在侧边栏选择 **"📝 模型质量测试"**
3. 配置参数:
   - 选择测试数据集 (MMLU, GSM8K, MATH-500, HumanEval)
   - 设置采样模式 (快速抽样/完整测试)
   - 调整 Few-shot 示例数
4. 点击 **"🚀 开始质量评估"**

### 2. 通过代码使用

```python
import asyncio
from core.quality_evaluator import QualityEvaluator, QualityTestConfig
from evaluators import EVALUATOR_REGISTRY

# 配置
config = QualityTestConfig(
    datasets=["mmlu", "gsm8k", "math500", "humaneval"],
    num_shots=5,
    max_samples=100,  # None = 全部
    temperature=0.0,
    max_tokens=256,
    concurrency=4
)

# 创建评估器
evaluator = QualityEvaluator(
    api_base_url="http://localhost:8000/v1",
    model_id="your-model-id",
    api_key="your-api-key",
    provider="OpenAI 兼容"
)

# 注册评估器类
for name, cls in EVALUATOR_REGISTRY.items():
    evaluator.register_evaluator(name, cls)

# 运行评估
async def main():
    results = await evaluator.run_evaluation(config)
    
    for dataset, result in results.items():
        print(f"{dataset}: {result.accuracy:.2%}")

asyncio.run(main())
```

### 3. 下载完整数据集

```bash
# 查看数据集状态
python utils/dataset_downloader.py --status

# 下载单个数据集
python utils/dataset_downloader.py -d math500
python utils/dataset_downloader.py -d humaneval

# 下载所有数据集
python utils/dataset_downloader.py --all
```

## 添加新测试集

1. 在 `evaluators/` 目录创建新评估器类:

```python
from evaluators.base_evaluator import BaseEvaluator

class NewDatasetEvaluator(BaseEvaluator):
    def load_dataset(self, subset=None):
        # 加载数据集
        pass
    
    def format_prompt(self, sample, include_answer=False):
        # 格式化 prompt
        pass
    
    def parse_response(self, response):
        # 解析模型响应
        pass
    
    def check_answer(self, predicted, correct):
        # 检查答案
        pass
```

2. 在 `evaluators/__init__.py` 中注册:

```python
from .new_dataset_evaluator import NewDatasetEvaluator

EVALUATOR_REGISTRY['new_dataset'] = NewDatasetEvaluator
```

3. 在 `datasets/new_dataset/` 目录放置数据文件

## 数据格式

### MMLU 格式
```json
{
  "question": "What is the capital of France?",
  "choices": ["London", "Paris", "Berlin", "Madrid"],
  "answer": 1,
  "subject": "geography"
}
```

### GSM8K 格式
```json
{
  "question": "Janet has 3 apples...",
  "answer": "...calculation steps...\n#### 8"
}
```

### MATH-500 格式
```json
{
  "problem": "Compute $\\dbinom{8}{4}$.",
  "solution": "...step by step...",
  "answer": "70",
  "subject": "Counting & Probability",
  "level": 1
}
```

### HumanEval 格式
```json
{
  "task_id": "HumanEval/0",
  "prompt": "def has_close_elements(...",
  "canonical_solution": "    for idx, elem in...",
  "test": "def check(candidate):...",
  "entry_point": "has_close_elements"
}
```

## 评估结果

结果自动保存到 `quality_results/{model_id}/` 目录:
- `{dataset}_{timestamp}.json` - 详细结果 (包含每题答案)
- `summary_{timestamp}.csv` - 汇总统计

## 注意事项

1. **API 消耗**: 完整测试会消耗大量 API 调用，建议先用快速抽样测试
2. **数据集下载**: 首次使用需要下载数据集，示例数据已包含在项目中
3. **Temperature**: 推荐使用 0.0 以获得确定性输出
4. **Few-shot 建议**:
   - MMLU: 5-shot
   - GSM8K: 8-shot
   - MATH-500: 4-shot
   - HumanEval: 0-shot (代码补全)
5. **代码执行安全**: HumanEval 会执行生成的 Python 代码，已在隔离环境中运行
