# LLM 性能基准测试平台 - 项目状态文档

> 版本: 2.0  
> 更新日期: 2025-12-23  
> 状态: 活跃开发中

---

## 📋 项目概述

**LLM 性能基准测试平台** 是一个全面的大语言模型评测工具，支持性能测试和质量评估两大核心功能。平台采用 Streamlit 构建，提供直观的 Web 界面，支持多平台、多模型的统一测试和对比分析。

### 核心特性

- 🚀 **性能测试**: 并发、Prefill、长上下文、稳定性等多维度性能评估
- 📝 **质量评估**: 支持 15+ 标准数据集的模型能力评测
- 🔍 **智能分析**: 自动失败分析、推理质量评估、一致性检测
- 🎯 **多平台支持**: MiMo、DeepSeek、OpenAI、智谱AI、火山引擎等
- 📊 **可视化报告**: 实时仪表板、交互式图表、多格式导出

---

## ✅ 已完成工作 (Phase 1-2)

### Phase 1: 核心功能建设 (v1.0)

| 模块 | 文件 | 功能描述 | 状态 |
|------|------|----------|------|
| 基准测试运行器 | `core/benchmark_runner.py` | 并发测试、Prefill测试、长上下文测试 | ✅ 完成 |
| 质量评估器 | `core/quality_evaluator.py` | 数据集加载、模型调用、结果统计 | ✅ 完成 |
| 评估器基类 | `evaluators/base_evaluator.py` | 统一的评估接口和数据结构 | ✅ 完成 |
| 数据集评估器 | `evaluators/*.py` | MMLU、GSM8K、MATH500 等 15 个数据集 | ✅ 完成 |
| 结果报告 | `ui/quality_reports.py` | 质量评估结果可视化 | ✅ 完成 |
| 配置管理 | `config/settings.py` | 供应商和模型配置 | ✅ 完成 |

### Phase 2: 质量测试增强 (v2.0)

| 模块 | 文件 | 功能描述 | 状态 |
|------|------|----------|------|
| 平台特性表 | `core/thinking_params.py` | 多平台思考参数标准化 | ✅ 完成 |
| 统一响应解析器 | `core/response_parser.py` | 流式响应解析、多平台适配 | ✅ 完成 |
| 推理指标计算 | `core/metrics.py` | TTUT、推理Token比例等 | ✅ 完成 |
| 重试机制 | `core/retry_handler.py` | 指数退避、错误处理 | ✅ 完成 |
| 增强评测报告 | `core/evaluation_report.py` | JSON/MD/HTML 多格式导出 | ✅ 完成 |
| UI 增强组件 | `ui/thinking_components.py` | 推理折叠展示、仪表盘 | ✅ 完成 |
| 智能答案解析器 | `core/smart_answer_parser.py` | 分层解析、LLM兜底 | ✅ 完成 |
| 推理质量评估器 | `core/reasoning_evaluator.py` | 5维度推理评估 | ✅ 完成 |
| 增强型评估器 | `core/enhanced_evaluator.py` | 统一评估接口 | ✅ 完成 |
| YAML配置系统 | `core/test_config.py` | 可复现测试配置 | ✅ 完成 |
| 失败分析系统 | `core/failure_analyzer.py` | 13类失败分类 | ✅ 完成 |
| 一致性测试 | `core/consistency_tester.py` | 答案稳定性评估 | ✅ 完成 |
| 鲁棒性测试 | `core/robustness_tester.py` | 输入扰动敏感性 | ✅ 完成 |
| 测试运行器 | `core/test_runner.py` | 统一测试执行器 | ✅ 完成 |
| 评测仪表板 | `ui/evaluation_dashboard.py` | 可视化组件集 | ✅ 完成 |

---

## 📁 项目结构

```
llm-benchmark-platform/
│
├── app.py                      # 主入口（Streamlit）
├── start.bat / start.sh        # 启动脚本
├── requirements.txt            # 依赖列表
│
├── core/                       # 核心模块
│   ├── benchmark_runner.py     # 性能测试运行器
│   ├── quality_evaluator.py    # 质量评估器
│   ├── smart_answer_parser.py  # 智能答案解析
│   ├── reasoning_evaluator.py  # 推理质量评估
│   ├── failure_analyzer.py     # 失败分析
│   ├── consistency_tester.py   # 一致性测试
│   ├── robustness_tester.py    # 鲁棒性测试
│   ├── test_config.py          # 配置系统
│   ├── test_runner.py          # 统一运行器
│   ├── thinking_params.py      # 平台参数
│   ├── response_parser.py      # 响应解析
│   ├── metrics.py              # 指标计算
│   ├── retry_handler.py        # 重试机制
│   └── evaluation_report.py    # 报告生成
│
├── evaluators/                 # 数据集评估器
│   ├── base_evaluator.py       # 基类
│   ├── mmlu_evaluator.py       # MMLU
│   ├── gsm8k_evaluator.py      # GSM8K
│   ├── math500_evaluator.py    # MATH-500
│   ├── humaneval_evaluator.py  # HumanEval
│   ├── gpqa_evaluator.py       # GPQA
│   └── ...                     # 更多评估器
│
├── ui/                         # UI 组件
│   ├── quality_reports.py      # 质量报告
│   ├── thinking_components.py  # 思考展示
│   ├── evaluation_dashboard.py # 评测仪表板
│   ├── reports.py              # 性能报告
│   └── log_viewer.py           # 日志查看器
│
├── config/                     # 配置
│   └── settings.py             # 供应商/模型设置
│
├── utils/                      # 工具模块
│   ├── helpers.py              # 通用辅助函数
│   ├── preset_manager.py       # 预设管理
│   └── custom_config.py        # 自定义配置
│
├── datasets/                   # 数据集存储
├── quality_results/            # 评测结果
├── tests/config/               # 测试配置文件
└── docs/                       # 文档
```

---

## 🔧 技术栈

| 类别 | 技术 |
|------|------|
| **后端** | Python 3.10+, asyncio, aiohttp |
| **前端** | Streamlit, Plotly, Pandas |
| **API 客户端** | httpx, OpenAI SDK |
| **NLP 工具** | HuggingFace Tokenizers |
| **配置** | YAML, JSON |
| **测试** | pytest, unittest |

---

## 📊 支持的数据集

| 数据集 | 类型 | 样本数 | 描述 |
|--------|------|--------|------|
| MMLU | 选择题 | 14,037 | 57学科通用知识 |
| GSM8K | 数学推理 | 1,314 | 小学数学应用题 |
| MATH-500 | 数学推理 | 496 | 竞赛级数学题 |
| HumanEval | 代码生成 | 164 | Python 编程题 |
| GPQA | 专家推理 | 195 | 研究生级科学题 |
| TruthfulQA | 真实性 | 812 | 误区/谎言检测 |
| HellaSwag | 常识推理 | 10,037 | 场景续写选择 |
| Winogrande | 代词消歧 | 1,262 | 常识推理 |
| ARC | 科学推理 | 1,167 | 小学科学题 |
| MBPP | 代码生成 | 497 | Python 基础编程 |
| LongBench | 长文本 | 4,150 | 长上下文理解 |
| Needle-Haystack | 检索 | 30 | 大海捞针测试 |
| C-Eval | 中文知识 | 13,000+ | 中文综合评估 |
| AIME | 数学竞赛 | 90 | 数学奥赛题目 |

---

## 🎯 支持的平台

| 平台 | 状态 | 特殊支持 |
|------|------|----------|
| OpenAI | ✅ | o1/o3 思考模式 |
| MiMo | ✅ | 原生思考支持 |
| DeepSeek | ✅ | R1 推理模型 |
| 智谱 AI | ✅ | GLM-4 系列 |
| 火山引擎 | ✅ | 豆包模型 |
| 阿里百炼 | ✅ | Qwen 系列 |
| OpenRouter | ✅ | 多模型聚合 |
| 自定义 | ✅ | OpenAI 兼容 API |

---

## 📈 质量测试改进对比

### 改进前 vs 改进后

| 维度 | 改进前 | 改进后 |
|------|--------|--------|
| **答案解析** | 简单正则匹配 | 分层解析 + LLM 兜底 |
| **评估维度** | 仅准确率 | 准确率 + 推理质量 + 一致性 |
| **失败分析** | 无 | 13类自动分类 + 建议 |
| **配置管理** | 硬编码 | YAML 可复现配置 |
| **稳定性测试** | 无 | 一致性 + 鲁棒性测试 |
| **报告格式** | 仅界面显示 | JSON/MD/HTML 导出 |

### 预期效果

- **假阴性减少**: 15-20%（智能解析器）
- **问题定位效率**: 提升 50%（失败分析）
- **测试可复现性**: 100%（YAML 配置）
- **评估维度**: 从 1 维扩展到 8+ 维度

---

## 🚀 未来开发路线图

### Phase 3: 深度评估能力 (计划中)

| 功能 | 描述 | 优先级 |
|------|------|--------|
| LLM-as-Judge 集成 | 使用强模型评估弱模型输出质量 | P0 |
| 多轮对话评估 | 支持对话一致性和上下文理解测试 | P0 |
| 幻觉检测 | 检测模型输出中的虚假信息 | P1 |
| 安全性测试 | 提示注入、越狱尝试检测 | P1 |
| 偏见测试 | 检测模型的社会偏见倾向 | P2 |

### Phase 4: 自动化与集成 (规划中)

| 功能 | 描述 | 优先级 |
|------|------|--------|
| CI/CD 集成 | GitHub Actions 自动评测 | P0 |
| 回归测试 | 模型版本对比自动化 | P0 |
| API 接口 | RESTful 评测服务 | P1 |
| 批量调度 | 定时任务、队列管理 | P1 |
| Slack/飞书通知 | 评测结果自动推送 | P2 |

### Phase 5: 高级分析 (远期规划)

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 统计显著性测试 | A/B 测试置信区间计算 | P1 |
| 成本效益分析 | Token 成本 vs 性能权衡 | P1 |
| 模型推荐系统 | 根据需求推荐最佳模型 | P2 |
| 自定义评估 DSL | 领域特定评估语言 | P2 |
| 多模态支持 | 图像理解、语音评估 | P3 |

---

## 🛠️ 快速开始

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd llm-benchmark-platform

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
# 启动 Streamlit 应用
streamlit run app.py

# 或使用启动脚本
./start.bat  # Windows
./start.sh   # Linux/Mac
```

### 使用 YAML 配置运行测试

```python
from core.test_runner import UnifiedTestRunner

runner = UnifiedTestRunner()
result = await runner.run_from_config(
    "tests/config/gsm8k_comparison.yaml",
    samples=test_samples,
    get_response_funcs={"mimo": mimo_api}
)
```

---

## 📝 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v2.0 | 2025-12-23 | 质量测试增强、智能解析、失败分析、一致性测试 |
| v1.5 | 2025-12-18 | UI 重构、新数据集支持 |
| v1.0 | 2025-11-20 | 初始版本、基础性能测试 |

---

## 👥 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

MIT License

---

*最后更新: 2025-12-23*
