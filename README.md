# LLM Benchmark Platform

> LLM 性能基准测试与质量评估平台
>
> **当前界面**: Streamlit UI (`app.py`) | **评估器**: 19 个质量数据集 | **Provider**: 15+

---

## Features

- **Performance Benchmarks**: 并发、Prefill、长上下文、矩阵、稳定性等多维度性能测试
- **Quality Evaluation**: 19 个数据集评估器 (MMLU, GSM8K, MATH500, HumanEval, GPQA 等)
- **Provider 支持**: DeepSeek, Moonshot, Gemini, MiMo, ZhiPu, Volcengine, Alibaba Bailian, SiliconFlow, OpenRouter, Ollama 等 15+ 推理服务
- **可视化报告**: 交互式图表 (Plotly), 静态图表导出 (Matplotlib), A+~D 评级
- **实时日志**: WebSocket 流式传输测试进度与日志
- **数据导出**: CSV / Excel 自动保存, 可配置列排序

---

## Quick Start

### Install

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac
pip install -r requirements.txt
```

### Configure API Keys

```bash
cp .env.example .env
# 编辑 .env 填入 API Key
```

### Run

```bash
streamlit run app.py
# → http://localhost:8501
```

或使用启动脚本:

```bash
./start.sh          # Linux/Mac
start.bat           # Windows
```

---

## Project Structure

```
├── app.py                 # Streamlit 主入口
├── core/                  # 核心引擎
│   ├── benchmark_runner.py    # 性能测试编排器
│   ├── quality_evaluator.py   # 质量评估逻辑
│   ├── providers/             # LLM Provider 适配器
│   ├── services/              # 业务逻辑层
│   ├── repositories/          # 数据访问层 (Repository Pattern)
│   ├── models/                # SQLAlchemy ORM 模型
│   ├── prompt_template.py     # Jinja2 模板引擎
│   └── safe_executor.py       # 安全代码执行 (AST 校验)
├── evaluators/            # 质量评估器插件 (19 个, @register_evaluator 自动发现)
├── ui/                    # Streamlit UI 组件
├── config/                # 配置 (Provider、Session State)
├── utils/                 # 工具函数 (日志、WebSocket 服务器)
├── tests/                 # 测试套件
├── presets/               # 测试预设 (JSON)
├── task_configs/          # 任务配置 (YAML)
└── scripts/               # 辅助脚本
```

---

## Quality Evaluators

| Evaluator | Dataset | Type |
|-----------|---------|------|
| `mmlu_evaluator` | MMLU | 57-subject knowledge |
| `gsm8k_evaluator` | GSM8K | Grade school math |
| `math500_evaluator` | MATH500 | Competition math |
| `humaneval_evaluator` | HumanEval | Python code generation |
| `mbpp_evaluator` | MBPP | Python code generation |
| `gpqa_evaluator` | GPQA | Graduate-level science |
| `aime_evaluator` | AIME 2025 | Competition math |
| `arc_evaluator` | ARC | Science reasoning |
| `hellaswag_evaluator` | HellaSwag | Commonsense reasoning |
| `winogrande_evaluator` | WinoGrande | Coreference resolution |
| `truthfulqa_evaluator` | TruthfulQA | Truthfulness |
| `needle_haystack_evaluator` | Needle Haystack | Long context retrieval |
| `longbench_evaluator` | LongBench | Long context understanding |
| `arena_hard_evaluator` | Arena Hard | Hard prompts |
| `global_piqa_evaluator` | Global PIQA | Physical reasoning |
| `swebench_evaluator` | SWE-Bench Lite | Software engineering |
| `custom_needle_evaluator` | Custom Needle | Custom retrieval tests |
| `open_ended_evaluator` | Open Ended | Open-ended generation |
| `yaml_evaluator` | YAML Config | Custom YAML-based eval |

---

## Supported Providers

| Provider | API Base |
|----------|----------|
| OpenAI Compatible (heyi, llama.cpp, LM Studio) | 本地 / 自定义 |
| DeepSeek | api.deepseek.com |
| Moonshot (Kimi) | api.moonshot.cn |
| Gemini | generativelanguage.googleapis.com |
| MiMo | api.xiaomimimo.com |
| ZhiPu (GLM) | open.bigmodel.cn |
| Volcengine | ark.cn-beijing.volces.com |
| Alibaba Bailian | dashscope.aliyuncs.com |
| SiliconFlow | api.siliconflow.cn |
| OpenRouter | openrouter.ai |
| MiniMax | api.minimax.chat |
| Ollama | localhost:11434 |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Specific modules
python -m pytest tests/engine/ tests/server/ -v

# With coverage
python -m pytest tests/ -v --cov=core --cov=evaluators --cov=utils
```

## Linting & Formatting

```bash
ruff check .
ruff format .
mypy core/ evaluators/ utils/
```

---

## Configuration

- **API Keys**: `.env` file (see `.env.example`)
- **Provider Settings**: `config/settings.py`
- **Test Presets**: `presets/` directory (JSON)
- **Task Configs**: `task_configs/` directory (YAML)

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture overview |
| [docs/API.md](docs/API.md) | REST API reference (FastAPI backend) |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Development workflow |
| [docs/SECURITY.md](docs/SECURITY.md) | Security guidelines |
| [CLAUDE.md](CLAUDE.md) | AI assistant instructions |

---

## License

MIT License
