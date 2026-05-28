# LLM Benchmark Platform

> A comprehensive LLM performance & quality evaluation platform
> **Tests**: 653+ unit tests | **Strategies**: 8 performance benchmarks | **Evaluators**: 17 quality datasets

---

## Features

- **8 Performance Benchmarks**: Concurrency, Prefill, Long Context, Matrix, Stability, and more
- **17 Quality Evaluators**: MMLU, GSM8K, MATH500, HumanEval, GPQA, etc.
- **Real-time Dashboard**: Live progress, throughput charts, and system insights
- **Reports**: Expert performance insights, grading, and interactive visual analysis
- **CSV Export**: Auto-saved results with configurable column ordering
- **10+ Providers**: DeepSeek, ZhiPu, MiniMax, OpenRouter, SiliconFlow, Gemini, and more

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Install

```bash
# Clone the repository
git clone https://github.com/leukocy/llm-test.git
cd llm-test

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### Run

```bash
streamlit run app.py
# → http://localhost:8501
```

---

## Project Structure

```
├── app.py               # Streamlit entry point
├── config/              # Configuration & session state
├── core/                # Core engine
│   ├── benchmark_runner.py  # Main test orchestrator
│   ├── providers/       # LLM provider adapters (OpenAI, Gemini, etc.)
│   ├── database/        # SQLite persistence layer
│   ├── results/         # Result history service
│   └── ...
├── evaluators/          # 17 quality dataset evaluators (auto-discovered)
├── ui/                  # Streamlit UI components
│   ├── sidebar.py       # Sidebar configuration
│   ├── test_panels.py   # Test execution panels
│   ├── charts.py        # Visualization
│   └── ...
├── utils/               # Utilities (logging, presets, tokenizers)
├── scripts/             # Development & debugging scripts
└── tests/               # Test suites (653 tests)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture diagram.

---

## Performance Tests

| Test | Description |
|------|-------------|
| Concurrency | Throughput across concurrency levels |
| Prefill | Input processing speed (TTFT) |
| Long Context | Long input retention & decoding |
| Segmented | Prefix caching effectiveness |
| Matrix | Concurrency x Context sweep |
| Stability | Time-based consistency |
| Custom Text | User-defined prompts |
| Dataset | Quality evaluation with real datasets |

## Quality Evaluators

MMLU - GSM8K - MATH500 - HumanEval - GPQA - TruthfulQA - HellaSwag -
WinoGrande - ARC - MBPP - LongBench - Needle Haystack - AIME 2025 -
Arena Hard - Global PIQA - SWE-Bench Lite - Custom Needle

---

## Supported Providers

OpenAI Compatible (Custom URL) - DeepSeek - Moonshot (Kimi) - MiMo - Gemini -
ZhiPu (GLM) - Volcengine - Alibaba Bailian - SiliconFlow -
OpenRouter - MiniMax

Any OpenAI-compatible endpoint is supported via the "Custom (OpenAI Compatible)" provider option.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific modules
python -m pytest tests/engine/ tests/test_providers.py -v

# Run with coverage
python -m pytest tests/ -v --cov=core --cov=evaluators --cov=utils
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture overview |
| [CLAUDE.md](CLAUDE.md) | Development guide for AI assistants |
| [docs/API.md](docs/API.md) | API reference (FastAPI backend) |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Development workflow |
| [docs/SECURITY.md](docs/SECURITY.md) | Security guidelines |

---

## License

MIT License
