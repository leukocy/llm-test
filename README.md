# LLM Benchmark Platform

> A comprehensive LLM performance & quality evaluation platform
> Automated regression tests cover performance, quality, security, and reporting.

---

## Features

- **Performance Benchmarks**: Concurrency, Prefill, Long Context, Matrix,
  Stability, and more
- **Quality Evaluators**: MMLU, GSM8K, MATH500, HumanEval, GPQA, etc.
- **Real-time Dashboard**: Live progress, throughput charts, and system insights
- **Reports**: Expert performance insights, grading, and interactive visual analysis
- **CSV Export**: Auto-saved results with configurable column ordering
- **Providers**: DeepSeek, ZhiPu, MiniMax, OpenRouter, SiliconFlow, Gemini,
  and more

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

## Datasets

The platform uses benchmark datasets for **quality evaluation** and **prompt-suffix
pools** (the builder-class tests draw questions from these pools).

### Self-contained (shipped)

- **AIME** (2024/2025/2026) — 90 math problems, pre-measured and bucketed in
  `aime_stable_pools.json` by stable decode-fill window. The **Math** prompt-suffix
  type and **Custom Text → Test Pool Problems** work immediately after clone. [OK]

### Not shipped (in `.gitignore`, optional)

These are **optional** — the platform runs without them, but the Science / Code /
Longform prompt-suffix types will have empty pools:

Quality benchmarks require their selected dataset. If it is missing, the run stops
instead of scoring built-in examples. Demonstration examples require
`LLM_TEST_ALLOW_EMBEDDED_SAMPLES=1` and are labeled in result files and reports.

<!-- markdownlint-disable MD013 -->

| Dataset | Type | Source |
|---------|------|--------|
| **GPQA Diamond** | Science | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) (gated) |
| **HumanEval** | Code | [openai_humaneval](https://huggingface.co/datasets/openai_human_eval) |
| **MBPP** | Code | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| **LongBench** | Longform | [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench) |
| **SWE-Bench Lite** | Code | [princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) |

<!-- markdownlint-enable MD013 -->

To download:

```bash
# Install huggingface datasets library
pip install datasets huggingface_hub

# For gated datasets (GPQA), log in first:
huggingface-cli login

# Download all
python scripts/download_datasets.py --all

# Or individual ones
python scripts/download_datasets.py --gpqa
python scripts/download_datasets.py --humaneval --mbpp

# Check status
python scripts/download_datasets.py --status
```

---

## Project Structure

```text
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

OpenAI-compatible endpoints are supported through the
"Custom (OpenAI Compatible)" provider option.

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
| [docs/METRICS_CONTRACT.md](docs/METRICS_CONTRACT.md) | Versioned performance metric definitions |
| [docs/INDUSTRIAL_READINESS.md](docs/INDUSTRIAL_READINESS.md) | Single-tenant readiness gates and rollout plan |

---

## License

MIT License
