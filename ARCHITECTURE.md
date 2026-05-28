# LLM Benchmark Platform — Architecture

> Version: 2.0
> Updated: 2026-05-28
> Architecture: Streamlit UI + Core Engine

---

## Project Structure

```
llm-test/
├── app.py                       # Streamlit entry point
├── config/                      # Configuration
│   ├── settings.py              # Provider endpoints, model options
│   ├── session_state.py         # Streamlit session state management
│   ├── secrets.py               # API key management
│   └── test_config_loader.py    # Test parameter presets
├── core/                        # Core engine
│   ├── benchmark_runner.py      # Main test orchestrator
│   ├── providers/               # LLM provider adapters
│   │   ├── base.py              # BaseProvider ABC
│   │   ├── openai.py            # OpenAI-compatible provider
│   │   ├── gemini.py            # Google Gemini provider
│   │   ├── factory.py           # Provider factory
│   │   └── stream_parser.py     # SSE stream parser
│   ├── database/                # SQLite persistence
│   │   ├── connection.py        # Database connection & project root resolution
│   │   ├── manager.py           # Database manager
│   │   ├── schema.py            # Schema definitions
│   │   └── backup.py            # Backup/restore
│   ├── results/                 # Result history service
│   ├── benchmark/               # Benchmark metrics calculation
│   ├── services/                # Business logic (data export/import)
│   ├── repositories/            # Data access layer (repository pattern)
│   ├── models/                  # SQLAlchemy ORM models
│   ├── request_logger.py        # API request logging
│   ├── tokenizer_utils.py       # Tokenizer management (tiktoken, HF)
│   ├── dataset_loader.py        # Dataset loading
│   ├── dataset_manager.py       # Dataset download & caching
│   ├── quality_evaluator.py     # Quality evaluation orchestration
│   ├── safe_executor.py         # Sandboxed code execution (AST validation)
│   └── ...
├── evaluators/                  # Quality evaluators (auto-discovered)
│   ├── __init__.py              # @register_evaluator decorator
│   ├── base_evaluator.py        # BaseEvaluator ABC
│   ├── mmlu_evaluator.py        # MMLU — 57-subject knowledge
│   ├── gsm8k_evaluator.py       # GSM8K — grade school math
│   ├── math500_evaluator.py     # MATH500 — competition math
│   ├── humaneval_evaluator.py   # HumanEval — Python coding
│   └── ... (17 total)
├── ui/                          # Streamlit UI components
│   ├── sidebar.py               # Configuration sidebar
│   ├── test_panels.py           # Test execution panels
│   ├── test_runner.py           # TestExecutor class
│   ├── charts.py                # Plotly visualization
│   ├── insights.py              # Performance insight display
│   ├── reporting/               # Report builders & columns
│   └── ...
├── utils/                       # Utilities
│   ├── test_config_manager.py   # Test configuration presets
│   ├── custom_config.py         # Custom provider/model management
│   ├── dataset_downloader.py    # HuggingFace dataset download
│   ├── helpers.py               # Helper functions
│   └── ...
├── scripts/                     # Development & utility scripts
└── tests/                       # Test suites (653 tests)
    ├── engine/                  # Engine module tests
    ├── server/                  # API endpoint tests
    └── test_*.py                # Individual test modules
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Sidebar  │  │  Test    │  │   Reports &      │  │
│  │ Config   │  │  Panels  │  │   Insights       │  │
│  └────┬─────┘  └────┬─────┘  └──────────────────┘  │
│       │              │                               │
│       ▼              ▼                               │
│  ┌──────────────────────────────────────────────┐   │
│  │           TestExecutor                        │   │
│  │  Config validation → Runner creation → Async  │   │
│  └────────────────────┬─────────────────────────┘   │
└───────────────────────┼─────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│                  Core Engine                           │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Benchmark  │  │  Provider    │  │  Database    │  │
│  │ Runner     │  │  Adapters    │  │  (SQLite)    │  │
│  │            │  │              │  │              │  │
│  │ • Concur.  │  │ • OpenAI     │  │ • Results    │  │
│  │ • Prefill  │  │ • Gemini     │  │ • Logs       │  │
│  │ • Long CTX │  │ • Custom URL │  │ • History    │  │
│  │ • Matrix   │  │              │  │              │  │
│  │ • Stability│  │              │  │              │  │
│  └────────────┘  └──────────────┘  └──────────────┘  │
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │         Evaluator Plugin System               │    │
│  │  @register_evaluator → auto-discovered        │    │
│  │  MMLU, GSM8K, HumanEval, ... (17 total)      │    │
│  └──────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
```

---

## Design Patterns

| Pattern | Implementation | Location |
|---------|---------------|----------|
| **Provider Abstraction** | BaseProvider ABC | `core/providers/` |
| **Plugin/Evaluator** | `@register_evaluator` decorator | `evaluators/` |
| **Repository** | Data access abstraction | `core/repositories/` |
| **Factory** | Provider creation | `core/providers/factory.py` |
| **Session State** | Centralized state management | `config/session_state.py` |
| **Safe Execution** | AST validation + sandbox | `core/safe_executor.py` |

---

## Data Flow

### Performance Test Flow

```
User (UI)                   TestExecutor              BenchmarkRunner
─────────                   ─────────────             ───────────────
Select config ──→ Validate config
                   Create runner ──→ __init__(provider, tokenizer)
Run test ────────→ Execute async ──→ run_concurrency_test()
                                       │
                                       ▼
                                   Provider.get_completion()
                                       │
                                       ▼
                                   Parse metrics (TTFT, TPS, etc.)
                                       │
                                       ▼
Display results ←── Collect results ←── DataFrame
```

### Quality Evaluation Flow

```
User (UI)                   Evaluator System
─────────                   ─────────────────
Select dataset ──→ BaseEvaluator.evaluate_batch()
                       │
                       ▼
                   Load dataset samples
                       │
                       ▼
                   For each sample:
                     LLM call → parse answer → compare
                       │
                       ▼
                   EvaluationResult (accuracy, details)
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **UI** | Streamlit |
| **HTTP** | httpx (async), requests |
| **Charts** | Plotly |
| **Database** | SQLAlchemy + SQLite |
| **API Server** | FastAPI + Uvicorn |
| **Tokenizers** | tiktoken, HuggingFace transformers |
| **Data** | pandas, numpy |
| **Quality Eval** | 17 dataset evaluators |
| **Testing** | pytest |

---

## Getting Started

```bash
# Install
pip install -e ".[dev]"

# Run
streamlit run app.py              # → http://localhost:8501

# Test
python -m pytest tests/ -v        # 653 tests
```
