# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

LLM Benchmark Platform - A comprehensive LLM performance & quality evaluation platform. Streamlit UI (app.py) is the main interface. Python 3.10+, install with `pip install -e ".[dev]"`.

## Commands

```bash
# Run the app
streamlit run app.py

# Run all tests
python -m pytest tests/ -v

# Run specific test file or module
python -m pytest tests/test_benchmark_runner.py -v
python -m pytest tests/engine/ tests/server/ -v

# Run with coverage
python -m pytest tests/ -v --cov=core --cov=evaluators --cov=utils

# Lint & format
ruff check .
ruff format .

# Type check
mypy core/ evaluators/ utils/
```

## Architecture

### Entry Point & UI Layer

`app.py` → Streamlit UI. UI components live in `ui/` (sidebar, test panels, charts, insights, reporting). Session state centralized in `config/session_state.py`.

### Provider System

Two provider implementations in `core/providers/`:
- **OpenAIProvider** (`openai.py`): OpenAI-compatible API adapter (covers DeepSeek, Moonshot, MiMo, ZhiPu, Volcengine, Alibaba, SiliconFlow, OpenRouter, Ollama, local models)
- **GeminiProvider** (`gemini.py`): Google Gemini native API
- **Factory** (`factory.py`): `get_provider()` routes by provider name — defaults to OpenAI-compatible

All providers inherit `LLMProvider` ABC from `base.py`. Key methods: `get_completion()`, `get_completion_stream()`.

### Benchmark Runner

`core/benchmark_runner.py` (~2,700 lines) is the main orchestrator. Test methods:
- `run_concurrency_test()` — throughput across concurrency levels
- `run_prefill_test()` — TTFT at different input token sizes
- `run_segmented_prefill_test()` — prefix caching effectiveness
- `run_long_context_test()` — long input retention
- `run_throughput_matrix_test()` — concurrency x context sweep
- `run_stability_test()` — time-based consistency
- `run_custom_text_test()` — user-defined prompts
- `run_dataset_test()` — quality evaluation with datasets

Metrics calculated in `core/benchmark/metrics.py`. Results saved to CSV via `utils/helpers.py`.

### Evaluator Plugin System

Evaluators in `evaluators/` are auto-discovered via `@register_evaluator` decorator (defined in `evaluators/__init__.py`). Each evaluator:
- Inherits `BaseEvaluator` (`evaluators/base_evaluator.py`)
- Implements `evaluate()` / `evaluate_batch()` for dataset-specific evaluation
- Returns `EvaluationResult` with accuracy metrics
- 17 evaluators: MMLU, GSM8K, MATH500, HumanEval, GPQA, TruthfulQA, HellaSwag, WinoGrande, ARC, MBPP, LongBench, Needle Haystack, AIME 2025, Arena Hard, Global PIQA, SWE-Bench Lite, Custom Needle

### Data Layer

- **SQLite** via SQLAlchemy: `core/database/` (connection, manager, schema)
- **Repository pattern**: `core/repositories/` for data access, `core/models/` for ORM models
- **Services**: `core/services/` for business logic

### Configuration

- `config/settings.py` — Provider endpoints (`PROVIDER_OPTIONS`), model list (`MODEL_OPTIONS`), tokenizer mappings (`HF_MODEL_MAPPING`)
- `config/secrets.py` — API key management
- `.env` file — API keys (never commit)
- `presets/` — JSON test configuration files

### Key Patterns

1. **Async HTTP**: All LLM calls use `httpx` async; benchmark runner is async throughout
2. **Tokenizers**: `core/tokenizer_utils.py` — auto-matches model name to local tokenizer via `HF_MODEL_MAPPING`, falls back to tiktoken
3. **Safe Execution**: `core/safe_executor.py` uses AST validation to block file I/O, imports, network in HumanEval code evaluation
4. **Request Logging**: `core/request_logger.py` logs all API calls as JSON to `api_logs/`

## Git Conventions

Branch naming: `feature/...`, `fix/...`
Commit format: `Type: Subject` (e.g., `Feat: add new chart export`)

## Code Style

- Ruff config in `pyproject.toml`: line length 100, double quotes, isort enabled
- First-party packages: `config`, `core`, `ui`, `utils`, `evaluators`
- pytest markers: `slow`, `integration`, `security`
