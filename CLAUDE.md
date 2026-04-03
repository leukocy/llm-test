# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Benchmark Platform - A comprehensive LLM performance & quality evaluation platform. Currently using Streamlit UI (app.py) as the main interface. The project is transitioning to a decoupled FastAPI + React architecture.

## Commands

### Running the Application

```bash
# Main Streamlit UI (current)
streamlit run app.py

# Or use the convenience scripts
./start.sh          # Linux/Mac
start.bat           # Windows
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/engine/ tests/server/ -v

# Run a single test file
python -m pytest tests/test_benchmark_runner.py -v

# Run with coverage
python -m pytest tests/ -v --cov=core --cov=evaluators --cov=utils
```

### Linting & Formatting

```bash
# Lint with ruff
ruff check .

# Format with ruff
ruff format .

# Type checking with mypy
mypy core/ evaluators/ utils/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Architecture

### Core Modules

- **core/benchmark_runner.py**: Main orchestrator for performance tests (~2,700 lines)
- **core/quality_evaluator.py**: Quality evaluation logic
- **core/providers/**: LLM provider adapters (10+ providers: OpenAI, DeepSeek, Moonshot, Gemini, MiMo, ZhiPu, Volcengine, Alibaba Bailian, SiliconFlow, OpenRouter, local models)
- **core/database/**: SQLite persistence layer
- **core/services/**: Business logic layer
- **core/repositories/**: Data access layer (repository pattern)

### Evaluator Plugin System

Evaluators in `evaluators/` are auto-discovered via `@register_evaluator` decorator. Each evaluator:
- Inherits from `BaseEvaluator` (evaluators/base_evaluator.py)
- Implements `evaluate()` method for dataset-specific evaluation
- Returns `EvaluationResult` with accuracy metrics

### Key Patterns

1. **Provider Abstraction**: All LLM calls go through `core/providers/` (10+ providers: OpenAI, DeepSeek, Moonshot, Gemini, MiMo, ZhiPu, Volcengine, Alibaba Bailian, SiliconFlow, OpenRouter, local models)
2. **Session State**: Streamlit session state managed via `config/session_state.py`
3. **Safe Execution**: Code evaluation uses `core/safe_executor.py` with AST validation (blocks file I/O, imports, network)
4. **Repository Pattern**: Data access via `core/repositories/`, SQLAlchemy ORM models in `core/models/`

### Dataset Evaluators (17 total)

| Evaluator | Dataset | Type |
|-----------|---------|------|
| `mmlu_evaluator.py` | MMLU | 57-subject knowledge |
| `gsm8k_evaluator.py` | GSM8K | Grade school math |
| `math500_evaluator.py` | MATH500 | Competition math |
| `humaneval_evaluator.py` | HumanEval | Python code generation |
| `gpqa_evaluator.py` | GPQA | Graduate-level science |
| `needle_haystack_evaluator.py` | Needle Haystack | Long context retrieval |
| `truthfulqa_evaluator.py` | TruthfulQA | Truthfulness |
| `hellaswag_evaluator.py` | HellaSwag | Commonsense reasoning |
| `winogrande_evaluator.py` | WinoGrande | Coreference resolution |
| `arc_evaluator.py` | ARC | Science reasoning |
| `mbpp_evaluator.py` | MBPP | Python code generation |
| `longbench_evaluator.py` | LongBench | Long context understanding |
| `aime_evaluator.py` | AIME 2025 | Competition math |
| `arena_hard_evaluator.py` | Arena Hard | Hard prompts |
| `global_piqa_evaluator.py` | Global PIQA | Physical reasoning |
| `swebench_evaluator.py` | SWE-Bench Lite | Software engineering |
| `custom_needle_evaluator.py` | Custom Needle | Custom retrieval tests |

## Configuration

- **API Keys**: Store in `.env` file (see `.env.example`). Never commit hardcoded keys.
- **Provider Settings**: `config/settings.py`
- **Test Presets**: `presets/` directory contains JSON test configurations

## Security Notes

- All API keys must be in environment variables (never hardcoded)
- Code execution in HumanEval uses restricted Python (AST validation)
- SSRF protection enabled by default (blocks private IPs)
- See `docs/SECURITY.md` for full security guidelines

## Git Workflow

Branch naming: `feature/...`, `fix/...`
Commit format: `Type: Subject` (e.g., `Feat: add new chart export`)

## Documentation

- `ARCHITECTURE.md` - Full architecture diagram
- `docs/API.md` - REST API reference (for FastAPI backend)
- `docs/DEVELOPMENT.md` - Development workflow (Chinese)
