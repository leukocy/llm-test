# LLM Benchmark Platform — Architecture

> Version: 3.0  
> Updated: 2026-02-14  
> Architecture: FastAPI Backend + React Frontend (decoupled)

---

## 📁 Project Structure

The project has been refactored from a Streamlit monolith to a decoupled
architecture. The old Streamlit code remains in `core/`, `ui/`, and `app.py`
and is still functional via the `legacy/streamlit_adapter.py` bridge.

```
llm-test/
├── engine/                     # ★ NEW: Core engine (zero UI dependency)
│   ├── events.py               # EventBus — thread-safe pub/sub
│   ├── models.py               # Pydantic data models
│   ├── metrics_calculator.py   # TTFT/TPS/TPOT calculation (pure functions)
│   ├── prompt_generator.py     # Prompt calibration & token targeting
│   ├── tokenizer.py            # Tokenizer management
│   ├── concurrency.py          # Concurrent request execution (asyncio)
│   ├── runner.py               # TestRunner orchestrator (~70 lines)
│   ├── strategies/             # ★ Plugin directory (auto-discovered)
│   │   ├── base.py             # TestStrategy ABC + @register_strategy
│   │   ├── concurrency.py      # ⚡ Concurrency test
│   │   ├── prefill.py          # 📥 Prefill speed test
│   │   ├── long_context.py     # 📏 Long context test
│   │   ├── segmented.py        # 🧩 Segmented/cached test
│   │   ├── matrix.py           # 📊 Throughput matrix
│   │   ├── stability.py        # 🔄 Stability test
│   │   ├── custom_text.py      # ✍️ Custom text test
│   │   └── dataset.py          # 📚 Dataset evaluation
│   ├── providers/              # LLM provider adapters
│   │   ├── base.py             # LLMProvider ABC
│   │   ├── openai_compat.py    # OpenAI-compatible provider
│   │   └── registry.py         # @register_provider + auto-discovery
│   └── persistence/            # ★ Data persistence
│       ├── csv_writer.py       # CSV output (batch + streaming)
│       ├── models.py           # SQLAlchemy ORM models
│       └── db_writer.py        # Database persistence
│
├── analysis/                   # ★ NEW: Report computation (pure logic, no UI)
│   ├── insights.py             # Performance insight generation
│   ├── grading.py              # A+…D grading system
│   ├── summary.py              # Report data computation
│   └── report_data.py          # Pydantic report models
│
├── server/                     # ★ NEW: FastAPI backend
│   ├── app.py                  # Application + CORS + SPA routing
│   ├── config.py               # Server settings
│   ├── state.py                # In-memory run registry
│   └── routes/
│       ├── tests.py            # POST /api/tests/run, /stop, GET /status, /history
│       ├── results.py          # GET /api/results/{id}, /csv
│       ├── reports.py          # GET /api/reports/{id}/analysis, /summary
│       ├── providers.py        # GET /api/providers, /models
│       ├── evaluators.py       # GET /api/evaluators, /datasets
│       └── ws.py               # WebSocket /ws/{run_id}
│
├── web/                        # ★ NEW: React frontend (Vite + TypeScript)
│   └── src/
│       ├── main.tsx            # Entry + routing
│       ├── hooks/useWebSocket.ts  # Real-time progress hook
│       ├── components/TestConfigPanel.tsx  # Dynamic form from param_schema
│       ├── pages/
│       │   ├── Home.tsx        # Dashboard + test execution
│       │   └── TestReport.tsx  # Report with charts, insights, grades
│       └── lib/
│           ├── api.ts          # TestService (Axios)
│           ├── types.ts        # Frontend type definitions
│           └── utils.ts        # Utilities
│
├── evaluators/                 # Quality evaluators (17 datasets)
│   ├── __init__.py             # Auto-discovery + @register_evaluator
│   ├── base_evaluator.py       # BaseEvaluator ABC
│   ├── mmlu_evaluator.py       # MMLU — 57-subject knowledge
│   ├── gsm8k_evaluator.py      # GSM8K — grade school math
│   ├── math500_evaluator.py    # MATH500 — competition math
│   ├── humaneval_evaluator.py  # HumanEval — Python coding
│   └── ... (17 total)
│
├── legacy/                     # ★ NEW: Transition bridge
│   └── streamlit_adapter.py    # Old Streamlit ↔ new engine bridge
│
├── core/                       # OLD: Monolithic core (still functional)
│   ├── benchmark_runner.py     # God Object (1,757 lines) — replaced by engine/
│   ├── database/               # SQLite database layer
│   └── providers/              # Old provider implementations
│
├── ui/                         # OLD: Streamlit UI components
│   ├── reports.py              # Report rendering (94K) — replaced by analysis/
│   ├── sidebar.py              # Config sidebar
│   └── ...
│
├── app.py                      # OLD: Streamlit entry point
├── start_server.py             # NEW: FastAPI entry point
└── tests/                      # Unit tests (62+ new architecture tests)
    ├── engine/                 # Engine module tests
    └── server/                 # API endpoint tests
```

---

## 🏗️ Architecture Overview

### New Architecture (v3.0)

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Home    │  │ Report   │  │ TestConfigPanel   │  │
│  │  Page    │  │ Page     │  │ (dynamic forms)   │  │
│  └────┬─────┘  └────┬─────┘  └──────────────────┘  │
│       │              │                               │
│       ▼              ▼                               │
│  ┌──────────────────────────────────────────────┐   │
│  │           TestService (api.ts)                │   │
│  │  HTTP: /api/*        WebSocket: /ws/{run_id} │   │
│  └────────────────────┬─────────────────────────┘   │
└───────────────────────┼─────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│                  FastAPI Backend                       │
│  ┌────────┐ ┌────────┐ ┌─────────┐ ┌──────────────┐ │
│  │ /tests │ │/reports│ │  /ws    │ │ /evaluators  │ │
│  └───┬────┘ └───┬────┘ └────┬────┘ └──────────────┘ │
│      │          │           │                         │
│      ▼          ▼           ▼                         │
│  ┌──────────────────────────────────────────────┐    │
│  │              TestRunner                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────┐ │    │
│  │  │ EventBus │  │ Strategy │  │ Provider   │ │    │
│  │  │ (pub/sub)│  │ (plugin) │  │ (adapter)  │ │    │
│  │  └──────────┘  └──────────┘  └────────────┘ │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │         Analysis Pipeline                     │    │
│  │  summary.py → insights.py → grading.py       │    │
│  │       ↓            ↓            ↓             │    │
│  │  BaseReportData + insights[] + grade (A+…D)  │    │
│  └──────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
```

### Design Patterns

| Pattern | Implementation | Location |
|---------|---------------|----------|
| **Plugin/Strategy** | `@register_strategy` decorator | `engine/strategies/` |
| **Event-Driven** | `EventBus` pub/sub | `engine/events.py` |
| **Adapter** | LLM provider abstraction | `engine/providers/` |
| **Registry** | Auto-discovery via `pkgutil` | strategies, providers, evaluators |
| **Bridge** | Streamlit compat layer | `legacy/streamlit_adapter.py` |
| **Repository** | Data access abstraction | `core/database/` |

### Key Architectural Decisions

1. **Zero UI Coupling**: `engine/` has zero imports from Streamlit, React, or any UI framework
2. **Plugin System**: Add a new test type by creating 1 file with `@register_strategy`
3. **Dynamic Forms**: Frontend auto-generates config forms from strategy `param_schema()`
4. **Structured Reports**: Analysis pipeline produces typed Pydantic models, not raw HTML

---

## 🔄 Data Flow

### Performance Test Flow (New Architecture)

```
Frontend (React)                    Backend (FastAPI)            Engine
─────────────────                   ────────────────            ──────
POST /api/tests/run ──────────────→ TestRunner.run()
                                         │
ws://host/ws/{run_id} ←── EventBus ←─────┤ Progress, Logs
                                         │
                                    Strategy.execute()
                                         │
                                    Provider.get_completion()
                                         │
GET /api/reports/{id}/analysis ←──→ generate_report(df)
                                    ├── insights.py
                                    └── grading.py
                                         │
                                    BaseReportData (JSON)
                                         │
Frontend renders ←────────────────── { sections, charts,
  charts + insights + grade              insights, grade }
```

### Quality Evaluation Flow

```
GET /api/evaluators ──→ list_evaluators() ──→ registry
                                                │
POST /api/tests/run   ──→ DatasetStrategy       │
  {test_type: "dataset"}     │                  │
                             ▼                  │
                    BaseEvaluator.evaluate_batch()
                             │
                    SampleResult[] → EvaluationResult
```

---

## 📦 API Endpoints (22 total)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/tests/run` | Start a test run |
| GET | `/api/tests/{id}/status` | Get run status |
| POST | `/api/tests/{id}/stop` | Stop a running test |
| POST | `/api/tests/{id}/pause` | Pause a test |
| POST | `/api/tests/{id}/resume` | Resume a paused test |
| GET | `/api/results/{id}` | Get test results |
| GET | `/api/results/{id}/csv` | Download results as CSV |
| GET | `/api/reports/{id}/analysis` | Get analysis report (charts + insights) |
| GET | `/api/reports/{id}/summary` | Get summary stats |
| GET | `/api/strategies` | List available test strategies |
| GET | `/api/providers` | List configured providers |
| GET | `/api/providers/models` | List available models |
| GET | `/api/evaluators/` | List evaluators with metadata |
| GET | `/api/evaluators/datasets` | List dataset names |
| GET | `/api/evaluators/yaml-tasks` | List YAML evaluator configs |
| WS | `/ws/{run_id}` | Real-time event stream |

---

## 🧪 Test Types

### Performance Benchmarks (8 strategies)

| Strategy | Key | Description |
|----------|-----|-------------|
| ⚡ Concurrency | `concurrency` | Throughput across concurrency levels |
| 📥 Prefill | `prefill` | Input processing speed |
| 📏 Long Context | `long_context` | Long input retention & speed |
| 🧩 Segmented | `segmented` | Prefix caching effectiveness |
| 📊 Matrix | `matrix` | Concurrency × Context sweep |
| 🔄 Stability | `stability` | Time-based consistency |
| ✍️ Custom Text | `custom_text` | User-defined prompts |
| 📚 Dataset | `dataset` | Quality evaluation |

### Quality Evaluators (17 datasets)

| Dataset | Type | Tests |
|---------|------|-------|
| MMLU | Multiple choice | 57-subject knowledge |
| GSM8K | Math reasoning | Grade school math |
| MATH500 | Competition math | Advanced problems |
| HumanEval | Code generation | Python programming |
| GPQA | Expert reasoning | Graduate-level science |
| TruthfulQA | Factuality | Common misconceptions |
| HellaSwag | Commonsense | Sentence completion |
| WinoGrande | Coreference | Pronoun resolution |
| ARC | Science | Elementary reasoning |
| MBPP | Code generation | Python basics |
| LongBench | Long context | Extended text tasks |
| Needle Haystack | Retrieval | Key fact extraction |
| AIME 2025 | Math competition | Olympiad problems |
| Arena Hard | Adversarial | Hard questions |
| Global PIQA | Physical | Physical reasoning |
| SWE-Bench Lite | Engineering | Bug fixing |
| Custom Needle | Custom | User-defined retrieval |

---

## 🔧 Tech Stack

| Category | Old (v2) | New (v3) |
|----------|----------|----------|
| **Frontend** | Streamlit | React + TypeScript + Vite |
| **Backend** | Streamlit (embedded) | FastAPI + Uvicorn |
| **Charts** | Plotly (Python) | Recharts (React) |
| **Real-time** | Streamlit rerun | WebSocket |
| **State** | st.session_state | Zustand / React hooks |
| **Styling** | Streamlit CSS hacks | Tailwind CSS |
| **API** | N/A (monolith) | REST + WebSocket |
| **Async** | asyncio | asyncio (unchanged) |
| **Data** | pandas, numpy | pandas, numpy (unchanged) |
| **Tokenizers** | tiktoken, transformers | tiktoken, transformers |
| **Testing** | pytest | pytest + tsc |

---

## 🚀 Getting Started

```bash
# Backend
pip install -r requirements.txt
python start_server.py           # → http://localhost:8000

# Frontend (development)
cd web && npm install && npm run dev  # → http://localhost:5173

# Production (single server)
cd web && npm run build
python start_server.py           # Serves both API + frontend

# Tests
python -m pytest tests/engine/ tests/server/ -v  # 62 tests
cd web && npx tsc --noEmit                        # Type check

# Legacy Streamlit (still works)
streamlit run app.py
```

---

*Last updated: 2026-02-14*
