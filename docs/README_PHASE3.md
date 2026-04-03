# LLM Benchmark Platform — New Architecture

This directory contains the refactored LLM Benchmark Platform with a decoupled
**FastAPI Backend** and a **React Frontend**.

## Architecture Overview

```
engine/                 # Core engine (zero UI dependency)
  ├── events.py         # EventBus — thread-safe pub/sub
  ├── models.py         # Pydantic data models (TestConfig, TestResult, ...)
  ├── metrics_calculator.py # TTFT/TPS/TPOT calculation (pure functions)
  ├── prompt_generator.py   # Prompt calibration
  ├── tokenizer.py          # Tokenizer management
  ├── concurrency.py        # Concurrent request execution
  ├── runner.py             # TestRunner orchestrator (~70 lines)
  ├── strategies/           # Plugin directory (auto-discovered)
  │   ├── base.py           # TestStrategy ABC + @register_strategy
  │   ├── concurrency.py    # ⚡ Concurrency test
  │   ├── prefill.py        # 📥 Prefill speed test
  │   ├── long_context.py   # 📏 Long context test
  │   ├── segmented.py      # 🧩 Segmented/cached test
  │   ├── matrix.py         # 📊 Throughput matrix
  │   ├── stability.py      # 🔄 Stability test
  │   ├── custom_text.py    # ✍️ Custom text test
  │   └── dataset.py        # 📚 Dataset evaluation
  └── providers/            # LLM provider adapters
      ├── base.py           # LLMProvider ABC
      ├── openai_compat.py  # OpenAI-compatible
      └── registry.py       # @register_provider

analysis/               # Report computation (pure logic, no UI)
  ├── insights.py       # Performance insight generation
  ├── grading.py        # A+…D grading system
  ├── summary.py        # Report data computation
  └── report_data.py    # Pydantic report models (BaseReportData, ChartConfig, ...)

server/                 # FastAPI backend
  ├── app.py            # Application + CORS + routing
  ├── config.py         # Settings
  ├── state.py          # In-memory run registry
  └── routes/
      ├── tests.py      # POST /api/tests/run, /stop, GET /status
      ├── results.py    # GET /api/results/{id}
      ├── reports.py    # GET /api/reports/{id}/analysis, /summary
      ├── providers.py  # GET /api/providers, /api/providers/models
      └── ws.py         # WebSocket /ws/{run_id}

web/                    # React frontend (Vite + TypeScript)
  └── src/
      ├── main.tsx      # Entry + routing
      ├── hooks/
      │   └── useWebSocket.ts  # Real-time progress hook
      ├── components/
      │   └── TestConfigPanel.tsx  # Dynamic form from param_schema
      ├── pages/
      │   ├── Home.tsx       # Dashboard + test execution
      │   └── TestReport.tsx # Report with charts, insights, grades
      └── lib/
          ├── api.ts     # TestService (Axios)
          ├── types.ts   # Frontend type definitions
          └── utils.ts   # Utilities
```

## How to Run

### 1. Start the Backend
```bash
pip install -r requirements.txt   # First time only
python start_server.py
```
→ API at `http://localhost:8000`
→ Swagger docs at `http://localhost:8000/docs`

### 2. Start the Frontend (Development)
```bash
cd web
npm install       # First time only
npm run dev
```
→ UI at `http://localhost:5173`
→ Vite proxy forwards `/api/*` and `/ws/*` → `localhost:8000`

### 3. Production Build
```bash
cd web && npm run build   # Outputs to web/dist/
python start_server.py    # Serves both API + frontend
```
→ Everything at `http://localhost:8000`

## Key Features

| Feature | Description |
|---------|-------------|
| **Plugin System** | Add a test by creating 1 file with `@register_strategy` |
| **Dynamic Forms** | Frontend auto-generates config forms from `param_schema` |
| **Real-time Updates** | WebSocket streams logs + progress |
| **Insights + Grading** | Auto-generated performance analysis (A+…D) |
| **Charts** | Recharts-based visualizations driven by structured data |
| **Zero UI Coupling** | Engine has zero UI framework imports |

## Testing
```bash
# All new architecture tests (56 tests)
python -m pytest tests/engine/ tests/server/ -v

# Frontend type check
cd web && npx tsc --noEmit
```
