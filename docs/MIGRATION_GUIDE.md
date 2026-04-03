# Migration Guide: Streamlit → FastAPI + React

This document explains how to transition from the old Streamlit-based architecture
to the new FastAPI + React architecture.

## Overview

| Component | Old Location | New Location | Status |
|-----------|-------------|-------------|--------|
| Test Runner | `core/benchmark_runner.py` | `engine/runner.py` + `engine/strategies/` | ✅ Replaced |
| Metrics | `core/metrics.py` | `engine/metrics_calculator.py` | ✅ Replaced |
| Providers | `core/providers/` | `engine/providers/` | ✅ Replaced |
| Reports | `ui/reports.py` (94KB) | `analysis/summary.py` + React `TestReport.tsx` | ✅ Replaced |
| Charts | `ui/charts.py` | React `Recharts` in `TestReport.tsx` | ✅ Replaced |
| Insights | `ui/insights.py` | `analysis/insights.py` | ✅ Moved |
| Grading | `analysis/grading.py` | `analysis/grading.py` | ✅ Same |
| Evaluators | `evaluators/` | `evaluators/` (auto-discovery added) | ✅ Enhanced |
| Database | `core/database/` | `core/database/` (unchanged) | ⏳ Future |
| Config UI | `ui/sidebar.py` | `web/src/components/TestConfigPanel.tsx` | ✅ Replaced |

## How to Use the Legacy Bridge

If you need to run old Streamlit UI code with the new engine:

```python
# In old Streamlit code, replace:
from core.benchmark_runner import BenchmarkRunner

# With:
from legacy.streamlit_adapter import run_benchmark_test

# Old way:
runner = BenchmarkRunner(client, model_id, ...)
await runner.run_concurrency_test(levels, rounds, max_tokens)
results_df = st.session_state.results_df

# New way (via bridge):
results_df = run_benchmark_test(
    test_type="concurrency",
    config={
        "api_base_url": "http://...",
        "model_id": "gpt-4",
        "api_key": "sk-...",
        "concurrency_levels": [1, 2, 4],
        "num_requests": 10,
        "max_tokens": 128,
    },
    progress_callback=lambda done, total, msg: progress_bar.progress(done/total)
)
```

## Adding a New Test Strategy

Create a single file in `engine/strategies/`:

```python
# engine/strategies/my_test.py
from engine.strategies.base import TestStrategy, register_strategy

@register_strategy("my_test", name="My Custom Test", icon="🧪")
class MyTestStrategy(TestStrategy):
    
    @classmethod
    def param_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "iterations": {"type": "integer", "default": 10, "title": "Iterations"},
            }
        }
    
    def calculate_total_requests(self, params: dict) -> int:
        return params.get("iterations", 10)
    
    async def execute(self, config, params, provider, tokenizer, prompt_generator):
        # Your test logic here
        results = []
        for i in range(params.get("iterations", 10)):
            result = await provider.get_completion(...)
            results.append(result)
        return results
    
    def csv_columns(self) -> list[str]:
        return ["session_id", "ttft", "tps", "output_tokens"]
```

The strategy will be automatically:
- Discovered and registered at import time
- Listed in `GET /api/strategies`
- Rendered with a dynamic form in the React frontend
- Available for execution via `POST /api/tests/run`

## Adding a New Evaluator

```python
# evaluators/my_evaluator.py
from evaluators import register_evaluator
from evaluators.base_evaluator import BaseEvaluator

@register_evaluator("my_dataset")
class MyDatasetEvaluator(BaseEvaluator):
    def load_dataset(self, subset=None):
        ...
    def format_prompt(self, sample, include_answer=False):
        ...
    def parse_response(self, response):
        ...
    def check_answer(self, predicted, correct):
        ...
```

## Files Safe to Delete (after full migration)

These files are fully replaced by the new architecture:

```
# Can be removed once Streamlit UI is no longer needed
app.py                          → start_server.py
ui/reports.py                   → analysis/summary.py + web/src/pages/TestReport.tsx
ui/charts.py                    → web/src/pages/TestReport.tsx (Recharts)
ui/sidebar.py                   → web/src/components/TestConfigPanel.tsx
ui/test_panels.py               → web/src/pages/Home.tsx
ui/test_runner.py               → engine/runner.py
ui/page_layout.py               → React routing
core/benchmark_runner.py        → engine/runner.py + engine/strategies/
```

## Environment Variables

```bash
# .env file
CORS_ORIGINS=*                  # Production: restrict to your domain
API_HOST=0.0.0.0
API_PORT=8000
```
