# LLM Benchmark Platform REST API Reference

> Version: 3.0
> Last Updated: 2026-02-14
> Base URL: `http://localhost:8000`

---

## 📚 Table of Contents

1. [Tests & Execution](#tests--execution)
2. [Results & Persistence](#results--persistence)
3. [Reporting & Analysis](#reporting--analysis)
4. [Providers & Models](#providers--models)
5. [Evaluators & Datasets](#evaluators--datasets)
6. [WebSockets](#websockets)

---

## Tests & Execution

### 🚀 Run Test
Starts a new benchmark test in the background.

- **URL**: `/api/tests/run`
- **Method**: `POST`
- **Body**: `TestRunConfig` (JSON)
- **Response**: `{"run_id": "8c2f1b", "status": "started"}`

### 📜 Test History
Retrieves recent test runs from the database.

- **URL**: `/api/tests/history`
- **Method**: `GET`
- **Query Params**: `limit=20`
- **Response**: List of `TestRunModel` summaries.

### 📊 Test Status
Gets in-memory status of an active test run.

- **URL**: `/api/tests/{run_id}/status`
- **Method**: `GET`
- **Response**: `{"run_id": "...", "status": "running", "results_count": 12}`

### 🛑 Control Operations
Interrupt or manage active test runs.

- **Stop**: `POST /api/tests/{run_id}/stop`
- **Pause**: `POST /api/tests/{run_id}/pause`
- **Resume**: `POST /api/tests/{run_id}/resume`

### 🧩 Strategies
Lists all available test strategies and their dynamic parameter schemas.

- **URL**: `/api/strategies`
- **Method**: `GET`
- **Response**: Map of strategy names to icons, labels, and JSON schemas.

---

## Results & Persistence

### 📥 Download CSV
Downloads the CSV results file for a specific run.

- **URL**: `/api/results/{run_id}/csv`
- **Method**: `GET`
- **Response**: File stream (`text/csv`)

### 📦 Get Raw Results
Returns the full result set as JSON.

- **URL**: `/api/results/{run_id}`
- **Method**: `GET`
- **Response**: List of `TestResult` objects.

---

## Reporting & Analysis

### 📝 Report Summary
Generates a structured report summary (stats, metadata, sections).

- **URL**: `/api/reports/{run_id}/summary`
- **Method**: `GET`
- **Response**: `BaseReportData` object.

### 💡 Performance Analysis
Generates nuanced performance insights and an A+…D grade.

- **URL**: `/api/reports/{run_id}/analysis`
- **Method**: `GET`
- **Response**: `{"insights": [...], "grade": "A", "color": "#..."}`

---

## Providers & Models

### 🏢 List Providers
Returns available LLM provider options.

- **URL**: `/api/providers/`
- **Method**: `GET`

### 🤖 List Models
Returns a list of pre-configured model IDs.

- **URL**: `/api/providers/models`
- **Method**: `GET`

---

## Evaluators & Datasets

### 📋 List Evaluators
Lists all registered quality evaluators (MMLU, GSM8K, etc.).

- **URL**: `/api/evaluators/`
- **Method**: `GET`

### 📂 List Datasets
Lists available local dataset folders for custom evaluation.

- **URL**: `/api/evaluators/datasets`
- **Method**: `GET`

---

## WebSockets

### 🔄 Real-time Updates
Subscribe to live progress and logs for a specific run.

- **URL**: `/ws/{run_id}`
- **Messages**:
    - `Event(type=PROGRESS)`: Iteration updates
    - `Event(type=LOG)`: Live execution logs
    - `Event(type=ERROR)`: Exception details
