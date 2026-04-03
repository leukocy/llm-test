# LLM Benchmark Platform — 架构重写计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 彻底重构 LLM 性能测试平台，消除 God Object / UI耦合 / 散弹枪手术等核心技术债务，使新增测试类型只需 **创建一个文件** 即可完成。

**架构方针:** 从 Streamlit 单体应用迁移到 **FastAPI 后端 + React/Vite 前端** 分离架构，核心引擎完全框架无关。

**Tech Stack:**
- **后端:** Python 3.10+, FastAPI, Pydantic, SQLAlchemy, asyncio, WebSocket
- **前端:** React 18 + TypeScript + Vite + Ant Design (或 shadcn/ui)
- **通信:** REST API + WebSocket (实时进度)
- **数据:** SQLite (开发) / PostgreSQL (生产), CSV 导出

---

## 📋 目录

- [Phase 0: 前置准备](#phase-0-前置准备)
- [Phase 1: 核心引擎重构 (无 UI 依赖)](#phase-1-核心引擎重构)
- [Phase 2: 测试策略插件系统](#phase-2-测试策略插件系统)
- [Phase 3: FastAPI 后端](#phase-3-fastapi-后端)
- [Phase 4: 报告引擎重构](#phase-4-报告引擎重构)
- [Phase 5: React 前端](#phase-5-react-前端)
- [Phase 6: 集成与迁移](#phase-6-集成与迁移)
- [附录: 新旧架构对比](#附录)

---

## 全局设计原则

1. **核心引擎零 UI 依赖** — `engine/` 不 import 任何 UI 框架
2. **测试类型插件化** — 新增测试 = 新增一个 `XxxTestStrategy` 文件 + 自动注册
3. **事件驱动通信** — 引擎通过 EventBus 发布进度，UI 层订阅消费
4. **数据模型统一** — 所有数据流使用 Pydantic model，JSON-in / JSON-out
5. **English-only 代码** — 注释、变量名、日志全部英文

---

## 新目录结构

```
llm-benchmark/
├── engine/                          # 核心引擎 (零 UI 依赖)
│   ├── __init__.py
│   ├── runner.py                    # TestRunner: 编排器 (~200行)
│   ├── events.py                    # EventBus + 事件类型定义
│   ├── models.py                    # Pydantic 数据模型 (TestConfig, TestResult, Metrics)
│   ├── tokenizer.py                 # Tokenizer 管理 (从 benchmark_runner 提取)
│   ├── prompt_generator.py          # Prompt 校准/生成 (从 benchmark_runner 提取)
│   ├── metrics_calculator.py        # TTFT/TPS/TPOT 计算 (从 benchmark_runner 提取)
│   ├── concurrency.py              # 并发执行引擎 (Semaphore/Batch)
│   │
│   ├── strategies/                  # 测试策略 (插件目录, 自动发现)
│   │   ├── __init__.py              # 自动注册所有策略
│   │   ├── base.py                  # TestStrategy 抽象基类 (~80行)
│   │   ├── concurrency.py           # ConcurrencyTestStrategy
│   │   ├── prefill.py               # PrefillTestStrategy
│   │   ├── long_context.py          # LongContextTestStrategy
│   │   ├── segmented.py             # SegmentedTestStrategy
│   │   ├── matrix.py                # MatrixTestStrategy
│   │   ├── stability.py             # StabilityTestStrategy
│   │   ├── custom_text.py           # CustomTextTestStrategy
│   │   └── dataset.py               # DatasetTestStrategy
│   │
│   ├── providers/                   # LLM Provider 适配器
│   │   ├── base.py                  # LLMProvider 抽象基类
│   │   ├── openai_compat.py         # OpenAI 兼容
│   │   ├── gemini.py                # Gemini 原生
│   │   └── registry.py              # Provider 注册表
│   │
│   └── persistence/                 # 数据持久化
│       ├── csv_writer.py            # CSV 输出
│       ├── db_writer.py             # 数据库写入
│       └── models.py                # SQLAlchemy ORM
│
├── analysis/                        # 分析与报告 (从 ui/ 提取纯逻辑)
│   ├── __init__.py
│   ├── insights.py                  # 性能洞察生成 (纯计算, 无 UI)
│   ├── grading.py                   # 性能评分
│   ├── summary.py                   # 测试摘要计算
│   └── report_data.py               # 报告数据模型
│
├── evaluators/                      # 数据集评估器 (保留, 改进注册)
│   ├── base.py
│   ├── registry.py                  # 评估器自动注册
│   ├── mmlu.py
│   ├── gsm8k.py
│   └── ...
│
├── server/                          # FastAPI 后端
│   ├── __init__.py
│   ├── app.py                       # FastAPI 应用
│   ├── routes/
│   │   ├── tests.py                 # 测试 CRUD + 执行
│   │   ├── results.py               # 结果查询
│   │   ├── reports.py               # 报告生成
│   │   ├── providers.py             # Provider 管理
│   │   └── ws.py                    # WebSocket 实时推送
│   ├── schemas.py                   # API 请求/响应 schema
│   ├── dependencies.py              # 依赖注入
│   └── config.py                    # 服务器配置
│
├── web/                             # React 前端
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/                     # API 客户端
│   │   ├── components/
│   │   │   ├── TestConfigPanel.tsx   # 测试配置 (动态表单)
│   │   │   ├── TestProgress.tsx      # 实时进度
│   │   │   ├── ResultsTable.tsx      # 结果表格
│   │   │   ├── Charts.tsx            # 图表组件
│   │   │   └── ReportView.tsx        # 报告视图
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts       # WS 实时数据 hook
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── TestRunner.tsx
│   │   │   ├── History.tsx
│   │   │   └── Compare.tsx
│   │   └── stores/                   # Zustand 状态管理
│   └── package.json
│
├── cli/                             # CLI 入口 (可选)
│   └── main.py                      # 命令行测试执行
│
├── legacy/                          # 旧代码备份 (迁移完成后删除)
│
├── tests/
│   ├── engine/
│   ├── server/
│   └── analysis/
│
└── pyproject.toml
```

---

## Phase 0: 前置准备

### Task 0.1: 创建项目骨架

**目的:** 在当前仓库内创建新的目录结构，旧代码暂时保留。

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/strategies/__init__.py`
- Create: `engine/providers/__init__.py`
- Create: `engine/persistence/__init__.py`
- Create: `analysis/__init__.py`
- Create: `server/__init__.py`
- Create: `server/routes/__init__.py`
- Create: `cli/__init__.py`

**Step 1:** 创建所有目录和 `__init__.py`

```bash
mkdir -p engine/strategies engine/providers engine/persistence analysis server/routes cli tests/engine tests/server tests/analysis
touch engine/__init__.py engine/strategies/__init__.py engine/providers/__init__.py engine/persistence/__init__.py
touch analysis/__init__.py server/__init__.py server/routes/__init__.py cli/__init__.py
```

**Step 2:** Commit

```bash
git add -A
git commit -m "chore: scaffold new architecture directories"
```

### Task 0.2: 安装新依赖

**Files:**
- Modify: `pyproject.toml`

**Step 1:** 添加新依赖

```toml
[project]
dependencies = [
    # Existing
    "httpx",
    "pandas",
    "numpy",
    "plotly",
    # New
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.0",
    "websockets>=12.0",
    "sqlalchemy>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
    "httpx",  # for TestClient
]
```

**Step 2:** 安装

```bash
pip install fastapi uvicorn[standard] pydantic websockets sqlalchemy pytest-asyncio
```

**Step 3:** Commit

```bash
git add pyproject.toml
git commit -m "chore: add FastAPI and architecture dependencies"
```

---

## Phase 1: 核心引擎重构

> **目标:** 将 `benchmark_runner.py` 的 2,658 行拆解为 5-6 个独立模块，每个 < 300 行，零 Streamlit 依赖。

### Task 1.1: 事件系统 (`engine/events.py`)

**目的:** 替代所有 `st.session_state` / `st.info()` / `st.error()` 调用，引擎通过事件总线通知外部。

**Files:**
- Create: `engine/events.py`
- Test: `tests/engine/test_events.py`

**Step 1: 写测试**

```python
# tests/engine/test_events.py
import pytest
from engine.events import EventBus, Event, EventType

def test_subscribe_and_emit():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.LOG, lambda e: received.append(e))
    bus.emit(Event(type=EventType.LOG, data={"message": "hello"}))
    assert len(received) == 1
    assert received[0].data["message"] == "hello"

def test_progress_event():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.PROGRESS, lambda e: received.append(e))
    bus.emit(Event(type=EventType.PROGRESS, data={"completed": 5, "total": 10}))
    assert received[0].data["completed"] == 5

def test_unsubscribe():
    bus = EventBus()
    received = []
    handler = lambda e: received.append(e)
    bus.subscribe(EventType.LOG, handler)
    bus.unsubscribe(EventType.LOG, handler)
    bus.emit(Event(type=EventType.LOG, data={"message": "ignored"}))
    assert len(received) == 0

def test_control_signal():
    bus = EventBus()
    bus.request_stop()
    assert bus.is_stop_requested()
    bus.clear_control()
    assert not bus.is_stop_requested()

def test_request_pause():
    bus = EventBus()
    bus.request_pause()
    assert bus.is_pause_requested()
```

**Step 2: 验证测试失败**

```bash
pytest tests/engine/test_events.py -v
# Expected: FAIL (module not found)
```

**Step 3: 实现**

```python
# engine/events.py
"""
Event system for decoupling engine from UI.

The engine emits events (progress, log, result, error, control).
UI layers (FastAPI WebSocket, Streamlit, CLI) subscribe and react.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    """All event types emitted by the engine."""
    LOG = "log"
    PROGRESS = "progress"
    RESULT = "result"           # Single test result completed
    TEST_START = "test_start"   # Test run started
    TEST_END = "test_end"       # Test run ended
    ERROR = "error"
    UI_UPDATE = "ui_update"     # Request UI refresh


@dataclass
class Event:
    """A single event."""
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Thread-safe event bus.

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.LOG, my_handler)
        bus.emit(Event(type=EventType.LOG, data={"message": "hello"}))
    """

    def __init__(self):
        self._handlers: dict[EventType, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        with self._lock:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable):
        with self._lock:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    def emit(self, event: Event):
        with self._lock:
            handlers = list(self._handlers.get(event.type, []))
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass  # Don't let subscriber errors crash the engine

    # --- Control signals (replaces st.session_state.stop_requested etc.) ---

    def request_stop(self):
        self._stop_requested.set()

    def request_pause(self):
        self._pause_requested.set()

    def is_stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def is_pause_requested(self) -> bool:
        return self._pause_requested.is_set()

    def clear_control(self):
        self._stop_requested.clear()
        self._pause_requested.clear()
```

**Step 4: 验证测试通过**

```bash
pytest tests/engine/test_events.py -v
# Expected: ALL PASS
```

**Step 5:** Commit

```bash
git add engine/events.py tests/engine/test_events.py
git commit -m "feat(engine): add event bus for UI-decoupled communication"
```

### Task 1.2: 数据模型 (`engine/models.py`)

**目的:** 用 Pydantic 定义所有数据结构，替代散落在各文件中的 dict 传递。

**Files:**
- Create: `engine/models.py`
- Test: `tests/engine/test_models.py`

**Step 1: 写测试**

```python
# tests/engine/test_models.py
import pytest
from engine.models import (
    TestConfig, TestResult, MetricsSnapshot,
    TestType, TestRunConfig, TestRunSummary
)

def test_test_type_enum():
    assert TestType.CONCURRENCY == "concurrency"
    assert TestType.PREFILL == "prefill"
    assert TestType.LONG_CONTEXT == "long_context"
    assert TestType.SEGMENTED == "segmented"
    assert TestType.MATRIX == "matrix"
    assert TestType.STABILITY == "stability"
    assert TestType.CUSTOM_TEXT == "custom_text"
    assert TestType.DATASET == "dataset"

def test_test_config_creation():
    cfg = TestConfig(
        api_base_url="http://localhost:8000/v1",
        model_id="qwen-2.5",
        api_key="sk-test",
        provider="OpenAI Compatible",
    )
    assert cfg.model_id == "qwen-2.5"

def test_test_result_serialization():
    r = TestResult(
        session_id=1,
        ttft=0.5,
        tps=30.0,
        prefill_tokens=100,
        decode_tokens=50,
    )
    d = r.model_dump()
    assert d["session_id"] == 1
    assert d["ttft"] == 0.5

def test_metrics_snapshot():
    m = MetricsSnapshot(
        ttft=0.3,
        tps=40.0,
        tpot=0.025,
        tpot_p95=0.03,
        tpot_p99=0.05,
        prefill_tokens=500,
        decode_tokens=200,
        decode_time=5.0,
        total_time=5.3,
    )
    assert m.tps == 40.0

def test_test_run_config_for_concurrency():
    run_cfg = TestRunConfig(
        test_type=TestType.CONCURRENCY,
        base=TestConfig(
            api_base_url="http://localhost:8000/v1",
            model_id="test",
            api_key="sk-test",
            provider="OpenAI Compatible",
        ),
        params={
            "concurrencies": [1, 2, 4],
            "rounds_per_level": 3,
            "max_tokens": 512,
            "input_tokens_target": 64,
        }
    )
    assert run_cfg.test_type == TestType.CONCURRENCY
    assert run_cfg.params["concurrencies"] == [1, 2, 4]
```

**Step 2: 验证测试失败**

**Step 3: 实现**

```python
# engine/models.py
"""
Pydantic data models for the benchmark engine.

All data flowing through the system uses these typed models
instead of raw dicts, enabling validation, serialization, and documentation.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class TestType(str, Enum):
    """All supported test types. Adding a new test = adding a value here."""
    CONCURRENCY = "concurrency"
    PREFILL = "prefill"
    LONG_CONTEXT = "long_context"
    SEGMENTED = "segmented"
    MATRIX = "matrix"
    STABILITY = "stability"
    CUSTOM_TEXT = "custom_text"
    DATASET = "dataset"
    ALL = "all"


class TestConfig(BaseModel):
    """Base configuration shared by all test types."""
    api_base_url: str
    model_id: str
    api_key: str
    provider: str = "OpenAI Compatible"
    tokenizer_option: str = "Auto"
    hf_tokenizer_model_id: Optional[str] = None
    latency_offset: float = 0.0
    thinking_enabled: Optional[bool] = None
    thinking_budget: Optional[int] = None
    reasoning_effort: Optional[str] = None
    random_seed: Optional[int] = None


class TestRunConfig(BaseModel):
    """Full configuration for a single test run."""
    test_type: TestType
    base: TestConfig
    params: dict[str, Any] = Field(default_factory=dict)


class MetricsSnapshot(BaseModel):
    """Metrics for a single request."""
    ttft: float = 0.0
    tps: float = 0.0
    tpot: float = 0.0
    tpot_p95: float = 0.0
    tpot_p99: float = 0.0
    prefill_speed: float = 0.0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    cache_hit_tokens: int = 0
    decode_time: float = 0.0
    total_time: float = 0.0
    system_output_throughput: float = 0.0
    system_input_throughput: float = 0.0
    token_calc_method: str = "unknown"


class TestResult(BaseModel):
    """Result of a single benchmark request."""
    session_id: int = 0
    concurrency: int = 1
    round: int = 1
    test_type: str = ""
    input_tokens_target: int = 0
    context_length_target: int = 0

    # Core metrics
    ttft: float = 0.0
    tps: float = 0.0
    tpot: float = 0.0
    prefill_speed: float = 0.0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    cache_hit_tokens: int = 0
    total_time: float = 0.0
    decode_time: float = 0.0

    # System throughput
    system_output_throughput: float = 0.0
    system_input_throughput: float = 0.0
    rps: float = 0.0
    tpot_p95: float = 0.0
    tpot_p99: float = 0.0

    # Meta
    token_calc_method: str = "unknown"
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class TestRunSummary(BaseModel):
    """Summary of a completed test run."""
    test_type: TestType
    model_id: str
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    duration_seconds: float = 0.0
    config: dict[str, Any] = Field(default_factory=dict)
    system_info: dict[str, str] = Field(default_factory=dict)
```

**Step 4: 验证测试通过**

**Step 5:** Commit

```bash
git add engine/models.py tests/engine/test_models.py
git commit -m "feat(engine): add Pydantic data models"
```

### Task 1.3: 指标计算器 (`engine/metrics_calculator.py`)

**目的:** 从 `benchmark_runner._calculate_metrics` 和 `_calculate_tokens` 提取纯函数。

**Files:**
- Create: `engine/metrics_calculator.py`
- Source: `core/benchmark_runner.py:647-778`
- Test: `tests/engine/test_metrics_calculator.py`

**核心提取:**

| 旧位置 | 新位置 | 说明 |
|--------|--------|------|
| `BenchmarkRunner._calculate_metrics()` L647-688 | `calculate_request_metrics()` | 纯函数，无 self |
| `BenchmarkRunner._calculate_tokens()` L718-778 | `calculate_tokens()` | 纯函数，接受 tokenizer |
| `BenchmarkRunner._get_cache_hit_tokens()` L690-716 | `extract_cache_hit_tokens()` | 纯函数 |
| `BenchmarkRunner._get_empty_metrics()` L644-645 | `empty_metrics()` | 工厂函数 |

**Step 1: 写测试** — 测试 `calculate_request_metrics` 的 TTFT/TPS/TPOT 计算正确性。

**Step 2: 实现** — 从 `benchmark_runner.py` L647-778 提取代码，移除 `self` 引用。

**Step 3:** Commit

### Task 1.4: Prompt 生成器 (`engine/prompt_generator.py`)

**目的:** 从 `benchmark_runner._calibrate_prompt` 和 `_get_text_for_token_count` 提取。

**Files:**
- Create: `engine/prompt_generator.py`
- Source: `core/benchmark_runner.py:397-609`
- Test: `tests/engine/test_prompt_generator.py`

**核心提取:**

| 旧位置 | 新位置 |
|--------|--------|
| `_calibrate_prompt()` L397-463 | `PromptGenerator.calibrate()` |
| `_get_text_for_token_count()` L465-609 | `PromptGenerator.generate_for_token_count()` |

**关键改变:** 构造函数接受 `tokenizer` 对象，而非通过 `self._get_tokenizer()` 延迟获取。

### Task 1.5: Tokenizer 管理 (`engine/tokenizer.py`)

**目的:** 从 `benchmark_runner._get_tokenizer` 和 `_infer_hf_model_id` 提取。

**Files:**
- Create: `engine/tokenizer.py`
- Source: `core/benchmark_runner.py:314-395`, `core/tokenizer_utils.py`

**关键改变:** 移除所有 `st.info()` / `st.error()` 调用，通过返回值或异常传递错误。

### Task 1.6: 并发执行引擎 (`engine/concurrency.py`)

**目的:** 从 `benchmark_runner._run_concurrency_batch`, `_run_continuous_batch`, `_run_time_based_batch` 提取。

**Files:**
- Create: `engine/concurrency.py`
- Source: `core/benchmark_runner.py:1023-1300, 2492-2606`

**核心提取:**

| 旧位置 | 新位置 | 说明 |
|--------|--------|------|
| `_run_concurrency_batch()` L1023-1111 | `ConcurrencyEngine.run_batch()` | 固定并发批处理 |
| `_run_continuous_batch()` L1113-1300 | `ConcurrencyEngine.run_continuous()` | Semaphore 持续并发 |
| `_run_time_based_batch()` L2492-2606 | `ConcurrencyEngine.run_timed()` | 定时并发 |

**关键改变:** 接受 `EventBus` 用于通知进度和检查停止信号，替代 `st.session_state.get('stop_requested')`。

---

## Phase 2: 测试策略插件系统

> **目标:** 新增测试类型 = 新增一个策略文件，无需修改任何其他文件。

### Task 2.1: 策略基类 (`engine/strategies/base.py`)

**Files:**
- Create: `engine/strategies/base.py`
- Test: `tests/engine/test_strategy_base.py`

**实现:**

```python
# engine/strategies/base.py
"""
Base class for test strategies.

To add a new test type:
1. Create a new file in this directory (e.g., my_test.py)
2. Create a class inheriting from TestStrategy
3. Decorate with @register_strategy("my_test")
4. Done! The strategy is automatically available via API and UI.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar
from engine.models import TestConfig, TestResult, TestType
from engine.events import EventBus


# Global strategy registry
_STRATEGY_REGISTRY: dict[str, type["TestStrategy"]] = {}


def register_strategy(test_type: str):
    """Decorator to register a test strategy."""
    def decorator(cls):
        _STRATEGY_REGISTRY[test_type] = cls
        cls.test_type_id = test_type
        return cls
    return decorator


def get_strategy(test_type: str) -> type["TestStrategy"]:
    """Get strategy class by test type."""
    if test_type not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown test type: {test_type}. "
            f"Available: {list(_STRATEGY_REGISTRY.keys())}"
        )
    return _STRATEGY_REGISTRY[test_type]


def list_strategies() -> dict[str, dict[str, Any]]:
    """List all registered strategies with metadata."""
    return {
        name: {
            "name": cls.display_name,
            "description": cls.description,
            "icon": cls.icon,
            "param_schema": cls.param_schema(),
        }
        for name, cls in _STRATEGY_REGISTRY.items()
    }


class TestStrategy(ABC):
    """
    Abstract base class for all test strategies.

    Each strategy defines:
    - What parameters it needs (param_schema)
    - How to calculate total requests
    - How to execute the test
    - What CSV columns it produces
    """

    # Metadata (override in subclass)
    test_type_id: ClassVar[str] = ""
    display_name: ClassVar[str] = "Unknown Test"
    description: ClassVar[str] = ""
    icon: ClassVar[str] = "🧪"

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    @classmethod
    @abstractmethod
    def param_schema(cls) -> dict[str, Any]:
        """
        Return JSON Schema describing this strategy's parameters.
        Used by the API for validation and the UI for dynamic form generation.

        Example:
        {
            "concurrencies": {"type": "array", "items": {"type": "integer"}, "default": [1, 2, 4]},
            "rounds_per_level": {"type": "integer", "default": 3, "min": 1},
            "max_tokens": {"type": "integer", "default": 512, "min": 1},
        }
        """
        ...

    @abstractmethod
    def calculate_total_requests(self, params: dict) -> int:
        """Calculate total number of requests for progress tracking."""
        ...

    @abstractmethod
    async def execute(
        self,
        config: TestConfig,
        params: dict,
        provider,
        tokenizer,
        prompt_generator,
    ) -> list[TestResult]:
        """
        Execute the test strategy.

        Args:
            config: Base test configuration
            params: Strategy-specific parameters (validated against param_schema)
            provider: LLM provider instance
            tokenizer: Tokenizer instance
            prompt_generator: Prompt generator instance

        Returns:
            List of TestResult objects
        """
        ...

    @abstractmethod
    def csv_columns(self) -> list[str]:
        """Return CSV column names for this test type's output."""
        ...
```

**Step 1:** 写测试验证注册机制和 `list_strategies()`。

**Step 2:** 实现以上代码。

**Step 3:** Commit。

### Task 2.2: 实现 ConcurrencyTestStrategy

**Files:**
- Create: `engine/strategies/concurrency.py`
- Source: `core/benchmark_runner.py:1314-1453` (run_concurrency_test)
- Test: `tests/engine/test_strategy_concurrency.py`

**实现要点:**

```python
# engine/strategies/concurrency.py
from engine.strategies.base import TestStrategy, register_strategy

@register_strategy("concurrency")
class ConcurrencyTestStrategy(TestStrategy):
    display_name = "Concurrency Test"
    description = "Test model performance under different concurrency levels"
    icon = "⚡"

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "concurrencies": {
                "type": "array", "items": {"type": "integer"},
                "default": [1, 2, 4], "title": "Concurrency Levels"
            },
            "rounds_per_level": {
                "type": "integer", "default": 3, "min": 1,
                "title": "Rounds Per Level"
            },
            "max_tokens": {
                "type": "integer", "default": 512, "min": 1,
                "title": "Max Output Tokens"
            },
            "input_tokens_target": {
                "type": "integer", "default": 64, "min": 1,
                "title": "Input Token Length"
            },
        }

    def calculate_total_requests(self, params: dict) -> int:
        return sum(
            c * params["rounds_per_level"]
            for c in params["concurrencies"]
        )

    async def execute(self, config, params, provider, tokenizer, prompt_generator):
        # ... 从 benchmark_runner.run_concurrency_test 迁移核心逻辑
        # 使用 self.event_bus.emit() 替代 st.xxx()
        # 使用 self.event_bus.is_stop_requested() 替代 st.session_state.get(...)
        ...

    def csv_columns(self) -> list[str]:
        return [
            "session_id", "concurrency", "round",
            "ttft", "tps", "tpot", "prefill_speed",
            "system_throughput", "system_input_throughput", "rps",
            "prefill_tokens", "decode_tokens", "total_time", "decode_time",
            "cache_hit_tokens", "token_calc_method", "input_tokens_target", "error"
        ]
```

### Task 2.3-2.8: 实现其余 6 个策略

按同样模式实现:

| Task | 策略 | 源代码位置 |
|------|------|-----------|
| 2.3 | PrefillTestStrategy | `benchmark_runner.py:1455-1614` |
| 2.4 | SegmentedTestStrategy | `benchmark_runner.py:1616-1997` |
| 2.5 | LongContextTestStrategy | `benchmark_runner.py:1999-2147` |
| 2.6 | MatrixTestStrategy | `benchmark_runner.py:2149-2294` |
| 2.7 | StabilityTestStrategy | `benchmark_runner.py:2608-2657` |
| 2.8 | CustomTextTestStrategy | `benchmark_runner.py:2246-2294` |

### Task 2.9: 策略自动发现 (`engine/strategies/__init__.py`)

**实现:**

```python
# engine/strategies/__init__.py
"""
Auto-discover and register all test strategies.

Any module in this directory that uses @register_strategy will be
automatically discovered when this package is imported.
"""

import importlib
import pkgutil
from pathlib import Path

# Import base first
from .base import TestStrategy, register_strategy, get_strategy, list_strategies

# Auto-discover all strategy modules in this directory
_package_dir = Path(__file__).parent
for _importer, _module_name, _is_pkg in pkgutil.iter_modules([str(_package_dir)]):
    if _module_name != "base":
        importlib.import_module(f".{_module_name}", package=__name__)
```

这样，只要在 `engine/strategies/` 下创建一个新文件并使用 `@register_strategy` 装饰器，它就会自动注册。

### Task 2.10: TestRunner 编排器 (`engine/runner.py`)

**目的:** 替代旧的 `BenchmarkRunner` 类，只做编排，不做具体测试逻辑。

**Files:**
- Create: `engine/runner.py` (~200 行)
- Test: `tests/engine/test_runner.py`

**核心逻辑:**

```python
# engine/runner.py
class TestRunner:
    """
    Orchestrates test execution.
    
    This is NOT the God Object. It only:
    1. Resolves the right TestStrategy
    2. Sets up tokenizer, provider, prompt generator
    3. Delegates execution to the strategy
    4. Collects results and emits events
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    async def run(self, run_config: TestRunConfig) -> list[TestResult]:
        """Execute a test run."""
        # 1. Resolve strategy
        strategy_cls = get_strategy(run_config.test_type.value)
        strategy = strategy_cls(self.event_bus)

        # 2. Setup dependencies
        tokenizer = TokenizerManager.get(run_config.base)
        provider = ProviderRegistry.get(run_config.base)
        prompt_gen = PromptGenerator(tokenizer)

        # 3. Calculate total and emit start
        total = strategy.calculate_total_requests(run_config.params)
        self.event_bus.emit(Event(
            type=EventType.TEST_START,
            data={"test_type": run_config.test_type, "total_requests": total}
        ))

        # 4. Delegate to strategy
        try:
            results = await strategy.execute(
                config=run_config.base,
                params=run_config.params,
                provider=provider,
                tokenizer=tokenizer,
                prompt_generator=prompt_gen,
            )
            self.event_bus.emit(Event(
                type=EventType.TEST_END,
                data={"success": True, "total_results": len(results)}
            ))
            return results
        except Exception as e:
            self.event_bus.emit(Event(
                type=EventType.ERROR,
                data={"error": str(e)}
            ))
            raise
```

---

## Phase 3: FastAPI 后端

> **目标:** 提供 REST API + WebSocket，替代 Streamlit 的交互模型。

### Task 3.1: FastAPI 应用骨架

**Files:**
- Create: `server/app.py`
- Create: `server/config.py`

```python
# server/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="LLM Benchmark Platform",
    version="3.0.0",
    description="LLM Performance & Quality Benchmark API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from server.routes import tests, results, reports, ws
app.include_router(tests.router, prefix="/api/tests", tags=["Tests"])
app.include_router(results.router, prefix="/api/results", tags=["Results"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
app.include_router(ws.router, prefix="/ws", tags=["WebSocket"])

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/strategies")
async def list_test_strategies():
    """List all available test strategies with their parameter schemas."""
    from engine.strategies import list_strategies
    return list_strategies()
```

### Task 3.2: 测试执行 API (`server/routes/tests.py`)

**核心端点:**

| Method | Path | 功能 |
|--------|------|------|
| `GET` | `/api/strategies` | 列出所有测试策略 + 参数 schema |
| `POST` | `/api/tests/run` | 启动测试 (返回 run_id) |
| `POST` | `/api/tests/{run_id}/stop` | 停止测试 |
| `POST` | `/api/tests/{run_id}/pause` | 暂停测试 |
| `GET` | `/api/tests/{run_id}/status` | 查询状态 |
| `GET` | `/api/results/{run_id}` | 获取结果 |
| `GET` | `/api/results/{run_id}/csv` | 下载 CSV |

```python
# server/routes/tests.py
from fastapi import APIRouter, BackgroundTasks
from engine.runner import TestRunner
from engine.events import EventBus
from engine.models import TestRunConfig

router = APIRouter()

# In-memory run registry (production: use Redis/DB)
_active_runs: dict[str, dict] = {}

@router.post("/run")
async def start_test(config: TestRunConfig, background_tasks: BackgroundTasks):
    """Start a benchmark test run."""
    import uuid
    run_id = str(uuid.uuid4())[:8]
    
    event_bus = EventBus()
    runner = TestRunner(event_bus)
    
    _active_runs[run_id] = {
        "event_bus": event_bus,
        "status": "running",
        "results": [],
    }
    
    background_tasks.add_task(_execute_run, run_id, runner, config)
    
    return {"run_id": run_id, "status": "started"}

async def _execute_run(run_id: str, runner: TestRunner, config: TestRunConfig):
    try:
        results = await runner.run(config)
        _active_runs[run_id]["results"] = [r.model_dump() for r in results]
        _active_runs[run_id]["status"] = "completed"
    except Exception as e:
        _active_runs[run_id]["status"] = "failed"
        _active_runs[run_id]["error"] = str(e)
```

### Task 3.3: WebSocket 实时推送 (`server/routes/ws.py`)

**目的:** 前端通过 WebSocket 接收实时进度、日志、结果。

```python
# server/routes/ws.py
from fastapi import APIRouter, WebSocket
import json

router = APIRouter()

@router.websocket("/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await websocket.accept()
    
    run_info = _active_runs.get(run_id)
    if not run_info:
        await websocket.close(code=4004)
        return
    
    event_bus = run_info["event_bus"]
    queue = asyncio.Queue()
    
    def on_event(event):
        queue.put_nowait(event)
    
    # Subscribe to all events
    for et in EventType:
        event_bus.subscribe(et, on_event)
    
    try:
        while True:
            event = await queue.get()
            await websocket.send_json({
                "type": event.type.value,
                "data": event.data,
            })
    except Exception:
        pass
```

### Task 3.4: 结果与报告 API

**Files:**
- Create: `server/routes/results.py`
- Create: `server/routes/reports.py`

---

## Phase 4: 报告引擎重构

> **目标:** 将 1,822 行的 `ui/reports.py` 拆解为数据计算 + 渲染分离。

### Task 4.1: 报告数据计算 (`analysis/report_data.py`)

**目的:** 纯 Python 计算，输出结构化数据，不包含任何 HTML/Streamlit 渲染。

**核心改变:**

旧: `generate_concurrency_report()` — 300 行混合计算 + `st.markdown()` + `st.plotly_chart()`  
新: `compute_concurrency_report_data()` — 50 行纯计算，返回 `ConcurrencyReportData(Pydantic model)`

```python
# analysis/report_data.py
class ConcurrencyReportData(BaseModel):
    summary_stats: dict[str, float]       # 汇总指标
    per_level_data: list[dict]            # 按并发级别的数据
    chart_configs: list[ChartConfig]      # 图表配置 (数据 + 类型)
    insights: list[str]                   # 洞察文本
    grade: tuple[str, str, str]           # (等级, 颜色, 描述)

def compute_concurrency_report(df: pd.DataFrame, model_id: str) -> ConcurrencyReportData:
    """Pure computation, no UI."""
    ...
```

### Task 4.2: 报告渲染策略

报告渲染由前端（React）或多种 exporter 负责：

| 渲染目标 | 实现 |
|----------|------|
| React UI | 前端组件消费 JSON |
| PDF | `analysis/exporters/pdf.py` |
| Markdown | `analysis/exporters/markdown.py` |
| 静态图片 | `analysis/exporters/image.py` (matplotlib) |

---

## Phase 5: React 前端

> **目标:** 现代化 SPA 替代 Streamlit，更好的交互体验。

### Task 5.1: 初始化 React 项目

```bash
cd web
npx -y create-vite@latest ./ --template react-ts
npm install antd @ant-design/icons recharts zustand
npm install -D @types/node
```

### Task 5.2: 动态测试配置面板

**核心创新:** 前端从 `/api/strategies` 获取所有测试策略及其 `param_schema`，**自动渲染配置表单**。新增测试类型后，前端无需修改。

```tsx
// web/src/components/TestConfigPanel.tsx
function TestConfigPanel({ strategy }) {
    // 根据 param_schema 动态生成表单
    return (
        <Form>
            {Object.entries(strategy.param_schema).map(([key, schema]) => (
                <DynamicFormField key={key} name={key} schema={schema} />
            ))}
            <Button type="primary" onClick={startTest}>
                🚀 Start Test
            </Button>
        </Form>
    );
}
```

### Task 5.3: 实时进度组件 (WebSocket)

```tsx
// web/src/hooks/useWebSocket.ts
function useTestProgress(runId: string) {
    const [progress, setProgress] = useState({ completed: 0, total: 0 });
    const [logs, setLogs] = useState<string[]>([]);
    const [results, setResults] = useState<TestResult[]>([]);

    useEffect(() => {
        const ws = new WebSocket(`ws://localhost:8000/ws/${runId}`);
        ws.onmessage = (e) => {
            const event = JSON.parse(e.data);
            switch (event.type) {
                case "progress":
                    setProgress(event.data);
                    break;
                case "log":
                    setLogs(prev => [...prev.slice(-99), event.data.message]);
                    break;
                case "result":
                    setResults(prev => [...prev, event.data]);
                    break;
            }
        };
        return () => ws.close();
    }, [runId]);

    return { progress, logs, results };
}
```

### Task 5.4: 图表与报告页面

使用 **Recharts** 或 **ECharts** 替代 Plotly（更轻量、React 友好）。

### Task 5.5: 测试历史与对比页面

---

## Phase 6: 集成与迁移

### Task 6.1: Streamlit 兼容层 (可选过渡)

如果需要平滑过渡，可以创建一个 Streamlit 适配器：

```python
# legacy/streamlit_adapter.py
"""Adapter that connects the new engine to the old Streamlit UI."""
from engine.events import EventBus, EventType, Event
import streamlit as st

class StreamlitEventAdapter:
    """Bridges EventBus to Streamlit UI updates."""
    
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe(EventType.LOG, self._on_log)
        event_bus.subscribe(EventType.PROGRESS, self._on_progress)
        event_bus.subscribe(EventType.ERROR, self._on_error)
        self._log_placeholder = None
        self._progress_bar = None

    def _on_log(self, event: Event):
        if self._log_placeholder:
            self._log_placeholder.info(event.data["message"])

    def _on_progress(self, event: Event):
        if self._progress_bar:
            pct = event.data["completed"] / max(event.data["total"], 1)
            self._progress_bar.progress(pct)

    def _on_error(self, event: Event):
        st.error(event.data.get("error", "Unknown error"))
```

### Task 6.2: 迁移 evaluators

`evaluators/` 目录的设计相对良好，只需：
1. 移除注释中的中英混杂
2. 添加 `registry.py` 实现自动发现
3. 确保 `BaseEvaluator` 不依赖 Streamlit

### Task 6.3: Provider 注册表

```python
# engine/providers/registry.py
_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {}

def register_provider(name: str):
    def decorator(cls):
        _PROVIDER_REGISTRY[name] = cls
        return cls
    return decorator

@register_provider("openai_compatible")
class OpenAICompatProvider(LLMProvider): ...

@register_provider("gemini")
class GeminiNativeProvider(LLMProvider): ...
```

### Task 6.4: 数据迁移

将现有 SQLite 数据库 schema 迁移到新的 SQLAlchemy ORM。

### Task 6.5: 清理旧代码

将旧的 `core/benchmark_runner.py`、`ui/reports.py`、`ui/test_panels.py` 等移至 `legacy/` 目录。

### Task 6.6: 文档更新

更新 `ARCHITECTURE.md`、`README.md`、`DEVELOPMENT.md`。

---

## 附录

### 新旧架构对比

| 维度 | 旧架构 (Streamlit) | 新架构 (FastAPI + React) |
|------|-------------------|------------------------|
| **核心引擎** | `benchmark_runner.py` 2,658 行 God Object | `engine/runner.py` ~200 行编排器 + 8 个策略文件各 ~150 行 |
| **UI 耦合** | 核心层直接调用 `st.xxx()` 20+ 处 | 核心层零 UI 依赖，通过 EventBus 通信 |
| **新增测试类型** | 修改 8-9 个文件 (散弹枪手术) | 创建 1 个策略文件 + `@register_strategy` 装饰器 |
| **新增 Provider** | 修改 `factory.py` 的 if/else | 创建 1 个文件 + `@register_provider` 装饰器 |
| **运行方式** | 仅 Streamlit Web | FastAPI API + React UI + CLI + CI/CD |
| **可测试性** | 难以单测 (Streamlit 依赖) | 核心引擎可完全单测 |
| **实时通信** | Streamlit 轮询 | WebSocket 推送 |
| **前端配置** | 硬编码 if/elif 分支 | 动态表单 (从 param_schema 自动生成) |
| **报告生成** | 1,822 行 UI 混合渲染 | 数据计算分离 + 多格式渲染器 |

### 新增测试类型完整流程 (重构后)

**只需 1 个文件!**

```python
# engine/strategies/my_new_test.py
from engine.strategies.base import TestStrategy, register_strategy

@register_strategy("my_new_test")
class MyNewTestStrategy(TestStrategy):
    display_name = "My New Test"
    description = "Description of what this test does"
    icon = "🆕"

    @classmethod
    def param_schema(cls):
        return {
            "param1": {"type": "integer", "default": 10, "title": "Parameter 1"},
            "param2": {"type": "string", "default": "hello", "title": "Parameter 2"},
        }

    def calculate_total_requests(self, params):
        return params["param1"]

    async def execute(self, config, params, provider, tokenizer, prompt_generator):
        results = []
        for i in range(params["param1"]):
            if self.event_bus.is_stop_requested():
                break
            # ... do the test ...
            results.append(TestResult(session_id=i, ...))
            self.event_bus.emit(Event(type=EventType.PROGRESS, data={
                "completed": i + 1, "total": params["param1"]
            }))
        return results

    def csv_columns(self):
        return ["session_id", "param1", "result_metric"]
```

然后:
- **API** 自动在 `/api/strategies` 中列出这个新测试
- **前端** 自动渲染对应配置表单 (根据 `param_schema`)
- **CLI** 自动支持 `llm-bench run --type my_new_test --param1 20`
- **无需修改任何其他文件**

### 执行时间估计

| Phase | 预估工时 | 优先级 |
|-------|---------|--------|
| Phase 0: 前置准备 | 0.5 天 | P0 |
| Phase 1: 核心引擎重构 | 3-4 天 | P0 |
| Phase 2: 策略插件系统 | 2-3 天 | P0 |
| Phase 3: FastAPI 后端 | 2-3 天 | P1 |
| Phase 4: 报告引擎重构 | 2 天 | P1 |
| Phase 5: React 前端 | 4-5 天 | P2 |
| Phase 6: 集成与迁移 | 2-3 天 | P2 |
| **总计** | **16-21 天** | |

### 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 迁移期间旧功能不可用 | Phase 6.1 提供 Streamlit 兼容层，可在新引擎上运行旧 UI |
| 前端开发周期长 | 先完成后端 API + CLI，React 可后续或并行开发 |
| 测试覆盖不足 | 每个 Phase 都包含测试，Phase 1-2 的核心引擎要求 >80% 覆盖 |
| Evaluators 迁移遗漏 | Evaluators 设计良好，仅需清理依赖，风险低 |

---

*计划制定日期: 2026-02-13*
*最后更新: 2026-02-13*
