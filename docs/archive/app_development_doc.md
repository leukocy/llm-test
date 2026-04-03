# LLM 基准测试平台 V2 - 开发文档

## 1. 项目概述
本项目是一个基于 Streamlit 的 LLM (Large Language Model) 性能基准测试平台 V2。采用模块化架构，旨在通过标准化的测试流程，评估不同 LLM API 供应商及模型在并发、首字延迟 (TTFT)、吞吐量 (TPS)、长上下文处理和质量评估等方面的性能表现。

### 技术栈
- **前端/框架**: Streamlit
- **异步处理**: Python `asyncio`
- **HTTP 客户端**: `httpx` (支持异步请求)
- **数据处理**: `pandas`, `numpy`
- **可视化**: `plotly.express`, `plotly.graph_objects`
- **Token 计算**: `tiktoken`, `transformers`
- **安全**: 自定义 SSRF 保护、速率限制、日志清理

## 2. 架构设计

### 2.1 模块化架构 (V2)
应用采用模块化架构，将原来的单文件 (2718 行) 拆分为多个专用模块：

```
llm-test/
├── app.py (245 行)          # 主入口 - 应用启动和路由
├── config/                  # 配置管理模块
│   ├── session_state.py     # Streamlit 会话状态集中管理
│   ├── settings.py          # 全局配置常量
│   ├── test_config_loader.py # 测试配置加载器
│   ├── secrets.py           # API 密钥安全管理
│   ├── auth.py              # 可选认证模块
│   └── development_settings.py # 开发环境配置
├── core/                    # 核心业务逻辑
│   ├── benchmark_runner.py  # 基准测试运行器
│   ├── batch_test.py        # 批量测试调度器 (~670 行)
│   ├── providers/           # LLM API 供应商适配器
│   │   ├── base.py          # 抽象基类
│   │   ├── openai.py        # OpenAI 兼容实现
│   │   ├── gemini.py        # Google Gemini 实现
│   │   └── factory.py       # 供应商工厂
│   ├── quality_evaluator.py # 质量评估器
│   ├── enhanced_parser.py   # 智能答案解析器
│   ├── consistency_tester.py # 一致性测试
│   ├── robustness_tester.py  # 鲁棒性测试
│   ├── failure_analyzer.py   # 失败分析器
│   ├── tokenizer_utils.py   # Token 工具
│   ├── dataset_loader.py    # 数据集加载器
│   ├── response_cache.py    # 响应缓存
│   ├── safe_executor.py     # 安全代码执行
│   ├── url_validator.py     # URL 验证器 (SSRF 保护)
│   ├── rate_limiter.py      # 速率限制器
│   └── ...                  # 其他核心模块
├── ui/                      # UI 组件模块
│   ├── sidebar.py           # 侧边栏配置 (328 行)
│   ├── test_panels.py       # 测试配置面板 (291 行)
│   ├── test_runner.py       # 测试执行器 (242 行)
│   ├── page_layout.py       # 页面布局 (209 行)
│   ├── advanced_panels.py   # 高级测试面板 (597 行)
│   ├── batch_test.py        # 批量测试 UI (~477 行)
│   ├── charts.py            # 图表生成
│   ├── reports.py           # 报告生成
│   ├── thinking_components.py # 推理组件
│   └── ...                  # 其他 UI 模块
├── utils/                   # 工具函数模块
│   ├── logger.py            # 日志工具
│   ├── helpers.py           # 辅助函数
│   ├── log_sanitizer.py     # 日志清理工具
│   └── ...                  # 其他工具
├── evaluators/              # 数据集评估器
│   ├── mmlu_evaluator.py    # MMLU 评估
│   ├── gsm8k_evaluator.py   # GSM8K 评估
│   └── ...                  # 其他评估器 (19 个数据集)
└── tests/                   # 测试套件
    ├── test_*.py            # 单元测试 (200+ tests)
    └── conftest.py          # Pytest 配置
```

### 2.2 架构优势
| 指标 | 重构前 | 重构后 | 改进 |
|------|--------|--------|------|
| app.py 行数 | 2718 | 245 | -91% |
| 模块化程度 | 单文件 | 12+ 模块 | ⬆️⬆️ |
| 可维护性 | 低 | 高 | ⬆️⬆️ |
| 可测试性 | 低 | 高 (200+ tests) | ⬆️⬆️ |
| 代码复用 | 困难 | 容易 | ⬆️⬆️ |

## 3. 核心组件详解

### 3.1 主入口 (app.py)
app.py 现在仅负责应用启动和路由，主要功能：
- **会话初始化**: 调用 `init_session_state()` 初始化所有 session_state
- **路由分发**: 根据测试类型分发到不同的 UI 面板
- **向后兼容**: 提供 `run_test()` 包装函数保持兼容性

**关键函数**:
```python
def main():
    """主程序入口"""
    init_session_state()           # 1. 初始化会话状态
    init_builtin_presets()         # 2. 初始化内置预设
    init_onboarding_state()        # 3. 初始化引导状态

    if show_onboarding():          # 4. 渲染新手引导
        render_onboarding_modal()

    sidebar_config = render_sidebar()  # 5. 渲染侧边栏
    render_onboarding_trigger()     # 6. 引导触发器

    # 7. 路由分发
    test_type = sidebar_config['test_type']
    if test_type == "📝 模型质量测试":
        render_quality_test_panel(sidebar_config, run_test)
    elif test_type == "🔄 A/B 模型对比":
        render_ab_comparison_panel(sidebar_config)
    elif test_type == "🔬 高级评测分析":
        render_advanced_eval_panel(sidebar_config)
    elif test_type == "📦 批量测试":
        render_batch_test_main()
    else:
        render_test_panels(test_type, run_test)  # 普通测试
```

### 3.2 配置管理 (config/)

#### session_state.py
集中管理所有 Streamlit session_state 变量：
```python
def init_session_state():
    """初始化所有 session_state 变量"""
    # 测试状态
    if 'test_running' not in st.session_state:
        st.session_state.test_running = False
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False

    # 结果数据
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    # 日志
    if 'log_content' not in st.session_state:
        st.session_state.log_content = []

    # ... 其他状态变量
```

#### secrets.py
安全的 API 密钥管理：
- 从环境变量或 UI 输入获取 API Key
- 不在源代码中硬编码密钥
- 支持开发/生产环境分离

#### url_validator.py
SSRF (Server-Side Request Forgery) 保护：
```python
def is_safe_url(url: str) -> bool:
    """验证 URL 是否安全，防止 SSRF 攻击"""
    parsed = urlparse(url)
    # 检查协议
    if parsed.scheme not in ('http', 'https'):
        return False
    # 检查内网 IP
    if is_private_ip(parsed.hostname):
        return False
    return True
```

### 3.3 核心业务逻辑 (core/)

#### benchmark_runner.py
核心测试引擎，包含所有测试方法：
- `run_concurrency_test()` - 并发性能测试
- `run_prefill_test()` - Prefill 压力测试
- `run_long_context_test()` - 长上下文测试
- `run_throughput_matrix_test()` - 综合矩阵测试
- `run_stability_test()` - 稳定性测试
- `run_all_tests()` - 全部测试

#### batch_test.py
批量测试调度器 (~670 行)：
- 支持多模型/多配置自动化测试
- 顺序执行和并行执行模式
- 自动生成对比报告
- 实时进度显示

#### providers/
LLM API 供应商适配器：
```
providers/
├── base.py          # LLMProvider 抽象基类
├── openai.py        # OpenAIProvider (兼容 DeepSeek, Kimi 等)
├── gemini.py        # GeminiProvider (Google Gemini)
└── factory.py       # get_provider(name, config) 工厂函数
```

**扩展新供应商**:
1. 继承 `LLMProvider` 基类
2. 实现 `get_completion()` 方法
3. 在 `factory.py` 中注册

#### safe_executor.py
安全代码执行模块：
- 用于执行用户提供的代码（如 HumanEval 评估）
- 沙箱环境执行
- 超时控制
- 资源限制

#### rate_limiter.py
令牌桶速率限制器：
```python
class RateLimiter:
    """令牌桶速率限制器"""
    def __init__(self, rate: float, capacity: int):
        self.rate = rate      # 令牌生成速率
        self.capacity = capacity  # 桶容量

    def acquire(self, tokens: int = 1) -> bool:
        """获取令牌，如果获取成功返回 True"""
```

### 3.4 UI 组件 (ui/)

#### sidebar.py (328 行)
侧边栏配置面板：
- API 供应商选择
- 模型 ID 选择/输入
- API Key 输入
- Token 计算方式选择
- 测试类型选择
- 自定义供应商/模型管理

#### test_panels.py (291 行)
7 种测试类型的配置面板：
- 并发性能测试
- Prefill 压力测试
- 长上下文测试
- 综合矩阵测试
- 自定义文本测试
- 全部测试
- 稳定性测试

#### test_runner.py (242 行)
测试执行流程封装：
- `TestExecutor` 类
- 异步任务执行
- 进度条更新
- 实时日志显示
- 错误处理

#### advanced_panels.py (597 行)
高级测试面板：
- 质量测试面板 (MMLU, GSM8K, GPQA 等)
- A/B 模型对比面板
- 高级评测分析面板 (一致性、鲁棒性)

#### charts.py
图表生成模块：
- `plot_plotly_bar()` - 柱状图
- `plot_plotly_line()` - 折线图
- `plot_relative_performance_bar()` - 相对性能对比图
- 热力图 (综合测试报告)

#### thinking_components.py
推理组件：
- 显示模型推理过程
- 支持流式显示
- Markdown/HTML 导出

### 3.5 工具模块 (utils/)

#### log_sanitizer.py
日志清理工具：
- 防止日志注入攻击
- 清理敏感信息 (API Key)
- 转义特殊字符

#### preset_manager.py
预设配置管理：
- 保存/加载测试配置
- 导入/导出配置
- 内置预设 (OpenAI GPT-4, DeepSeek, Kimi 等)

## 3. 核心组件详解

### 3.1 全局状态管理
应用使用 `st.session_state` 维护以下关键状态：
- `test_running`: 标记测试是否正在运行，控制按钮禁用状态。
- `stop_requested`: 用于响应用户的“停止测试”操作。
- `results_df`: 存储测试结果的 DataFrame。
- `log_content`: 存储实时日志列表。
- `current_csv_file` / `current_log_file`: 当前测试对应的文件名。

### 3.2 配置模块 (Sidebar)
用户可以在侧边栏配置：
- **API 供应商**: 支持 OpenAI 兼容接口 (DeepSeek, Kimi, Local 等) 和 Google Gemini (特殊适配)。
- **模型 ID**: 提供预设列表和自定义输入。
- **API Key**: 用于鉴权。
- **Token 计算方式**: 支持 API 返回 usage、Tiktoken 估算或字符数回退。
- **测试类型**: 选择具体的测试模式。

### 3.3 BenchmarkRunner 类
这是系统的核心引擎。

#### 初始化 (`__init__`)
接收 UI 占位符、进度条、配置参数等，初始化内部状态。

#### API 调度 (`get_completion`)
根据 `provider` 字段分发请求：
- **Gemini**: 调用 `_get_completion_gemini` (处理 SSE 流式响应的特殊格式)。
- **其他 (OpenAI 兼容)**: 调用 `_get_completion_openai` (标准 SSE 格式)。

#### 指标计算
- **`_calculate_metrics`**: 计算 TTFT (首字延迟) 和 TPS (生成吞吐量)。包含对 TTFT 的校准 (`TTFT_CALIBRATION_OFFSET`)。
- **`_calculate_tokens`**: 根据配置策略 (API usage vs Tiktoken) 计算 Prompt 和 Completion 的 Token 数。

#### 测试方法
- **`run_concurrency_test`**: 并发性能测试。使用 `asyncio.gather` 并行发送请求，测量高并发下的系统表现。
- **`run_prefill_test`**: Prefill 压力测试。通过构造不同长度的 Prompt，测试模型对长输入的处理速度。
- **`run_long_context_test`**: 长上下文测试。类似 Prefill 测试，但侧重于长文境下的生成表现。
- **`run_throughput_matrix_test`**: 综合矩阵测试。测试 [并发数 x 上下文长度] 的所有组合。
- **`run_all_tests`**: 一键串行执行上述所有测试。

### 3.4 可视化模块
使用 Plotly 绘制交互式图表，主要包含：
- **`plot_plotly_bar`**: 柱状图，用于展示 TTFT、吞吐量等对比。支持相对性能 (%) 显示。
- **`plot_plotly_line`**: 折线图，用于展示随并发数或上下文长度变化的性能趋势。
- **`plot_relative_performance_bar`**: 专门的相对性能对比图。
- **热力图**: 在综合测试报告中，展示并发与上下文长度对性能的联合影响。

## 4. 数据流与执行流程

### 4.1 完整执行流程
```
┌─────────────────┐
│   用户在 UI     │
│   配置参数      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  render_sidebar │  ← ui/sidebar.py
│  (侧边栏配置)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ init_session_   │  ← config/session_state.py
│     state       │     初始化所有状态
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ render_test_    │  ← ui/test_panels.py
│     panels      │     显示测试配置面板
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   用户点击      │
│  "开始测试"     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TestExecutor.  │  ← ui/test_runner.py
│    run_test     │     创建执行器
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BenchmarkRunner │  ← core/benchmark_runner.py
│  .run_xxx_test  │     执行具体测试
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   get_provider  │  ← core/providers/factory.py
│   (供应商工厂)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Provider.get_   │  ← core/providers/openai.py
│   completion    │     异步 API 调用
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   实时反馈      │
│ - 日志更新      │  ← utils/logger.py
│ - 进度条        │  ← ui/test_runner.py
│ - 结果表格      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   结果处理      │
│ - 计算指标      │  ← core/metrics.py
│ - 保存 CSV      │
│ - 更新图表      │  ← ui/charts.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   报告生成      │
│ - Markdown      │  ← ui/reports.py
│ - Plotly 图表   │  ← ui/charts.py
└─────────────────┘
```

### 4.2 请求处理流程
```
用户请求 → URL 验证 (url_validator.py)
         ↓
    速率检查 (rate_limiter.py)
         ↓
    获取供应商 (factory.py)
         ↓
    API 调用 (providers/openai.py 或 gemini.py)
         ↓
    流式响应处理
         ↓
    指标计算 (TTFT, TPS, Tokens)
         ↓
    结果存储 (CSV + DataFrame)
         ↓
    UI 更新 (图表 + 表格)
```

## 5. 关键逻辑说明

### 5.1 Token 计算策略
多级回退策略 (`core/tokenizer_utils.py`)：
1. **API usage 字段**: 优先使用 API 响应中的 `usage` 字段 (最准确)
2. **Tiktoken**: 如果 API 未返回 usage，使用 `tiktoken` 库进行本地计算
3. **HuggingFace Transformers**: 使用指定的 tokenizer 模型
4. **字符数回退**: 如果以上都不可用，回退到基于字符长度的估算

### 5.2 停止机制
通过 `st.session_state.stop_requested` 实现：
- 用户点击"停止测试"按钮，标志位置为 True
- 在 Provider 的流式读取循环中检测标志位
- 抛出 `asyncio.CancelledError` 中断请求
- 使用线程安全的事件机制

### 5.3 Gemini 适配
`core/providers/gemini.py` 单独处理 Gemini API 的特殊格式：
- URL 路径: `:streamGenerateContent`
- Payload 结构: `contents`, `generationConfig`
- 响应格式: 嵌套的 JSON 结构
- SSE 格式: `data: ` 前缀

### 5.4 安全机制

#### SSRF 保护
```python
# core/url_validator.py
def is_safe_url(url: str) -> bool:
    """验证 URL 是否安全，防止 SSRF 攻击"""
    parsed = urlparse(url)
    # 检查协议
    if parsed.scheme not in ('http', 'https'):
        return False
    # 检查内网 IP
    if is_private_ip(parsed.hostname):
        return False
    return True
```

#### 速率限制
```python
# core/rate_limiter.py
class RateLimiter:
    """令牌桶速率限制器"""
    def acquire(self, tokens: int = 1) -> bool:
        """获取令牌"""
```

#### 日志清理
```python
# utils/log_sanitizer.py
def sanitize_log(message: str) -> str:
    """清理日志中的敏感信息和危险字符"""
```

### 5.5 批量测试流程
```
批量测试配置 (ui/batch_test.py)
         ↓
创建测试项列表
         ↓
┌────────────────┐
│ 执行模式选择   │
├────────────────┤
│ 顺序执行       │  并行执行 │
└───────┬────────┴──────┐
        │               │
        ▼               ▼
   依次执行         asyncio.gather
   每个测试         并行运行所有测试
        │               │
        └───────┬───────┘
                ▼
         汇总结果
                ▼
         生成对比报告
```

## 6. 已实现的改进 (2026-01)

### 6.1 安全改进
| 改进项 | 状态 | 说明 |
|--------|------|------|
| API 密钥管理 | ✅ | 移除硬编码密钥，使用 `secrets.py` |
| SSRF 保护 | ✅ | `url_validator.py` 验证所有 URL |
| 安全代码执行 | ✅ | `safe_executor.py` 沙箱执行 |
| 速率限制 | ✅ | `rate_limiter.py` 令牌桶算法 |
| 日志清理 | ✅ | `log_sanitizer.py` 防止注入 |
| 路径遍历保护 | ✅ | `dataset_loader.py` 验证文件路径 |

### 6.2 代码质量改进
| 改进项 | 状态 | 说明 |
|--------|------|------|
| 模块化重构 | ✅ | 2718 行 → 245 行 (app.py) |
| 移除死代码 | ✅ | 删除 68 行未使用的函数 |
| 修复异常处理 | ✅ | 9 个 bare except → 具体异常类型 |
| 修复资源泄漏 | ✅ | 临时文件泄漏已修复 |
| 单元测试 | ✅ | 200+ 测试用例 |

### 6.3 测试覆盖
```
tests/
├── test_security.py        # 21 个安全测试
├── test_benchmark_runner.py # 核心测试
├── test_providers/         # 供应商测试
├── test_evaluators/        # 评估器测试
└── conftest.py            # Pytest 配置
```

## 7. 扩展指南

### 7.1 添加新的 LLM 供应商

**步骤**:
1. 在 `core/providers/` 创建新文件 `new_provider.py`
2. 继承 `LLMProvider` 基类:
```python
from core.providers.base import LLMProvider

class NewProvider(LLMProvider):
    async def get_completion(self, prompt: str, **kwargs) -> dict:
        # 实现异步 API 调用
        pass
```
3. 在 `core/providers/factory.py` 中注册:
```python
def get_provider(name: str, config: dict) -> LLMProvider:
    if name == "NewProvider":
        return NewProvider(config)
    # ...
```
4. 在 `config/settings.py` 或 `config/development_settings.py` 添加配置

### 7.2 添加新的测试类型

**步骤**:
1. 在 `core/benchmark_runner.py` 添加测试方法:
```python
async def run_new_test(self, params: dict):
    """新的测试方法"""
    # 实现测试逻辑
    pass
```
2. 在 `ui/test_panels.py` 添加配置面板:
```python
def render_new_test_panel():
    """新测试的配置面板"""
    # 添加配置选项
    pass
```
3. 在 `config/settings.py` 的 `TEST_TYPES` 添加选项
4. 在 `ui/charts.py` 添加图表生成函数 (如需要)

### 7.3 添加新的评估数据集

**步骤**:
1. 在 `evaluators/` 创建新的评估器文件
2. 继承基类或实现标准接口:
```python
class NewDatasetEvaluator:
    def evaluate(self, model_responses: list) -> dict:
        # 评估逻辑
        pass
```
3. 在 `core/quality_evaluator.py` 中集成
4. 下载数据集到 `datasets/` 目录

### 7.4 添加新的 UI 组件

**步骤**:
1. 在 `ui/` 创建新组件文件
2. 遵循现有命名规范 (如 `new_component.py`)
3. 使用 Streamlit 组件构建 UI
4. 在 `app.py` 或相应的面板中导入使用

## 8. 开发环境设置

### 8.1 安装依赖
```bash
# 生产依赖
pip install -r requirements.txt

# 开发依赖
pip install -r requirements-dev.txt
```

### 8.2 运行测试
```bash
# 所有测试
pytest tests/ -v

# 安全测试
pytest tests/test_security.py -v

# 覆盖率
pytest --cov=. --cov-report=html
```

### 8.3 代码检查
```bash
# Ruff 检查
ruff check .

# Ruff 格式化
ruff format .

# 类型检查
mypy core/ ui/ config/
```

### 8.4 启动应用
```bash
streamlit run app.py
```

## 9. 常见问题

### Q1: 如何调试异步代码？
A: 使用 `asyncio.run()` 在测试中运行异步函数，或在 Streamlit 中使用 `TestExecutor` 包装。

### Q2: 如何添加新的 API Key？
A: 通过侧边栏的 API Key 输入框，或设置环境变量 `OPENAI_API_KEY`。

### Q3: 测试数据保存在哪里？
A: CSV 文件保存在 `raw_data/` 目录，图表保存在根目录。

### Q4: 如何配置自定义供应商？
A: 在侧边栏"自定义配置管理"中添加，或编辑 `config/development_settings.py`。

### Q5: 批量测试失败怎么办？
A: 检查 `batch_tests/` 目录下的日志文件，查看具体错误信息。

## 10. 参考资源

- **Streamlit 文档**: https://docs.streamlit.io
- **Plotly 文档**: https://plotly.com/python/
- **httpx 文档**: https://www.python-httpx.org/
- **项目 README**: README.md
- **Git 工作流**: DEVELOPMENT.md
- **代码审查报告**: CODE_REVIEW_REPORT.md

---

**文档版本**: v2.0
**最后更新**: 2026-01-31
**维护者**: llm-test 项目组
