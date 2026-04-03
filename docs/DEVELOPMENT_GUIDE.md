# LLM-Test 开发指南

> 版本: 2.0
> 更新日期: 2026-02-02

---

## 📖 目录

1. [快速开始](#快速开始)
2. [开发环境设置](#开发环境设置)
3. [代码规范](#代码规范)
4. [Git 工作流](#git-工作流)
5. [添加新功能](#添加新功能)
6. [测试指南](#测试指南)
7. [调试技巧](#调试技巧)

---

## 快速开始

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd llm-test

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 启动开发服务器

```bash
streamlit run app.py
```

应用将在 http://localhost:8501 启动。

---

## 开发环境设置

### 推荐工具

- **IDE**: VS Code / PyCharm
- **代码格式化**: Black
- **代码检查**: Ruff
- **类型检查**: MyPy
- **测试**: Pytest

### VS Code 配置

创建 `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true
  }
}
```

---

## 代码规范

### 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 类名 | PascalCase | `BenchmarkRunner`, `QualityEvaluator` |
| 函数名 | snake_case | `run_test`, `calculate_metrics` |
| 变量名 | snake_case | `api_key`, `max_tokens` |
| 常量 | UPPER_SNAKE_CASE | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| 私有成员 | 前缀下划线 | `_internal_method`, `_private_var` |

### 文档字符串

```python
def calculate_metrics(
    start_time: float,
    end_time: float,
    tokens: int
) -> dict:
    """
    计算性能指标

    Args:
        start_time: 开始时间戳
        end_time: 结束时间戳
        tokens: 生成的 token 数量

    Returns:
        包含 ttft, tps, tpot 等指标的字典

    Example:
        >>> metrics = calculate_metrics(100.0, 105.0, 100)
        >>> print(metrics['tps'])
        20.0
    """
```

### 类型注解

```python
from typing import List, Dict, Optional, Callable

def process_results(
    results: List[dict],
    callback: Optional[Callable] = None
) -> Dict[str, float]:
    """处理结果并返回统计数据"""
    pass
```

---

## Git 工作流

### 分支策略

```
main (主分支，保持稳定)
├── feature/* (功能开发)
├── fix/* (Bug 修复)
└── refactor/* (代码重构)
```

### 提交规范

```
<Type>: <Subject>

<Body>

<Footer>
```

| Type | 说明 |
|------|------|
| `Feat` | 新功能 |
| `Fix` | Bug 修复 |
| `Docs` | 文档修改 |
| `Style` | 格式调整 |
| `Refactor` | 代码重构 |
| `Test` | 测试相关 |
| `Chore` | 构建/工具相关 |

**示例**:

```
Feat: 添加批量测试调度器

- 实现 BatchTestScheduler 类
- 支持顺序和并行执行模式
- 添加进度回调接口

Closes #123
```

### 开发新功能流程

```bash
# 1. 切换到主分支并更新
git checkout main
git pull origin main

# 2. 创建功能分支
git checkout -b feature/add-metric-type

# 3. 开发和提交
git add .
git commit -m "Feat: 添加新的指标类型"

# 4. 推送到远程
git push origin feature/add-metric-type

# 5. 创建 Pull Request
```

---

## 添加新功能

### 添加新的测试类型

**1. 在 `ui/test_panels.py` 添加配置面板**

```python
elif test_type == "新测试类型":
    st.header("🔬 新测试类型")

    with st.sidebar.expander("📊 参数设置", expanded=True):
        param1 = st.number_input("参数1", min_value=1, value=10)
        param2 = st.slider("参数2", 0, 100, 50)

    if st.button("🚀 开始测试"):
        run_test_func(
            BenchmarkRunner.run_new_test,
            BenchmarkRunner,
            param1, param2
        )
```

**2. 在 `core/benchmark_runner.py` 添加测试方法**

```python
async def run_new_test(
    self,
    param1: int,
    param2: int,
    log_callback: Optional[Callable] = None
) -> None:
    """执行新的测试类型"""
    # 实现测试逻辑
    pass
```

**3. 在 `config/settings.py` 更新测试类型列表**

```python
TEST_TYPES = [
    # ... 现有测试
    "新测试类型",
]
```

### 添加新的质量评估器

**1. 创建 `evaluators/new_dataset_evaluator.py`**

```python
from .base_evaluator import BaseEvaluator, EvaluationResult, DatasetType

class NewDatasetEvaluator(BaseEvaluator):
    """新数据集评估器"""

    def __init__(self, data_path: str = "datasets/new_dataset"):
        self.data_path = data_path
        # 初始化代码

    async def evaluate(
        self,
        model: str,
        samples: List[dict],
        config: dict
    ) -> EvaluationResult:
        """评估模型在新数据集上的表现"""
        # 实现评估逻辑
        pass
```

**2. 在 `evaluators/__init__.py` 注册**

```python
from .new_dataset_evaluator import NewDatasetEvaluator

EVALUATOR_REGISTRY = {
    # ... 现有评估器
    'new_dataset': NewDatasetEvaluator,
}
```

### 添加新的 API 供应商

**1. 创建 `core/providers/new_provider.py`**

```python
from .base import LLMProvider

class NewProvider(LLMProvider):
    """新 API 供应商适配器"""

    async def get_completion(
        self,
        prompt: str,
        max_tokens: int = 1000,
        **kwargs
    ) -> dict:
        """获取模型完成"""
        # 实现 API 调用逻辑
        pass
```

**2. 在 `core/providers/factory.py` 注册**

```python
def get_provider(
    provider_name: str,
    api_base_url: str,
    api_key: str,
    model_id: str
) -> LLMProvider:
    if provider_name == "new_provider":
        return NewProvider(api_base_url, api_key, model_id)
    # ... 其他供应商
```

---

## 测试指南

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_benchmark_runner.py -v

# 运行特定测试
pytest tests/test_metrics.py::TestMetrics::test_calculate_tps -v

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

### 编写测试

```python
import pytest
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_runner():
    """创建测试用的 BenchmarkRunner 实例"""
    return BenchmarkRunner(
        placeholder=MagicMock(),
        progress_bar=MagicMock(),
        # ... 其他必需参数
    )

class TestNewFeature:
    """新功能的测试类"""

    def test_basic_functionality(self, mock_runner):
        """测试基本功能"""
        result = mock_runner.new_method()
        assert result == expected_value

    @pytest.mark.asyncio
    async def test_async_functionality(self, mock_runner):
        """测试异步功能"""
        result = await mock_runner.async_method()
        assert result is not None
```

---

## 调试技巧

### Streamlit 调试

```python
import streamlit as st

# 使用 st.write 输出调试信息
st.write("Debug:", variable)

# 使用 st.json 查看结构
st.json(complex_object)

# 使用 st.exception 显示异常
try:
    risky_operation()
except Exception as e:
    st.exception(e)
```

### 日志调试

```python
from utils.logger import get_logger

logger = get_logger(__name__)

# 不同级别的日志
logger.debug("调试信息")
logger.info("普通信息")
logger.warning("警告信息")
logger.error("错误信息")
```

### WebSocket 日志查看

在测试运行时，点击 "🔍 打开日志查看器" 查看实时日志流。

---

## 常见问题

### Q: 如何解决 asyncio 事件循环错误?

```python
# 使用 nest_asyncio 在 Jupyter/Streamlit 中
import nest_asyncio
nest_asyncio.apply()
```

### Q: 如何处理不同平台的 API 差异?

使用 `providers/` 中的适配器，通过工厂模式统一接口。

### Q: 如何添加新的 Tokenizer?

在 `core/tokenizer_utils.py` 中添加新的 tokenizer 加载逻辑。

---

*最后更新: 2026-02-02*
