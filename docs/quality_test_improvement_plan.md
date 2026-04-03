# 模型质量测试优化方案

> 基于针对 MiMo、DeepSeek、智谱AI、Gemini、火山引擎、阿里百炼、OpenRouter 等平台的深度 API 测试结果制定。

## ✅ 已完成的改进

| 改进项 | 文件 | 状态 |
|-------|------|------|
| Phase 1: 平台特性表 | `core/thinking_params.py` | ✅ 完成 |
| Phase 2: 统一响应解析器 | `core/response_parser.py` | ✅ 完成 |
| Phase 3: 推理指标计算 | `core/metrics.py` | ✅ 完成 |
| Phase 4: 重试机制 | `core/retry_handler.py` | ✅ 完成 |
| Phase 5: 增强评测报告 | `core/evaluation_report.py` | ✅ 完成 |
| Phase 6: UI 增强组件 | `ui/thinking_components.py` | ✅ 完成 |
| 智能答案解析器 | `core/smart_answer_parser.py` | ✅ 完成 |
| 推理质量评估器 | `core/reasoning_evaluator.py` | ✅ 完成 |
| 增强型评估器 | `core/enhanced_evaluator.py` | ✅ 完成 |
| SampleResult 扩展 | `evaluators/base_evaluator.py` | ✅ 完成 |
| YAML 测试配置系统 | `core/test_config.py` | ✅ 完成 |
| 失败案例分析系统 | `core/failure_analyzer.py` | ✅ 完成 |
| 一致性测试系统 | `core/consistency_tester.py` | ✅ 完成 |
| 统一测试运行器 | `core/test_runner.py` | ✅ 完成 |
| 鲁棒性测试系统 | `core/robustness_tester.py` | ✅ 完成 |
| 评测结果仪表板 | `ui/evaluation_dashboard.py` | ✅ 完成 |

**示例配置文件**:
- `tests/config/gsm8k_comparison.yaml` - 多模型对比测试
- `tests/config/consistency_test.yaml` - 一致性测试

---





## 一、核心问题诊断

通过 `api_tests/` 目录下的系统性测试，我们发现当前质量测试存在以下关键问题：

| 问题类别 | 具体表现 | 影响 |
|---------|---------|------|
| **参数不一致** | 各平台思考参数命名差异大（`thinking`、`enable_thinking`、`thinkingConfig`） | 部分模型未真正启用推理模式，导致对比失真 |
| **推理内容丢失** | 未捕获 `reasoning_content`、`thought` 等字段 | 无法评估模型的"思考质量" |
| **Token 统计不全** | 未区分推理 Token 与正文 Token | 无法评估"质量/成本"平衡 |
| **延迟指标单一** | 仅统计 TTFT，忽略推理延迟 | 无法反映用户真实等待体验 |
| **错误处理不足** | 遇到 429/400 直接失败 | 大规模测试中断，结果不完整 |

---

## 二、改进目标

1. **公平性**：确保每个模型都在其"最佳配置"下被评估
2. **完整性**：捕获完整的推理过程和最终输出
3. **可量化**：新增推理相关的量化指标
4. **稳定性**：增强错误处理和重试机制
5. **可解释性**：提供详细的评测报告，支持人工复核

---

## 三、具体改进方案

### 3.1 参数标准化层 (ThinkingParams Enhancement)

**目标**：自动适配各平台的思考参数格式

**改进点**：
```python
# core/thinking_params.py 改进

# 1. 新增平台特性表
PLATFORM_FEATURES = {
    "mimo": {
        "thinking_param_location": "top_level",
        "thinking_field": "thinking",
        "thinking_format": {"type": "enabled|disabled"},
        "supports_budget": False,
        "reasoning_output_field": "reasoning_content"
    },
    "deepseek": {
        "thinking_param_location": "extra_body",
        "thinking_field": "thinking",
        "thinking_format": {"type": "enabled|disabled"},
        "supports_budget": False,
        "reasoning_output_field": "reasoning_content"
    },
    "gemini": {
        "thinking_param_location": "generationConfig.thinkingConfig",
        "thinking_field": "thinkingConfig",
        "thinking_format": {"includeThoughts": True, "thinkingBudget": -1},
        "supports_budget": True,
        "reasoning_output_field": "thought"
    },
    "zhipu": {
        "thinking_param_location": "top_level",
        "thinking_field": "thinking",
        "thinking_format": {"type": "enabled|disabled"},
        "supports_budget": False,
        "reasoning_output_field": "reasoning_content"
    },
    "volcano": {
        "thinking_param_location": "extra_body",
        "thinking_field": "thinking",
        "thinking_format": {"type": "enabled|disabled"},
        "supports_effort": True,  # 支持 reasoning.effort
        "effort_levels": ["minimal", "low", "medium", "high"],
        "reasoning_output_field": "reasoning_content"
    },
    "aliyun": {
        "thinking_param_location": "extra_body",
        "thinking_field": "enable_thinking",
        "thinking_format": True,  # 布尔值
        "supports_budget": True,
        "budget_field": "thinking_budget",
        "reasoning_output_field": "reasoning_content"
    }
}

# 2. 新增推理输出字段自动检测
def get_reasoning_field(platform: str) -> str:
    """根据平台返回推理内容的字段名"""
    return PLATFORM_FEATURES.get(platform, {}).get("reasoning_output_field", "reasoning_content")
```

---

### 3.2 响应解析器增强 (Response Parser)

**目标**：统一捕获各平台的推理内容

**新增文件**：`core/response_parser.py`

```python
class UnifiedResponseParser:
    """统一的响应解析器，支持多平台推理内容提取"""
    
    def __init__(self, platform: str):
        self.platform = platform
        self.reasoning_field = get_reasoning_field(platform)
    
    def parse_stream_chunk(self, chunk: dict) -> dict:
        """解析流式响应块，返回标准化结构"""
        result = {
            "content": "",
            "reasoning": "",
            "usage": None,
            "finish_reason": None
        }
        
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            
            # 提取正文内容
            result["content"] = delta.get("content", "") or ""
            
            # 提取推理内容（平台适配）
            if self.platform == "gemini":
                # Gemini 的推理在 parts 的 thought 字段
                parts = delta.get("parts", [])
                for part in parts:
                    result["reasoning"] += part.get("thought", "") or ""
                    result["content"] += part.get("text", "") or ""
            else:
                # 其他平台使用 reasoning_content
                result["reasoning"] = delta.get(self.reasoning_field, "") or ""
            
            result["finish_reason"] = chunk["choices"][0].get("finish_reason")
        
        if "usage" in chunk:
            result["usage"] = chunk["usage"]
        
        return result
```

---

### 3.3 新增评测指标

**目标**：量化推理模型的"思考价值"

| 指标名称 | 计算方式 | 意义 |
|---------|---------|------|
| **TTUT** (Time To User Text) | 从请求发出到第一个正文 Token 到达的时间 | 反映用户真实等待体验 |
| **Reasoning Token Ratio** | 推理 Token / 总 Token | 衡量模型"思考开销" |
| **Reasoning Density** | 推理字符数 / 正文字符数 | 衡量推理过程的详细程度 |
| **Quality/Cost Score** | 质量分 / (推理 Token × 单价) | 评估性价比 |

**实现**：

```python
# core/metrics.py 新增

class ThinkingMetrics:
    """推理模型专用指标"""
    
    def __init__(self):
        self.first_reasoning_time = None
        self.first_content_time = None
        self.reasoning_tokens = 0
        self.content_tokens = 0
    
    def record_reasoning_start(self, timestamp: float):
        if self.first_reasoning_time is None:
            self.first_reasoning_time = timestamp
    
    def record_content_start(self, timestamp: float):
        if self.first_content_time is None:
            self.first_content_time = timestamp
    
    def calculate_ttut(self, request_start: float) -> float:
        """计算 Time To User Text"""
        if self.first_content_time:
            return self.first_content_time - request_start
        return None
    
    def calculate_reasoning_ratio(self) -> float:
        """计算推理 Token 占比"""
        total = self.reasoning_tokens + self.content_tokens
        if total == 0:
            return 0.0
        return self.reasoning_tokens / total
```

---

### 3.4 错误处理与重试机制

**目标**：确保大规模测试的稳定性

```python
# core/retry_handler.py 新增

import asyncio
from typing import Callable, Any

class RetryHandler:
    """指数退避重试处理器"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        retryable_errors: tuple = (429, 500, 502, 503, 504),
        **kwargs
    ) -> Any:
        """带重试的执行器"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_code = getattr(e, 'status_code', None)
                
                if error_code in retryable_errors:
                    delay = self.base_delay * (2 ** attempt)
                    # 对于 429，尝试解析 Retry-After
                    if error_code == 429:
                        retry_after = getattr(e, 'retry_after', None)
                        if retry_after:
                            delay = max(delay, float(retry_after))
                    
                    await asyncio.sleep(delay)
                else:
                    raise
        
        raise last_error
```

---

### 3.5 评测报告增强

**目标**：提供详细的、可复核的评测报告

**报告结构**：

```json
{
  "test_id": "uuid",
  "timestamp": "2025-12-23T02:30:00+08:00",
  "model_config": {
    "platform": "mimo",
    "model_id": "mimo-v2-flash",
    "thinking_enabled": true,
    "thinking_budget": null,
    "reasoning_effort": "high"
  },
  "results": {
    "overall_score": 8.5,
    "latency": {
      "ttft_ms": 450,
      "ttut_ms": 2300,
      "total_ms": 5600
    },
    "tokens": {
      "reasoning_tokens": 1200,
      "content_tokens": 800,
      "total_tokens": 2000,
      "reasoning_ratio": 0.6
    },
    "quality": {
      "accuracy_score": 9.0,
      "reasoning_coherence": 8.0,
      "response_completeness": 8.5
    },
    "cost": {
      "estimated_cost_usd": 0.0024,
      "quality_per_dollar": 3541.67
    }
  },
  "raw_data": {
    "prompt": "...",
    "reasoning_content": "...",
    "final_content": "...",
    "stream_snapshot": [...]
  }
}
```

---

## 四、实施计划

| 阶段 | 任务 | 预计工作量 | 优先级 |
|-----|------|----------|-------|
| **Phase 1** | 完善 `thinking_params.py` 平台特性表 | 2h | P0 |
| **Phase 2** | 实现 `UnifiedResponseParser` | 3h | P0 |
| **Phase 3** | 新增 `ThinkingMetrics` 指标计算 | 2h | P1 |
| **Phase 4** | 实现 `RetryHandler` 重试机制 | 1h | P1 |
| **Phase 5** | 增强评测报告结构与导出 | 3h | P2 |
| **Phase 6** | UI 展示改进（推理内容折叠、指标可视化） | 4h | P2 |

---

## 五、验证标准

改进完成后，需通过以下验证：

1. **参数验证**：对每个支持的平台发送测试请求，确认思考参数被正确应用（通过日志/响应确认）
2. **解析验证**：确保各平台的推理内容都能被正确提取并展示
3. **指标验证**：TTUT、Reasoning Ratio 等指标计算正确
4. **稳定性验证**：模拟 429 错误，验证重试机制有效
5. **报告验证**：导出的 JSON 报告包含所有必要字段

---

## 六、附录：测试结果关键发现摘要

| 平台 | 发现 | 已应用修复 |
|-----|------|----------|
| MiMo | `thinking` 必须是顶级参数，不能放 `extra_body` | ✅ |
| MiMo | Header 必须用 `api-key` 而非 `Authorization` | ✅ |
| MiMo | 必须用 `max_completion_tokens` 而非 `max_tokens` | ✅ |
| DeepSeek | `thinking` 需放在 `extra_body` 内 | ✅ |
| Gemini | REST API 不支持 `thinkingConfig`（v1beta），需用 SDK 或 v1alpha | ⚠️ 部分 |
| Gemini | 免费 Key 限流严重，需加重试 | 待 Phase 4 |
| 智谱 | `thinking` 是顶级参数，与 MiMo 类似 | ✅ |
| 火山 | 支持 `reasoning.effort` 控制思考深度 | ✅ |

---

*Last Updated: 2025-12-23*
