# 混合推理模型支持文档

## 概述

系统现在支持不同平台的混合推理模型（Thinking/Reasoning Models），这些模型可以在简单对话和复杂推理之间动态切换模式以优化性能和成本。

## 支持的平台

### 1. 小米 MiMo 平台
- **API 文档**: https://platform.xiaomimimo.com (OpenRouter文档)
- **推理参数** (通过 `extra_body` 传递):
  - `thinking`: 对象格式 `{"type": "enabled" | "disabled"}`
  - 注意：不支持 `thinking_budget` 参数
- **使用方式**: `extra_body={"thinking": {"type": "enabled"}}`
- **推理模型**: `mimo-v2-flash`
- **响应字段**: 推理内容在 `reasoning_details` 数组中返回

### 2. 硅基流动 (SiliconFlow)
- **API 文档**: https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
- **推理参数**:
  - `enable_thinking`: bool - 是否启用思考模式（默认 True）
  - `thinking_budget`: int - 推理token预算（128-32768，默认 4096）
- **推理模型**: 
  - `zai-org/GLM-4.6`
  - `Qwen/Qwen3-*` 系列
  - `tencent/Hunyuan-A13B-Instruct`
  - `deepseek-ai/DeepSeek-V3.1-Terminus`

### 3. 火山引擎 (Volcano Engine / 豆包)
- **API 文档**: https://www.volcengine.com/docs/82379/1956279
- **推理参数** (通过 `extra_body` 传递):
  - `thinking`: 对象格式 `{"type": "enabled" | "disabled" | "auto"}`
  - `reasoning.effort`: 思考深度 `"minimal" | "low" | "medium" | "high"` (可选)
- **使用方式**: `extra_body={"thinking": {"type": "enabled"}, "reasoning": {"effort": "medium"}}`
- **推理模型**: `doubao-seed-*`, `deepseek-v3-*` 系列
- **注意**:
  - 仅部分模型支持 `reasoning.effort`（如 `doubao-seed-1-6-251015`）
  - `thinking.type="disabled"` 时，`reasoning.effort` 仅支持 `"minimal"`

### 4. 阿里百炼 (Aliyun DashScope)
- **API 文档**: https://help.aliyun.com/zh/model-studio/deep-thinking
- **推理参数** (通过 `extra_body` 传递):
  - `enable_thinking`: bool - 是否启用思考模式
  - `thinking_budget`: int - 推理过程最大token数
- **使用方式**: `extra_body={"enable_thinking": True, "thinking_budget": 4096}`
- **推理模型**:
  - **Qwen3**: `qwen3-*`, `qwen-plus-*`, `qwen-turbo-*`, `qwen-flash-*` 系列
  - **QwQ**: `qwq-plus`, `qwq-32b`
  - **DeepSeek**: `deepseek-v3.*`, `deepseek-r1*`
  - **GLM**: `glm-4.6`, `glm-4.5`, `glm-4.5-air`
  - **Kimi**: `kimi-k2-thinking`
- **响应字段**: 推理内容在 `reasoning_content` 字段中返回

### 5. MiniMax
- **API 文档**: https://platform.minimaxi.com/docs/guides/text-m2-function-call
- **推理参数**:
  - `reasoning_split`: bool - 是否启用交错思考模式（Interleaved Thinking）
  - 注意：此参数需要通过 `extra_body` 传递
- **推理模型**: `MiniMax-M2`
- **特点**: 
  - 使用 Interleaved Thinking（交错思维），遵循"计划→行动→反思"循环
  - 思考内容在 `reasoning_details` 字段中返回
  - 不支持 `thinking_budget` 参数

### 6. OpenAI
- **推理参数**:
  - `reasoning_effort`: str - 推理强度 ("low", "medium", "high")
- **推理模型**: `o1-*`, `o3-*` 系列

## 使用方法

### 在质量测试中配置

在 UI 的"模型质量测试"面板中:

1. 选择**模型类型**为 "思考模型 (Reasoning/CoT)"
2. 系统会自动显示推理配置选项:
   - **思考预算** (thinking_budget): 最大思考token数
   - **思考强度** (reasoning_effort): low/medium/high

### 示例配置

#### 阿里百炼 - Qwen3:
```python
config = QualityTestConfig(
    model_type="thinking",
    thinking_enabled=True,      # 启用思考模式
    thinking_budget=8192,       # 允许最多8192个思考token
    reasoning_effort="medium"   # 会被转换为 thinking 参数
)
```

#### 硅基流动 - DeepSeek:
```python
config = QualityTestConfig(
    model_type="thinking",
    thinking_enabled=False,     # 关闭思考模式（用于简单问题）
    thinking_budget=0,
    reasoning_effort="low"
)
```

#### OpenAI - O1:
```python
config = QualityTestConfig(
    model_type="thinking",
    thinking_enabled=True,      # 会被忽略
    thinking_budget=0,          # 会被忽略
    reasoning_effort="high"     # 使用 reasoning_effort
)
```

#### MiniMax - M2:
```python
config = QualityTestConfig(
    model_type="thinking",
    thinking_enabled=True,      # 会被转换为 reasoning_split=True
    thinking_budget=0,          # MiniMax不支持此参数
    reasoning_effort="medium"
)
```

## 自动平台检测

系统会根据 `api_base_url` 和 `model_id` 自动检测平台并使用正确的参数格式:

```python
# 小米平台
api_base_url = "https://api.xiaomimimo.com/v1"
model_id = "mimo-v2-flash"
# → 自动使用: extra_body={"thinking": {"type": "enabled"}}

# 火山引擎
api_base_url = "https://ark.cn-beijing.volces.com/api/v3"
model_id = "doubao-seed-1-6-251015"
# → 自动使用: extra_body={"thinking": {"type": "enabled"}, "reasoning": {"effort": "medium"}}

# 硅基流动
api_base_url = "https://api.siliconflow.cn/v1"
model_id = "Qwen/Qwen3-235b-a22b"
# → 自动使用: enable_thinking, thinking_budget

# 阿里云
api_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model_id = "qwen3-max-preview"
# → 自动使用: extra_body={"enable_thinking": True, "thinking_budget": 4096}

# MiniMax
api_base_url = "https://api.minimaxi.com/v1"
model_id = "MiniMax-M2"
# → 自动使用: extra_body={"reasoning_split": True}

# OpenAI
api_base_url = "https://api.openai.com/v1"
model_id = "o1-mini"
# → 自动使用: reasoning_effort
```

## 注意事项

1. **成本考虑**: 启用思考模式会增加token消耗和API成本
2. **延迟影响**: 思考过程会增加响应时间
3. **场景选择**:
   - 简单对话/问答 → `enable_thinking=False`
   - 复杂推理/代码/数学 → `enable_thinking=True`
4. **默认配置** (已设置为最高性能):
   - 思考预算: **32768 tokens** (最大值)
   - 思考强度: **high** (最高档位)
   - 如需降低成本，可手动调整为较低配置

## 技术实现

核心文件:
- `core/thinking_params.py`: 参数管理和平台检测
- `core/providers/openai.py`: 参数应用和API调用
- `evaluators/quality_evaluator.py`: 配置传递

## 版本兼容性

- 向后兼容: 不使用推理模型时，参数会被忽略
- 新平台扩展: 在 `thinking_params.py` 中添加新平台规则即可
