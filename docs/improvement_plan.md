# 质量测试系统改进计划

## 执行摘要

本文档基于与 `lm-evaluation-harness`、`OpenCompass` 等主流评估框架的对比，
列出本项目质量测试系统的改进方向和具体实施建议。

---

## 🔴 高优先级改进

### 1. 答案解析增强 (Critical)

**现状问题:**
- 当前主要依赖简单正则匹配 (`####` 模式, `\boxed{}` 等)
- 对于复杂数学表达式、代码输出等支持不足
- 误判率较高，依赖 AI Judge 二次确认

**改进方案:**
```python
# 建议实现多层答案解析器
class MultiStageAnswerParser:
    def parse(self, response: str, answer_type: str) -> str:
        # 1. 结构化提取 (boxed, ####, 等)
        result = self._extract_structured(response)
        if result: return result
        
        # 2. 语义理解 (最后一句话中的数字/选项)
        result = self._extract_semantic(response)
        if result: return result
        
        # 3. LLM 辅助提取 (作为回退)
        return self._llm_extract(response)
```

**参考实现:** lm-eval 的 filter chain 机制

---

### 2. 评估指标多样化 (Important)

**现状问题:**
- 主要使用 `exact_match` (准确率)
- 缺少 F1、BLEU、ROUGE 等其他指标
- 没有置信区间/统计显著性测试

**改进方案:**
```yaml
# 在 YAML 配置中支持多指标
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
  - metric: f1_score
    aggregation: mean
  - metric: pass@k
    k: [1, 5, 10]  # 代码生成专用
  - metric: code_execution
    sandbox: true
```

**建议新增指标:**
- `pass@k` - 代码生成评估
- `F1/BLEU/ROUGE` - 开放式问答
- `Bootstrap CI` - 置信区间估计
- `McNemar Test` - 模型对比显著性

---

### 3. 数据集自动下载 (High Priority)

**现状问题:**
- 需要用户手动下载数据集
- 缺少自动版本管理
- HuggingFace 数据集支持有限

**改进方案:**
```python
# 自动数据集管理器
class DatasetManager:
    def ensure_dataset(self, name: str, version: str = "latest"):
        """确保数据集可用，自动下载如果不存在"""
        local_path = self.get_local_path(name)
        if not self.is_valid(local_path, version):
            self.download(name, version)
        return local_path
    
    def download(self, name: str, version: str):
        if name in HF_DATASETS:
            # 从 HuggingFace 下载
            from datasets import load_dataset
            ds = load_dataset(name)
            self.save_local(ds, name)
        elif name in CUSTOM_SOURCES:
            # 从自定义源下载
            self.download_from_url(name)
```

---

### 4. Few-shot 模板标准化 (Medium)

**现状问题:**
- 每个 Evaluator 硬编码 few-shot 格式
- 缺少系统消息、角色消息的支持
- 不支持 Chat 模板 vs Completion 模板切换

**改进方案:**
```yaml
# 支持多种 prompt 格式
prompt_format:
  type: chat  # chat | completion
  system_message: "You are a helpful math assistant."
  
  fewshot_template: |
    Question: {{question}}
    Answer: Let me solve this step by step.
    {{solution}}
    #### {{answer}}
    
  doc_to_text: |
    Question: {{question}}
    Answer:

# 或者使用 lm-eval 的 chat template 支持
use_chat_template: true
fewshot_as_multiturn: true
```

---

## 🟡 中等优先级改进

### 5. 结果可复现性 (Medium)

**改进项:**
- 固定随机种子链
- 记录完整环境配置
- 样本级别结果保存

```python
# 增强的结果元数据
class EvaluationMetadata:
    seed: int
    python_version: str
    library_versions: Dict[str, str]
    prompt_hash: str  # 确保 prompt 完全一致
    sample_order: List[str]  # 样本顺序
```

---

### 6. 并行评估优化 (Medium)

**现状问题:**
- 单数据集并发控制
- 缺少批量推理支持
- 没有请求去重/缓存

**改进方案:**
```python
# 添加结果缓存
class ResponseCache:
    def __init__(self, cache_dir: str):
        self.cache = diskcache.Cache(cache_dir)
    
    def get_or_compute(self, prompt_hash: str, compute_fn):
        if prompt_hash in self.cache:
            return self.cache[prompt_hash]
        result = compute_fn()
        self.cache[prompt_hash] = result
        return result
```

---

### 7. 报告生成增强 (Medium)

**改进项:**
- 生成标准化的 JSON 报告格式
- 支持与其他框架结果对比
- 添加详细的失败案例分析

```python
class StandardReport:
    def export_lm_eval_format(self):
        """导出 lm-evaluation-harness 兼容格式"""
        return {
            "results": {
                dataset: {
                    "acc": accuracy,
                    "acc_stderr": stderr,
                    "alias": alias
                }
            },
            "config": {...},
            "git_hash": get_git_hash()
        }
```

---

## 🟢 低优先级改进

### 8. 多语言支持 (Low)

- 添加 C-Eval、CMMLU 等中文数据集完整支持
- 国际化 UI 界面

### 9. 分布式评估 (Low)

- 支持多 GPU/多机器并行
- 任务队列管理

### 10. A/B 对比测试 (Low)

- 支持同时评估多个模型
- 自动生成对比报告

---

## 📋 实施计划

### Phase 1 (1-2 周) ✅ 完成
- [x] 实现增强答案解析器 ✅ `core/enhanced_parser.py`
- [x] 添加数据集自动下载 ✅ `core/dataset_manager.py`
- [x] 改进 prompt 模板系统 ✅ `core/prompt_template.py`

### Phase 2 (2-4 周) ✅ 完成
- [x] 添加更多评估指标 ✅ `core/metrics.py` (扩展)
- [x] 实现结果缓存 ✅ `core/response_cache.py`
- [x] 标准化报告格式 ✅ `core/standard_report.py`

### Phase 3 (4+ 周)
- [x] A/B 对比功能 ✅ `core/model_comparator.py`
- [ ] 分布式评估支持
- [ ] 完整国际化

### 已完成的改进

| 日期 | 改进项 | 文件 | 描述 |
|------|--------|------|------|
| 2024-12-29 | 增强答案解析器 | `core/enhanced_parser.py` | 多层解析策略、LLM 回退、置信度评估 |
| 2024-12-29 | 响应缓存系统 | `core/response_cache.py` | SQLite 缓存、断点续评、自动过期 |
| 2024-12-29 | Prompt 模板系统 | `core/prompt_template.py` | Jinja2 模板、Chat/Completion 格式、模板库 |
| 2024-12-29 | 评估指标多样化 | `core/metrics.py` | F1/BLEU/ROUGE、pass@k、置信区间、Wilson 分数 |
| 2024-12-29 | 数据集自动下载 | `core/dataset_manager.py` | HuggingFace 下载、版本管理、本地缓存 |
| 2024-12-29 | 标准化报告格式 | `core/standard_report.py` | lm-eval/OpenCompass 兼容、Markdown/CSV 导出 |
| 2024-12-29 | A/B 对比功能 | `core/model_comparator.py` | 多模型对比、McNemar 检验、差异分析 |

---

## 参考资源

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [OpenCompass](https://github.com/open-compass/opencompass)
- [Eval-Scope (阿里)](https://github.com/modelscope/eval-scope)
- [HELM (Stanford)](https://crfm.stanford.edu/helm/)
