"""
YAML 任务ConfigureLoad器
借鉴 lm-evaluation-harness  YAML 任务定义方式

支持via YAML 文件定义Evaluation任务，no需编写 Python 代码i.e.可Add新任务。
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template


@dataclass
class TaskConfig:
    """任务ConfigureData类"""

    # 基本信息
    task: str                                    # 任务名称
    task_alias: str | None = None             # 任务别名 (Display用)
    tag: list[str] | None = None              # 任务Label
    description: str = ""                        # 任务描述

    # Data集Configure
    dataset_path: str = ""                       # HuggingFace Dataset路径
    dataset_name: str | None = None           # Data集子集名称
    dataset_local_path: str | None = None     # 本地Dataset路径
    test_split: str = "test"                     # Test集 split
    training_split: str | None = None         # 训练集 split
    fewshot_split: str | None = None          # few-shot 示例 split

    # Prompt Configure
    doc_to_text: str = ""                        # 输入模板 (Jinja2)
    doc_to_target: str = ""                      # 目标Answer模板
    doc_to_choice: list[str] | None = None    # Options列表 (选择题)
    system_prompt: str | None = None          # 系统Tip
    fewshot_delimiter: str = "\n\n"              # few-shot 分隔符
    target_delimiter: str = " "                  # 输入与目标之间分隔符

    # 评估Configure
    output_type: str = "generate_until"          # generate_until / multiple_choice
    num_fewshot: int = 0                         # few-shot 数量
    metric_list: list[dict] = field(default_factory=list)  # Score指标

    # GenerateConfigure
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    # Filter器 (Postprocess)
    filter_list: list[dict] = field(default_factory=list)

    # 元信息
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskConfigLoader:
    """YAML 任务ConfigureLoad器"""

    def __init__(self, config_dir: str = "task_configs"):
        """
        InitializeLoad器

        Args:
            config_dir: 任务Configure目录
        """
        self.config_dir = config_dir
        self.configs: dict[str, TaskConfig] = {}
        self._load_builtin_configs()

    def _load_builtin_configs(self):
        """Load内置Configure"""
        builtin_dir = os.path.join(os.path.dirname(__file__), "..", "task_configs")
        if os.path.exists(builtin_dir):
            self._load_from_directory(builtin_dir)

        # 同时Load用户Custom Configuration
        if os.path.exists(self.config_dir):
            self._load_from_directory(self.config_dir)

    def _load_from_directory(self, directory: str):
        """从目录Load所has YAML Configure"""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.yaml', '.yml')):
                    filepath = os.path.join(root, file)
                    try:
                        config = self.load_config_file(filepath)
                        if config and config.task:
                            self.configs[config.task] = config
                    except Exception as e:
                        print(f"Load Config {filepath} 失败: {e}")

    def load_config_file(self, filepath: str) -> TaskConfig | None:
        """
        Load单 YAML Configure文件

        Args:
            filepath: ConfigureFile path

        Returns:
            TaskConfig 对象
        """
        with open(filepath, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        # Convertis TaskConfig
        config = TaskConfig(
            task=data.get('task', os.path.splitext(os.path.basename(filepath))[0]),
            task_alias=data.get('task_alias'),
            tag=data.get('tag', []) if isinstance(data.get('tag'), list) else [data.get('tag')] if data.get('tag') else None,
            description=data.get('description', ''),
            dataset_path=data.get('dataset_path', ''),
            dataset_name=data.get('dataset_name'),
            dataset_local_path=data.get('dataset_local_path'),
            test_split=data.get('test_split', 'test'),
            training_split=data.get('training_split'),
            fewshot_split=data.get('fewshot_split'),
            doc_to_text=data.get('doc_to_text', ''),
            doc_to_target=data.get('doc_to_target', ''),
            doc_to_choice=data.get('doc_to_choice'),
            system_prompt=data.get('system_prompt'),
            fewshot_delimiter=data.get('fewshot_delimiter', '\n\n'),
            target_delimiter=data.get('target_delimiter', ' '),
            output_type=data.get('output_type', 'generate_until'),
            num_fewshot=data.get('num_fewshot', 0),
            metric_list=data.get('metric_list', [{'metric': 'accuracy', 'aggregation': 'mean'}]),
            generation_kwargs=data.get('generation_kwargs', {}),
            filter_list=data.get('filter_list', []),
            metadata=data.get('metadata', {})
        )

        return config

    def get_config(self, task_name: str) -> TaskConfig | None:
        """Get任务Configure"""
        return self.configs.get(task_name)

    def list_tasks(self) -> list[str]:
        """列出所has已Load任务"""
        return list(self.configs.keys())

    def list_tasks_by_tag(self, tag: str) -> list[str]:
        """列出指定Label所has任务"""
        return [
            name for name, config in self.configs.items()
            if config.tag and tag in config.tag
        ]


class PromptRenderer:
    """Prompt Render器 - use Jinja2 模板"""

    @staticmethod
    def render(template: str, doc: dict[str, Any]) -> str:
        """
        Render Jinja2 模板

        Args:
            template: Jinja2 模板字符串
            doc: 文档Data

        Returns:
            Render后字符串
        """
        if not template:
            return ""

        try:
            jinja_template = Template(template)
            return jinja_template.render(**doc)
        except Exception:
            # 回退到简单字符串替换
            result = template
            for key, value in doc.items():
                result = result.replace(f"{{{{{key}}}}}", str(value))
            return result

    @staticmethod
    def build_prompt(
        config: TaskConfig,
        doc: dict[str, Any],
        few_shot_docs: list[dict] | None = None,
        include_target: bool = False
    ) -> str:
        """
        Build完整 prompt

        Args:
            config: 任务Configure
            doc: 当前文档
            few_shot_docs: few-shot 示例文档列表
            include_target: is否包含目标Answer

        Returns:
            完整 prompt
        """
        parts = []

        # 1. 系统Tip / 描述
        if config.system_prompt:
            parts.append(PromptRenderer.render(config.system_prompt, doc))
        elif config.description:
            parts.append(PromptRenderer.render(config.description, doc))

        # 2. Few-shot 示例
        if few_shot_docs and config.num_fewshot > 0:
            for example in few_shot_docs[:config.num_fewshot]:
                example_text = PromptRenderer.render(config.doc_to_text, example)
                example_target = PromptRenderer.render(config.doc_to_target, example)
                parts.append(f"{example_text}{config.target_delimiter}{example_target}")

        # 3. 当前问题
        current_text = PromptRenderer.render(config.doc_to_text, doc)
        if include_target:
            current_target = PromptRenderer.render(config.doc_to_target, doc)
            parts.append(f"{current_text}{config.target_delimiter}{current_target}")
        else:
            parts.append(current_text)

        return config.fewshot_delimiter.join(parts)


class ResponseFilter:
    """响应Filter器 - PostprocessModel输出"""

    @staticmethod
    def apply_filters(response: str, filter_list: list[dict]) -> str:
        """
        ApplyFilter器链

        Args:
            response: 原始响应
            filter_list: Filter器Configure列表

        Returns:
            Filter后响应
        """
        result = response

        for filter_spec in filter_list:
            if not isinstance(filter_spec, dict):
                continue

            filters = filter_spec.get('filter', [])
            for f in filters:
                func = f.get('function', '')

                if func == 'regex':
                    pattern = f.get('regex_pattern', '')
                    group_select = f.get('group_select', 0)
                    if pattern:
                        match = re.search(pattern, result)
                        if match:
                            try:
                                result = match.group(group_select if group_select >= 0 else len(match.groups()) + group_select + 1)
                            except:
                                result = match.group(0)

                elif func == 'take_first':
                    # 取一Result (用于多次匹配)
                    pass

                elif func == 'remove_whitespace':
                    result = result.strip()

                elif func == 'lowercase':
                    result = result.lower()

                elif func == 'uppercase':
                    result = result.upper()

                elif func == 'strip_until':
                    until = f.get('until', [])
                    for u in until:
                        if u in result:
                            result = result.split(u)[0]

        return result.strip()


class MetricCalculator:
    """ScoreMetric calculation器"""

    @staticmethod
    def calculate(
        predictions: list[str],
        references: list[str],
        metric_list: list[dict]
    ) -> dict[str, float]:
        """
        CalculateScore指标

        Args:
            predictions: 预测列表
            references: 参考Answer列表
            metric_list: 指标Configure

        Returns:
            指标Result字典
        """
        results = {}

        for metric_spec in metric_list:
            metric_name = metric_spec.get('metric', 'accuracy')
            metric_spec.get('aggregation', 'mean')
            metric_spec.get('higher_is_better', True)
            ignore_case = metric_spec.get('ignore_case', False)
            ignore_punctuation = metric_spec.get('ignore_punctuation', False)

            # Preprocess
            preds = predictions.copy()
            refs = references.copy()

            if ignore_case:
                preds = [p.lower() for p in preds]
                refs = [r.lower() for r in refs]

            if ignore_punctuation:
                import string
                preds = [p.translate(str.maketrans('', '', string.punctuation)) for p in preds]
                refs = [r.translate(str.maketrans('', '', string.punctuation)) for r in refs]

            # Calculate
            if metric_name in ['accuracy', 'acc', 'exact_match']:
                correct = sum(1 for p, r in zip(preds, refs, strict=False) if p.strip() == r.strip())
                value = correct / len(preds) if preds else 0.0

            elif metric_name == 'contains':
                correct = sum(1 for p, r in zip(preds, refs, strict=False) if r.strip() in p)
                value = correct / len(preds) if preds else 0.0

            elif metric_name == 'f1':
                # 简化 F1 Calculate
                from collections import Counter
                scores = []
                for p, r in zip(preds, refs, strict=False):
                    p_tokens = p.split()
                    r_tokens = r.split()
                    common = Counter(p_tokens) & Counter(r_tokens)
                    num_common = sum(common.values())
                    if num_common == 0:
                        scores.append(0.0)
                    else:
                        precision = num_common / len(p_tokens) if p_tokens else 0
                        recall = num_common / len(r_tokens) if r_tokens else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        scores.append(f1)
                value = sum(scores) / len(scores) if scores else 0.0

            else:
                value = 0.0

            results[metric_name] = value

        return results


# Export
__all__ = [
    'TaskConfig',
    'TaskConfigLoader',
    'PromptRenderer',
    'ResponseFilter',
    'MetricCalculator'
]
