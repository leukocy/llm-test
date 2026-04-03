"""
YAML 驱动通用Evaluator
借鉴 lm-evaluation-harness 设计，via YAML Configure自动Create评估任务

use方式:
1. in task_configs/ 目录underCreate YAML Configure文件
2. use YAMLEvaluator Load并运行Evaluation
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_choice_answer, extract_number_answer

# Lazy import避免循环依赖
TaskConfig = None
TaskConfigLoader = None
PromptRenderer = None
ResponseFilter = None


def _import_task_config():
    """Lazy import task_config 模块"""
    global TaskConfig, TaskConfigLoader, PromptRenderer, ResponseFilter
    if TaskConfig is None:
        from core.task_config import PromptRenderer as PR
        from core.task_config import ResponseFilter as RF
        from core.task_config import TaskConfig as TC
        from core.task_config import TaskConfigLoader as TCL
        TaskConfig = TC
        TaskConfigLoader = TCL
        PromptRenderer = PR
        ResponseFilter = RF


class YAMLEvaluator(BaseEvaluator):
    """
    YAML 驱动通用Evaluator

    based on YAML Configure文件自动:
    - LoadDataset
    - Build prompt
    - Parse响应
    - Score
    """

    def __init__(
        self,
        task_name: str,
        config_path: str | None = None,
        dataset_path: str = "datasets",
        num_shots: int = None,
        max_samples: int | None = None,
        seed: int = 42
    ):
        """
        Initialize YAML Evaluator

        Args:
            task_name: 任务名称 (对应 YAML 文件in task 字段)
            config_path: ConfigureFile path (optional，default从 task_configs 目录查找)
            dataset_path: Dataset根目录
            num_shots: few-shot 数量 (覆盖 YAML Configure)
            max_samples: 最大Sample count
            seed: Random Seed
        """
        _import_task_config()

        self.task_name = task_name
        self.config: TaskConfig = None

        # LoadConfigure
        if config_path and os.path.exists(config_path):
            loader = TaskConfigLoader()
            self.config = loader.load_config_file(config_path)
        else:
            loader = TaskConfigLoader("task_configs")
            self.config = loader.get_config(task_name)

        if not self.config:
            raise ValueError(f"Not found任务Configure: {task_name}")

        # 确定 num_shots
        effective_num_shots = num_shots if num_shots is not None else self.config.num_fewshot

        # 确定Dataset路径
        local_path = self.config.dataset_local_path or os.path.join(dataset_path, task_name)

        super().__init__(
            dataset_name=task_name,
            dataset_path=local_path,
            num_shots=effective_num_shots,
            max_samples=max_samples,
            seed=seed
        )

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """LoadDataset"""
        import random
        random.seed(self.seed)

        samples = []

        # 尝试从本地Load
        possible_files = [
            os.path.join(self.dataset_path, f"{self.task_name}.json"),
            os.path.join(self.dataset_path, "test.json"),
            os.path.join(self.dataset_path, "data.json"),
            os.path.join(self.dataset_path, f"{self.task_name}.jsonl"),
            os.path.join(self.dataset_path, "test.jsonl"),
        ]

        for filepath in possible_files:
            if os.path.exists(filepath):
                try:
                    if filepath.endswith('.jsonl'):
                        with open(filepath, encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    samples.append(json.loads(line))
                    else:
                        with open(filepath, encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            samples = data
                        elif isinstance(data, dict) and 'data' in data:
                            samples = data['data']
                    break
                except Exception as e:
                    print(f"Load {filepath} 失败: {e}")

        # if本地没has，尝试从 HuggingFace Load
        if not samples and self.config.dataset_path:
            try:
                from datasets import load_dataset
                ds_path = self.config.dataset_path
                ds_name = self.config.dataset_name
                split = self.config.test_split or "test"

                if ds_name:
                    ds = load_dataset(ds_path, ds_name, split=split)
                else:
                    ds = load_dataset(ds_path, split=split)

                samples = [dict(item) for item in ds]
                print(f"从 HuggingFace Load {len(samples)}  samples")
            except Exception as e:
                print(f"从 HuggingFace Load失败: {e}")

        if not samples:
            print(f"Warning: Not foundDataset {self.task_name}")
            return []

        # 随机打乱
        random.shuffle(samples)

        # Process few-shot
        if self.num_shots > 0 and len(samples) > self.num_shots:
            self.few_shot_examples = samples[:self.num_shots]
            samples = samples[self.num_shots:]

        # 限制Sample count
        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """use YAML 模板Format prompt"""
        _import_task_config()

        text = PromptRenderer.render(self.config.doc_to_text, sample)

        if include_answer:
            target = PromptRenderer.render(self.config.doc_to_target, sample)
            text += self.config.target_delimiter + target

        return text

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt"""
        _import_task_config()

        return PromptRenderer.build_prompt(
            self.config,
            sample,
            self.few_shot_examples,
            include_target=False
        )

    def parse_response(self, response: str) -> str:
        """Parse响应"""
        _import_task_config()

        # ApplyFilter器
        if self.config.filter_list:
            for filter_spec in self.config.filter_list:
                result = ResponseFilter.apply_filters(response, [filter_spec])
                if result and result.strip():
                    return result

        # 回退: based on输出类型Parse
        if self.config.output_type == "multiple_choice":
            choices = self.config.doc_to_choice or ['A', 'B', 'C', 'D']
            return extract_choice_answer(response, choices)

        elif self.config.output_type == "generate_until":
            # 尝试提取数字
            number = extract_number_answer(response)
            if number is not None:
                return str(int(number) if number == int(number) else number)

            # 尝试提取 boxed
            boxed_match = re.search(r'\\boxed\{([^{}]+)\}', response)
            if boxed_match:
                return boxed_match.group(1).strip()

        return response.strip()

    def check_answer(self, predicted: str, correct: str) -> bool:
        """CheckAnswer"""
        if not predicted or not correct:
            return False

        # 规范化
        pred = predicted.strip().lower()
        corr = correct.strip().lower()

        # 直接匹配
        if pred == corr:
            return True

        # 数值匹配
        try:
            pred_num = float(pred.replace(',', ''))
            corr_num = float(corr.replace(',', ''))
            if abs(pred_num - corr_num) < 1e-6:
                return True
        except:
            pass

        # 选择题匹配
        if self.config.output_type == "multiple_choice":
            # A and a etc.价
            if pred.upper() == corr.upper():
                return True

        return False

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get样本类别"""
        # 尝试多常见字段
        for field in ['category', 'subject', 'type', 'source']:
            if field in sample:
                return str(sample[field])
        return ""

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """GetCorrect answer"""
        _import_task_config()

        return PromptRenderer.render(self.config.doc_to_target, sample)


def create_evaluator_from_yaml(
    task_name: str,
    config_path: str | None = None,
    **kwargs
) -> YAMLEvaluator:
    """
    从 YAML ConfigureCreateEvaluatorFactory函数

    Args:
        task_name: 任务名称
        config_path: ConfigureFile path
        **kwargs: other参数传递给 YAMLEvaluator

    Returns:
        YAMLEvaluator 实例
    """
    return YAMLEvaluator(task_name, config_path, **kwargs)


def list_yaml_tasks(config_dir: str = "task_configs") -> list[str]:
    """列出所has可用 YAML 任务"""
    _import_task_config()

    loader = TaskConfigLoader(config_dir)
    return loader.list_tasks()
