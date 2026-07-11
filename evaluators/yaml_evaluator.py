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
from typing import TYPE_CHECKING, Any

from .base_evaluator import BaseEvaluator, extract_choice_answer, extract_number_answer

if TYPE_CHECKING:
    from core.task_config import TaskConfig as _TaskConfigType

# Lazy import避免循环依赖（运行时在 _import_task_config 中赋值）
# 用 Any 避免运行时引用 TYPE_CHECKING-only 的类型别名
TaskConfig: Any = None
TaskConfigLoader: Any = None
PromptRenderer: Any = None
ResponseFilter: Any = None

_lazy_imported = False


def _import_task_config() -> None:
    """Lazy import task_config 模块"""
    global TaskConfig, TaskConfigLoader, PromptRenderer, ResponseFilter, _lazy_imported
    if not _lazy_imported:
        from core.task_config import PromptRenderer as PR
        from core.task_config import ResponseFilter as RF
        from core.task_config import TaskConfig as TC
        from core.task_config import TaskConfigLoader as TCL

        TaskConfig = TC
        TaskConfigLoader = TCL
        PromptRenderer = PR
        ResponseFilter = RF
        _lazy_imported = True


def _get_loader() -> Any:
    """Return TaskConfigLoader class (non-None) for type narrowing."""
    _import_task_config()
    assert TaskConfigLoader is not None
    return TaskConfigLoader


def _get_prompt_renderer() -> Any:
    """Return PromptRenderer class (non-None) for type narrowing."""
    _import_task_config()
    assert PromptRenderer is not None
    return PromptRenderer


def _get_response_filter() -> Any:
    """Return ResponseFilter class (non-None) for type narrowing."""
    _import_task_config()
    assert ResponseFilter is not None
    return ResponseFilter


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
        num_shots: int | None = None,
        max_samples: int | None = None,
        seed: int = 42,
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
        self.config: _TaskConfigType | None = None

        # LoadConfigure
        loader_cls = _get_loader()
        if config_path and os.path.exists(config_path):
            loader = loader_cls()
            self.config = loader.load_config_file(config_path)
        else:
            loader = loader_cls("task_configs")
            self.config = loader.get_config(task_name)

        if not self.config:
            raise ValueError(f"Not found任务Configure: {task_name}")

        config = self.config

        # 确定 num_shots
        effective_num_shots = num_shots if num_shots is not None else config.num_fewshot

        # 确定Dataset路径
        local_path = config.dataset_local_path or os.path.join(dataset_path, task_name)

        super().__init__(
            dataset_name=task_name,
            dataset_path=local_path,
            num_shots=effective_num_shots,
            max_samples=max_samples,
            seed=seed,
        )

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """LoadDataset"""
        import random

        random.seed(self.seed)
        assert self.config is not None
        config = self.config

        samples: list[dict[str, Any]] = []

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
                    if filepath.endswith(".jsonl"):
                        with open(filepath, encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    samples.append(json.loads(line))
                    else:
                        with open(filepath, encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            samples = data
                        elif isinstance(data, dict) and "data" in data:
                            samples = data["data"]
                    break
                except Exception as e:
                    print(f"Load {filepath} 失败: {e}")

        # if本地没has，尝试从 HuggingFace Load
        if not samples and config.dataset_path:
            try:
                from datasets import load_dataset  # type: ignore[attr-defined]

                ds_path = config.dataset_path
                ds_name = config.dataset_name
                split = config.test_split or "test"

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
            self.few_shot_examples = samples[: self.num_shots]
            samples = samples[self.num_shots :]

        # 限制Sample count
        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[: self.max_samples]

        self.samples = samples
        return samples

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """use YAML 模板Format prompt"""
        renderer = _get_prompt_renderer()
        assert self.config is not None
        config = self.config

        text = renderer.render(config.doc_to_text, sample)

        if include_answer:
            target = renderer.render(config.doc_to_target, sample)
            text += config.target_delimiter + target

        return str(text)

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt"""
        renderer = _get_prompt_renderer()
        assert self.config is not None

        return str(
            renderer.build_prompt(self.config, sample, self.few_shot_examples, include_target=False)
        )

    def parse_response(self, response: str) -> str:
        """Parse响应"""
        assert self.config is not None
        config = self.config

        # ApplyFilter器
        if config.filter_list:
            response_filter = _get_response_filter()
            for filter_spec in config.filter_list:
                result = response_filter.apply_filters(response, [filter_spec])
                if result and result.strip():
                    return str(result)

        # 回退: based on输出类型Parse
        if config.output_type == "multiple_choice":
            choices = config.doc_to_choice or ["A", "B", "C", "D"]
            return extract_choice_answer(response, choices)

        elif config.output_type == "generate_until":
            # 尝试提取数字
            number = extract_number_answer(response)
            if number is not None:
                fnum = float(number)
                return str(int(fnum) if fnum == int(fnum) else number)

            # 尝试提取 boxed
            boxed_match = re.search(r"\\boxed\{([^{}]+)\}", response)
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
            pred_num = float(pred.replace(",", ""))
            corr_num = float(corr.replace(",", ""))
            if abs(pred_num - corr_num) < 1e-6:
                return True
        except Exception:
            pass

        # 选择题匹配
        if self.config is not None and self.config.output_type == "multiple_choice":
            # A and a etc.价
            if pred.upper() == corr.upper():
                return True

        return False

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get样本类别"""
        # 尝试多常见字段
        for field in ["category", "subject", "type", "source"]:
            if field in sample:
                return str(sample[field])
        return ""

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """GetCorrect answer"""
        renderer = _get_prompt_renderer()
        assert self.config is not None

        return str(renderer.render(self.config.doc_to_target, sample))


def create_evaluator_from_yaml(
    task_name: str, config_path: str | None = None, **kwargs
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
    loader_cls = _get_loader()

    loader = loader_cls(config_dir)
    return list(loader.list_tasks())
