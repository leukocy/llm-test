"""
Quality Evaluators Package

Provides evaluation capabilities for various benchmark datasets.
Evaluators are auto-discovered from this directory using the @register_evaluator decorator.

Usage:
    from evaluators import get_evaluator, list_available_datasets
    
    evaluator_cls = get_evaluator("mmlu")
    evaluator = evaluator_cls(dataset_path="...", num_shots=5)
    results = await evaluator.evaluate_batch(samples, get_response)
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Optional, Type, Dict

from .base_evaluator import (
    BaseEvaluator,
    DatasetType,
    EvaluationResult,
    SampleResult,
    extract_choice_answer,
    extract_number_answer,
    normalize_text,
)

# === Evaluator Registry ===

_EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}


def register_evaluator(name: str):
    """
    Decorator to register an evaluator class.
    
    Usage:
        @register_evaluator("mmlu")
        class MMLUEvaluator(BaseEvaluator):
            ...
    """
    def decorator(cls):
        _EVALUATOR_REGISTRY[name] = cls
        cls._registry_name = name
        return cls
    return decorator


def get_evaluator(dataset_name: str) -> Optional[Type[BaseEvaluator]]:
    """Get evaluator class by dataset name."""
    return _EVALUATOR_REGISTRY.get(dataset_name.lower())


def list_available_datasets() -> list[str]:
    """List all registered dataset evaluator names."""
    return sorted(_EVALUATOR_REGISTRY.keys())


def list_evaluators() -> Dict[str, Dict]:
    """List all evaluators with metadata."""
    result = {}
    for name, cls in _EVALUATOR_REGISTRY.items():
        result[name] = {
            "name": name,
            "class": cls.__name__,
            "module": cls.__module__,
            "doc": (cls.__doc__ or "").strip().split("\n")[0],
        }
    return result


# === Auto-discover all evaluator modules in this directory ===
_package_dir = Path(__file__).parent
for _importer, _module_name, _is_pkg in pkgutil.iter_modules([str(_package_dir)]):
    if _module_name not in ("base_evaluator", "__init__"):
        try:
            importlib.import_module(f".{_module_name}", package=__name__)
        except ImportError as e:
            # Gracefully skip evaluators with missing dependencies
            pass


# === Legacy compatibility: manually register evaluators that don't use @register_evaluator yet ===
def _legacy_register():
    """Register evaluators that haven't been updated with the decorator yet."""
    legacy_mappings = {
        'mmlu': 'mmlu_evaluator.MMLUEvaluator',
        'gsm8k': 'gsm8k_evaluator.GSM8KEvaluator', 
        'math500': 'math500_evaluator.MATH500Evaluator',
        'humaneval': 'humaneval_evaluator.HumanEvalEvaluator',
        'gpqa': 'gpqa_evaluator.GPQAEvaluator',
        'arc': 'arc_evaluator.ARCEvaluator',
        'truthfulqa': 'truthfulqa_evaluator.TruthfulQAEvaluator',
        'hellaswag': 'hellaswag_evaluator.HellaSwagEvaluator',
        'winogrande': 'winogrande_evaluator.WinoGrandeEvaluator',
        'mbpp': 'mbpp_evaluator.MBPPEvaluator',
        'longbench': 'longbench_evaluator.LongBenchEvaluator',
        'swebench_lite': 'swebench_evaluator.SWEBenchLiteEvaluator',
        'needle_haystack': 'needle_haystack_evaluator.NeedleHaystackEvaluator',
        'custom_needle': 'custom_needle_evaluator.CustomNeedleEvaluator',
        'aime2025': 'aime_evaluator.AIME2025Evaluator',
        'arena_hard': 'arena_hard_evaluator.ArenaHardEvaluator',
        'global_piqa': 'global_piqa_evaluator.GlobalPIQAEvaluator',
    }
    
    for name, class_path in legacy_mappings.items():
        if name not in _EVALUATOR_REGISTRY:
            module_name, class_name = class_path.rsplit('.', 1)
            try:
                mod = importlib.import_module(f".{module_name}", package=__name__)
                cls = getattr(mod, class_name)
                _EVALUATOR_REGISTRY[name] = cls
            except (ImportError, AttributeError):
                pass


_legacy_register()


# === YAML Evaluator support ===

def list_yaml_tasks():
    """List available YAML task configurations."""
    try:
        from core.task_config import TaskConfigLoader
        loader = TaskConfigLoader("task_configs")
        return loader.list_tasks()
    except Exception:
        return []


def get_yaml_evaluator(task_name: str, **kwargs):
    """Get a YAML-driven evaluator instance."""
    try:
        from .yaml_evaluator import YAMLEvaluator
        return YAMLEvaluator(task_name, **kwargs)
    except Exception as e:
        print(f"Failed to create YAML Evaluator: {e}")
        return None


__all__ = [
    'BaseEvaluator',
    'EvaluationResult',
    'SampleResult',
    'DatasetType',
    'register_evaluator',
    'get_evaluator',
    'list_available_datasets',
    'list_evaluators',
    'extract_choice_answer',
    'extract_number_answer',
    'normalize_text',
    'list_yaml_tasks',
    'get_yaml_evaluator',
]
