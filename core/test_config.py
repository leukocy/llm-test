"""
YAML Test Configuration系统 (Test Configuration System)

支持via YAML 文件定义可复现Test Configuration。
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model Configuration"""
    platform: str
    model_id: str
    api_base_url: str = ""
    api_key_env: str = ""  # 环境变量名
    thinking_enabled: bool = False
    thinking_budget: int | None = None
    reasoning_effort: str = "medium"
    temperature: float = 0.7
    max_tokens: int = 2048

    def get_api_key(self) -> str:
        """从环境变量Get API Key"""
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""


@dataclass
class DatasetConfig:
    """DatasetConfigure"""
    name: str
    path: str = ""
    subset: str = ""
    samples: int = 0  # 0 = use全部
    seed: int = 42
    num_shots: int = 0


@dataclass
class MetricsConfig:
    """指标Configure"""
    accuracy: bool = True
    reasoning_quality: bool = True
    ttft: bool = True
    ttut: bool = True
    tps: bool = True
    token_usage: bool = True
    cost: bool = True
    consistency: bool = False  # 一致性Test
    consistency_runs: int = 3


@dataclass
class OutputConfig:
    """输出Configure"""
    report_dir: str = "reports"
    json_export: bool = True
    markdown_export: bool = True
    html_export: bool = True
    include_raw_responses: bool = False
    include_reasoning_content: bool = True


@dataclass
class TestConfig:
    """完整Test Configuration"""
    name: str
    description: str = ""
    version: str = "1.0"
    created_at: str = ""

    # ModelConfigure（支持多Model Comparison）
    models: list[ModelConfig] = field(default_factory=list)

    # Data集Configure
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # 指标Configure
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # 输出Configure
    output: OutputConfig = field(default_factory=OutputConfig)

    # 运行Configure
    concurrency: int = 4
    timeout_seconds: int = 120
    retry_count: int = 3
    enable_llm_judge: bool = False
    enable_llm_parser: bool = False


class TestConfigLoader:
    """
    Test ConfigurationLoad器

    Usage:
        loader = TestConfigLoader()

        # 从 YAML 文件Load
        config = loader.load("tests/config/gsm8k_mimo.yaml")

        # ValidateConfigure
        errors = loader.validate(config)
        if errors:
            print("Configuration error:", errors)

        # Save Config
        loader.save(config, "tests/config/new_config.yaml")
    """

    def __init__(self, config_dir: str = "tests/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load(self, filepath: str) -> TestConfig:
        """
        从 YAML 文件Load Config

        Args:
            filepath: YAML File path

        Returns:
            TestConfig
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.config_dir / path

        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return self._parse_config(data)

    def _parse_config(self, data: dict[str, Any]) -> TestConfig:
        """ParseConfigure字典"""
        # ParseModel Configuration
        models = []
        for model_data in data.get('models', []):
            models.append(ModelConfig(
                platform=model_data.get('platform', 'unknown'),
                model_id=model_data.get('model_id', ''),
                api_base_url=model_data.get('api_base_url', ''),
                api_key_env=model_data.get('api_key_env', ''),
                thinking_enabled=model_data.get('thinking_enabled', False),
                thinking_budget=model_data.get('thinking_budget'),
                reasoning_effort=model_data.get('reasoning_effort', 'medium'),
                temperature=model_data.get('temperature', 0.7),
                max_tokens=model_data.get('max_tokens', 2048)
            ))

        # ParseDatasetConfigure
        dataset_data = data.get('dataset', {})
        dataset = DatasetConfig(
            name=dataset_data.get('name', ''),
            path=dataset_data.get('path', ''),
            subset=dataset_data.get('subset', ''),
            samples=dataset_data.get('samples', 0),
            seed=dataset_data.get('seed', 42),
            num_shots=dataset_data.get('num_shots', 0)
        )

        # Parse指标Configure
        metrics_data = data.get('metrics', {})
        metrics = MetricsConfig(
            accuracy=metrics_data.get('accuracy', True),
            reasoning_quality=metrics_data.get('reasoning_quality', True),
            ttft=metrics_data.get('ttft', True),
            ttut=metrics_data.get('ttut', True),
            tps=metrics_data.get('tps', True),
            token_usage=metrics_data.get('token_usage', True),
            cost=metrics_data.get('cost', True),
            consistency=metrics_data.get('consistency', False),
            consistency_runs=metrics_data.get('consistency_runs', 3)
        )

        # Parse输出Configure
        output_data = data.get('output', {})
        output = OutputConfig(
            report_dir=output_data.get('report_dir', 'reports'),
            json_export=output_data.get('json_export', True),
            markdown_export=output_data.get('markdown_export', True),
            html_export=output_data.get('html_export', True),
            include_raw_responses=output_data.get('include_raw_responses', False),
            include_reasoning_content=output_data.get('include_reasoning_content', True)
        )

        return TestConfig(
            name=data.get('name', 'Unnamed Test'),
            description=data.get('description', ''),
            version=data.get('version', '1.0'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            models=models,
            dataset=dataset,
            metrics=metrics,
            output=output,
            concurrency=data.get('concurrency', 4),
            timeout_seconds=data.get('timeout_seconds', 120),
            retry_count=data.get('retry_count', 3),
            enable_llm_judge=data.get('enable_llm_judge', False),
            enable_llm_parser=data.get('enable_llm_parser', False)
        )

    def save(self, config: TestConfig, filepath: str):
        """
        Save Config到 YAML 文件

        Args:
            config: Test Configuration
            filepath: Save路径
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.config_dir / path

        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._config_to_dict(config)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _config_to_dict(self, config: TestConfig) -> dict[str, Any]:
        """willConfigureConvertis字典"""
        return {
            'name': config.name,
            'description': config.description,
            'version': config.version,
            'created_at': config.created_at,

            'models': [
                {
                    'platform': m.platform,
                    'model_id': m.model_id,
                    'api_base_url': m.api_base_url,
                    'api_key_env': m.api_key_env,
                    'thinking_enabled': m.thinking_enabled,
                    'thinking_budget': m.thinking_budget,
                    'reasoning_effort': m.reasoning_effort,
                    'temperature': m.temperature,
                    'max_tokens': m.max_tokens
                }
                for m in config.models
            ],

            'dataset': {
                'name': config.dataset.name,
                'path': config.dataset.path,
                'subset': config.dataset.subset,
                'samples': config.dataset.samples,
                'seed': config.dataset.seed,
                'num_shots': config.dataset.num_shots
            },

            'metrics': {
                'accuracy': config.metrics.accuracy,
                'reasoning_quality': config.metrics.reasoning_quality,
                'ttft': config.metrics.ttft,
                'ttut': config.metrics.ttut,
                'tps': config.metrics.tps,
                'token_usage': config.metrics.token_usage,
                'cost': config.metrics.cost,
                'consistency': config.metrics.consistency,
                'consistency_runs': config.metrics.consistency_runs
            },

            'output': {
                'report_dir': config.output.report_dir,
                'json_export': config.output.json_export,
                'markdown_export': config.output.markdown_export,
                'html_export': config.output.html_export,
                'include_raw_responses': config.output.include_raw_responses,
                'include_reasoning_content': config.output.include_reasoning_content
            },

            'concurrency': config.concurrency,
            'timeout_seconds': config.timeout_seconds,
            'retry_count': config.retry_count,
            'enable_llm_judge': config.enable_llm_judge,
            'enable_llm_parser': config.enable_llm_parser
        }

    def validate(self, config: TestConfig) -> list[str]:
        """
        ValidateConfigurehas效性

        Args:
            config: Test Configuration

        Returns:
            Error列表（空列表表示Validatevia）
        """
        errors = []

        # Check名称
        if not config.name:
            errors.append("Config namenot能is空")

        # CheckModel
        if not config.models:
            errors.append("至少needConfigure一 models")
        else:
            for i, model in enumerate(config.models):
                if not model.platform:
                    errors.append(f"Model {i+1} 缺少 platform")
                if not model.model_id:
                    errors.append(f"Model {i+1} 缺少 model_id")

        # CheckDataset
        if not config.dataset.name:
            errors.append("Dataset名称not能is空")

        # CheckConcurrency
        if config.concurrency < 1:
            errors.append("Concurrencymust大于 0")

        return errors

    def list_configs(self) -> list[str]:
        """列出所hasConfigure文件"""
        return [f.name for f in self.config_dir.glob("*.yaml")]

    def create_template(self, name: str = "template") -> str:
        """
        Create模板Configure文件

        Args:
            name: 模板名称

        Returns:
            CreateFile path
        """
        template = TestConfig(
            name="模板Test Configuration",
            description="这is一模板Configure，请based onneed修改",
            models=[
                ModelConfig(
                    platform="mimo",
                    model_id="mimo-v2-flash",
                    api_key_env="MIMO_API_KEY",
                    thinking_enabled=True
                )
            ],
            dataset=DatasetConfig(
                name="gsm8k",
                samples=100,
                seed=42,
                num_shots=0
            )
        )

        filepath = f"{name}.yaml"
        self.save(template, filepath)
        return str(self.config_dir / filepath)


def load_test_config(filepath: str) -> TestConfig:
    """便捷函数：LoadTest Configuration"""
    loader = TestConfigLoader()
    return loader.load(filepath)
