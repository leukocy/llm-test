"""
Dataset Manager (Dataset Manager)

提供Dataset自动under载、Version管理and本地缓存功能。

支持Data源:
1. HuggingFace Datasets
2. Custom URL under载
3. 本地文件

use方式:
    from core.dataset_manager import DatasetManager, get_dataset

    # GetDataset (自动under载ifnot存in)
    manager = DatasetManager()
    samples = manager.load("gsm8k", split="test")

    # 快捷函数
    samples = get_dataset("mmlu", split="test", max_samples=100)
"""

import gzip
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================
# Config
# ============================================

@dataclass
class DatasetConfig:
    """DatasetConfigure"""
    name: str
    hf_path: str | None = None  # HuggingFace Dataset路径
    url: str | None = None  # 直接under载 URL
    local_path: str = ""  # 本地存储路径
    split_mapping: dict[str, str] = field(default_factory=dict)  # 分割映射
    version: str = "latest"
    description: str = ""


# 预定义DatasetConfigure
DATASET_CONFIGS = {
    # 数学推理
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_path="openai/gsm8k",
        local_path="datasets/gsm8k",
        split_mapping={"test": "test", "train": "train"},
        description="Grade School Math 8K - 小学数学Apply题"
    ),

    # 多任务语言理解
    "mmlu": DatasetConfig(
        name="mmlu",
        hf_path="cais/mmlu",
        local_path="datasets/mmlu",
        split_mapping={"test": "test", "dev": "validation", "train": "dev"},
        description="Massive Multitask Language Understanding"
    ),

    # 高级数学
    "math500": DatasetConfig(
        name="math500",
        hf_path="hendrycks/competition_math",
        local_path="datasets/math500",
        split_mapping={"test": "test"},
        description="Competition Math - 竞赛数学"
    ),

    # 代码Generate
    "humaneval": DatasetConfig(
        name="humaneval",
        hf_path="openai/openai_humaneval",
        local_path="datasets/humaneval",
        split_mapping={"test": "test"},
        description="HumanEval - Python 代码Generate"
    ),

    # 研究生问答
    "gpqa": DatasetConfig(
        name="gpqa",
        hf_path="Idavidrein/gpqa",
        local_path="datasets/gpqa",
        split_mapping={"test": "train"},  # GPQA 只has train split
        description="Graduate-Level Google-Proof Q&A"
    ),

    # 科学问答
    "arc": DatasetConfig(
        name="arc",
        hf_path="allenai/ai2_arc",
        local_path="datasets/arc",
        split_mapping={"test": "test", "train": "train"},
        description="AI2 Reasoning Challenge"
    ),

    # 真实性问答
    "truthfulqa": DatasetConfig(
        name="truthfulqa",
        hf_path="truthfulqa/truthful_qa",
        local_path="datasets/truthfulqa",
        split_mapping={"test": "validation"},
        description="TruthfulQA - 真实性问答"
    ),

    # 常识推理
    "hellaswag": DatasetConfig(
        name="hellaswag",
        hf_path="Rowan/hellaswag",
        local_path="datasets/hellaswag",
        split_mapping={"test": "validation", "train": "train"},
        description="HellaSwag - 常识推理"
    ),

    # Winogrande
    "winogrande": DatasetConfig(
        name="winogrande",
        hf_path="allenai/winogrande",
        local_path="datasets/winogrande",
        split_mapping={"test": "validation", "train": "train"},
        description="Winogrande - 代词消解"
    ),

    # Python 代码
    "mbpp": DatasetConfig(
        name="mbpp",
        hf_path="google-research-datasets/mbpp",
        local_path="datasets/mbpp",
        split_mapping={"test": "test", "train": "train"},
        description="Mostly Basic Python Problems"
    ),
}


# ============================================
# Data集管理器
# ============================================

class DatasetManager:
    """
    Dataset Manager

    自动under载、缓存andLoad评估Dataset
    """

    def __init__(
        self,
        cache_dir: str = "datasets",
        auto_download: bool = True,
        log_callback: Callable[[str], None] | None = None
    ):
        """
        InitializeDataset Manager

        Args:
            cache_dir: Dataset缓存目录
            auto_download: is否自动under载缺失Dataset
            log_callback: LogCallback函数
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_download = auto_download
        self.log_callback = log_callback

        # Data集Configure
        self.configs = DATASET_CONFIGS.copy()

        # under载Status
        self._download_progress: dict[str, float] = {}

    def _log(self, message: str):
        """输出Log"""
        if self.log_callback:
            self.log_callback(message)
        print(f"[DatasetManager] {message}")

    def register(self, config: DatasetConfig):
        """RegisterCustomDatasetConfigure"""
        self.configs[config.name] = config

    def list_datasets(self) -> list[str]:
        """列出所has可用Dataset"""
        return list(self.configs.keys())

    def get_config(self, name: str) -> DatasetConfig | None:
        """GetDatasetConfigure"""
        return self.configs.get(name)

    def get_local_path(self, name: str) -> Path:
        """GetDataset本地路径"""
        config = self.configs.get(name)
        if config and config.local_path:
            return Path(config.local_path)
        return self.cache_dir / name

    def is_available(self, name: str) -> bool:
        """CheckDatasetis否本地可用"""
        local_path = self.get_local_path(name)
        if not local_path.exists():
            return False

        # Checkis否hasData文件
        data_files = list(local_path.glob("*.json")) + list(local_path.glob("*.jsonl"))
        return len(data_files) > 0

    def download(
        self,
        name: str,
        force: bool = False,
        progress_callback: Callable[[float, str], None] | None = None
    ) -> bool:
        """
        under载Dataset

        Args:
            name: Dataset名称
            force: is否强制重新under载
            progress_callback: 进度Callback (progress: 0-1, message: str)

        Returns:
            is否succeeded
        """
        config = self.configs.get(name)
        if not config:
            self._log(f"未知Dataset: {name}")
            return False

        self.get_local_path(name)

        # Checkis否needunder载
        if self.is_available(name) and not force:
            self._log(f"Dataset {name} 已存in")
            return True

        self._log(f"开始under载Dataset: {name}")

        if progress_callback:
            progress_callback(0.0, f"准备under载 {name}...")

        try:
            # 优先use HuggingFace
            if config.hf_path:
                return self._download_from_hf(name, config, progress_callback)

            # use URL under载
            if config.url:
                return self._download_from_url(name, config, progress_callback)

            self._log(f"Dataset {name} 没hasConfigureunder载源")
            return False

        except Exception as e:
            self._log(f"Download failed: {e}")
            return False

    def _download_from_hf(
        self,
        name: str,
        config: DatasetConfig,
        progress_callback: Callable[[float, str], None] | None = None
    ) -> bool:
        """从 HuggingFace under载Dataset"""
        try:
            from datasets import load_dataset
        except ImportError:
            self._log("请安装 datasets: pip install datasets")
            return False

        local_path = self.get_local_path(name)
        local_path.mkdir(parents=True, exist_ok=True)

        try:
            if progress_callback:
                progress_callback(0.1, f"currently从 HuggingFace Load {config.hf_path}...")

            # LoadDataset
            # Process特殊情况
            if name == "arc":
                dataset = load_dataset(config.hf_path, "ARC-Challenge")
            elif name == "mmlu":
                dataset = load_dataset(config.hf_path, "all")
            elif name == "winogrande":
                dataset = load_dataset(config.hf_path, "winogrande_xl")
            elif name == "truthfulqa":
                dataset = load_dataset(config.hf_path, "generation")
            else:
                dataset = load_dataset(config.hf_path)

            if progress_callback:
                progress_callback(0.5, "currentlySave到本地...")

            # Save各 split
            splits_saved = 0
            total_splits = len(dataset)

            for split_name in dataset:
                split_data = dataset[split_name]
                output_file = local_path / f"{split_name}.json"

                # Convertis JSON
                samples = [dict(row) for row in split_data]

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)

                splits_saved += 1
                if progress_callback:
                    progress = 0.5 + 0.5 * (splits_saved / total_splits)
                    progress_callback(progress, f"Saved {split_name} ({len(samples)} 样本)")

                self._log(f"Saved {split_name}: {len(samples)} 样本")

            # Save元信息
            meta = {
                "name": name,
                "hf_path": config.hf_path,
                "downloaded_at": datetime.now().isoformat(),
                "splits": list(dataset.keys()),
                "total_samples": sum(len(dataset[s]) for s in dataset)
            }
            with open(local_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            if progress_callback:
                progress_callback(1.0, "Download complete")

            self._log(f"Dataset {name} Download complete")
            return True

        except Exception as e:
            self._log(f"从 HuggingFace Download failed: {e}")
            return False

    def _download_from_url(
        self,
        name: str,
        config: DatasetConfig,
        progress_callback: Callable[[float, str], None] | None = None
    ) -> bool:
        """从 URL under载Dataset"""
        local_path = self.get_local_path(name)
        local_path.mkdir(parents=True, exist_ok=True)

        try:
            url = config.url
            filename = url.split("/")[-1]
            download_path = local_path / filename

            if progress_callback:
                progress_callback(0.1, f"Downloading {filename}...")

            # under载文件
            urllib.request.urlretrieve(url, download_path)

            if progress_callback:
                progress_callback(0.7, "currently解压...")

            # 解压
            if filename.endswith(".zip"):
                with zipfile.ZipFile(download_path, 'r') as zf:
                    zf.extractall(local_path)
                download_path.unlink()
            elif filename.endswith(".gz"):
                with gzip.open(download_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
                output_file = local_path / filename.replace(".gz", "")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                download_path.unlink()

            if progress_callback:
                progress_callback(1.0, "Download complete")

            return True

        except Exception as e:
            self._log(f"从 URL Download failed: {e}")
            return False

    def load(
        self,
        name: str,
        split: str = "test",
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 42
    ) -> list[dict[str, Any]]:
        """
        LoadDataset

        Args:
            name: Dataset名称
            split: Data分割 (test, train, dev, validation)
            max_samples: 最大Sample count
            shuffle: is否打乱
            seed: Random Seed

        Returns:
            样本列表
        """
        # Checkis否needunder载
        if not self.is_available(name):
            if self.auto_download:
                self._log(f"Dataset {name} not存in，currently自动under载...")
                if not self.download(name):
                    return []
            else:
                self._log(f"Dataset {name} not存in，请先under载")
                return []

        local_path = self.get_local_path(name)
        config = self.configs.get(name)

        # 映射 split 名称
        actual_split = split
        if config and config.split_mapping:
            actual_split = config.split_mapping.get(split, split)

        # 尝试多种文件格式
        possible_files = [
            local_path / f"{actual_split}.json",
            local_path / f"{split}.json",
            local_path / f"{name}_{split}.json",
            local_path / f"{actual_split}.jsonl",
            local_path / f"{split}.jsonl",
        ]

        samples = []
        for filepath in possible_files:
            if filepath.exists():
                try:
                    if filepath.suffix == ".jsonl":
                        with open(filepath, encoding='utf-8') as f:
                            samples = [json.loads(line) for line in f if line.strip()]
                    else:
                        with open(filepath, encoding='utf-8') as f:
                            data = json.load(f)
                            samples = data if isinstance(data, list) else data.get("data", [])

                    self._log(f"从 {filepath.name} Load {len(samples)}  samples")
                    break
                except Exception as e:
                    self._log(f"Load {filepath} 失败: {e}")

        if not samples:
            self._log(f"Not found {name}  {split} 分割")
            return []

        # 打乱
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(samples)

        # 限制数量
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]

        return samples

    def get_info(self, name: str) -> dict[str, Any]:
        """GetDataset信息"""
        config = self.configs.get(name)
        local_path = self.get_local_path(name)

        info = {
            "name": name,
            "available": self.is_available(name),
            "local_path": str(local_path),
            "description": config.description if config else "",
            "hf_path": config.hf_path if config else None,
        }

        # 读取元信息
        meta_file = local_path / "metadata.json"
        if meta_file.exists():
            with open(meta_file, encoding='utf-8') as f:
                info["metadata"] = json.load(f)

        # Statistics文件
        if local_path.exists():
            info["files"] = [f.name for f in local_path.iterdir() if f.is_file()]
            info["total_size_mb"] = sum(
                f.stat().st_size for f in local_path.iterdir() if f.is_file()
            ) / (1024 * 1024)

        return info

    def delete(self, name: str) -> bool:
        """Delete本地Dataset"""
        local_path = self.get_local_path(name)
        if local_path.exists():
            shutil.rmtree(local_path)
            self._log(f"已DeleteDataset: {name}")
            return True
        return False


# ============================================
# 全局实例and便捷函数
# ============================================

_global_manager: DatasetManager | None = None


def get_manager(cache_dir: str = "datasets", **kwargs) -> DatasetManager:
    """Get全局Dataset Manager实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = DatasetManager(cache_dir=cache_dir, **kwargs)
    return _global_manager


def get_dataset(
    name: str,
    split: str = "test",
    max_samples: int | None = None,
    **kwargs
) -> list[dict[str, Any]]:
    """
    快速GetDataset

    Args:
        name: Dataset名称
        split: Data分割
        max_samples: 最大Sample count

    Returns:
        样本列表
    """
    manager = get_manager()
    return manager.load(name, split=split, max_samples=max_samples, **kwargs)


def ensure_dataset(name: str, **kwargs) -> bool:
    """
    确保Dataset可用 (ifnot存in则under载)

    Returns:
        is否可用
    """
    manager = get_manager()
    if manager.is_available(name):
        return True
    return manager.download(name, **kwargs)


def list_available_datasets() -> list[dict[str, Any]]:
    """列出所has可用Datasetand其Status"""
    manager = get_manager()
    result = []
    for name in manager.list_datasets():
        info = manager.get_info(name)
        result.append({
            "name": name,
            "available": info["available"],
            "description": info["description"],
            "size_mb": info.get("total_size_mb", 0)
        })
    return result
