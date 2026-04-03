"""
Dataset Downloader
Datasetunder载工具 - 自动从 HuggingFace under载公开Test set
"""

import json
import os
from typing import Any, Dict, Optional

import requests


class DatasetDownloader:
    """
    Datasetunder载器
    支持从 HuggingFace orother源under载评估Dataset
    """

    # HuggingFace API Configure
    HF_API_BASE = "https://datasets-server.huggingface.co"

    # 支持Datasetand其 HuggingFace ID
    DATASET_REGISTRY = {
        "math500": {
            "hf_id": "HuggingFaceH4/MATH-500",
            "config": "default",
            "split": "test",
            "local_path": "datasets/math500",
            "filename": "test.json"
        },
        "gsm8k": {
            "hf_id": "openai/gsm8k",
            "config": "main",
            "split": "test",
            "local_path": "datasets/gsm8k",
            "filename": "test.json"
        },
        "mmlu": {
            "hf_id": "cais/mmlu",
            "config": "all",
            "split": "test",
            "local_path": "datasets/mmlu",
            "filename": "test.json"
        },
        "humaneval": {
            "hf_id": "openai/openai_humaneval",
            "config": "default",
            "split": "test",
            "local_path": "datasets/humaneval",
            "filename": "test.json"
        }
    }



    def __init__(self, base_dir: str = "."):
        """
        Initializeunder载器

        Args:
            base_dir: 基础目录
        """
        self.base_dir = base_dir

    def download_dataset(
        self,
        dataset_name: str,
        max_rows: int | None = None,
        force: bool = False
    ) -> bool:
        """
        under载指定Dataset

        Args:
            dataset_name: Dataset名称 ("math500", "gsm8k", "mmlu")
            max_rows: 最大行数限制 (None = 全部)
            force: is否强制重新under载

        Returns:
            is否succeeded
        """
        if dataset_name not in self.DATASET_REGISTRY:
            print(f"未知Dataset: {dataset_name}")
            print(f"支持Dataset: {list(self.DATASET_REGISTRY.keys())}")
            return False

        config = self.DATASET_REGISTRY[dataset_name]
        local_path = os.path.join(self.base_dir, config["local_path"])
        full_path = os.path.join(local_path, config["filename"])

        # Checkis否已存in
        if os.path.exists(full_path) and not force:
            print(f"Dataset已存in: {full_path}")
            return True

        print(f"Downloading {dataset_name} 从 HuggingFace...")

        try:
            # Build API URL
            hf_id = config["hf_id"]
            split = config["split"]
            hf_config = config.get("config", "default")

            # use datasets-server API，分页under载
            url = f"{self.HF_API_BASE}/rows"
            all_samples = []

            # 每次最多请求 100 条，避免超时
            page_size = 100
            offset = 0
            max_total = max_rows or 100000  # default最大 10 万条

            while len(all_samples) < max_total:
                params = {
                    "dataset": hf_id,
                    "config": hf_config,
                    "split": split,
                    "offset": offset,
                    "length": min(page_size, max_total - len(all_samples))
                }

                response = requests.get(url, params=params, timeout=60)

                if response.status_code == 422:
                    # 可能already到达末尾
                    break

                response.raise_for_status()
                data = response.json()

                # 提取行Data
                rows = data.get("rows", [])
                if not rows:
                    break  # 没has更多Data

                samples = [row["row"] for row in rows]
                all_samples.extend(samples)

                print(f"  已under载 {len(all_samples)} 条Data...")

                if len(samples) < page_size:
                    break  # 这is最后一页

                offset += page_size

            if not all_samples:
                print("未Get到Data")
                return False

            # 确保目录存in
            os.makedirs(local_path, exist_ok=True)

            # Save到文件
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(all_samples, f, ensure_ascii=False, indent=2)

            print(f"✅ succeededunder载 {len(all_samples)} 条Data到 {full_path}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Download failed (Network error): {e}")
            return False
        except Exception as e:
            print(f"Download failed: {e}")
            import traceback
            traceback.print_exc()
            return False


    def download_all(self, force: bool = False) -> dict[str, bool]:
        """
        under载所hasRegisterDataset

        Returns:
            各DatasetDownload Results
        """
        results = {}
        for name in self.DATASET_REGISTRY:
            results[name] = self.download_dataset(name, force=force)
        return results

    def check_datasets(self) -> dict[str, dict[str, Any]]:
        """
        Check本地DatasetStatus

        Returns:
            各DatasetStatus信息
        """
        status = {}

        for name, config in self.DATASET_REGISTRY.items():
            local_path = os.path.join(self.base_dir, config["local_path"])
            full_path = os.path.join(local_path, config["filename"])

            info = {
                "name": name,
                "hf_id": config["hf_id"],
                "local_path": full_path,
                "exists": os.path.exists(full_path),
                "samples": 0
            }

            if info["exists"]:
                try:
                    with open(full_path, encoding='utf-8') as f:
                        data = json.load(f)
                    info["samples"] = len(data) if isinstance(data, list) else 0
                except:
                    pass

            status[name] = info

        return status

    def print_status(self):
        """打印DatasetStatus"""
        status = self.check_datasets()

        print("\n" + "=" * 60)
        print("📊 DatasetStatus")
        print("=" * 60)

        for name, info in status.items():
            status_icon = "✅" if info["exists"] else "❌"
            samples_str = f"({info['samples']} 样本)" if info["samples"] > 0 else ""

            print(f"{status_icon} {name:15} | {info['hf_id']:30} | {samples_str}")

        print("=" * 60)


def download_math500(output_dir: str = "datasets/math500", max_samples: int = 500) -> bool:
    """
    专门under载 MATH-500 Dataset便捷函数
    """
    downloader = DatasetDownloader()
    return downloader.download_dataset("math500", max_rows=max_samples)


def download_gsm8k(output_dir: str = "datasets/gsm8k", max_samples: int = 1319) -> bool:
    """
    专门under载 GSM8K Dataset便捷函数
    """
    downloader = DatasetDownloader()
    return downloader.download_dataset("gsm8k", max_rows=max_samples)


# CLI 入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Datasetunder载工具")
    parser.add_argument("--dataset", "-d", type=str, help="要under载Dataset名称")
    parser.add_argument("--all", "-a", action="store_true", help="under载所hasDataset")
    parser.add_argument("--status", "-s", action="store_true", help="DisplayDatasetStatus")
    parser.add_argument("--force", "-f", action="store_true", help="强制重新under载")
    parser.add_argument("--max-rows", "-m", type=int, default=None, help="最大行数")

    args = parser.parse_args()

    downloader = DatasetDownloader()

    if args.status:
        downloader.print_status()
    elif args.all:
        downloader.download_all(force=args.force)
    elif args.dataset:
        downloader.download_dataset(args.dataset, max_rows=args.max_rows, force=args.force)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python dataset_downloader.py --status")
        print("  python dataset_downloader.py -d math500")
        print("  python dataset_downloader.py --all")
