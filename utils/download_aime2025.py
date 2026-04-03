"""
AIME 2025 Datasetunder载工具

从 HuggingFace under载 AIME 2025 Dataset
支持多Data源：
- opencompass/AIME2025 (推荐)
- math-ai/aime25 (备选)

用法:
    python utils/download_aime2025.py
"""

import json
import os
import sys

# Add items目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def download_from_huggingface():
    """从 HuggingFace under载 AIME 2025 Dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("need安装 datasets 库: pip install datasets")
        return False

    output_dir = os.path.join(project_root, "datasets", "aime2025")
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []

    # 尝试从 opencompass/AIME2025 under载
    try:
        print("currently从 opencompass/AIME2025 under载...")

        # 尝试Load两子集
        for subset in ["AIME2025-I", "AIME2025-II"]:
            try:
                ds = load_dataset("opencompass/AIME2025", subset, split="test")
                source = "2025-I" if "I" in subset and "II" not in subset else "2025-II"

                for item in ds:
                    sample = {
                        "id": f"aime_{source.replace('-', '_')}_{len(all_samples) + 1}",
                        "problem": item.get("problem", item.get("question", "")),
                        "answer": str(item.get("answer", "")),
                        "source": source
                    }
                    if item.get("solution"):
                        sample["solution"] = item["solution"]
                    all_samples.append(sample)

                print(f"  - {subset}: Load {len([s for s in all_samples if s['source'] == source])} 道题")

            except Exception as e:
                print(f"  - {subset}: Load failed ({e})")

        if all_samples:
            print(f"从 opencompass/AIME2025 共under载 {len(all_samples)} 道题")

    except Exception as e:
        print(f"从 opencompass/AIME2025 Download failed: {e}")

    # if opencompass 失败，尝试 math-ai/aime25
    if not all_samples:
        try:
            print("currently从 math-ai/aime25 under载...")
            ds = load_dataset("math-ai/aime25", split="train")

            for i, item in enumerate(ds):
                sample = {
                    "id": item.get("id", f"aime_2025_{i+1}"),
                    "problem": item.get("problem", item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                    "source": "2025-I" if i < 15 else "2025-II"
                }
                if item.get("solution"):
                    sample["solution"] = item["solution"]
                all_samples.append(sample)

            print(f"从 math-ai/aime25 共under载 {len(all_samples)} 道题")

        except Exception as e:
            print(f"从 math-ai/aime25 Download failed: {e}")

    if not all_samples:
        print("所hasData源均失败，use内置示例Data")
        return False

    # SaveData
    output_file = os.path.join(output_dir, "aime2025.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\n✅ DatasetSaved到 {output_file}")
    print(f"   - AIME I: {len([s for s in all_samples if '2025-I' in s.get('source', '') and 'II' not in s.get('source', '')])} 道题")
    print(f"   - AIME II: {len([s for s in all_samples if '2025-II' in s.get('source', '')])} 道题")
    print(f"   - 总计: {len(all_samples)} 道题")

    return True


def download_from_url():
    """从 URL 直接under载 (备用方案)"""
    try:
        import requests
    except ImportError:
        print("need安装 requests 库: pip install requests")
        return False

    output_dir = os.path.join(project_root, "datasets", "aime2025")
    os.makedirs(output_dir, exist_ok=True)

    # 尝试直接Get HuggingFace  JSON 文件
    urls = [
        "https://huggingface.co/datasets/opencompass/AIME2025/resolve/main/AIME2025-I/test.json",
        "https://huggingface.co/datasets/opencompass/AIME2025/resolve/main/AIME2025-II/test.json",
    ]

    all_samples = []

    for url in urls:
        try:
            print(f"Downloading: {url}")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                source = "2025-I" if "AIME2025-I" in url else "2025-II"

                for item in data:
                    sample = {
                        "id": f"aime_{source.replace('-', '_')}_{len(all_samples) + 1}",
                        "problem": item.get("problem", item.get("question", "")),
                        "answer": str(item.get("answer", "")),
                        "source": source
                    }
                    all_samples.append(sample)

                print(f"  - succeededLoad {len([s for s in all_samples if s['source'] == source])} 道题")
            else:
                print(f"  - 失败 (HTTP {resp.status_code})")
        except Exception as e:
            print(f"  - 失败 ({e})")

    if all_samples:
        output_file = os.path.join(output_dir, "aime2025.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        print(f"\n✅ DatasetSaved到 {output_file}")
        return True

    return False


def main():
    print("=" * 60)
    print("AIME 2025 Datasetunder载工具")
    print("=" * 60)
    print()

    # 优先use HuggingFace datasets 库
    if download_from_huggingface():
        return

    # 备用方案：直接 HTTP under载
    print("\n尝试备用under载方案...")
    if download_from_url():
        return

    print("\n⚠️  no法自动under载Dataset")
    print("请手动under载Data并放置到 datasets/aime2025/ 目录")
    print("Data源:")
    print("  - https://huggingface.co/datasets/opencompass/AIME2025")
    print("  - https://huggingface.co/datasets/math-ai/aime25")


if __name__ == "__main__":
    main()
