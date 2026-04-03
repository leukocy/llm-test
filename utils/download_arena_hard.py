"""
Arena-Hard Datasetunder载工具

从 HuggingFace under载 Arena-Hard Dataset
Data源: lmarena-ai/arena-hard-auto

用法:
    python utils/download_arena_hard.py
"""

import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def download_from_huggingface():
    """从 HuggingFace under载 Arena-Hard Dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("need安装 datasets 库: pip install datasets")
        return False

    output_dir = os.path.join(project_root, "datasets", "arena_hard")
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []

    try:
        print("currently从 lmarena-ai/arena-hard-auto under载...")
        # 尝试not同Configure
        try:
            ds = load_dataset("lmarena-ai/arena-hard-auto", split="train")
        except Exception:
            try:
                ds = load_dataset("lmarena-ai/arena-hard-auto", "default", split="train")
            except Exception:
                ds = load_dataset("lmarena-ai/arena-hard-auto", trust_remote_code=True, split="train")

        for i, item in enumerate(ds):
            sample = {
                "id": item.get("question_id", f"arena_{i}"),
                "question": "",
                "category": item.get("category", "general"),
                "reference_answer": ""
            }

            # 提取问题
            turns = item.get("turns", [])
            if turns:
                sample["question"] = turns[0].get("content", "")

            if sample["question"]:  # 只保留has问题样本
                all_samples.append(sample)

        print(f"under载 {len(all_samples)}  samples")

    except Exception as e:
        print(f"Download failed: {e}")
        return False

    if not all_samples:
        print("未能Get到Data")
        return False

    # SaveData
    output_file = os.path.join(output_dir, "arena_hard.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    # Statistics类别
    categories = {}
    for s in all_samples:
        cat = s.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n✅ Arena-Hard DatasetSaved到 {output_file}")
    print(f"总计: {len(all_samples)}  samples")
    print("类别分布:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  - {cat}: {count}")

    return True


def main():
    print("=" * 60)
    print("Arena-Hard Datasetunder载工具")
    print("=" * 60)
    print()

    if download_from_huggingface():
        return

    print("\n⚠️  自动Download failed，willuse内置示例Data")


if __name__ == "__main__":
    main()
