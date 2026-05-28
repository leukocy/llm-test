"""
Global PIQA Datasetunder载工具

从 HuggingFace under载 Global PIQA Dataset
Data源: mrlbenchmarks/global-piqa-nonparallel

用法:
    python utils/download_global_piqa.py
"""

import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def download_from_huggingface():
    """从 HuggingFace under载 Global PIQA Dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("need安装 datasets 库: pip install datasets")
        return False

    output_dir = os.path.join(project_root, "datasets", "global_piqa")
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []

    try:
        print("currently从 mrlbenchmarks/global-piqa-nonparallel under载...")

        # 常见语言Configure列表
        language_configs = [
            ("eng_latn", "English"),
            ("zho_hans", "Chinese"),
            ("jpn_jpan", "Japanese"),
            ("spa_latn", "Spanish"),
            ("fra_latn", "French"),
            ("deu_latn", "German"),
            ("arb_arab", "Arabic"),
            ("hin_deva", "Hindi"),
            ("kor_hang", "Korean"),
            ("por_latn", "Portuguese"),
            ("rus_cyrl", "Russian"),
            ("tha_thai", "Thai"),
            ("vie_latn", "Vietnamese"),
        ]

        for config, lang_name in language_configs:
            try:
                ds = load_dataset("mrlbenchmarks/global-piqa-nonparallel", config, split="test")

                for i, item in enumerate(ds):
                    # Data集use字段名: prompt, solution0, solution1, label
                    sample = {
                        "id": f"piqa_{config}_{i}",
                        "goal": item.get("prompt", item.get("goal", "")),
                        "sol1": item.get("solution0", item.get("sol1", "")),
                        "sol2": item.get("solution1", item.get("sol2", "")),
                        "label": int(item.get("label", 0)) if item.get("label") is not None else 0,
                        "language": lang_name,
                        "country": item.get("country", "")
                    }

                    if sample["goal"] and sample["sol1"] and sample["sol2"]:
                        all_samples.append(sample)

                print(f"  - {lang_name} ({config}): {len([s for s in all_samples if s['language'] == lang_name])} 样本")

            except Exception as e:
                print(f"  - {lang_name} ({config}): 跳过 ({e})")

        print(f"总计under载 {len(all_samples)}  samples")

    except Exception as e:
        print(f"Download failed: {e}")
        return False

    if not all_samples:
        print("未能Get到Data")
        return False

    # SaveData
    output_file = os.path.join(output_dir, "global_piqa.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    # Statistics语言
    languages = {}
    for s in all_samples:
        lang = s.get("language", "Unknown")
        languages[lang] = languages.get(lang, 0) + 1

    print(f"\n✅ Global PIQA DatasetSaved到 {output_file}")
    print(f"总计: {len(all_samples)}  samples")
    print(f"覆盖 {len(languages)} 种语言:")
    for lang, count in sorted(languages.items(), key=lambda x: -x[1])[:10]:
        print(f"  - {lang}: {count}")
    if len(languages) > 10:
        print(f"  ... 以andother {len(languages) - 10} 种语言")

    return True


def main():
    print("=" * 60)
    print("Global PIQA Datasetunder载工具")
    print("=" * 60)
    print()

    if download_from_huggingface():
        return

    print("\n⚠️  自动Download failed，willuse内置示例Data")


if __name__ == "__main__":
    main()
