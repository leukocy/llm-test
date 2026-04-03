"""
新增Test setDataunder载脚本
Downloads GPQA, ARC-Challenge, TruthfulQA, LongBench, SWE-Bench Lite, Needle-in-a-Haystack datasets

use方法:
    python utils/download_new_datasets.py
    python utils/download_new_datasets.py --dataset gpqa
    python utils/download_new_datasets.py --all
"""

import argparse
import json
import os
import sys

# Add items目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")


def ensure_datasets_installed():
    """确保 datasets 库已安装"""
    try:
        from datasets import load_dataset
        return True
    except ImportError:
        print("📦 currently安装 datasets 库...")
        os.system(f"{sys.executable} -m pip install datasets -q")
        try:
            from datasets import load_dataset
            return True
        except ImportError:
            print("❌ datasets 库安装失败，请手动运行: pip install datasets")
            return False


def download_gpqa():
    """under载 GPQA Diamond Dataset"""
    print("\n" + "=" * 50)
    print("📥 under载 GPQA Diamond (研究生级别科学推理)")
    print("=" * 50)

    try:
        from datasets import load_dataset

        output_dir = os.path.join(DATASETS_DIR, "gpqa")
        os.makedirs(output_dir, exist_ok=True)

        print("🔄 currently从 HuggingFace under载...")

        # GPQA need同意use条款，尝试多种方式
        try:
            ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', trust_remote_code=True)
            split = 'train' if 'train' in ds else list(ds.keys())[0]
        except Exception:
            # 备选方案
            print("⚠️ 官方Datasetneed授权，尝试备选来源...")
            ds = load_dataset('fingertap/GPQA-Diamond', trust_remote_code=True)
            split = list(ds.keys())[0]

        # Convertis标准格式
        samples = []
        for item in ds[split]:
            try:
                # 尝试not同字段名
                question = item.get('Question', item.get('question', ''))

                # Options
                choices = []
                if 'choices' in item:
                    choices = item['choices']
                else:
                    choices = [
                        item.get('Incorrect Answer 1', item.get('A', item.get('choice_a', ''))),
                        item.get('Incorrect Answer 2', item.get('B', item.get('choice_b', ''))),
                        item.get('Incorrect Answer 3', item.get('C', item.get('choice_c', ''))),
                        item.get('Correct Answer', item.get('D', item.get('choice_d', '')))
                    ]

                # Answer
                ans_raw = item.get('Correct Answer', item.get('answer', item.get('Answer', 'D')))
                if isinstance(ans_raw, int):
                    answer = ans_raw
                elif isinstance(ans_raw, str):
                    if ans_raw.upper() in 'ABCD':
                        answer = ord(ans_raw.upper()) - ord('A')
                    else:
                        answer = 3  # defaultis D
                else:
                    answer = 3

                sample = {
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'domain': item.get('Subdomain', item.get('domain', 'science'))
                }
                samples.append(sample)
            except Exception:
                continue

        output_file = os.path.join(output_dir, "gpqa_diamond.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"✅ GPQA Download complete: {len(samples)}  samples")
        print(f"   Save位置: {output_file}")
        return True

    except Exception as e:
        print(f"❌ GPQA Download failed: {e}")
        return False


def download_arc():
    """under载 ARC-Challenge Dataset"""
    print("\n" + "=" * 50)
    print("📥 under载 ARC-Challenge (科学常识推理)")
    print("=" * 50)

    try:
        from datasets import load_dataset

        output_dir = os.path.join(DATASETS_DIR, "arc")
        os.makedirs(output_dir, exist_ok=True)

        print("🔄 currently从 HuggingFace under载...")
        ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge')

        # Saveis JSONL 格式
        output_file = os.path.join(output_dir, "ARC-Challenge-Test.jsonl")

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in ds['test']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✅ ARC-Challenge Download complete: {len(ds['test'])}  samples")
        print(f"   Save位置: {output_file}")
        return True

    except Exception as e:
        print(f"❌ ARC Download failed: {e}")
        return False


def download_truthfulqa():
    """under载 TruthfulQA Dataset"""
    print("\n" + "=" * 50)
    print("📥 under载 TruthfulQA (真实性Test)")
    print("=" * 50)

    try:
        from datasets import load_dataset

        output_dir = os.path.join(DATASETS_DIR, "truthfulqa")
        os.makedirs(output_dir, exist_ok=True)

        print("🔄 currently从 HuggingFace under载...")
        ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')

        # Convertis标准格式
        samples = []
        for item in ds['validation']:
            sample = {
                'question': item.get('question', ''),
                'mc1_targets': item.get('mc1_targets', {}),
                'mc2_targets': item.get('mc2_targets', {}),
                'category': item.get('category', 'unknown')
            }
            samples.append(sample)

        output_file = os.path.join(output_dir, "truthfulqa_mc.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"✅ TruthfulQA Download complete: {len(samples)}  samples")
        print(f"   Save位置: {output_file}")
        return True

    except Exception as e:
        print(f"❌ TruthfulQA Download failed: {e}")
        return False


def download_longbench():
    """under载 LongBench Dataset"""
    print("\n" + "=" * 50)
    print("📥 under载 LongBench (长onunder文理解)")
    print("=" * 50)

    try:
        from datasets import load_dataset

        output_dir = os.path.join(DATASETS_DIR, "longbench")
        os.makedirs(output_dir, exist_ok=True)

        print("🔄 currently从 HuggingFace under载 (这可能need一些时间)...")
        # LongBench has多子任务，我们先under载 narrativeqa 作is示例
        try:
            ds = load_dataset('THUDM/LongBench', 'narrativeqa', split='test', trust_remote_code=True)

            output_file = os.path.join(output_dir, "narrativeqa.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in ds:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"✅ LongBench (narrativeqa) Download complete: {len(ds)}  samples")
            print(f"   Save位置: {output_file}")
            print("   注意: LongBench 包含多子任务，这里仅under载 narrativeqa 示例")
            return True
        except Exception as e:
            print(f"⚠️ LongBench 官方源Download failed: {e}")
            return False

    except Exception as e:
        print(f"❌ LongBench Download failed: {e}")
        return False


def download_swebench_lite():
    """under载 SWE-Bench Lite Dataset"""
    print("\n" + "=" * 50)
    print("📥 under载 SWE-Bench Lite (软件工程问题)")
    print("=" * 50)

    try:
        from datasets import load_dataset

        output_dir = os.path.join(DATASETS_DIR, "swebench_lite")
        os.makedirs(output_dir, exist_ok=True)

        print("🔄 currently从 HuggingFace under载...")
        try:
            ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test', trust_remote_code=True)

            output_file = os.path.join(output_dir, "swe-bench-lite.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in ds:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"✅ SWE-Bench Lite Download complete: {len(ds)}  samples")
            print(f"   Save位置: {output_file}")
            return True
        except Exception as e:
            print(f"⚠️ SWE-Bench Lite Download failed: {e}")
            return False

    except Exception as e:
        print(f"❌ SWE-Bench Lite Download failed: {e}")
        return False


def download_needle_haystack():
    """Generate/under载 Needle-in-a-Haystack test data"""
    print("\n" + "=" * 50)
    print("📥 准备 Needle-in-a-Haystack (大海捞针Test)")
    print("=" * 50)

    try:
        output_dir = os.path.join(DATASETS_DIR, "needle_haystack")
        os.makedirs(output_dir, exist_ok=True)

        print("ℹ️ Needle-in-a-Haystack test data通常is动态Generate")
        print("✅ 目录已准备就绪，Evaluatorwill自动Generatetest data")
        return True

    except Exception as e:
        print(f"❌ 准备目录失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="under载新增Test setData")
    parser.add_argument('--dataset', '-d', type=str,
                       choices=['gpqa', 'arc', 'truthfulqa', 'longbench', 'swebench_lite', 'needle_haystack', 'all'],
                       default='all',
                       help='要under载Dataset (default: all)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='under载所hasDataset')

    args = parser.parse_args()

    print("\n" + "🚀" * 20)
    print("    LLM Quality Assessment - 新增Test setunder载工具")
    print("🚀" * 20)

    # Check datasets 库
    if not ensure_datasets_installed():
        return

    results = {}

    if args.all or args.dataset == 'all':
        results['gpqa'] = download_gpqa()
        results['arc'] = download_arc()
        results['truthfulqa'] = download_truthfulqa()
        results['longbench'] = download_longbench()
        results['swebench_lite'] = download_swebench_lite()
        results['needle_haystack'] = download_needle_haystack()
    elif args.dataset == 'gpqa':
        results['gpqa'] = download_gpqa()
    elif args.dataset == 'arc':
        results['arc'] = download_arc()
    elif args.dataset == 'truthfulqa':
        results['truthfulqa'] = download_truthfulqa()
    elif args.dataset == 'longbench':
        results['longbench'] = download_longbench()
    elif args.dataset == 'swebench_lite':
        results['swebench_lite'] = download_swebench_lite()
    elif args.dataset == 'needle_haystack':
        results['needle_haystack'] = download_needle_haystack()

    # 打印汇总
    print("\n" + "=" * 50)
    print("📊 under载汇总")
    print("=" * 50)

    for name, success in results.items():
        status = "✅ succeeded" if success else "❌ 失败"
        print(f"  {name.upper()}: {status}")

    success_count = sum(results.values())
    total_count = len(results)

    if success_count == total_count:
        print(f"\n✨ 全部Download complete! ({success_count}/{total_count})")
    else:
        print(f"\n⚠️ 部分Download failed ({success_count}/{total_count})")
        print("   Please check网络Connector手动Download failedDataset")


if __name__ == "__main__":
    main()
