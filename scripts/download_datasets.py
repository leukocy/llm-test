"""
Dataset Downloader using HuggingFace datasets library
use HuggingFace datasets 库under载Full data集 - 更可靠方法
"""

import json
import os
import sys


def check_datasets_installed():
    """Check datasets 库is否安装"""
    import importlib.util

    return importlib.util.find_spec("datasets") is not None


def install_datasets():
    """安装 datasets 库"""
    import subprocess

    print("currently安装 datasets 库...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
    print("datasets 库安装完成")


def download_math500(output_dir="datasets/math500", force=False):
    """under载 MATH-500 Dataset"""
    output_file = os.path.join(output_dir, "test.json")

    if os.path.exists(output_file) and not force:
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= 500:
            print(f"MATH-500 Dataset已存in ({len(data)} 样本)")
            return True

    print("Downloading MATH-500 Dataset...")
    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

        samples = []
        for item in ds:
            samples.append(
                {
                    "problem": item.get("problem", ""),
                    "solution": item.get("solution", ""),
                    "answer": item.get("answer", ""),
                    "subject": item.get("subject", ""),
                    "level": item.get("level", 0),
                    "unique_id": item.get("unique_id", ""),
                }
            )

        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"MATH-500 Download complete: {len(samples)} 样本")
        return True
    except Exception as e:
        print(f"MATH-500 Download failed: {e}")
        return False


def download_gsm8k(output_dir="datasets/gsm8k", force=False):
    """under载 GSM8K Dataset"""
    output_file = os.path.join(output_dir, "test.json")

    if os.path.exists(output_file) and not force:
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= 1000:
            print(f"GSM8K Dataset已存in ({len(data)} 样本)")
            return True

    print("Downloading GSM8K Dataset...")
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")

        samples = []
        for item in ds:
            samples.append(
                {"question": item.get("question", ""), "answer": item.get("answer", "")}
            )

        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"GSM8K Download complete: {len(samples)} 样本")
        return True
    except Exception as e:
        print(f"GSM8K Download failed: {e}")
        return False


def download_mmlu(output_dir="datasets/mmlu", force=False, subset="all"):
    """under载 MMLU Dataset"""
    output_file = os.path.join(output_dir, "test.json")

    if os.path.exists(output_file) and not force:
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= 1000:
            print(f"MMLU Dataset已存in ({len(data)} 样本)")
            return True

    print("Downloading MMLU Dataset (这可能need几minutes)...")
    try:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", subset, split="test")

        samples = []
        for item in ds:
            samples.append(
                {
                    "question": item.get("question", ""),
                    "choices": item.get("choices", []),
                    "answer": item.get("answer", 0),
                    "subject": item.get("subject", ""),
                }
            )

        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"MMLU Download complete: {len(samples)} 样本")
        return True
    except Exception as e:
        print(f"MMLU Download failed: {e}")
        return False


def download_humaneval(output_dir="datasets/humaneval", force=False):
    """under载 HumanEval Dataset"""
    output_file = os.path.join(output_dir, "test.json")

    if os.path.exists(output_file) and not force:
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= 100:
            print(f"HumanEval Dataset已存in ({len(data)} 样本)")
            return True

    print("Downloading HumanEval Dataset...")
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test")

        samples = []
        for item in ds:
            samples.append(
                {
                    "task_id": item.get("task_id", ""),
                    "prompt": item.get("prompt", ""),
                    "canonical_solution": item.get("canonical_solution", ""),
                    "test": item.get("test", ""),
                    "entry_point": item.get("entry_point", ""),
                }
            )

        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"HumanEval Download complete: {len(samples)} 样本")
        return True
    except Exception as e:
        print(f"HumanEval Download failed: {e}")
        return False


def download_all(force=False):
    """under载所hasDataset"""
    print("=" * 60)
    print("开始under载所hasQuality AssessmentDataset")
    print("=" * 60)

    results = {}

    results["math500"] = download_math500(force=force)
    results["gsm8k"] = download_gsm8k(force=force)
    results["mmlu"] = download_mmlu(force=force)
    results["humaneval"] = download_humaneval(force=force)
    # Prompt-suffix builder pools (Science/Code/Longform types)
    results["gpqa"] = download_gpqa(force=force)
    results["mbpp"] = download_mbpp(force=force)
    results["longbench"] = download_longbench(force=force)
    results["swebench_lite"] = download_swebench_lite(force=force)

    print("\n" + "=" * 60)
    print("Download Results汇总")
    print("=" * 60)

    for name, success in results.items():
        status = "成功" if success else "失败"
        print(f"  {name}: {status}")

    return results


def download_gpqa(output_dir="datasets/gpqa", force=False):
    """Download GPQA Diamond (used by the Science prompt-suffix type).

    NOTE: GPQA is gated on HuggingFace — you must accept its license at
    https://huggingface.co/datasets/Idavidrein/gpqa before this works,
    and be logged in (`huggingface-cli login`).
    """
    output_file = os.path.join(output_dir, "gpqa_diamond.json")
    if os.path.exists(output_file) and not force:
        print(f"GPQA Diamond已存in: {output_file}")
        return True
    print("Downloading GPQA Diamond...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True
        )
        samples = []
        for item in ds:
            samples.append(
                {
                    "question": item.get("Question", ""),
                    "choices": [
                        item.get(k, "")
                        for k in (
                            "Correct Answer",
                            "Incorrect Answer 1",
                            "Incorrect Answer 2",
                            "Incorrect Answer 3",
                        )
                    ],
                    **item,
                }
            )
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"GPQA Diamond Download complete: {len(samples)} 样本")
        return True
    except Exception as e:
        print(f"GPQA Download failed: {e}")
        print(
            "   GPQA is gated — accept the license at "
            "https://huggingface.co/datasets/Idavidrein/gpqa then `huggingface-cli login`."
        )
        return False


def download_mbpp(output_dir="datasets/mbpp", force=False):
    """Download MBPP (used by the Code prompt-suffix type)."""
    output_file = os.path.join(output_dir, "mbpp.json")
    if os.path.exists(output_file) and not force:
        print(f"MBPP已存in: {output_file}")
        return True
    print("Downloading MBPP...")
    try:
        from datasets import load_dataset

        ds = load_dataset("google-research-datasets/mbpp", split="test")
        samples = []
        for item in ds:
            samples.append(
                {
                    "text": item.get("text", ""),
                    "code": item.get("code", ""),
                    "test_list": item.get("test_list", []),
                    "task_id": item.get("task_id"),
                }
            )
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"MBPP Download complete: {len(samples)} 样本")
        return True
    except Exception as e:
        print(f"MBPP Download failed: {e}")
        return False


def download_longbench(output_dir="datasets/longbench", force=False):
    """Download LongBench subsets (used by the Longform prompt-suffix type)."""
    subsets = ["repobench-p", "lcc", "qasper"]
    all_ok = True
    for sub in subsets:
        out = os.path.join(output_dir, f"{sub}.jsonl")
        if os.path.exists(out) and not force:
            print(f"LongBench/{sub}已存in: {out}")
            continue
        print(f"Downloading LongBench/{sub}...")
        try:
            from datasets import load_dataset

            ds = load_dataset(
                "THUDM/LongBench", sub, split="test", trust_remote_code=True
            )
            os.makedirs(output_dir, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                for item in ds:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"LongBench/{sub} complete: {len(ds)} 行")
        except Exception as e:
            print(f"LongBench/{sub} failed: {e}")
            all_ok = False
    return all_ok


def download_swebench_lite(output_dir="datasets/swebench_lite", force=False):
    """Download SWE-Bench Lite (used by the Code prompt-suffix type)."""
    output_file = os.path.join(output_dir, "swe-bench-lite.jsonl")
    if os.path.exists(output_file) and not force:
        print(f"SWE-Bench Lite已存in: {output_file}")
        return True
    print("Downloading SWE-Bench Lite...")
    try:
        from datasets import load_dataset

        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"SWE-Bench Lite complete: {len(ds)} 行")
        return True
    except Exception as e:
        print(f"SWE-Bench Lite failed: {e}")
        return False


def check_status():
    """CheckDatasetStatus"""
    datasets_info = {
        "math500": ("datasets/math500/test.json", 500),
        "gsm8k": ("datasets/gsm8k/test.json", 1319),
        "mmlu": ("datasets/mmlu/test.json", 14042),
        "humaneval": ("datasets/humaneval/test.json", 164),
        "gpqa": ("datasets/gpqa/gpqa_diamond.json", 198),
        "mbpp": ("datasets/mbpp/mbpp.json", 500),
        "longbench": ("datasets/longbench/repobench-p.jsonl", 500),
        "swebench_lite": ("datasets/swebench_lite/swe-bench-lite.jsonl", 300),
    }

    print("\n" + "=" * 60)
    print("DatasetStatus")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Status':<10} {'Sample count':<10} {'期望':<10}")
    print("-" * 60)

    for name, (path, expected) in datasets_info.items():
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                count = len(data)
                if count >= expected * 0.9:  # 允许 10% 误差
                    status = "完整"
                else:
                    status = "not完整"
            except Exception:
                status = "损坏"
                count = 0
        else:
            status = "缺失"
            count = 0

        print(f"{name:<15} {status:<10} {count:<10} {expected:<10}")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Datasetunder载工具 (use HuggingFace datasets 库)"
    )
    parser.add_argument("--all", "-a", action="store_true", help="under载所hasDataset")
    parser.add_argument("--math500", action="store_true", help="under载 MATH-500")
    parser.add_argument("--gsm8k", action="store_true", help="under载 GSM8K")
    parser.add_argument("--mmlu", action="store_true", help="under载 MMLU")
    parser.add_argument("--humaneval", action="store_true", help="under载 HumanEval")
    parser.add_argument(
        "--gpqa", action="store_true", help="under载 GPQA Diamond (Science题型)"
    )
    parser.add_argument("--mbpp", action="store_true", help="under载 MBPP (Code题型)")
    parser.add_argument(
        "--longbench", action="store_true", help="under载 LongBench (Longform题型)"
    )
    parser.add_argument(
        "--swebench", action="store_true", help="under载 SWE-Bench Lite (Code题型)"
    )
    parser.add_argument(
        "--status", "-s", action="store_true", help="CheckDatasetStatus"
    )
    parser.add_argument("--force", "-f", action="store_true", help="强制重新under载")

    args = parser.parse_args()

    # Check datasets 库
    if not check_datasets_installed():
        print("datasets 库未安装")
        install_datasets()

    if args.status:
        check_status()
    elif args.all:
        download_all(force=args.force)
    elif args.math500:
        download_math500(force=args.force)
    elif args.gsm8k:
        download_gsm8k(force=args.force)
    elif args.mmlu:
        download_mmlu(force=args.force)
    elif args.humaneval:
        download_humaneval(force=args.force)
    elif args.gpqa:
        download_gpqa(force=args.force)
    elif args.mbpp:
        download_mbpp(force=args.force)
    elif args.longbench:
        download_longbench(force=args.force)
    elif args.swebench:
        download_swebench_lite(force=args.force)
    else:
        check_status()
        print("\nuse示例:")
        print("  python download_datasets.py --status    # Check status")
        print("  python download_datasets.py --all       # under载所has")
        print("  python download_datasets.py --math500   # 只under载 MATH-500")
        print("  python download_datasets.py --all -f    # 强制重新under载")
