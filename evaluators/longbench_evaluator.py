"""
LongBench Evaluator
LongBench 长onunder文理解DatasetEvaluator

LongBench is一 testsModel长onunder文理解能力Dataset，
包含多种任务类型：单文档QA、多文档QA、摘要、代码etc.。
平均onunder文长度in 5K-15K tokens。
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_choice_answer, normalize_text


class LongBenchEvaluator(BaseEvaluator):
    """
    LongBench DatasetEvaluator

    Data格式:
    {
        "input": "长文本内容...",
        "context": "optional额外onunder文",
        "answers": ["answer1", "answer2"],  # 可接受Answer列表
        "length": 12345,  # onunder文长度
        "dataset": "qasper",  # 子任务名称
        "language": "en",
        "all_classes": null,  # 用于分类任务
        "_id": "xxx"
    }

    子任务包括:
    - 单文档QA: narrativeqa, qasper, multifieldqa_en/zh
    - 多文档QA: hotpotqa, 2wikimqa, musique
    - 摘要: gov_report, qmsum, multi_news
    - Few-shot: trec, triviaqa, samsum
    - 代码: lcc, repobench-p
    """

    # 支持子任务
    TASKS = {
        'single_doc_qa': ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh'],
        'multi_doc_qa': ['hotpotqa', '2wikimqa', 'musique'],
        'summarization': ['gov_report', 'qmsum', 'multi_news'],
        'few_shot': ['trec', 'triviaqa', 'samsum'],
        'code': ['lcc', 'repobench-p'],
        'synthetic': ['passage_count', 'passage_retrieval_en', 'passage_retrieval_zh']
    }

    def __init__(
        self,
        dataset_name: str = "longbench",
        dataset_path: str = "datasets/longbench",
        num_shots: int = 0,  # LongBench 通常use 0-shot
        max_samples: int | None = None,
        seed: int = 42
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_shots=num_shots,
            max_samples=max_samples,
            seed=seed
        )
        random.seed(seed)

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """Load LongBench Dataset"""
        samples = []

        # if指定子集，只Load该子集
        if subset:
            subsets_to_load = [subset]
        else:
            # Load所has子集
            subsets_to_load = []
            for task_subsets in self.TASKS.values():
                subsets_to_load.extend(task_subsets)

        # 尝试LoadData
        for task_name in subsets_to_load:
            possible_files = [
                os.path.join(self.dataset_path, f"{task_name}.jsonl"),
                os.path.join(self.dataset_path, f"{task_name}.json"),
            ]

            for filepath in possible_files:
                if os.path.exists(filepath):
                    try:
                        if filepath.endswith('.jsonl'):
                            with open(filepath, encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        item = json.loads(line)
                                        item['task'] = task_name
                                        samples.append(item)
                        else:
                            with open(filepath, encoding='utf-8') as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    item['task'] = task_name
                                    samples.append(item)
                        break
                    except Exception as e:
                        print(f"Load {filepath} 失败: {e}")

        if not samples:
            samples = self._create_sample_data()

        samples = self._normalize_samples(samples)
        random.shuffle(samples)

        total_needed = self.num_shots + (self.max_samples if self.max_samples else len(samples))
        if len(samples) > total_needed:
            samples = samples[:total_needed]

        if self.num_shots > 0:
            self.few_shot_examples = samples[:self.num_shots]
            samples = samples[self.num_shots:]

        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def _normalize_samples(self, samples: list[dict]) -> list[dict]:
        """Standardize样本格式"""
        normalized = []

        for sample in samples:
            try:
                # Get输入andonunder文
                input_text = sample.get('input', '')
                context = sample.get('context', '')

                # Mergeis完整onunder文
                full_context = f"{context}\n\n{input_text}" if context else input_text

                # GetAnswer
                answers = sample.get('answers', [])
                if isinstance(answers, str):
                    answers = [answers]

                normalized.append({
                    'context': full_context,
                    'answers': answers,
                    'task': sample.get('task', sample.get('dataset', 'unknown')),
                    'length': sample.get('length', len(full_context)),
                    'language': sample.get('language', 'en')
                })
            except Exception as e:
                continue

        return normalized

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Create示例Data"""
        return [
            {
                "context": "The quick brown fox jumps over the lazy dog. " * 100 +
                          "\n\nQuestion: What animal jumps over the dog?",
                "answers": ["fox", "the fox", "brown fox"],
                "task": "narrativeqa",
                "length": 5000,
                "language": "en"
            },
            {
                "context": "In a scientific study published in Nature, researchers found that " +
                          "climate change affects biodiversity significantly. " * 50 +
                          "\n\nQuestion: What does climate change affect?",
                "answers": ["biodiversity", "species diversity"],
                "task": "qasper",
                "length": 3000,
                "language": "en"
            },
            {
                "context": "Document 1: Paris is the capital of France.\n" +
                          "Document 2: Berlin is the capital of Germany.\n" +
                          "Document 3: London is the capital of UK.\n" * 20 +
                          "\n\nQuestion: What is the capital of France?",
                "answers": ["Paris"],
                "task": "hotpotqa",
                "length": 2000,
                "language": "en"
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format LongBench 样本"""
        context = sample.get('context', '')

        if include_answer:
            answers = sample.get('answers', [])
            answer = answers[0] if answers else ""
            return f"{context}\n\nAnswer: {answer}"
        else:
            return context

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt"""
        task = sample.get('task', 'qa')

        # based on任务类型选择指令
        if 'summary' in task or 'summarization' in task:
            instruction = "Please summarize the following document:\n\n"
        elif 'code' in task:
            instruction = "Complete the following code:\n\n"
        else:
            instruction = "Answer the following question based on the given context:\n\n"

        return instruction + self.format_prompt(sample, include_answer=False)

    def parse_response(self, response: str) -> str:
        """Parse响应"""
        return response.strip()

    def check_answer(self, predicted: str, correct: str) -> bool:
        """CheckAnswer - use包含匹配"""
        if not predicted:
            return False

        # correct 可能isAnswer列表字符串表示
        if isinstance(correct, str):
            try:
                correct_list = eval(correct) if correct.startswith('[') else [correct]
            except:
                correct_list = [correct]
        else:
            correct_list = correct if isinstance(correct, list) else [correct]

        predicted_lower = normalize_text(predicted.lower())

        return any(normalize_text(ans.lower()) in predicted_lower for ans in correct_list)

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """GetCorrect answer"""
        answers = sample.get('answers', [])
        return str(answers)

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get任务类型"""
        return sample.get('task', 'unknown')
