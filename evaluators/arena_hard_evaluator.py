"""
Arena-Hard Evaluator
Arena-Hard 创意写作/指令遵循Evaluation器

Arena-Hard is一高质量 LLM Evaluation基准，use LLM-as-a-judge 方式评估：
- 软件工程问题
- 数学Question
- 创意写作任务
- 指令遵循能力

特点:
- use LLM 作is裁判进行 pairwise 对比
- 高度相关人类偏好 (89.1% agreement)
- 适合内部Model A/B Test

Data来源: HuggingFace lmarena-ai/arena-hard-auto
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator


class ArenaHardEvaluator(BaseEvaluator):
    """
    Arena-Hard Evaluator

    Data格式:
    {
        "question_id": "xxx",
        "category": "creative_writing" / "coding" / "math" / etc,
        "turns": [
            {"content": "用户提问..."}
        ],
        "reference_answer": "optional参考Answer"
    }

    Evaluation方式:
    - use LLM-as-judge 评估响应质量
    - Score维度: has用性、相关性、准确性、深度、创造性、细节
    """

    # Evaluation类别
    CATEGORIES = [
        "creative_writing",
        "coding",
        "math",
        "reasoning",
        "extraction",
        "stem",
        "humanities",
        "roleplay"
    ]

    def __init__(
        self,
        dataset_name: str = "arena_hard",
        dataset_path: str = "datasets/arena_hard",
        num_shots: int = 0,  # Arena-Hard 通常 0-shot
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

        # Arena-Hard 专用Score模板
        self.judge_prompt_template = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]

Please rate the response on a scale of 1-10, where:
- 1-3: Poor (unhelpful, incorrect, or off-topic)
- 4-5: Below Average (partially helpful but with significant issues)
- 6-7: Average (helpful but could be improved)
- 8-9: Good (helpful, accurate, and well-structured)
- 10: Excellent (exceptionally helpful, insightful, and comprehensive)

Please output your rating in the following format:
Rating: [[X]]

Then provide a brief explanation of your rating."""

    def load_dataset(self, subset: str | None = None) -> list[dict[str, Any]]:
        """
        Load Arena-Hard Dataset
        """
        samples = []

        possible_files = [
            os.path.join(self.dataset_path, "arena_hard.json"),
            os.path.join(self.dataset_path, "test.json"),
            os.path.join(self.dataset_path, "data.json"),
            os.path.join(self.dataset_path, "arena_hard.jsonl"),
        ]

        for filepath in possible_files:
            if os.path.exists(filepath):
                try:
                    if filepath.endswith('.jsonl'):
                        with open(filepath, encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    samples.append(json.loads(line))
                    else:
                        with open(filepath, encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            samples = data
                        elif isinstance(data, dict) and 'data' in data:
                            samples = data['data']
                    break
                except Exception as e:
                    print(f"Load {filepath} 失败: {e}")

        if not samples:
            samples = self._create_sample_data()

        # Standardize格式
        samples = self._normalize_samples(samples)

        # 按类别Filter
        if subset:
            samples = [s for s in samples if s.get('category', '').lower() == subset.lower()]

        random.shuffle(samples)

        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def _normalize_samples(self, samples: list[dict]) -> list[dict]:
        """Standardize样本格式"""
        normalized = []
        for i, s in enumerate(samples):
            # 提取问题
            question = ""
            if "turns" in s and len(s["turns"]) > 0:
                question = s["turns"][0].get("content", "")
            elif "question" in s:
                question = s["question"]
            elif "prompt" in s:
                question = s["prompt"]

            sample = {
                "id": s.get("question_id", s.get("id", f"arena_{i}")),
                "question": question,
                "category": s.get("category", "general"),
                "reference_answer": s.get("reference_answer", ""),
            }
            normalized.append(sample)
        return normalized

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Create Arena-Hard 示例Data"""
        return [
            {
                "id": "arena_1",
                "question": "Write a creative short story about a robot who discovers emotions for the first time. The story should be engaging, have a clear narrative arc, and explore themes of identity and consciousness.",
                "category": "creative_writing",
                "reference_answer": ""
            },
            {
                "id": "arena_2",
                "question": "Explain the concept of quantum entanglement to a 10-year-old using simple analogies and examples. Make it fun and engaging while remaining scientifically accurate.",
                "category": "stem",
                "reference_answer": ""
            },
            {
                "id": "arena_3",
                "question": "You are a medieval knight who has just been transported to modern-day New York City. Describe your first day, your reactions to technology, and how you try to adapt. Stay in character throughout.",
                "category": "roleplay",
                "reference_answer": ""
            },
            {
                "id": "arena_4",
                "question": "Write a Python function that implements a binary search tree with insert, delete, and search operations. Include comprehensive error handling and docstrings.",
                "category": "coding",
                "reference_answer": ""
            },
            {
                "id": "arena_5",
                "question": "A farmer has 100 meters of fencing and wants to enclose a rectangular area along a river (which serves as one side, requiring no fence). What dimensions will maximize the enclosed area? Show your work step by step.",
                "category": "math",
                "reference_answer": "Width = 50m, Length = 25m, Maximum Area = 1250 sq meters"
            },
            {
                "id": "arena_6",
                "question": "Compare and contrast the philosophical approaches of Stoicism and Existentialism. Discuss their views on meaning, suffering, and human agency.",
                "category": "humanities",
                "reference_answer": ""
            },
            {
                "id": "arena_7",
                "question": "Analyze the following argument and identify any logical fallacies: 'Everyone I know uses this product, so it must be good. Besides, the company has been around for 50 years, and old companies are always trustworthy.'",
                "category": "reasoning",
                "reference_answer": ""
            },
            {
                "id": "arena_8",
                "question": "Extract all the key dates, people, and events mentioned in this text: 'On July 20, 1969, Neil Armstrong and Buzz Aldrin became the first humans to walk on the Moon as part of NASA's Apollo 11 mission. Michael Collins orbited above in the command module.'",
                "category": "extraction",
                "reference_answer": "Dates: July 20, 1969. People: Neil Armstrong, Buzz Aldrin, Michael Collins. Events: Moon landing, Apollo 11 mission."
            },
            {
                "id": "arena_9",
                "question": "Write a haiku about artificial intelligence that captures both its promise and potential dangers. Then write a limerick on the same theme.",
                "category": "creative_writing",
                "reference_answer": ""
            },
            {
                "id": "arena_10",
                "question": "Design a REST API for a library management system. Include endpoints for books, users, and borrowing operations. Specify HTTP methods, request/response formats, and error handling.",
                "category": "coding",
                "reference_answer": ""
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format样本is Prompt"""
        question = sample.get('question', '')
        return question

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt"""
        return self.format_prompt(sample, include_answer=False)

    def parse_response(self, response: str) -> str:
        """ParseModel响应 - Arena-Hard Return完整响Apply于 judge"""
        return response.strip()

    def check_answer(self, predicted: str, correct: str) -> bool:
        """
        CheckAnswer - Arena-Hard use质量Score

        注意: Arena-Hard 主要依赖 LLM-as-judge，
        这里Return True 表示hashas效响应，实际质量由 judge 评估
        """
        # ifhas响应内容，就算has效
        return bool(predicted and len(predicted.strip()) > 20)

    def build_judge_prompt(self, question: str, answer: str) -> str:
        """Build judge 评估 prompt"""
        return self.judge_prompt_template.format(
            question=question,
            answer=answer
        )

    def parse_judge_score(self, judge_response: str) -> float | None:
        """从 judge 响应in提取Score"""
        # 匹配 Rating: [[X]] 格式
        match = re.search(r'Rating:\s*\[\[(\d+(?:\.\d+)?)\]\]', judge_response)
        if match:
            return float(match.group(1))

        # 回退：匹配任意数字Score
        match = re.search(r'(\d+(?:\.\d+)?)\s*/?\s*10', judge_response)
        if match:
            return float(match.group(1))

        return None

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get样本类别"""
        return sample.get('category', 'general')

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """Get参考Answer"""
        return sample.get('reference_answer', '')
