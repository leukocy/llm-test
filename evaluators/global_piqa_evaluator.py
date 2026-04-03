"""
Global PIQA Evaluator
Global PIQA 多语言物理常识推理Evaluation器

Global PIQA is一多语言常识推理基准：
- 覆盖 116 种语言变体
- 由 335 位研究者手工Build
- 50%+ 包含本地化文化内容
- Test物理常识推理能力

特点:
- 非翻译，原生多语言
- 文化特定日常物理知识
- 二选一格式

Data来源: HuggingFace mrlbenchmarks/global-piqa-nonparallel
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_choice_answer


class GlobalPIQAEvaluator(BaseEvaluator):
    """
    Global PIQA Evaluator

    Data格式:
    {
        "goal": "如何保持米饭not粘锅",
        "sol1": "in煮饭前加一点油",
        "sol2": "煮饭时not断搅拌",
        "label": 0,  # 0 表示 sol1 正确, 1 表示 sol2 正确
        "language": "Chinese",
        "country": "China"
    }

    Score方式:
    - 二选一Accuracy (Accuracy)
    - 随机基准 50%
    """

    def __init__(
        self,
        dataset_name: str = "global_piqa",
        dataset_path: str = "datasets/global_piqa",
        num_shots: int = 3,  # use few-shot HelpModel理解格式
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
        """
        Load Global PIQA Dataset

        Args:
            subset: optional语言Filter (如 "Chinese", "English", "Japanese")
        """
        samples = []

        possible_files = [
            os.path.join(self.dataset_path, "global_piqa.json"),
            os.path.join(self.dataset_path, "test.json"),
            os.path.join(self.dataset_path, "data.json"),
            os.path.join(self.dataset_path, "global_piqa.jsonl"),
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

        # 按语言Filter
        if subset:
            samples = [
                s for s in samples
                if s.get('language', '').lower() == subset.lower()
            ]

        random.shuffle(samples)

        # Calculate总需Sample count
        total_needed = self.num_shots + (self.max_samples if self.max_samples else len(samples))
        if len(samples) > total_needed:
            samples = samples[:total_needed]

        # Set few-shot 示例
        if self.num_shots > 0 and len(samples) > self.num_shots:
            self.few_shot_examples = samples[:self.num_shots]
            samples = samples[self.num_shots:]

        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def _normalize_samples(self, samples: list[dict]) -> list[dict]:
        """Standardize样本格式"""
        normalized = []
        for i, s in enumerate(samples):
            sample = {
                "id": s.get("id", f"piqa_{i}"),
                "goal": s.get("goal", s.get("question", "")),
                "sol1": s.get("sol1", s.get("solution1", s.get("choice_a", ""))),
                "sol2": s.get("sol2", s.get("solution2", s.get("choice_b", ""))),
                "label": s.get("label", 0),
                "language": s.get("language", s.get("lang", "English")),
                "country": s.get("country", s.get("region", ""))
            }
            normalized.append(sample)
        return normalized

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Create Global PIQA 示例Data (多语言)"""
        return [
            # 英语示例
            {
                "id": "piqa_en_1",
                "goal": "How to keep bread fresh for longer?",
                "sol1": "Store it in the refrigerator",
                "sol2": "Keep it in a bread box at room temperature",
                "label": 1,
                "language": "English",
                "country": "USA"
            },
            {
                "id": "piqa_en_2",
                "goal": "How to remove a stuck ring from your finger?",
                "sol1": "Apply soap and water to lubricate",
                "sol2": "Heat the ring with a lighter",
                "label": 0,
                "language": "English",
                "country": "UK"
            },
            {
                "id": "piqa_en_3",
                "goal": "How to prevent onions from making you cry while cutting?",
                "sol1": "Chill the onion in the refrigerator before cutting",
                "sol2": "Heat the onion in the microwave before cutting",
                "label": 0,
                "language": "English",
                "country": "Australia"
            },
            # in文示例
            {
                "id": "piqa_zh_1",
                "goal": "如何煮出not粘锅米饭？",
                "sol1": "in煮饭前往锅里加一点油or黄油",
                "sol2": "用冷水而notis热水开始煮",
                "label": 0,
                "language": "Chinese",
                "country": "China"
            },
            {
                "id": "piqa_zh_2",
                "goal": "如何让饺子皮notwill粘in一起？",
                "sol1": "in饺子皮on撒一些面粉",
                "sol2": "把饺子皮放in温水里泡一under",
                "label": 0,
                "language": "Chinese",
                "country": "China"
            },
            {
                "id": "piqa_zh_3",
                "goal": "如何快速剥蒜？",
                "sol1": "用刀面轻轻拍一under蒜瓣",
                "sol2": "把蒜放in冰箱里冻一will儿",
                "label": 0,
                "language": "Chinese",
                "country": "China"
            },
            # 日语示例
            {
                "id": "piqa_ja_1",
                "goal": "お米を洗う正しい方法は？",
                "sol1": "水を入れて軽くかき混ぜ、水を捨てる作業を2-3回繰り返す",
                "sol2": "一度たっぷりの水で強くこすり洗いする",
                "label": 0,
                "language": "Japanese",
                "country": "Japan"
            },
            # 西班牙语示例
            {
                "id": "piqa_es_1",
                "goal": "¿Cómo evitar que el aguacate se oxide?",
                "sol1": "Rociar jugo de limón sobre la superficie cortada",
                "sol2": "Dejarlo al sol para que se seque",
                "label": 0,
                "language": "Spanish",
                "country": "Mexico"
            },
            # 法语示例
            {
                "id": "piqa_fr_1",
                "goal": "Comment garder le fromage frais plus longtemps?",
                "sol1": "L'envelopper dans du papier sulfurisé puis dans du plastique",
                "sol2": "Le mettre dans un sac en plastique hermétique directement",
                "label": 0,
                "language": "French",
                "country": "France"
            },
            # 阿拉伯语示例
            {
                "id": "piqa_ar_1",
                "goal": "كيف تحافظ على الخبز طازجاً؟",
                "sol1": "لفه في قطعة قماش قطنية",
                "sol2": "وضعه في كيس بلاستيكي محكم الإغلاق",
                "label": 0,
                "language": "Arabic",
                "country": "Egypt"
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """Format样本is Prompt"""
        goal = sample.get('goal', '')
        sol1 = sample.get('sol1', '')
        sol2 = sample.get('sol2', '')

        prompt = f"""Goal: {goal}

Which solution is better to achieve the goal?
A. {sol1}
B. {sol2}

Answer:"""

        if include_answer:
            answer = "A" if sample.get('label', 0) == 0 else "B"
            prompt += f" {answer}"

        return prompt

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """Build完整 prompt"""
        instruction = (
            "You are given a goal and two possible solutions. "
            "Choose the solution that better achieves the goal. "
            "Answer with only 'A' or 'B'.\n\n"
        )

        # Few-shot 示例
        examples = []
        for example in self.few_shot_examples[:self.num_shots]:
            examples.append(self.format_prompt(example, include_answer=True))

        # 待Evaluation问题
        question = self.format_prompt(sample, include_answer=False)

        full_prompt = instruction
        if examples:
            full_prompt += "\n\n".join(examples) + "\n\n"
        full_prompt += question

        return full_prompt

    def parse_response(self, response: str) -> str:
        """ParseModel响应，提取Options A/B"""
        response = response.strip().upper()

        # 直接匹配 A or B
        if response.startswith('A'):
            return 'A'
        if response.startswith('B'):
            return 'B'

        # 尝试从完整响应in提取
        answer = extract_choice_answer(response, choices=['A', 'B'])
        if answer:
            return answer

        # 查找 "answer is A/B" 模式
        match = re.search(r'(?:answer|choice|solution)[:\s]*([AB])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 最后尝试：取一 A or B
        for char in response[:20]:
            if char in ['A', 'B']:
                return char

        return ""

    def check_answer(self, predicted: str, correct: str) -> bool:
        """CheckAnsweris否正确"""
        if not predicted or not correct:
            return False

        pred = predicted.strip().upper()
        corr = correct.strip().upper()

        return pred == corr

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get样本类别 (语言)"""
        return sample.get('language', 'Unknown')

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """GetCorrect answer"""
        label = sample.get('label', 0)
        return 'A' if label == 0 else 'B'
