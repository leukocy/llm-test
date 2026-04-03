"""
GPQA Evaluator
GPQA (Graduate-Level Google-Proof Q&A) DatasetEvaluator

GPQA is一研究生级别科学问答Dataset，包含生物、物理、化学etc.领域高难度问题。
由专家精心设计，用于TestModel高级推理能力。
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_choice_answer


class GPQAEvaluator(BaseEvaluator):
    """
    GPQA DatasetEvaluator

    Data格式 (GPQA Diamond):
    {
        "question": "In quantum mechanics, what is the expectation value of...",
        "choices": ["Option A", "Option B", "Option C", "Option D"],
        "answer": 2,  // 0-indexed, C
        "domain": "physics"  // physics, chemistry, biology
    }

    GPQA has三难度级别:
    - GPQA Extended: 全部问题
    - GPQA Main: inetc.难度子集
    - GPQA Diamond: 最高难度子集 (约450题)
    """

    # 学科领域分类
    DOMAINS = {
        "physics": "物理学",
        "chemistry": "化学",
        "biology": "生物学",
        "organic_chemistry": "has机化学",
        "genetics": "遗传学",
        "quantum_mechanics": "量子力学"
    }

    def __init__(
        self,
        dataset_name: str = "gpqa",
        dataset_path: str = "datasets/gpqa",
        num_shots: int = 3,  # GPQA 推荐较少 few-shot
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
        Load GPQA Dataset

        Args:
            subset: optional "diamond", "main", "extended" (default diamond)

        Returns:
            样本列表
        """
        samples = []

        # 确定难度级别
        level = subset if subset in ['diamond', 'main', 'extended'] else 'diamond'

        # 尝试多种文件格式
        possible_files = [
            os.path.join(self.dataset_path, f"gpqa_{level}.json"),
            os.path.join(self.dataset_path, f"{level}.json"),
            os.path.join(self.dataset_path, f"gpqa_{level}.jsonl"),
            os.path.join(self.dataset_path, "gpqa_diamond.json"),
            os.path.join(self.dataset_path, "test.json"),
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
                        elif isinstance(data, dict) and 'examples' in data:
                            samples = data['examples']
                    break
                except Exception as e:
                    print(f"Load {filepath} 失败: {e}")

        # if没has本地Data，Create示例
        if not samples:
            samples = self._create_sample_data()

        # StandardizeData格式
        samples = self._normalize_samples(samples)

        # 随机打乱
        random.shuffle(samples)

        # Calculate总需Sample count
        total_needed = self.num_shots + (self.max_samples if self.max_samples else len(samples))
        if len(samples) > total_needed:
            samples = samples[:total_needed]

        # Set few-shot 示例
        if self.num_shots > 0:
            self.few_shot_examples = samples[:self.num_shots]
            samples = samples[self.num_shots:]

        # 限制TestSample count
        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        self.samples = samples
        return samples

    def _normalize_samples(self, samples: list[dict]) -> list[dict]:
        """Standardize样本格式"""
        normalized = []
        for sample in samples:
            # Processnot同字段名
            question = sample.get('question', sample.get('Question', ''))

            # ProcessOptions - 可能is列表or单独字段
            if 'choices' in sample:
                choices = sample['choices']
            elif 'options' in sample:
                choices = sample['options']
            else:
                # GPQA 原始格式可能use A/B/C/D 字段
                choices = [
                    sample.get('A', sample.get('Incorrect Answer 1', '')),
                    sample.get('B', sample.get('Incorrect Answer 2', '')),
                    sample.get('C', sample.get('Incorrect Answer 3', '')),
                    sample.get('D', sample.get('Correct Answer', ''))
                ]

            # ProcessAnswer
            answer = sample.get('answer', sample.get('Answer', 0))
            if isinstance(answer, str):
                if answer.upper() in 'ABCD':
                    answer = ord(answer.upper()) - ord('A')
                elif answer.isdigit():
                    answer = int(answer)

            # Process领域
            domain = sample.get('domain', sample.get('Domain', sample.get('subdomain', 'unknown')))

            normalized.append({
                'question': question,
                'choices': choices[:4] if len(choices) >= 4 else choices + [''] * (4 - len(choices)),
                'answer': answer,
                'domain': domain
            })

        return normalized

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """Create示例Data"""
        return [
            {
                "question": "In quantum mechanics, the Heisenberg uncertainty principle states that which pairs of physical properties cannot be simultaneously measured with arbitrary precision?",
                "choices": [
                    "Position and velocity",
                    "Position and momentum",
                    "Energy and time",
                    "Both B and C are correct"
                ],
                "answer": 3,  # D
                "domain": "physics"
            },
            {
                "question": "Which of the following best describes the mechanism of CRISPR-Cas9 gene editing?",
                "choices": [
                    "It uses RNA interference to silence genes",
                    "It creates double-strand breaks guided by a complementary RNA sequence",
                    "It uses restriction enzymes to cut at specific sequences",
                    "It modifies histones to change gene expression"
                ],
                "answer": 1,  # B
                "domain": "biology"
            },
            {
                "question": "In organic chemistry, what is the main product when benzene undergoes Friedel-Crafts acylation with acetyl chloride in the presence of AlCl3?",
                "choices": [
                    "Toluene",
                    "Acetophenone",
                    "Phenol",
                    "Anisole"
                ],
                "answer": 1,  # B
                "domain": "chemistry"
            },
            {
                "question": "The Hardy-Weinberg equilibrium in population genetics assumes which of the following conditions?",
                "choices": [
                    "Natural selection favors heterozygotes",
                    "Random mating, no mutation, no migration, large population, no selection",
                    "Genetic drift is the primary mechanism of evolution",
                    "Inbreeding increases genetic diversity"
                ],
                "answer": 1,  # B
                "domain": "biology"
            },
            {
                "question": "In thermodynamics, the Gibbs free energy change (ΔG) for a spontaneous process at constant temperature and pressure must be:",
                "choices": [
                    "Positive",
                    "Negative",
                    "Zero",
                    "Equal to the enthalpy change"
                ],
                "answer": 1,  # B
                "domain": "chemistry"
            }
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """
        Format GPQA 样本is Prompt
        """
        question = sample.get('question', '')
        choices = sample.get('choices', [])

        # 确保has 4 Options
        while len(choices) < 4:
            choices.append("")

        prompt_lines = [
            f"Question: {question}",
            f"A. {choices[0]}",
            f"B. {choices[1]}",
            f"C. {choices[2]}",
            f"D. {choices[3]}",
        ]

        if include_answer:
            answer_idx = sample.get('answer', 0)
            if isinstance(answer_idx, str):
                answer_letter = answer_idx.upper()
            else:
                answer_letter = chr(ord('A') + answer_idx)
            prompt_lines.append(f"Answer: {answer_letter}")
        else:
            prompt_lines.append("Answer:")

        return "\n".join(prompt_lines)

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """
        Build完整 GPQA prompt
        """
        domain = sample.get('domain', 'science')
        domain_display = self.DOMAINS.get(domain, domain.replace('_', ' ').title())

        # 系统指令 - 强调need专业知识and推理
        instruction = (
            f"The following are challenging {domain_display} questions that require graduate-level knowledge. "
            "Think carefully and choose the best answer.\n\n"
        )

        # Few-shot 示例
        examples = []
        for example in self.few_shot_examples[:self.num_shots]:
            examples.append(self.format_prompt(example, include_answer=True))

        # 待评估问题
        question = self.format_prompt(sample, include_answer=False)

        full_prompt = instruction + "\n\n".join(examples)
        if examples:
            full_prompt += "\n\n"
        full_prompt += question

        return full_prompt

    def parse_response(self, response: str) -> str:
        """ParseModel响应，提取Options字母"""
        return extract_choice_answer(response, ['A', 'B', 'C', 'D'])

    def check_answer(self, predicted: str, correct: str) -> bool:
        """CheckAnsweris否正确"""
        if not predicted:
            return False

        # if correct is数字，Convertis字母
        if isinstance(correct, int) or (isinstance(correct, str) and correct.isdigit()):
            correct = chr(ord('A') + int(correct))

        return predicted.upper() == correct.upper()

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get样本领域分类"""
        return sample.get('domain', 'unknown')

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """GetCorrect answer (Convertis字母)"""
        answer = sample.get('answer', 0)
        if isinstance(answer, int):
            return chr(ord('A') + answer)
        return str(answer).upper()
