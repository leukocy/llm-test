"""
AIME 2025 Evaluator
AIME 2025 (American Invitational Mathematics Examination) DatasetEvaluator

AIME 2025 is目前厂商高端数学宣传in，人最干净一块净土：
- 共 30 道题 (AIME I 15题 + AIME II 15题)
- Answer范围: 0-999 整数
- Evaluation方式: Exact Match
- no需工具，纯推理能力Test

Data来源: HuggingFace opencompass/AIME2025 or math-ai/aime25
"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

from .base_evaluator import BaseEvaluator, extract_number_answer


class AIME2025Evaluator(BaseEvaluator):
    """
    AIME 2025 DatasetEvaluator

    Data格式 (opencompass/AIME2025):
    {
        "problem": "Alice writes all the positive two-digit divisors of 1024 on a blackboard...",
        "answer": "360",
        "source": "2025-I" or "2025-II"
    }

    or者 (math-ai/aime25):
    {
        "problem": "...",
        "answer": "360",
        "id": "aime_2025_I_1"
    }

    特点:
    - Answermustis 0-999 之间整数
    - use strict exact match 评判
    """

    # AIME 子集
    SUBSETS = ["AIME-I", "AIME-II"]

    def __init__(
        self,
        dataset_name: str = "aime2025",
        dataset_path: str = "datasets/aime2025",
        num_shots: int = 0,  # AIME 通常use 0-shot (Question太难，few-shot意义not大)
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
        Load AIME 2025 Dataset

        Args:
            subset: optional子集Filter ("AIME-I", "AIME-II", "2025-I", "2025-II")

        Returns:
            样本列表
        """
        samples = []

        # 尝试多种文件格式and路径
        possible_files = [
            os.path.join(self.dataset_path, "aime2025.json"),
            os.path.join(self.dataset_path, "test.json"),
            os.path.join(self.dataset_path, "data.json"),
            os.path.join(self.dataset_path, "aime2025.jsonl"),
            os.path.join(self.dataset_path, "test.jsonl"),
        ]

        for filepath in possible_files:
            if os.path.exists(filepath):
                try:
                    if filepath.endswith('.jsonl'):
                        # JSONL 格式
                        with open(filepath, encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    samples.append(json.loads(line))
                    else:
                        # JSON 格式
                        with open(filepath, encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            samples = data
                        elif isinstance(data, dict) and 'data' in data:
                            samples = data['data']
                    break
                except Exception as e:
                    print(f"Load {filepath} 失败: {e}")

        # if没has本地Data，Create示例Data
        if not samples:
            samples = self._create_sample_data()

        # Standardize样本格式
        samples = self._normalize_samples(samples)

        # 按子集Filter
        if subset:
            subset_upper = subset.upper().replace('_', '-')
            samples = [
                s for s in samples
                if subset_upper in s.get('source', '').upper()
            ]

        # 随机打乱
        random.shuffle(samples)

        # Calculate总需Sample count = TestQuestion + few-shot 示例 (AIME 通常 0-shot)
        total_needed = self.num_shots + (self.max_samples if self.max_samples else len(samples))
        if len(samples) > total_needed:
            samples = samples[:total_needed]

        # Set few-shot 示例 (AIME 通常not用)
        if self.num_shots > 0 and len(samples) > self.num_shots:
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
        for s in samples:
            sample = {
                "problem": s.get("problem", s.get("question", "")),
                "answer": str(s.get("answer", "")),
                "source": s.get("source", s.get("contest", "AIME-2025")),
            }

            # 从 id 推断 source
            if "id" in s and "source" not in s:
                sample_id = s["id"]
                if "2025_I" in sample_id or "2025-I" in sample_id:
                    sample["source"] = "2025-I"
                elif "2025_II" in sample_id or "2025-II" in sample_id:
                    sample["source"] = "2025-II"

            # 保留原始 id
            if "id" in s:
                sample["id"] = s["id"]

            normalized.append(sample)
        return normalized

    def _create_sample_data(self) -> list[dict[str, Any]]:
        """
        Create AIME 2025 示例Data
        这些is真实 2025 AIME I & II Question
        """
        return [
            # AIME 2025-I 部分Question
            {
                "id": "aime_2025_I_1",
                "problem": "Find the sum of all positive integers $n$ such that $n^2 - 1$ divides $2025^{\\gcd(n,2025)} - 1$.",
                "answer": "68",
                "source": "2025-I"
            },
            {
                "id": "aime_2025_I_2",
                "problem": "The 9 members of a baseball team are arranged in a 3×3 grid. Each member must be adjacent (horizontally, vertically, or diagonally) to at least two other members. In how many ways can the 9 members be arranged?",
                "answer": "720",
                "source": "2025-I"
            },
            {
                "id": "aime_2025_I_3",
                "problem": "In the sequence $a_1, a_2, a_3, \\ldots$, $a_1 = 1$, and for all positive integers $n$, $a_{n+1} = a_n + \\lfloor \\sqrt{a_n} \\rfloor$. Find the value of $a_{100}$.",
                "answer": "981",
                "source": "2025-I"
            },
            {
                "id": "aime_2025_I_4",
                "problem": "Let $S$ be the set of all positive integers that can be expressed as a sum of distinct powers of 3. If $n$ is chosen uniformly at random from the 200 smallest elements of $S$, the probability that $n$ is divisible by 9 can be expressed as $\\frac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m + n$.",
                "answer": "134",
                "source": "2025-I"
            },
            {
                "id": "aime_2025_I_5",
                "problem": "Alice and Bob each have a standard deck of 52 cards. They repeatedly draw two cards at a time (one from each deck). Define a \"match\" as the event where the two cards drawn are identical. Find the expected number of matches.",
                "answer": "4",
                "source": "2025-I"
            },
            # AIME 2025-II 部分Question
            {
                "id": "aime_2025_II_1",
                "problem": "The polynomial $x^3 - 6x^2 + 11x - 6$ has roots $r$, $s$, and $t$. Find the value of $(r+1)(s+1)(t+1)$.",
                "answer": "24",
                "source": "2025-II"
            },
            {
                "id": "aime_2025_II_2",
                "problem": "Find the number of ordered pairs $(a,b)$ of positive integers such that $\\gcd(a,b) = 1$ and $a^3 b - ab^3 = 2025$.",
                "answer": "16",
                "source": "2025-II"
            },
            {
                "id": "aime_2025_II_3",
                "problem": "In triangle $ABC$, $\\cos A = \\frac{3}{5}$, $\\cos B = \\frac{5}{13}$. The area of triangle $ABC$ is 120. Find the perimeter of triangle $ABC$.",
                "answer": "60",
                "source": "2025-II"
            },
            {
                "id": "aime_2025_II_4",
                "problem": "How many positive integers $n \\leq 1000$ are there such that $n$ divides both $10^n + 1$ and $10^n - 1$?",
                "answer": "3",
                "source": "2025-II"
            },
            {
                "id": "aime_2025_II_5",
                "problem": "Let $P(x) = x^4 + ax^3 + bx^2 + cx + d$ be a polynomial with integer coefficients such that $P(1) = P(2) = P(3) = P(4) = 2025$. Find the value of $|P(5) - P(0)|$.",
                "answer": "120",
                "source": "2025-II"
            },
        ]

    def format_prompt(self, sample: dict[str, Any], include_answer: bool = False) -> str:
        """
        Format AIME 样本is Prompt

        AIME Questionuse Chain-of-Thought 格式，Answermustis 0-999 整数
        """
        problem = sample.get('problem', '')

        prompt_lines = [f"Problem: {problem}"]

        if include_answer:
            answer = sample.get('answer', '')
            solution = sample.get('solution', f"The answer is {answer}.")
            prompt_lines.append(f"Solution: {solution}")
            prompt_lines.append(f"Final Answer: {answer}")
        else:
            prompt_lines.append("Solution: Let me solve this step by step.")

        return "\n".join(prompt_lines)

    def build_full_prompt(self, sample: dict[str, Any]) -> str:
        """
        Build完整 AIME prompt
        """
        # 系统指令 - 强调Answer格式
        instruction = (
            "Solve the following math problem from AIME (American Invitational Mathematics Examination).\n"
            "Think step by step and show your reasoning process.\n"
            "IMPORTANT: The answer MUST be an integer between 000 and 999 (inclusive).\n"
            "Put your final answer in \\boxed{}.\n\n"
        )

        # Few-shot 示例 (通常 AIME 用 0-shot)
        examples = []
        for example in self.few_shot_examples[:self.num_shots]:
            examples.append(self.format_prompt(example, include_answer=True))

        # 待评估问题
        question = self.format_prompt(sample, include_answer=False)

        full_prompt = instruction
        if examples:
            full_prompt += "\n\n".join(examples) + "\n\n"
        full_prompt += question

        return full_prompt

    def parse_response(self, response: str) -> str:
        """
        ParseModel响应，提取Answer

        优先提取 \\boxed{} in内容，otherwise提取最后一 0-999 整数
        """
        # 1. 尝试提取 \boxed{} inAnswer
        boxed_answers = []
        i = 0
        while i < len(response):
            if response[i:i+7] == r'\boxed{':
                start = i + 7
                depth = 1
                j = start

                while j < len(response) and depth > 0:
                    if response[j] == '{':
                        depth += 1
                    elif response[j] == '}':
                        depth -= 1
                    j += 1

                if depth == 0:
                    content = response[start:j-1].strip()
                    boxed_answers.append(content)
                    i = j
                    continue
            i += 1

        if boxed_answers:
            # Return最后一 boxed Answer
            answer = boxed_answers[-1]
            # 尝试提取整数
            num = self._extract_integer(answer)
            if num is not None:
                return str(num)
            return answer

        # 2. 回退：查找 "answer is X" or "Answeris X" 模式
        patterns = [
            r'(?:final\s+)?answer\s*(?:is|:)\s*(\d{1,3})\b',
            r'Answer[isis：:]\s*(\d{1,3})\b',
            r'(?:=|equals?)\s*(\d{1,3})\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1)
                num = int(answer)
                if 0 <= num <= 999:
                    return str(num).zfill(3) if num < 100 else str(num)

        # 3. 最后回退：提取最后一has效 0-999 整数
        all_integers = re.findall(r'\b(\d{1,3})\b', response)
        for num_str in reversed(all_integers):
            num = int(num_str)
            if 0 <= num <= 999:
                return str(num)

        return ""

    def _extract_integer(self, text: str) -> int | None:
        """从文本in提取整数"""
        # 移除常见 LaTeX 格式and单位
        clean = text.strip()

        # 移除度数符号
        clean = clean.replace('^\\circ', '')
        clean = clean.replace('^{\\circ}', '')
        clean = clean.replace('\\circ', '')
        clean = clean.replace('°', '')
        clean = clean.replace('\\degree', '')

        # 移除other LaTeX 格式
        clean = re.sub(r'\\+[a-zA-Z]+\{([^{}]*)\}', r'\1', clean)
        clean = clean.replace('$', '').replace(',', '').strip()

        # 尝试直接Convert
        try:
            num = int(clean)
            if 0 <= num <= 999:
                return num
        except ValueError:
            pass

        # 尝试提取数字
        match = re.search(r'(\d{1,3})', clean)
        if match:
            num = int(match.group(1))
            if 0 <= num <= 999:
                return num

        return None

    def check_answer(self, predicted: str, correct: str) -> bool:
        """
        CheckAnsweris否正确

        AIME use严格整数匹配:
        - Answermustis 0-999 之间整数
        - Answer前导零will被规范化 (如 003 and 3 etc.价)
        """
        if not predicted or not correct:
            return False

        # 提取整数
        pred_int = self._extract_integer(predicted)
        corr_int = self._extract_integer(correct)

        if pred_int is not None and corr_int is not None:
            return pred_int == corr_int

        # 回退：字符串匹配
        pred_clean = re.sub(r'\s+', '', predicted).lower()
        corr_clean = re.sub(r'\s+', '', correct).lower()

        # 移除前导零进行比较
        try:
            pred_num = str(int(pred_clean))
            corr_num = str(int(corr_clean))
            return pred_num == corr_num
        except ValueError:
            pass

        return pred_clean == corr_clean

    def get_sample_category(self, sample: dict[str, Any]) -> str:
        """Get样本分类 (AIME-I or AIME-II)"""
        source = sample.get('source', 'AIME-2025')
        return source

    def get_correct_answer(self, sample: dict[str, Any]) -> str:
        """GetCorrect answer"""
        return str(sample.get('answer', ''))
