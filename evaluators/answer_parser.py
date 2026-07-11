"""
Centralized Answer Parsers for Quality Evaluation

Provides enhanced answer extraction for multiple-choice, math, code, and text answers.
Largely inspired by / ported from EvalScope's approach:

- Multi-choice: ``ANSWER: X`` strict format + Chinese ``答案：X`` + fallback
  (ref: evalscope/utils/multi_choices.py parse_answers / parse_answers_zh)
- Math: ``\\boxed{}`` extraction with brace-depth tracking, ``strip_answer_string``
  LaTeX normalization, and ``math_equal`` symbolic comparison via SymPy
  (ref: evalscope/metrics/math_parser.py extract_answer / math_equal)
- Code: markdown fence extraction
- Text: normalized / substring / fuzzy comparison

Supports SymPy as an optional dependency for symbolic equivalence.
"""

import importlib.util
import re
from difflib import SequenceMatcher
from math import isclose

# ---------------------------------------------------------------------------
# Optional dependency availability
# ---------------------------------------------------------------------------
_sympy_available = importlib.util.find_spec("sympy") is not None


# ===========================================================================
# MultiChoiceParser  (ref: EvalScope multi_choices.py)
# ===========================================================================


class MultiChoiceParser:
    """Enhanced multiple-choice answer extraction.

    Pattern priority (first match wins):
    1. ``ANSWER: X`` (strict, EvalScope-style)
    2. ``答案[:：]X`` (Chinese strict)
    3. ``The answer is X``, ``I choose X``, etc. (English verbose)
    4. ``答案是X``, ``选择X``, etc. (Chinese verbose)
    5. ``(A)``, ``（B）``, ``[C]``, ``A.`` (structured)
    6. Fallback: last valid uppercase letter (EvalScope _fallback_parse_answer)
    """

    def __init__(self):
        C = "A-Za-z"
        self._patterns: list[tuple[str, re.Pattern]] = [
            # --- EvalScope-style strict ANSWER: X (highest confidence) ---
            (
                "strict_answer",
                re.compile(rf"(?i)^ANSWER\s*:\s*([{C}])\s*(?:$|\n|\.)", re.MULTILINE),
            ),
            # Looser version for mid-text
            ("answer_colon", re.compile(rf"(?i)ANSWER\s*:\s*([{C}])(?:[^\w]|\n|$|\.)")),
            # --- Chinese strict 答案：X ---
            ("zh_answer", re.compile(rf"答案\s*[:：]\s*([{C}])")),
            # --- English verbose ---
            (
                "explicit_en",
                re.compile(
                    rf'(?:ANSWER|Answer|answer)\s*(?:is|:)\s*[<>"\']?\s*([{C}])\b',
                    re.IGNORECASE,
                ),
            ),
            (
                "choose",
                re.compile(
                    rf"(?:I\s+(?:choose|select|pick)|My answer is)\s*[（(]?\s*([{C}])\s*[）)]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "the_answer_is",
                re.compile(
                    rf"(?:The answer is|the correct answer is)\s*[：: ]?\s*[（(]?\s*([{C}])\b",
                    re.IGNORECASE,
                ),
            ),
            # --- Chinese verbose ---
            (
                "explicit_cn",
                re.compile(
                    rf"(?:答案是|选择|选项|正确答案为|答案为)\s*[（(]?\s*([{C}])\s*[）)]?"
                ),
            ),
            # --- Structured formats ---
            ("paren_cn", re.compile(r"[（(]\s*([{C}])\s*[）)]")),
            ("bracket", re.compile(r"[\[]\s*([{C}])\s*[\]]")),
            ("dot_prefix", re.compile(rf"^([{C}])\s*[.。:：]", re.MULTILINE)),
            ("dot_end", re.compile(rf"^([{C}])\s*[.。]?\s*$", re.MULTILINE)),
        ]

    def parse(self, response: str, choices: list[str] | None = None) -> str:
        """Extract a choice letter from *response*.

        Returns uppercase letter (e.g. ``"A"``) or empty string.
        """
        if not response:
            return ""
        valid = {c.upper() for c in (choices or ["A", "B", "C", "D"])}
        response_stripped = response.strip()

        for _name, pat in self._patterns:
            m = pat.search(response_stripped)
            if m:
                ch = m.group(1).upper()
                if ch in valid:
                    return ch

        # EvalScope fallback: last valid uppercase letter
        for letter in reversed(response_stripped):
            if letter.isupper() and letter in valid:
                return letter

        return ""


# ===========================================================================
# MathAnswerParser  (ref: EvalScope math_parser.py)
# ===========================================================================


class MathAnswerParser:
    """Consolidated math answer extraction with symbolic equivalence.

    Extraction priority:
    1. ``\\boxed{...}`` with brace-depth tracking (EvalScope extract_answer)
    2. ``####`` marker (GSM8K)
    3. ``he answer is``, ``final answer is``, ``ANSWER:``, ``答案是`` (EvalScope patterns)
    4. Last number fallback

    Comparison (``check_answer``):
    1. Exact / normalized string match
    2. Numeric: float tolerance + percentage (EvalScope numeric_equal)
    3. Symbolic: SymPy ``simplify(a - b) == 0`` (EvalScope symbolic_equal)
    """

    _BOXED_RE = re.compile(r"\\boxed\s*\{", re.DOTALL)
    _HASH_RE = re.compile(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)")

    def parse(self, response: str) -> str:
        """Extract a math answer string from *response*."""
        if not response:
            return ""
        response = response.strip()

        # 1. \boxed{} with brace-depth tracking (EvalScope pattern)
        boxed = self._extract_boxed(response)
        if boxed:
            return boxed

        # 2. #### marker (GSM8K style)
        m = self._HASH_RE.search(response)
        if m:
            return m.group(1).replace(",", "").replace("$", "")

        # 3. EvalScope-style explicit patterns
        for pat, grp in [
            (re.compile(r"he answer is\s*(.+?)(?:\n|$)", re.IGNORECASE), 1),
            (re.compile(r"final answer is\s*\$?\s*(.+?)(?:\n|$)", re.IGNORECASE), 1),
            (re.compile(r"答案是\s*(.+?)(?:\n|$)"), 1),
            (re.compile(r"ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE), 1),
        ]:
            m = pat.search(response)
            if m:
                return m.group(grp).strip().rstrip(".").rstrip("/")

        # 4. English / Chinese explicit declarations (broader)
        _explicit = [
            re.compile(
                r'(?:final answer|answer|result|solution)\s*(?:is|=|:)\s*["\']?\s*([-+]?\$?[\d,]+(?:\.\d+)?)\b',
                re.IGNORECASE,
            ),
            re.compile(
                r'(?:答案|结果|解)\s*(?:是|为|：|:|=)\s*["\']?\s*([-+]?\$?[\d,]+(?:\.\d+)?)\b'
            ),
            re.compile(
                r"(?:therefore|thus|hence|所以)\s*,?\s*(?:the answer is)?\s*([-+]?\$?[\d,]+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
        ]
        for pat in _explicit:
            m = pat.search(response)
            if m:
                return m.group(1).replace(",", "").replace("$", "")

        # 5. Last number on last line
        last_line = response.strip().split("\n")[-1]
        nums = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", last_line)
        if nums:
            return str(nums[-1]).replace(",", "").replace("$", "")

        # 6. Last number anywhere
        all_nums = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", response)
        if all_nums:
            return str(all_nums[-1]).replace(",", "").replace("$", "")

        return ""

    # -- comparison (ref: EvalScope math_equal) ------------------------------

    @staticmethod
    def check_answer(
        predicted: str, correct: str, tolerance: float = 1e-6
    ) -> bool:  # noqa: ARG004
        """Compare predicted and correct math answers.

        Strategy: exact string → float tolerance (+ percentage) → SymPy symbolic.
        """
        pred = (predicted or "").strip()
        exp = (correct or "").strip()

        if not pred:
            return False

        # 1. Exact string match (case-insensitive)
        if pred.lower() == exp.lower():
            return True

        # 2. Normalized match
        def _norm(s: str) -> str:
            return (
                s.replace(",", "")
                .replace(" ", "")
                .replace("$", "")
                .replace("−", "-")
                .replace("\u043a\u0438", "")
                .strip()
            )

        if _norm(pred) == _norm(exp):
            return True

        # 3. Numeric comparison with percentage tolerance (EvalScope numeric_equal)
        try:
            pred_f = float(_norm(pred))
            exp_f = float(_norm(exp))

            # Direct tolerance
            if isclose(exp_f, pred_f, rel_tol=1e-4):
                return True

            # Percentage variants (EvalScope: gt_result = [ref/100, ref, ref*100])
            for candidate in [exp_f / 100, exp_f, exp_f * 100]:
                try:
                    if isclose(candidate, pred_f, rel_tol=1e-4):
                        return True
                except Exception:
                    continue

            # Rounded match (e.g. 15400.0 == 15400)
            if round(pred_f) == round(exp_f):
                return True

        except (ValueError, TypeError, ZeroDivisionError):
            pass

        # 4. SymPy symbolic equivalence (EvalScope symbolic_equal)
        if _sympy_available:
            try:
                import sympy

                pred_expr = sympy.sympify(_norm(pred).replace("^", "**"))
                exp_expr = sympy.sympify(_norm(exp).replace("^", "**"))
                if sympy.simplify(pred_expr - exp_expr) == 0:
                    return True
                # Also try .equals() method (EvalScope)
                try:
                    if pred_expr.equals(exp_expr):
                        return True
                except Exception:
                    pass
            except Exception:
                pass

        return False

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_boxed(text: str) -> str:
        """Extract content from \\boxed{...} with brace-depth tracking (EvalScope)."""
        match = MathAnswerParser._BOXED_RE.search(text)
        if not match:
            return ""
        start = match.end()  # position right after the opening {
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start : i - 1].strip()
        return ""

    @staticmethod
    def strip_answer_string(string: str) -> str:
        """Clean up a math answer string (simplified from EvalScope strip_answer_string).

        Handles: LaTeX cleanup, commas, units, dollar signs, trailing periods.
        """
        if not string:
            return ""
        string = str(string).strip()
        string = string.replace("\n", "")
        string = string.rstrip(".")
        string = string.replace("\\!", "")
        string = string.replace("tfrac", "frac").replace("dfrac", "frac")
        string = string.replace("\\left", "").replace("\\right", "")
        string = string.replace("\\{", "{").replace("\\}", "}")
        # Remove units
        string = re.sub(r"\\text\{.*?\}$", "", string).strip()
        string = string.replace("^{\\circ}", "").replace("^\\circ", "")
        string = string.replace("\\$", "").replace("$", "")
        string = string.replace("\\%", "").replace("%", "")
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        # Remove unnecessary backslash before integers
        string = re.sub(r"\\(?=\-?\d+(\\|\)|,|\]|$))", "", string)
        # Normalize thousands separators
        if re.fullmatch(r"\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*", string):
            string = string.replace(",", "")
        return string


# ===========================================================================
# CodeAnswerParser
# ===========================================================================


class CodeAnswerParser:
    """Extract code from model responses."""

    _FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)

    def parse(self, response: str) -> str:
        """Extract code from *response*."""
        if not response:
            return ""

        blocks = self._FENCE_RE.findall(response)
        if blocks:
            return str(blocks[-1]).strip()

        lines = response.split("\n")
        code_lines = [l for l in lines if l.startswith("    ") or l.startswith("\t")]
        if code_lines:
            return "\n".join(code_lines)

        return response.strip()


# ===========================================================================
# TextAnswerParser
# ===========================================================================


class TextAnswerParser:
    """Text answer comparison for LongBench, Arena Hard, etc."""

    def parse(self, response: str) -> str:
        """Extract text answer from response (returns stripped response)."""
        if not response:
            return ""
        return response.strip()

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def check_answer(
        predicted: str, correct: str, fuzzy_threshold: float = 0.8
    ) -> bool:
        if not predicted or not correct:
            return False
        pred_norm = TextAnswerParser.normalize(predicted)
        exp_norm = TextAnswerParser.normalize(correct)

        if pred_norm == exp_norm:
            return True
        if exp_norm in pred_norm:
            return True
        if pred_norm in exp_norm:
            return True

        ratio = SequenceMatcher(None, pred_norm, exp_norm).ratio()
        return ratio >= fuzzy_threshold


# ===========================================================================
# Factory
# ===========================================================================

CHOICE_DATASETS = {
    "mmlu",
    "gpqa",
    "hellaswag",
    "truthfulqa",
    "winogrande",
    "arc",
    "ceval",
    "cmmlu",
}
MATH_DATASETS = {"gsm8k", "math500", "aime2025", "aime"}
CODE_DATASETS = {"humaneval", "mbpp"}
TEXT_DATASETS = {
    "longbench",
    "arena_hard",
    "swebench_lite",
    "swebench",
    "piqa",
    "global_piqa",
}


def get_parser_for_dataset(dataset_name: str):
    """Return the appropriate parser instance for a dataset."""
    name = dataset_name.lower().replace("-", "_").replace(" ", "")
    if name in CHOICE_DATASETS:
        return MultiChoiceParser()
    if name in MATH_DATASETS:
        return MathAnswerParser()
    if name in CODE_DATASETS:
        return CodeAnswerParser()
    return TextAnswerParser()
