"""
Enhanced Answer Parser (增强AnswerParse器)

借鉴 lm-evaluation-harness  Filter Chain 设计，提供更强大、更灵活AnswerParse能力。

主要改进:
1. 多层Parse策略 (结构化 → 模式匹配 → 语义 → LLM)
2. 可ConfigureFilter器链
3. 数学表达式etc.价性判断 (SymPy)
4. 代码执行Validate
5. in英文混合支持
6. 详细ParseLog
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


class AnswerType(Enum):
    """Answer类型"""
    NUMBER = "number"           # 数值Answer
    INTEGER = "integer"         # 整数Answer (AIMEetc.)
    CHOICE = "choice"           # 选择题 (A/B/C/D)
    MULTI_CHOICE = "multi_choice"  # 多选题
    TEXT = "text"               # 文本Answer
    BOOLEAN = "boolean"         # is/否
    CODE = "code"               # 代码
    MATH_EXPR = "math_expr"     # 数学表达式
    LIST = "list"               # 列表/数组
    JSON = "json"               # JSON 结构


@dataclass
class ParseResult:
    """Parse result"""
    extracted: str                      # 提取Answer
    normalized: Any                     # 规范化后值
    confidence: float                   # 置信度 (0-1)
    method: str                         # Parse方法
    matched_pattern: str = ""           # 匹配模式
    all_candidates: list[str] = field(default_factory=list)  # 所has候选Answer
    parse_log: list[str] = field(default_factory=list)       # Parse过程Log
    error: str | None = None


@dataclass
class FilterConfig:
    """Filter器Configure"""
    name: str
    function: str           # regex, take_first, take_last, remove, normalize
    pattern: str | None = None
    group: int = 1
    flags: int = 0


class EnhancedAnswerParser:
    """
    增强AnswerParse器

    use多层Parse策略:
    Layer 1: 结构化标记提取 (boxed, ####, XML tags)
    Layer 2: 显式声明匹配 (answer is, Answeris)
    Layer 3: 位置启发式 (最后一行, 最后一数字)
    Layer 4: LLM 辅助提取 (optional)

    Usage:
        parser = EnhancedAnswerParser()

        # Parse数值
        result = parser.parse("CalculateResultis \\boxed{42}", AnswerType.NUMBER)
        print(result.extracted)  # "42"
        print(result.confidence)  # 0.95

        # useCustomFilter器
        result = parser.parse_with_filters(
            response="The answer is approximately 3.14159",
            filters=[
                FilterConfig(name="extract_number", function="regex", pattern=r"[-+]?\\d+\\.?\\d*")
            ]
        )
    """

    # 结构化标记模式 (Highest优先级)
    STRUCTURED_PATTERNS = {
        'boxed': r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',  # 支持嵌套
        'boxed_simple': r'\\boxed\{([^{}]+)\}',
        'hash_answer': r'####\s*([-+]?\$?[\d,]+(?:\.\d+)?)',
        'xml_answer': r'<answer>\s*(.*?)\s*</answer>',
        'final_answer_tag': r'<final_answer>\s*(.*?)\s*</final_answer>',
        'think_answer': r'</think>\s*\n*(.*?)$',  # DeepSeek think 模式
    }

    # 显式声明模式 (高优先级)
    EXPLICIT_PATTERNS = {
        'answer_is': r'(?:answer|result|solution)\s*(?:is|=|:)\s*["\']?\s*([-+]?\$?[\d,]+(?:\.\d+)?|[A-Fa-f])\b',
        'answer_cn': r'(?:Answer|Result|解)\s*[isis：:=]\s*["\']?\s*([-+]?\$?[\d,]+(?:\.\d+)?|[A-Fa-f])\b',
        'therefore': r'(?:therefore|thus|hence|so)\s*,?\s*(?:the answer is)?\s*([-+]?\$?[\d,]+(?:\.\d+)?)',
        'equals_final': r'=\s*([-+]?\$?[\d,]+(?:\.\d+)?)\s*[。.;,]?\s*$',
        'final_colon': r'(?:Final|最终|Answer)[：:]\s*([-+]?\$?[\d,]+(?:\.\d+)?|[A-Fa-f])',
    }

    # 选择题模式
    CHOICE_PATTERNS = [
        r'(?:answer|选|Answer)\s*(?:is|:)?\s*[（\(]?\s*([A-Fa-f])\s*[）\)]?',
        r'^[（\(]\s*([A-Fa-f])\s*[）\)]',  # 开头 (A)
        r'^([A-Fa-f])\s*[.。:：]',  # 开头 A.
        r'\b([A-Fa-f])\s*(?:is correct|正确)',
        r'(?:选择|choose|select)\s*[（\(]?\s*([A-Fa-f])\s*[）\)]?',
    ]

    def __init__(
        self,
        llm_fallback_threshold: float = 0.5,
        enable_math_eval: bool = True,
        enable_code_eval: bool = False,
        verbose: bool = False
    ):
        """
        Initialize增强Parse器

        Args:
            llm_fallback_threshold: LLM 兜底阈值
            enable_math_eval: is否启用数学表达式求值
            enable_code_eval: is否启用代码执行Validate
            verbose: is否输出Verbose Logging
        """
        self.llm_fallback_threshold = llm_fallback_threshold
        self.enable_math_eval = enable_math_eval
        self.enable_code_eval = enable_code_eval
        self.verbose = verbose

        # 尝试Load SymPy (optional)
        self.sympy_available = False
        try:
            import sympy
            self.sympy_available = True
        except ImportError:
            pass

    def parse(
        self,
        response: str,
        answer_type: AnswerType,
        expected: str | None = None,
        choices: list[str] | None = None
    ) -> ParseResult:
        """
        ParseModel响应

        Args:
            response: Model响应
            answer_type: 期望Answer类型
            expected: 期望Correct answer (用于Validate)
            choices: 选择题Options列表

        Returns:
            ParseResult
        """
        if not response or not response.strip():
            return ParseResult(
                extracted="",
                normalized=None,
                confidence=0.0,
                method="empty_response",
                error="Empty response"
            )

        response = response.strip()
        log = []

        # based onAnswer类型选择Parse策略
        if answer_type in [AnswerType.NUMBER, AnswerType.INTEGER, AnswerType.MATH_EXPR]:
            return self._parse_numeric(response, answer_type, log)
        elif answer_type in [AnswerType.CHOICE, AnswerType.MULTI_CHOICE]:
            return self._parse_choice(response, choices or ['A', 'B', 'C', 'D'], log)
        elif answer_type == AnswerType.BOOLEAN:
            return self._parse_boolean(response, log)
        elif answer_type == AnswerType.CODE:
            return self._parse_code(response, log)
        elif answer_type == AnswerType.LIST:
            return self._parse_list(response, log)
        else:
            return self._parse_text(response, log)

    def _parse_numeric(
        self,
        response: str,
        answer_type: AnswerType,
        log: list[str]
    ) -> ParseResult:
        """Parse数值/数学表达式"""
        candidates = []

        # Layer 1: 结构化标记
        for name, pattern in self.STRUCTURED_PATTERNS.items():
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                # 取最后一匹配 (通常is最终Answer)
                value = matches[-1].strip()
                log.append(f"L1 [{name}] matched: {value[:50]}")

                # 尝试规范化
                normalized = self._normalize_number(value)
                if normalized is not None:
                    candidates.append((value, normalized, 0.95, f"L1_{name}"))
                elif name in ['boxed', 'boxed_simple']:
                    # boxed 内可能is表达式
                    expr_result = self._evaluate_expression(value)
                    if expr_result is not None:
                        candidates.append((value, expr_result, 0.92, f"L1_{name}_expr"))

        # Layer 2: 显式声明
        for name, pattern in self.EXPLICIT_PATTERNS.items():
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                value = matches[-1].strip()
                log.append(f"L2 [{name}] matched: {value}")
                normalized = self._normalize_number(value)
                if normalized is not None:
                    candidates.append((value, normalized, 0.85, f"L2_{name}"))

        # Layer 3: 位置启发式
        # 3a: 最后一行数字
        last_line = response.strip().split('\n')[-1]
        last_line_nums = re.findall(r'[-+]?\$?[\d,]+(?:\.\d+)?', last_line)
        if last_line_nums:
            value = last_line_nums[-1]
            log.append(f"L3 [last_line] found: {value}")
            normalized = self._normalize_number(value)
            if normalized is not None:
                candidates.append((value, normalized, 0.7, "L3_last_line"))

        # 3b: etc.号后数字
        eq_matches = re.findall(r'=\s*([-+]?\$?[\d,]+(?:\.\d+)?)', response)
        if eq_matches:
            value = eq_matches[-1]
            log.append(f"L3 [equation] found: {value}")
            normalized = self._normalize_number(value)
            if normalized is not None:
                candidates.append((value, normalized, 0.65, "L3_equation"))

        # 3c: 最后出现数字 (Lowest置信度)
        all_nums = re.findall(r'[-+]?[\d,]+(?:\.\d+)?', response)
        if all_nums:
            value = all_nums[-1]
            log.append(f"L3 [last_number] found: {value}")
            normalized = self._normalize_number(value)
            if normalized is not None:
                candidates.append((value, normalized, 0.4, "L3_last_number"))

        # 选择Highest置信度候选
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best = candidates[0]

            # 对于整数类型，额外Validate
            if answer_type == AnswerType.INTEGER:
                normalized = int(round(best[1])) if isinstance(best[1], float) else best[1]
            else:
                normalized = best[1]

            return ParseResult(
                extracted=best[0],
                normalized=normalized,
                confidence=best[2],
                method=best[3],
                all_candidates=[c[0] for c in candidates],
                parse_log=log
            )

        log.append("No numeric answer found")
        return ParseResult(
            extracted="",
            normalized=None,
            confidence=0.0,
            method="failed",
            parse_log=log,
            error="No numeric answer found"
        )

    def _parse_choice(
        self,
        response: str,
        choices: list[str],
        log: list[str]
    ) -> ParseResult:
        """Parse选择题"""
        response_clean = response.strip()
        choices_upper = [c.upper() for c in choices]
        candidates = []

        # Layer 1: 显式声明Answer
        for pattern in self.CHOICE_PATTERNS:
            matches = re.findall(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if matches:
                choice = matches[-1].upper()
                if choice in choices_upper:
                    log.append(f"L1 explicit choice matched: {choice}")
                    candidates.append((choice, choice, 0.95, "L1_explicit"))

        # Layer 2: 开头/结尾Options
        # 开头
        first_match = re.match(rf'^[（\(]?\s*([{",".join(choices)}])\s*[）\).]?', response_clean, re.IGNORECASE)
        if first_match:
            choice = first_match.group(1).upper()
            if choice in choices_upper:
                log.append(f"L2 first char: {choice}")
                candidates.append((choice, choice, 0.85, "L2_first"))

        # 结尾
        last_match = re.search(rf'[（\(]?\s*([{",".join(choices)}])\s*[）\).]?\s*$', response_clean, re.IGNORECASE)
        if last_match:
            choice = last_match.group(1).upper()
            if choice in choices_upper:
                log.append(f"L2 last char: {choice}")
                candidates.append((choice, choice, 0.8, "L2_last"))

        # Layer 3: 任意位置一Options字母
        for char in response_clean[:200]:
            if char.upper() in choices_upper:
                log.append(f"L3 any position: {char.upper()}")
                candidates.append((char.upper(), char.upper(), 0.5, "L3_any"))
                break

        # 选择Best
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best = candidates[0]
            return ParseResult(
                extracted=best[0],
                normalized=best[1],
                confidence=best[2],
                method=best[3],
                all_candidates=[c[0] for c in candidates],
                parse_log=log
            )

        log.append("No choice found")
        return ParseResult(
            extracted="",
            normalized=None,
            confidence=0.0,
            method="failed",
            parse_log=log,
            error="No choice found"
        )

    def _parse_boolean(self, response: str, log: list[str]) -> ParseResult:
        """Parseis/否Answer"""
        response_lower = response.lower().strip()

        yes_patterns = [
            r'\b(yes|true|correct|right|is|对|正确|确实|can)\b',
        ]
        no_patterns = [
            r'\b(no|false|incorrect|wrong|否|not|Error|not对|notcan)\b',
        ]

        # Check明确肯定
        for pattern in yes_patterns:
            if re.search(pattern, response_lower):
                log.append("Boolean YES matched")
                return ParseResult(
                    extracted="Yes",
                    normalized=True,
                    confidence=0.85,
                    method="keyword_yes",
                    parse_log=log
                )

        # Check明确否定
        for pattern in no_patterns:
            if re.search(pattern, response_lower):
                log.append("Boolean NO matched")
                return ParseResult(
                    extracted="No",
                    normalized=False,
                    confidence=0.85,
                    method="keyword_no",
                    parse_log=log
                )

        log.append("No boolean indicator found")
        return ParseResult(
            extracted="",
            normalized=None,
            confidence=0.0,
            method="failed",
            parse_log=log,
            error="No boolean indicator found"
        )

    def _parse_code(self, response: str, log: list[str]) -> ParseResult:
        """Parse代码Answer"""
        # 提取代码块
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            code = code_blocks[-1].strip()
            log.append(f"Code block extracted: {len(code)} chars")
            return ParseResult(
                extracted=code,
                normalized=code,
                confidence=0.9,
                method="code_block",
                parse_log=log
            )

        # 尝试识别缩进代码
        lines = response.split('\n')
        code_lines = [l for l in lines if l.startswith('    ') or l.startswith('\t')]
        if code_lines:
            code = '\n'.join(code_lines)
            log.append(f"Indented code extracted: {len(code)} chars")
            return ParseResult(
                extracted=code,
                normalized=code,
                confidence=0.7,
                method="indented_code",
                parse_log=log
            )

        # Return整响应作is代码
        log.append("Using full response as code")
        return ParseResult(
            extracted=response,
            normalized=response,
            confidence=0.5,
            method="full_response",
            parse_log=log
        )

    def _parse_list(self, response: str, log: list[str]) -> ParseResult:
        """Parse列表Answer"""
        # 尝试 JSON 数组
        json_match = re.search(r'\[([^\[\]]+)\]', response)
        if json_match:
            try:
                import json
                arr = json.loads(f"[{json_match.group(1)}]")
                log.append(f"JSON array parsed: {arr}")
                return ParseResult(
                    extracted=str(arr),
                    normalized=arr,
                    confidence=0.9,
                    method="json_array",
                    parse_log=log
                )
            except:
                pass

        # 逗号分隔
        items = re.findall(r'(?:^|\n)\s*[-•*]?\s*(.+?)(?:,|$|\n)', response)
        if items:
            items = [i.strip() for i in items if i.strip()]
            log.append(f"List items: {items}")
            return ParseResult(
                extracted=str(items),
                normalized=items,
                confidence=0.7,
                method="list_items",
                parse_log=log
            )

        return ParseResult(
            extracted=response,
            normalized=[response],
            confidence=0.5,
            method="single_item",
            parse_log=log
        )

    def _parse_text(self, response: str, log: list[str]) -> ParseResult:
        """Parse文本Answer"""
        # 取最后一句作isAnswer
        sentences = re.split(r'[.。!！?？\n]+', response.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            answer = sentences[-1][:500]  # 限制长度
            log.append(f"Last sentence: {answer[:100]}...")
            return ParseResult(
                extracted=answer,
                normalized=answer,
                confidence=0.6,
                method="last_sentence",
                parse_log=log
            )

        return ParseResult(
            extracted=response[:500],
            normalized=response[:500],
            confidence=0.4,
            method="full_text",
            parse_log=log
        )

    def _normalize_number(self, value: str) -> float | None:
        """规范化数值字符串"""
        if not value:
            return None

        try:
            # Cleanup
            clean = value.strip()
            clean = re.sub(r'[\$,\s%]', '', clean)  # 移除 $, 逗号, 空格, %
            clean = clean.replace('−', '-')  # Unicode 负号

            # Process分数
            frac_match = re.match(r'^(-?\d+)/(\d+)$', clean)
            if frac_match:
                return float(frac_match.group(1)) / float(frac_match.group(2))

            # Process科学计数法
            if 'e' in clean.lower() or '×' in clean or '*' in clean:
                clean = clean.lower().replace('×', 'e').replace('*', 'e').replace(' ', '')

            return float(clean)
        except:
            return None

    def _evaluate_expression(self, expr: str) -> float | None:
        """安全求值数学表达式"""
        if not expr:
            return None

        try:
            # Cleanup LaTeX
            clean = expr
            clean = re.sub(r'\\(?:text|mathrm|mathbf)\{[^}]*\}', '', clean)
            clean = re.sub(r'\\(?:frac)\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', clean)
            clean = re.sub(r'\\(?:sqrt)\{([^}]+)\}', r'math.sqrt(\1)', clean)
            clean = clean.replace(r'\times', '*').replace(r'\cdot', '*')
            clean = clean.replace(r'\div', '/').replace('×', '*')
            clean = clean.replace('^', '**')

            # 只保留安全字符
            safe = re.sub(r'[^0-9+\-*/().mathsqrt\s]', '', clean)

            if safe:
                result = eval(safe, {"__builtins__": {}, "math": math})
                return float(result)
        except:
            pass

        # 尝试 SymPy (if可用)
        if self.sympy_available:
            try:
                import sympy
                result = sympy.sympify(expr)
                return float(result.evalf())
            except:
                pass

        return None

    async def parse_with_llm_fallback(
        self,
        response: str,
        answer_type: AnswerType,
        llm_func: Callable,
        expected: str | None = None,
        choices: list[str] | None = None
    ) -> ParseResult:
        """
        带 LLM 兜底Parse

        Args:
            response: Model响应
            answer_type: Answer类型
            llm_func: async (prompt) -> str
            expected: 期望Answer
            choices: 选择题Options

        Returns:
            ParseResult
        """
        # 先尝试规则Parse
        rule_result = self.parse(response, answer_type, expected, choices)

        # 置信度足够高，直接Return
        if rule_result.confidence >= self.llm_fallback_threshold:
            return rule_result

        # LLM 兜底
        try:
            llm_result = await self._llm_extract(response, answer_type, llm_func, choices)

            if llm_result.confidence > rule_result.confidence:
                llm_result.parse_log = rule_result.parse_log + ["LLM fallback used"] + llm_result.parse_log
                return llm_result

            return rule_result

        except Exception as e:
            rule_result.error = f"LLM fallback failed: {e}"
            return rule_result

    async def _llm_extract(
        self,
        response: str,
        answer_type: AnswerType,
        llm_func: Callable,
        choices: list[str] | None = None
    ) -> ParseResult:
        """use LLM 提取Answer"""
        type_instructions = {
            AnswerType.NUMBER: "Extract the final numerical answer. Return ONLY the number (e.g., 42 or 3.14). No units, no explanation.",
            AnswerType.INTEGER: "Extract the final integer answer. Return ONLY the integer (e.g., 42). No decimals.",
            AnswerType.CHOICE: f"Extract the selected option from {choices or ['A','B','C','D']}. Return ONLY the letter.",
            AnswerType.BOOLEAN: "Is the answer Yes or No? Return ONLY 'Yes' or 'No'.",
            AnswerType.TEXT: "Extract the final answer. Be concise.",
        }

        prompt = f"""Extract the final answer from this response.

{type_instructions.get(answer_type, "Extract the final answer.")}

Response:
\"\"\"
{response[:3000]}
\"\"\"

Final Answer:"""

        llm_response = await llm_func(prompt)
        extracted = llm_response.strip()

        # Validate
        log = [f"LLM extracted: {extracted}"]

        if answer_type in [AnswerType.NUMBER, AnswerType.INTEGER, AnswerType.MATH_EXPR]:
            normalized = self._normalize_number(extracted)
            if normalized is not None:
                if answer_type == AnswerType.INTEGER:
                    normalized = int(round(normalized))
                return ParseResult(
                    extracted=extracted,
                    normalized=normalized,
                    confidence=0.85,
                    method="llm",
                    parse_log=log
                )
        elif answer_type in [AnswerType.CHOICE, AnswerType.MULTI_CHOICE]:
            choices_upper = [c.upper() for c in (choices or ['A','B','C','D'])]
            if extracted.upper() in choices_upper:
                return ParseResult(
                    extracted=extracted.upper(),
                    normalized=extracted.upper(),
                    confidence=0.85,
                    method="llm",
                    parse_log=log
                )
        else:
            return ParseResult(
                extracted=extracted,
                normalized=extracted,
                confidence=0.8,
                method="llm",
                parse_log=log
            )

        return ParseResult(
            extracted=extracted,
            normalized=extracted,
            confidence=0.5,
            method="llm_uncertain",
            parse_log=log
        )


# ============================================
# Answer比较函数
# ============================================

def compare_answers(
    predicted: Any,
    expected: Any,
    answer_type: AnswerType,
    tolerance: float = 1e-6,
    ignore_case: bool = True
) -> tuple[bool, float, str]:
    """
    比较两Answeris否etc.价

    Returns:
        (is_correct, similarity_score, comparison_method)
    """
    if predicted is None or expected is None:
        return False, 0.0, "null_value"

    if answer_type in [AnswerType.NUMBER, AnswerType.INTEGER, AnswerType.MATH_EXPR]:
        return _compare_numeric(predicted, expected, tolerance)
    elif answer_type in [AnswerType.CHOICE, AnswerType.MULTI_CHOICE]:
        return _compare_choice(predicted, expected)
    elif answer_type == AnswerType.BOOLEAN:
        return _compare_boolean(predicted, expected)
    else:
        return _compare_text(predicted, expected, ignore_case)


def _compare_numeric(predicted: Any, expected: Any, tolerance: float) -> tuple[bool, float, str]:
    """比较数值"""
    try:
        pred_val = float(predicted) if not isinstance(predicted, (int, float)) else predicted
        exp_val = float(str(expected).replace(',', '').replace('$', ''))

        # 完全相etc.
        if pred_val == exp_val:
            return True, 1.0, "exact_match"

        # 绝对容差
        if abs(pred_val - exp_val) <= tolerance:
            return True, 0.99, "within_tolerance"

        # 相对容差 (0.1%)
        if exp_val != 0 and abs((pred_val - exp_val) / exp_val) < 0.001:
            return True, 0.95, "relative_tolerance"

        # 整数舍入
        if round(pred_val) == round(exp_val):
            return True, 0.9, "rounded_match"

        return False, 0.0, "mismatch"

    except (ValueError, TypeError):
        return False, 0.0, "conversion_error"


def _compare_choice(predicted: Any, expected: Any) -> tuple[bool, float, str]:
    """比较选择题"""
    pred_str = str(predicted).upper().strip()
    exp_str = str(expected).upper().strip()

    # Process数字Index (0,1,2,3 -> A,B,C,D)
    if exp_str.isdigit():
        idx = int(exp_str)
        if 0 <= idx <= 25:
            exp_str = chr(ord('A') + idx)

    if pred_str == exp_str:
        return True, 1.0, "exact_match"

    return False, 0.0, "mismatch"


def _compare_boolean(predicted: Any, expected: Any) -> tuple[bool, float, str]:
    """比较布尔值"""
    def to_bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).lower().strip()
        return s in ['true', 'yes', 'is', '对', '1']

    pred_bool = to_bool(predicted)
    exp_bool = to_bool(expected)

    if pred_bool == exp_bool:
        return True, 1.0, "exact_match"
    return False, 0.0, "mismatch"


def _compare_text(predicted: Any, expected: Any, ignore_case: bool) -> tuple[bool, float, str]:
    """比较文本"""
    pred_str = str(predicted).strip()
    exp_str = str(expected).strip()

    if ignore_case:
        pred_str = pred_str.lower()
        exp_str = exp_str.lower()

    # 完全匹配
    if pred_str == exp_str:
        return True, 1.0, "exact_match"

    # 包含关系
    if exp_str in pred_str:
        return True, 0.8, "contains"

    if pred_str in exp_str:
        return True, 0.7, "partial"

    # 规范化后比较 (移除标点)
    pred_norm = re.sub(r'[^\w\s]', '', pred_str)
    exp_norm = re.sub(r'[^\w\s]', '', exp_str)

    if pred_norm == exp_norm:
        return True, 0.9, "normalized_match"

    return False, 0.0, "mismatch"


# ============================================
# 便捷函数
# ============================================

# 全局Parse器实例
_default_parser = None

def get_parser() -> EnhancedAnswerParser:
    """Get全局Parse器实例"""
    global _default_parser
    if _default_parser is None:
        _default_parser = EnhancedAnswerParser()
    return _default_parser


def quick_parse(response: str, answer_type: str = "number") -> ParseResult:
    """
    快速ParseAnswer

    Usage:
        result = quick_parse("The answer is 42", "number")
        print(result.normalized)  # 42.0
    """
    type_map = {
        "number": AnswerType.NUMBER,
        "int": AnswerType.INTEGER,
        "integer": AnswerType.INTEGER,
        "choice": AnswerType.CHOICE,
        "bool": AnswerType.BOOLEAN,
        "text": AnswerType.TEXT,
        "code": AnswerType.CODE,
        "math": AnswerType.MATH_EXPR,
    }

    at = type_map.get(answer_type.lower(), AnswerType.TEXT)
    return get_parser().parse(response, at)
