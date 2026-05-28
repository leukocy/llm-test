import asyncio
import random
import string
import threading
import time

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from config.settings import HF_MODEL_MAPPING
from core.benchmark.metrics import calculate_request_metrics, empty_metrics
from core.error_messages import get_error_info
from core.providers.factory import get_provider
from core.tokenizer_utils import get_cached_tokenizer
from ui.formatters import format_results_for_display
from ui.log_viewer import render_log_viewer
from utils.get_logger import get_logger
from utils.helpers import append_to_csv, initialize_csv, reorder_dataframe_columns
from utils.log_server import log_server
from utils.logger import BenchmarkLogger, LogLevel

try:
    import psutil
except ImportError:
    psutil = None

# Prefill calibration constant (token overhead for calibration prompts)
PREFILL_PROMPT_OVERHEAD = 0

# ---------------------------------------------------------------------------
# Lazy-loaded suffix prompt pool from evaluation datasets
# ---------------------------------------------------------------------------
_SUFFIX_PROMPT_POOL: list[str] | None = None
_SUFFIX_POOL_LOCK = threading.Lock()


def _load_suffix_prompt_pool() -> list[str]:
    """
    Load hard benchmark tasks to use as diverse suffix prompts.
    Returns a list of question strings.
    """
    global _SUFFIX_PROMPT_POOL
    if _SUFFIX_PROMPT_POOL is not None:
        return _SUFFIX_PROMPT_POOL

    pool: list[str] = []
    import json as _json

    def _load_json(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = _json.load(f)
            return data if isinstance(data, list) else data.get("data", [])
        except Exception:
            return []

    def _load_jsonl(path, limit=None):
        try:
            with open(path, encoding="utf-8") as f:
                rows = []
                for line in f:
                    if line.strip():
                        rows.append(_json.loads(line))
                        if limit and len(rows) >= limit:
                            break
                return rows
        except Exception:
            return []

    long_answer_instruction = (
        "\n\nIMPORTANT: You MUST write a very long, detailed, step-by-step answer. "
        "Do NOT just state the final answer. You must:\n"
        "1. Show all your reasoning and intermediate steps in detail.\n"
        "2. Discuss alternative approaches and explain why they would or would not work.\n"
        "3. Analyze edge cases and potential pitfalls.\n"
        "4. Provide a thorough conclusion that summarizes everything.\n"
        "Your response should be at least 2000 words. Short answers will be rejected.\n"
        "请务必写出详尽、完整的长篇回答，包含所有推理步骤、详细分析和总结。"
        "不要只给出最终答案，必须展示完整的思考过程。回答至少2000字以上。"
    )

    def _clip_task_text(task: str, max_chars: int = 12000) -> str:
        task = (task or "").strip()
        if len(task) <= max_chars:
            return task
        return task[:max_chars].rstrip() + "\n[Truncated for benchmark prompt generation.]"

    def _add_hard_prompt(source: str, task: str):
        task = _clip_task_text(task)
        if task:
            pool.append(f"{source}:\n{task}{long_answer_instruction}")

    def _add_longbench_prompt(source: str, row: dict):
        context = (row.get("context") or "").strip()
        task_input = (row.get("input") or "").strip()
        parts = []
        if context:
            parts.append(f"Context:\n{context}")
        if task_input:
            parts.append(f"Task:\n{task_input}")
        _add_hard_prompt(source, "\n\n".join(parts))

    # Prefer hard benchmarks whose prompts tend to require long reasoning or analysis.
    for row in _load_json("datasets/aime2024/aime2024.json")[:200]:
        _add_hard_prompt("AIME 2024 competition math problem", row.get("problem", ""))
    for row in _load_json("datasets/aime2025/aime2025.json")[:200]:
        _add_hard_prompt("AIME 2025 competition math problem", row.get("problem", ""))
    for row in _load_json("datasets/aime2026/aime2026.json")[:200]:
        _add_hard_prompt("AIME 2026 competition math problem", row.get("problem", ""))

    import re as _re
    for row in _load_json("datasets/gpqa/gpqa_diamond.json")[:200]:
        raw_q = (row.get("question") or "").strip()
        # Strip trailing A./B./C./D. answer selection block (keeps a)/b)/c)/d) descriptions)
        cleaned = _re.sub(r'\n[A-D]\.\s+.*$', '', raw_q, flags=_re.DOTALL).strip()
        # Check if the question has option descriptions like a) ... b) ... c) ... d)
        if _re.search(r'\ba\)\s', cleaned) and _re.search(r'\bd\)\s', cleaned):
            # Rewrite as "analyze each option" task to force long output
            cleaned = (
                cleaned
                + "\n\nFor each option above, analyze in detail whether it is correct or incorrect. "
                "Provide thorough scientific reasoning for your analysis of every option, then "
                "conclude with which option is best and why."
            )
        else:
            cleaned = cleaned + (
                "\n\nProvide a comprehensive scientific analysis explaining the underlying "
                "principles, derive any relevant equations, and discuss the implications."
            )
        _add_hard_prompt("GPQA Diamond graduate-level science analysis", cleaned)

    for row in _load_jsonl("datasets/swebench_lite/swe-bench-lite.jsonl", limit=500):
        _add_hard_prompt("SWE-Bench Lite software engineering issue", row.get("problem_statement", ""))

    for row in _load_jsonl("datasets/longbench/repobench-p.jsonl", limit=200):
        _add_longbench_prompt("LongBench RepoBench code reasoning task", row)

    for row in _load_jsonl("datasets/longbench/lcc.jsonl", limit=200):
        _add_longbench_prompt("LongBench LCC code completion task", row)

    for row in _load_jsonl("datasets/longbench/qasper.jsonl", limit=200):
        _add_longbench_prompt("LongBench Qasper research QA task", row)

    # --- Additional open-ended datasets that naturally elicit long responses ---

    # AIME problems with varied solving approach instructions to maximize output length
    _aime_variant_suffixes = [
        "\n\nSolve this step by step. Show ALL intermediate calculations. "
        "After finding the answer, verify it by substituting back, then discuss "
        "at least 2 alternative approaches that could also solve this problem.",
        "\n\nProvide a complete solution with detailed reasoning for every step. "
        "Then generalize this problem: create a harder variant and solve it as well. "
        "Explain the connections between the original and the generalized version.",
        "\n\nFirst, solve this problem showing all work. Then explain the underlying "
        "mathematical principles and theorems used. Finally, discuss how this type of "
        "problem appears in advanced mathematics and provide 2 related examples with solutions.",
        "\n\nSolve this problem rigorously. For each step, justify why the approach is valid. "
        "After reaching the answer, analyze what happens when the constraints are modified. "
        "Provide a thorough discussion of edge cases and special scenarios.",
        "\n\nGive a detailed solution. Then write a complete proof explaining why your method "
        "works and why no simpler approach suffices. Discuss the historical context of this "
        "type of problem and its significance in competition mathematics.",
    ]
    aime_rows = (
        _load_json("datasets/aime2024/aime2024.json")[:200]
        + _load_json("datasets/aime2025/aime2025.json")[:200]
        + _load_json("datasets/aime2026/aime2026.json")[:200]
    )
    for row in aime_rows:
        problem = (row.get("problem") or "").strip()
        if problem:
            # Each AIME problem gets multiple variant instructions
            for variant in _aime_variant_suffixes:
                _add_hard_prompt("AIME competition math problem", problem + variant)

    for row in _load_json("datasets/humaneval/test.json")[:164]:
        prompt_text = (row.get("prompt") or "").strip()
        entry_point = row.get("entry_point", "")
        if prompt_text:
            task_desc = (
                f"{prompt_text}\n\n"
                "Write a complete, well-documented implementation of the function above. "
                "Include: type hints, docstrings, inline comments explaining the logic, "
                "edge case handling, and at least 3 additional test cases beyond the ones shown. "
                "Also write a brief explanation of your approach."
            )
            _add_hard_prompt(f"HumanEval code generation task ({entry_point})", task_desc)

    for row in _load_json("datasets/mbpp/mbpp.json")[:500]:
        text = (row.get("text") or "").strip()
        if text:
            task_desc = (
                f"{text}\n\n"
                "Write a complete Python implementation with thorough documentation. "
                "Include: the full function with type hints and docstring, inline comments, "
                "error handling for edge cases, and at least 3 example usages with assertions."
            )
            _add_hard_prompt("MBPP Python programming task", task_desc)

    if pool:
        _SUFFIX_PROMPT_POOL = pool
        return pool

    pool = [
        (
            "Advanced proof-writing task: Prove that for any positive integer n, "
            "the sum of the first n cubes equals the square of the sum of the first n integers. "
            "Provide a complete proof by mathematical induction, then also provide an alternative "
            "combinatorial proof. Discuss generalizations and related identities. "
            f"Write at least 2000 words showing all steps in full detail.{long_answer_instruction}"
        ),
        (
            "Software engineering task: Design and implement a thread-safe LRU cache in Python "
            "with support for expiration, batch operations, and event callbacks. Include the full "
            "implementation, unit tests, performance benchmarks, thread-safety analysis, and a "
            f"design document explaining all trade-offs. Write at least 2000 words.{long_answer_instruction}"
        ),
        (
            "请写一篇关于人工智能在医疗领域应用的详细研究报告。包括：1）当前主要应用场景分析；"
            "2）技术原理详解；3）伦理和隐私问题讨论；4）未来发展趋势预测；5）具体案例分析。"
            f"全文至少3000字，每个部分都需要详细展开论述。{long_answer_instruction}"
        ),
        (
            "Creative writing task: Write a complete science fiction short story (at least 3000 words) "
            "set in a future where humans have colonized Mars. Include detailed world-building, "
            "character development, a central conflict, and a resolution. Describe the technology, "
            f"social structures, and daily life in vivid detail.{long_answer_instruction}"
        ),
    ]
    _SUFFIX_PROMPT_POOL = pool
    return pool

def _get_random_suffix_prompt(target_token_budget: int, tokenizer) -> str:
    """
    Pick a random real question from the dataset pool and trim it to fit
    within the given token budget. Falls back to a short generic prompt
    if the pool is unavailable.
    """
    try:
        pool = _load_suffix_prompt_pool()
    except Exception:
        pool = []

    if not pool:
        return "Continue writing a detailed response."

    # Shuffle pick
    candidates = random.sample(pool, min(10, len(pool)))

    for candidate in candidates:
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        if len(tokens) <= target_token_budget:
            return candidate

    # If all candidates too long, trim the shortest one
    best = min(candidates, key=lambda c: len(tokenizer.encode(c, add_special_tokens=False)))
    tokens = tokenizer.encode(best, add_special_tokens=False)
    if len(tokens) > target_token_budget:
        trimmed = tokenizer.decode(tokens[:target_token_budget], skip_special_tokens=True)
        return trimmed

    return best

# Module logger
logger = get_logger(__name__)

class BenchmarkRunner:
    def __init__(self, placeholder, progress_bar, status_text, api_base_url, model_id, tokenizer_option, csv_filename, api_key, log_placeholder, provider, dashboard=None, output_placeholder=None, hf_tokenizer_model_id=None, latency_offset=0.0, thinking_enabled=None, thinking_budget=None, reasoning_effort=None, random_seed=None, skip_first_token_for_tps=False, template_tokens=0):
        self.placeholder, self.progress_bar, self.status_text = placeholder, progress_bar, status_text
        self.api_base_url = api_base_url
        self.model_id = model_id
        self.tokenizer_option = tokenizer_option
        self.latency_offset = latency_offset
        self.skip_first_token_for_tps = skip_first_token_for_tps
        self.template_tokens = self._normalize_template_tokens(template_tokens)
        self.completed_requests, self.total_requests = 0, 0
        self.tokenizer = None
        self.csv_file = csv_filename

        self.api_key = api_key
        self.log_placeholder = log_placeholder
        self.output_placeholder = output_placeholder
        self.hf_tokenizer_model_id = hf_tokenizer_model_id

        # Thinking/Reasoning parameters
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort

        # Random seed for reproducibility
        self.random_seed = random_seed

        # Initialize Structured Logger
        self.logger = BenchmarkLogger(max_entries=500)

        # Initialize Request Logger
        import os

        from core.request_logger import init_request_logger
        log_dir = os.path.join(os.path.dirname(self.csv_file), "api_logs")
        self.request_logger = init_request_logger(
            log_dir=log_dir,
            enabled=True,
            max_total_size_mb=500,  # default限制 500MB
        )

        # Capture the current script run context to pass to background threads
        self.ctx = get_script_run_ctx()

        # Start WebSocket Server (Singleton, safe to call multiple times)
        try:
            log_server.start()
        except Exception as e:
            logger.warning(f"Failed to start WebSocket server: {e}")

        # Use provider factory to create provider instance
        self.provider = get_provider(provider, api_base_url, api_key, model_id)

        # Real-time dashboard (optional)
        self.dashboard = dashboard

        self.results_list = []
        self.all_outputs = [] # Store all outputs for review
        self.last_output = None
        self._last_rendered_output = None

        self.combined_csv_columns = [
            "test_type", "concurrency", "round",
            "input_tokens_target",
            "context_length_target",
            "session_id", "ttft", "tps", "prefill_speed",
            "prefill_tokens", "decode_tokens", "api_prefill", "api_decode", "cache_hit_tokens",
            "token_calc_method", "error", "system_output_throughput", "system_input_throughput", "rps", "tpot_p95", "tpot_p99"
        ]

        # Cache for transformers tokenizer
        self._transformers_tokenizer = None

        # Database integration
        self._db_run = None  # Current TestRun in database
        self._db_manager = None  # Lazy loaded DatabaseManager
        self._test_type_for_db = None  # Current test type for database

    @staticmethod
    def _normalize_template_tokens(template_tokens):
        try:
            return max(0, int(template_tokens or 0))
        except (TypeError, ValueError):
            return 0

    def _prompt_generation_target(self, target_tokens, extra_overhead=0):
        target_tokens = int(target_tokens or 0)
        extra_overhead = max(0, int(extra_overhead or 0))
        return max(1, target_tokens - self.template_tokens - extra_overhead)

    def _get_db_manager(self):
        """GetDatabase管理器（LatencyLoad）"""
        if self._db_manager is None:
            from core.database import db_manager
            self._db_manager = db_manager
        return self._db_manager

    def _apply_seed(self):
        """
        Apply全局Random Seed以确保可复现性

        Set random, numpy, torch Random Seed。
        if未指定种子，use当前时间戳（not可复现）。
        """
        seed = self.random_seed

        if seed is None:
            # 未指定种子，not强制Set（保持原has随机行is）
            logger.debug("未SetRandom Seed，usedefault随机行is")
            return

        # Set Python random 模块种子
        random.seed(seed)

        # Set NumPy 种子
        np.random.seed(seed)

        # Set PyTorch 种子（if可用）
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # 确保确定性（可能影响性能）
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.debug(f"PyTorch 种子已Set: {seed}")
        except ImportError:
            pass

        logger.info(f"全局Random Seed已Apply: {seed}")

    def get_seed_info(self) -> dict:
        """Get种子信息用于记录"""
        return {
            "random_seed": self.random_seed,
            "seed_applied": self.random_seed is not None,
        }

    def _start_db_run(self, test_type: str, config: dict = None):
        """开始DatabaseTest运行记录"""
        try:
            db = self._get_db_manager()

            # ApplyRandom Seed（确保可复现性）
            self._apply_seed()

            # Get自动捕获系统信息
            from core.system_info import capture_system_info
            auto_sys_info = capture_system_info()

            # Merge用户Custom系统信息
            user_sys_info = self.get_system_info()
            merged_sys_info = {**auto_sys_info, **user_sys_info}

            # BuildConfigure
            full_config = {
                "api_base_url": self.api_base_url,
                "model_id": self.model_id,
                "tokenizer_option": self.tokenizer_option,
                "latency_offset": self.latency_offset,
                "template_tokens": self.template_tokens,
                "hf_tokenizer_model_id": self.hf_tokenizer_model_id,
                "thinking_enabled": self.thinking_enabled,
                "thinking_budget": self.thinking_budget,
                "reasoning_effort": self.reasoning_effort,
                "random_seed": self.random_seed,
            }
            if config:
                full_config.update(config)

            self._db_run = db.start_test_run(
                test_type=test_type,
                model_id=self.model_id,
                provider=getattr(self.provider, 'name', 'unknown') if self.provider else 'unknown',
                config=full_config,
                system_info=merged_sys_info,
            )
            self._test_type_for_db = test_type

            logger.info(f"DatabaseTest运行已Create: ID={self._db_run.id}")
            return self._db_run

        except Exception as e:
            logger.warning(f"CreateDatabaseTest运行失败: {e}")
            return None

    def _save_result_to_db(self, result: dict):
        """Save单Result到Database"""
        if self._db_run is None:
            return

        try:
            db = self._get_db_manager()
            db.save_result(self._db_run, result)
        except Exception as e:
            logger.warning(f"SaveResult到Database失败: {e}")

    def _update_db_progress(self):
        """UpdateDatabasein进度"""
        if self._db_run is None:
            return

        try:
            db = self._get_db_manager()
            db.update_run_progress(
                self._db_run,
                self.completed_requests,
                self.total_requests,
                0  # failed count
            )
        except Exception as e:
            logger.warning(f"UpdateDatabase进度失败: {e}")

    def _complete_db_run(self, success: bool = True):
        """完成DatabaseTest运行"""
        if self._db_run is None:
            return

        try:
            db = self._get_db_manager()
            db.complete_test_run(self._db_run, success, calculate_stats=True)
            logger.info(f"DatabaseTest运行Completed: ID={self._db_run.id}, success={success}")
        except Exception as e:
            logger.warning(f"完成DatabaseTest运行失败: {e}")
        finally:
            self._db_run = None

    def _add_result(self, result: dict, csv_columns: list):
        """
        AddTest Results（统一Process CSV、列表andDatabase）

        Args:
            result: Result字典
            csv_columns: CSV 列定义
        """
        # Add到列表
        self.results_list.append(result)

        # Save到 CSV
        append_to_csv(result, csv_columns, self.csv_file)

        # Save到Database
        self._save_result_to_db(result)

    def _batch_save_results_to_db(self):
        """批量Save所hasResult到Database"""
        if self._db_run is None or not self.results_list:
            return

        try:
            db = self._get_db_manager()
            # 只Save还没hasSaveResult（viaCheckDatabaseinResult数量）
            existing_count = db.results.count("run_id = ?", (self._db_run.id,))
            new_results = self.results_list[existing_count:]

            if new_results:
                db.save_results_batch(self._db_run, new_results)
                logger.info(f"批量Save {len(new_results)} 条Result到Database")
        except Exception as e:
            logger.warning(f"批量SaveResult失败: {e}")

    def get_system_info(self):
        """Get system information (Model & Provider + Custom User Input)."""
        # Default empty
        info = {
            "system": "",
            "processor": "",
            "python": "",
            "hostname": "",
            "memory": "",
            "cpu_count": "",
            "gpu": "",
            "mainboard": ""
        }

        # Try to load user custom overrides from Session State
        try:
            if hasattr(st, 'session_state') and 'custom_sys_info' in st.session_state:
                custom = st.session_state.custom_sys_info
                for key in ['processor', 'mainboard', 'memory', 'gpu', 'system', 'engine_name']:
                    if custom.get(key):
                        info[key] = custom.get(key)
        except Exception:
            pass # Safe fallback

        # Default engine_name to empty string - provider name is NOT the inference engine
        # Only use it if not already set by custom overrides
        if 'engine_name' not in info:
            info['engine_name'] = ""

        info['model_name'] = self.model_id

        return info

    def _infer_hf_model_id(self):
        """Infer HuggingFace model ID from model_id using mapping."""
        model_id_lower = self.model_id.lower()
        logger.debug(f"Inferring HF ID for '{model_id_lower}'")
        for key, hf_id in HF_MODEL_MAPPING.items():
            if key.lower() in model_id_lower:
                logger.debug(f"Match found: {key} -> {hf_id}")
                return hf_id
        logger.debug("No match found in HF_MODEL_MAPPING")
        return None

    def _safe_log(self, log_func, message):
        """Safely log to Streamlit placeholder, handling NoSessionContext in threads."""
        if log_func is None:
            return
        try:
            log_func(message)
        except Exception:
            # Silently ignore if no Streamlit context (e.g., in thread pool)
            logger.debug(f"Streamlit log skipped (no context): {message[:50]}...")

    def _get_tokenizer(self):
        """Get tokenizer: Custom HF -> Auto-Inferred HF -> transformers fallback"""

        # Priority 1: Custom HuggingFace Tokenizer
        if self.tokenizer_option == "HuggingFace Tokenizer" and self.hf_tokenizer_model_id:
            try:
                if not hasattr(self, '_transformers_tokenizer') or self._transformers_tokenizer is None or \
                   (hasattr(self._transformers_tokenizer, 'name_or_path') and self._transformers_tokenizer.name_or_path != self.hf_tokenizer_model_id):
                    self._safe_log(self.log_placeholder.info if self.log_placeholder else None,
                                   f"currentlyLoad HuggingFace Tokenizer: {self.hf_tokenizer_model_id}...")

                # Use shared cached loader
                self._transformers_tokenizer = get_cached_tokenizer(self.hf_tokenizer_model_id)

                if self._transformers_tokenizer:
                    self._safe_log(self.log_placeholder.success if self.log_placeholder else None,
                                   f"succeededLoad Tokenizer: {self.hf_tokenizer_model_id}")
                    return self._transformers_tokenizer
                else:
                    raise ValueError("Loader returned None")

            except Exception as e:
                # Enhanced error message for tokenizer loading
                error_info = get_error_info(
                    e,
                    context=f"Tokenizer: {self.hf_tokenizer_model_id}",
                    language="zh"
                )
                self._safe_log(self.log_placeholder.error if self.log_placeholder else None,
                               f"{error_info['title']}\n\n{error_info['details']}\n\nSolution:\n" + "\n".join(f"• {s}" for s in error_info['solutions']))
                # Fall through to other methods or return None

        # Priority 2: Auto-infer HF Tokenizer (Universal)
        inferred_id = self._infer_hf_model_id()
        if inferred_id:
            try:
                if not hasattr(self, '_transformers_tokenizer') or self._transformers_tokenizer is None or \
                   (hasattr(self._transformers_tokenizer, 'name_or_path') and self._transformers_tokenizer.name_or_path != inferred_id):
                    self._safe_log(self.log_placeholder.info if self.log_placeholder else None,
                                   f"自动检测并Load HuggingFace Tokenizer: {inferred_id}...")

                self._transformers_tokenizer = get_cached_tokenizer(inferred_id)

                if self._transformers_tokenizer:
                    self._safe_log(self.log_placeholder.success if self.log_placeholder else None,
                                   f"succeededLoad自动推断 Tokenizer: {inferred_id}")
                    return self._transformers_tokenizer
                else:
                     raise ValueError("Loader returned None")

            except Exception as e:
                # Enhanced error message for auto-inferred tokenizer
                error_info = get_error_info(
                    e,
                    context=f"Auto-inferred Tokenizer: {inferred_id}",
                    language="zh"
                )
                self._safe_log(self.log_placeholder.warning if self.log_placeholder else None,
                               f"{error_info['title']}: {error_info['details']}\nwill回退到估算模式。")

        # Priority 3: Transformers Fallback (GPT-2)
        try:
            from transformers import AutoTokenizer
            # Use GPT-2 tokenizer as a reasonable default
            if not hasattr(self, '_transformers_tokenizer') or self._transformers_tokenizer is None:
                self._transformers_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self._safe_log(st.info, "已Load transformers GPT-2 tokenizer 作is托底。")
            return self._transformers_tokenizer
        except Exception as tf_error:
            self._safe_log(st.error, f"transformers tokenizer 也Load failed: {tf_error}")
            return None

    def _calibrate_prompt(self, target_tokens, suffix="", _tokenizer=None):
        """
        Strictly calibrate prompt length by adding/removing random noise at the beginning.
        Target Error: 0 tokens (Strict).

        Args:
            target_tokens: Target token count
            suffix: Suffix to append
            _tokenizer: Pre-loaded tokenizer (internal use, avoids thread-safety issues)
        """
        # Use pre-loaded tokenizer if provided (for thread-safe parallel execution)
        # Otherwise fall back to _get_tokenizer() for backward compatibility
        tokenizer = _tokenizer if _tokenizer is not None else self._get_tokenizer()
        if not tokenizer:
            return suffix

        # Helper to encode
        def get_count(text):
            try:
                if hasattr(tokenizer, 'encode_plus'):
                    return len(tokenizer.encode(text, add_special_tokens=False))
                return len(tokenizer.encode(text))
            except Exception:
                return 0

        # 1. Initial Coarse Generation
        suffix_count = get_count(suffix)
        body_target_tokens = target_tokens - suffix_count
        if body_target_tokens < 1:
            body_target_tokens = 1

        # Generate initial body (try to get close)
        # Pass the pre-loaded tokenizer to avoid thread-safety issues
        body_text, _, _ = self._get_text_for_token_count(body_target_tokens, force_random=True, _tokenizer=tokenizer)

        # 2. Fine-grained Character Adjustment Loop
        # Only needed when body + suffix doesn't exactly match target.
        # For large texts, avoid re-encoding the full text — adjust the prefix only.

        chars = "The quick brown fox jumps over the lazy dog. "

        # Initial full encode
        current_count = get_count(body_text + suffix)
        diff = current_count - target_tokens

        if diff == 0:
            return body_text + suffix

        # For small diffs (typical ±few tokens), adjust by adding/removing a
        # small prefix. Only encode the adjusted prefix, not the full text.
        max_iter = 20
        for _i in range(max_iter):
            if diff == 0:
                return body_text + suffix

            if diff > 0:
                # Too long: trim from start
                chars_to_remove = max(1, int(diff * 2))
                body_text = body_text[chars_to_remove:]
            else:
                # Too short: prepend chars
                chars_to_add = max(1, int(abs(diff) * 3))
                noise = "".join(random.choices(chars, k=chars_to_add))
                body_text = noise + body_text

            current_count = get_count(body_text + suffix)
            diff = current_count - target_tokens

        return body_text + suffix

    def _get_text_for_token_count(self, target_tokens, force_random=False, _tokenizer=None):
        """
        Generate text that strictly matches target_tokens using the current tokenizer.

        Args:
            target_tokens: Target token count
            force_random: Whether to force random generation
            _tokenizer: Pre-loaded tokenizer (internal use, avoids thread-safety issues)
        """
        target_tokens = int(target_tokens)
        # Use pre-loaded tokenizer if provided (for thread-safe parallel execution)
        tokenizer = _tokenizer if _tokenizer is not None else self._get_tokenizer()

        if not tokenizer:
            self._safe_log(st.error, "所has tokenizer（tiktoken, transformers）都not可用。no法Generate精确 Token 长度文本。")
            raise RuntimeError("No tokenizer available - cannot generate token-based prompts")

        try:
            # Helper for clean encoding
            def encode_no_special(text):
                if hasattr(tokenizer, 'encode'):
                    return tokenizer.encode(text, add_special_tokens=False)
                return []

            # 1. Define Suffix Options — use real questions from datasets

            # Separator to mark the actual task after noise padding
            suffix_separator = "---\n\n"

            # "Heavy" Suffix: Real question from dataset pool (>= 200 tokens)
            # "Medium" Suffix: Shorter real question (60 - 200 tokens)
            # "Micro" Suffix: Short prompt (20 - 60 tokens)
            suffix_micro = "请直接创作一部超长篇科幻小说，描写人类在火星建立殖民地的故事。"

            # "Nano" Suffix (~4-8 tokens): For tiny prefill (8 - 20)
            suffix_nano = "续写万字小说"

            # "Pico" Suffix (~1 token): For single-digit prefill (1 - 8)
            suffix_pico = "写"

            # 2. Select Suffix based on available space
            # Wrapper Context to avoid refusal
            prefix_str = ""
            suffix_wrapper = "\n"

            prefix_tokens = encode_no_special(prefix_str)
            suffix_wrapper_tokens = encode_no_special(suffix_wrapper)

            wrapper_overhead = len(prefix_tokens) + len(suffix_wrapper_tokens)
            effective_target = target_tokens - wrapper_overhead

            selected_suffix = ""
            suffix_tokens = []

            if target_tokens >= 200:
                # Use real question from dataset pool with separator
                sep_tokens = encode_no_special(suffix_separator)
                budget = effective_target - len(sep_tokens)
                if budget > 10:
                    question = _get_random_suffix_prompt(budget, tokenizer)
                    question_tokens = encode_no_special(question)
                    # Append minimum-output enforcement instruction
                    output_enforcement = (
                        "\n\n请务必写出2000字以上的详细回答。你必须展示完整的推理过程，"
                        "不能只给出简短答案。"
                    )
                    enforcement_tokens = encode_no_special(output_enforcement)
                    if len(sep_tokens) + len(question_tokens) + len(enforcement_tokens) <= effective_target:
                        selected_suffix = suffix_separator + question + output_enforcement
                        suffix_tokens = sep_tokens + question_tokens + enforcement_tokens
                    else:
                        # Not enough room for enforcement; use question as-is
                        selected_suffix = suffix_separator + question
                        suffix_tokens = sep_tokens + question_tokens
                else:
                    # Not enough budget for separator + question, fall back
                    selected_suffix = suffix_micro
                    suffix_tokens = encode_no_special(selected_suffix)

            if not suffix_tokens and target_tokens >= 60:
                # Medium: try a short real question without separator
                budget = effective_target
                question = _get_random_suffix_prompt(budget, tokenizer)
                question_tokens = encode_no_special(question)
                if len(question_tokens) <= effective_target:
                    selected_suffix = question
                    suffix_tokens = question_tokens

            if not suffix_tokens and target_tokens >= 20:
                selected_suffix = suffix_micro
                suffix_tokens = encode_no_special(selected_suffix)

            if not suffix_tokens and target_tokens >= 8:
                selected_suffix = suffix_nano
                suffix_tokens = encode_no_special(selected_suffix)

            if not suffix_tokens and target_tokens >= 1:
                selected_suffix = suffix_pico
                suffix_tokens = encode_no_special(selected_suffix)

            # Double check: Does it fit?
            if len(suffix_tokens) > effective_target:
                # Trim to fit
                suffix_tokens = suffix_tokens[:max(effective_target, 0)]
                if suffix_tokens:
                    selected_suffix = tokenizer.decode(suffix_tokens, skip_special_tokens=True)
                else:
                    selected_suffix = ""
                    suffix_tokens = []

            # 3. Calculate Noise
            noise_target_tokens = effective_target - len(suffix_tokens)

            # Generate unique natural-looking filler text for each prompt to avoid cache hits.
            # Strategy: parameterized sentence templates with random numbers/names produce
            # virtually infinite unique sentences. Generate all text first, encode once.

            _names = [
                "Chen", "Patel", "Mueller", "Tanaka", "Silva", "Johansson", "Kim",
                "Dubois", "Navarro", "Ivanova", "Okafor", "Lindqvist", "Bergstrom",
                "Fernandez", "Nakamura", "Andersen", "Kowalski", "Morales", "Sato", "Larsson",
            ]
            _fields = [
                "computational biology", "quantum information theory", "materials science",
                "climate modeling", "neural network optimization", "signal processing",
                "distributed systems", "computational linguistics", "robotics",
                "cryptography", "computer vision", "reinforcement learning",
                "graph theory", "optimization theory", "biomedical engineering",
                "fluid dynamics", "astrophysics", "genomics", "topology",
                "information retrieval", "natural language processing", "database systems",
            ]
            _methods = [
                "cross-validation with 10-fold splits", "Bayesian hierarchical modeling",
                "spectral clustering analysis", "Monte Carlo simulation",
                "gradient-based optimization", "principal component analysis",
                "kernel density estimation", "Markov chain Monte Carlo sampling",
                "finite element analysis", "transfer learning from pretrained models",
                "ensemble methods with bagging", "attention-based neural architectures",
                "variational inference", "convex relaxation techniques",
                "stochastic gradient descent with momentum",
            ]
            _metrics = [
                "F1 score", "root mean squared error", "area under the curve",
                "mean absolute percentage error", "R-squared coefficient",
                "likelihood ratio", "Cohen's kappa", "Matthews correlation coefficient",
                "normalized mutual information", "Silhouette score",
            ]
            _adjs = [
                "significant", "notable", "remarkable", "substantial", "measurable",
                "pronounced", "moderate", "consistent", "unexpected", "intriguing",
            ]

            def _gen_sentence():
                """Generate one unique natural sentence using randomized parameters."""
                r = random.random()
                if r < 0.15:
                    return (f"相关研究由{random.choice(_names)}团队在{random.randint(2021, 2026)}年开展，"
                            f"针对{random.choice(_fields)}领域进行了深入探索，"
                            f"实验在{random.randint(10, 1000)}组对照条件下完成了验证。")
                elif r < 0.3:
                    return (f"Subsequent analysis by {random.choice(_names)} and {random.choice(_names)} "
                            f"achieved a {random.choice(_metrics)} of {random.uniform(0.7, 0.99):.3f} "
                            f"on the held-out test set after {random.randint(50, 500)} training epochs.")
                elif r < 0.45:
                    return (f"Phase {random.randint(1, 4)} focused on {random.choice(_fields)} "
                            f"with {random.randint(10, 200)} concurrent workers, "
                            f"yielding a {random.choice(_adjs)} {random.uniform(5, 40):.1f}% improvement.")
                elif r < 0.6:
                    return (f"The ablation study showed removing the {random.choice(_methods)} component "
                            f"reduced {random.choice(_metrics)} by {random.uniform(5, 25):.1f}%, "
                            f"confirming its importance for overall performance.")
                elif r < 0.75:
                    return (f"A comparative evaluation across {random.randint(3, 8)} baselines "
                            f"on {random.choice(_fields)} benchmarks gave a margin of "
                            f"{random.uniform(1, 15):.1f}% (p < {random.uniform(0.001, 0.05):.4f}).")
                elif r < 0.85:
                    return (f"为进一步验证鲁棒性，研究团队在{random.randint(3, 10)}个公开数据集上"
                            f"进行了交叉验证，{random.choice(_metrics)}保持在"
                            f"{random.uniform(0.8, 0.98):.3f}的稳定水平。")
                else:
                    return (f"In their {random.randint(2020, 2026)} study on {random.choice(_fields)}, "
                            f"{random.choice(_names)} et al. reported a {random.choice(_adjs)} improvement "
                            f"of {random.uniform(5, 45):.1f}% using {random.choice(_methods)}. "
                            f"The experiment involved {random.randint(10, 2000)} samples.")

            # Estimate chars/token ratio from a small sample (2 encode calls total)
            sample_text = " ".join(_gen_sentence() for _ in range(50))
            sample_tokens = encode_no_special(sample_text)
            if sample_tokens:
                chars_per_token = len(sample_text) / len(sample_tokens)
            else:
                chars_per_token = 3.0

            # Generate enough text with 10% margin, then encode once
            est_chars = int(noise_target_tokens * chars_per_token * 1.1) + 500
            sentences = []
            total_chars = 0
            while total_chars < est_chars:
                s = _gen_sentence()
                sentences.append(s)
                total_chars += len(s)

            noise_text = " ".join(sentences)
            noise_tokens_all = encode_no_special(noise_text)
            if not noise_tokens_all:
                noise_tokens_all = encode_no_special("DefaultSeed")
            # Trim to exact target
            final_noise_tokens = noise_tokens_all[:noise_target_tokens]

            # 4. Combine
            if noise_target_tokens > 0:
                final_tokens = prefix_tokens + final_noise_tokens + suffix_wrapper_tokens + suffix_tokens
            else:
                # If no space for noise, just return prefix+suffix (might overshoot if overhead > target, but minimal risk for >100 tokens)
                # If target is extremely small (e.g. 10), effective might be negative.
                if effective_target < 0:
                     # Fallback for tiny targets: Just return random noise of target length, no wrapper
                     final_tokens = (final_noise_tokens if len(final_noise_tokens)>0 else []) + suffix_tokens
                     final_tokens = final_tokens[:target_tokens] # Hard clip
                else:
                     final_tokens = prefix_tokens + final_noise_tokens + suffix_wrapper_tokens + suffix_tokens

            # Decode
            prompt_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
            actual_tokens = len(final_tokens)

            # Extract a short summary of the question used (for display)
            question_summary = ""
            if selected_suffix and target_tokens >= 60:
                # Strip the separator prefix to get just the question
                display_text = selected_suffix
                if display_text.startswith("--- BENCHMARK PADDING END ---"):
                    display_text = display_text.split("\n", 2)[-1] if "\n" in display_text else display_text
                first_line = display_text.split("\n")[0].strip()
                if first_line:
                    question_summary = first_line[:80]

            return prompt_text, actual_tokens, question_summary

        except Exception as e:
            st.error(f"use tokenizer Process文本失败: {e}。")
            raise

    def _update_log(self, message, level=LogLevel.INFO, **kwargs):
        """
        Enhanced log update method.
        1. Logs to BenchmarkLogger
        2. Broadcasts via WebSocket
        3. Updates Streamlit UI
        """
        try:
            # 1. Log to memory
            entry = self.logger.log(level, message, **kwargs)

            # 2. Broadcast via WebSocket
            log_server.broadcast(entry.to_dict())

            # 3. Update UI (Compact Mode)
            # Ensure we have the script context if running in a background thread
            if self.log_placeholder:
                try:
                    if not get_script_run_ctx() and self.ctx:
                        add_script_run_ctx(threading.current_thread(), self.ctx)

                    render_log_viewer(
                        self.logger,
                        placeholder=self.log_placeholder,
                        max_display=50,
                        compact_mode=True
                    )
                except Exception as ui_error:
                    logger.debug(f"UI update failed (likely thread context issue): {ui_error}")

        except Exception as e:
            logger.debug(f"Failed to update log: {e}")

    def _get_empty_metrics(self):
        return empty_metrics()

    def _calculate_metrics(self, start_time, first_token_time, end_time, completion_tokens, token_timestamps=None):
        """Calculate TTFT, TPS, and TPOT using monotonic timestamps."""
        return calculate_request_metrics(
            start_time,
            first_token_time,
            end_time,
            completion_tokens,
            latency_offset=self.latency_offset,
            token_timestamps=token_timestamps,
            skip_first_token=self.skip_first_token_for_tps,
        ).as_tuple()

    def _decode_tokens_for_tps(self, completion_tokens, token_timestamps=None):
        """Return the decode token count that matches TPS/TPOT timing semantics."""
        if completion_tokens <= 0:
            return 0

        timestamps = list(token_timestamps or [])
        if self.skip_first_token_for_tps and len(timestamps) >= 2:
            return max(0, completion_tokens - 1)

        return completion_tokens

    @staticmethod
    def _result_decode_tokens_for_tps(res):
        """Read the skip-adjusted decode token count from a request result."""
        return res.get('decode_tokens_for_tps', res.get('decode_tokens', 0)) or 0

    def _get_cache_hit_tokens(self, usage_info):
        """从 usage_info in提取Cache Hit Token 数，兼容not同 API 结构

        Priority:
        1. OpenAI Standard (prompt_tokens_details.cached_tokens)
        2. Direct keys (cache_hit_tokens, etc.)
        3. Anthropic (cache_read_input_tokens)
        """
        if not usage_info:
            return 0

        # 1. OpenAI/vLLM 标准嵌套结构 (Qwen3-Coder, DeepSeek-V3)
        prompt_details = usage_info.get("prompt_tokens_details")
        if prompt_details and isinstance(prompt_details, dict):
            hit = prompt_details.get("cached_tokens", 0)
            if hit:
                return hit

        # 2. 直接in usage 根目录 (MiMo, 部分 SiliconFlow)
        for key in ["cache_hit_tokens", "prompt_cache_hit_tokens", "disk_cache_hit_tokens"]:
            if usage_info.get(key):
                return usage_info.get(key)

        # 3. Anthropic 风格
        if usage_info.get("cache_read_input_tokens"):
             return usage_info.get("cache_read_input_tokens")

        return 0

    def _calculate_tokens(self, prompt, full_response_content, usage_info=None):
        prompt_tokens = 0
        completion_tokens = 0
        token_calc_method = "未知"
        cache_hit_tokens = 0

        # 记录 API Raw dataBackup
        api_usage = {
            "prompt_tokens": usage_info.get("prompt_tokens") if usage_info else None,
            "completion_tokens": usage_info.get("completion_tokens") if usage_info else None
        }

        # Priority 1: API Usage (最准确，包含 Chat Template 开销)
        # if API Returnhas效 usage 信息，直接采用，not再use本地 tokenizer
        if usage_info and usage_info.get("prompt_tokens") is not None:
             prompt_tokens = usage_info.get("prompt_tokens", 0)
             completion_tokens = usage_info.get("completion_tokens", 0)
             cache_hit_tokens = self._get_cache_hit_tokens(usage_info)
             token_calc_method = "API (usage field)"
             return prompt_tokens, completion_tokens, token_calc_method, cache_hit_tokens, api_usage

        # Get Tokenizer (Explicit or Fallback)
        tokenizer = self._get_tokenizer()
        is_hf_ready = tokenizer and hasattr(tokenizer, 'encode_plus')

        # Priority 2: HuggingFace Tokenizer (Manual or Auto-Inferred)
        # 仅当 API 未Return usage 时，才use匹配本地 Tokenizer
        if is_hf_ready:
            inferred_id = self._infer_hf_model_id()
            if (self.tokenizer_option == "HuggingFace Tokenizer" and self.hf_tokenizer_model_id) or inferred_id:
                try:
                    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                    completion_tokens = len(tokenizer.encode(full_response_content, add_special_tokens=False))

                    method_id = self.hf_tokenizer_model_id if (self.tokenizer_option == "HuggingFace Tokenizer" and self.hf_tokenizer_model_id) else inferred_id
                    token_calc_method = f"HF ({method_id})" if self.tokenizer_option == "HuggingFace Tokenizer" else f"HF-Auto ({method_id})"

                    cache_hit_tokens = self._get_cache_hit_tokens(usage_info)
                    return prompt_tokens, completion_tokens, token_calc_method, cache_hit_tokens, api_usage
                except Exception:
                    pass

        # Priority 3: Fallback Tokenizer (GPT-2 or tiktoken)
        if tokenizer:
            is_transformers = hasattr(tokenizer, 'encode_plus')
            if is_transformers:
                token_calc_method = "transformers (GPT-2)"
                try:
                    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                    completion_tokens = len(tokenizer.encode(full_response_content, add_special_tokens=False))
                except Exception:
                    pass
            else:
                token_calc_method = f"tiktoken ({getattr(tokenizer, 'name', 'unknown')})"
                try:
                    prompt_tokens = len(tokenizer.encode(prompt))
                    completion_tokens = len(tokenizer.encode(full_response_content))
                except Exception:
                    pass

        return prompt_tokens, completion_tokens, token_calc_method, cache_hit_tokens, api_usage

    async def get_completion(self, client, session_id, prompt, max_tokens, barrier=None):
        """Get completion from the configured provider."""
        # Update dashboard - request starting
        if self.dashboard:
            self.dashboard.update_request_state(session_id, 'running')

        # Log start
        provider_name = self.provider.__class__.__name__.replace("Provider", "")
        self._update_log(f"Session {session_id} started ({provider_name})", level=LogLevel.INFO, session_id=str(session_id))

        # Call provider's get_completion with thinking parameters
        def simple_log_callback(msg):
            self._update_log(msg, level=LogLevel.DEBUG, session_id=str(session_id))

        # Build kwargs with thinking parameters
        kwargs = {}
        if self.thinking_enabled is not None:
            kwargs['thinking_enabled'] = self.thinking_enabled
        if self.thinking_budget is not None:
            kwargs['thinking_budget'] = self.thinking_budget
        if self.reasoning_effort is not None:
            kwargs['reasoning_effort'] = self.reasoning_effort
        if barrier is not None:
            kwargs['_barrier'] = barrier

        result = await self.provider.get_completion(
            client, session_id, prompt, max_tokens,
            log_callback=simple_log_callback,
            **kwargs
        )

        # Check if error occurred
        if result.get("error"):
            error_msg = result["error"]

            # Enhanced error logging with detailed information
            error_info = result.get("error_info")
            if error_info:
                # Log enhanced error message
                solutions_text = "\n".join(f"• {s}" for s in error_info.get("solutions", [])[:3])  # Show first 3 solutions
                enhanced_log = (
                    f"Session {session_id} 失败: {error_info['title']}\n"
                    f"详情: {error_info['details']}\n"
                    f"Solution:\n{solutions_text}"
                )
                self._update_log(enhanced_log, level=LogLevel.ERROR, session_id=str(session_id), error=error_msg)
            else:
                self._update_log(f"Session {session_id} failed: {error_msg}", level=LogLevel.ERROR, session_id=str(session_id), error=error_msg)

            # Update dashboard - request failed
            if self.dashboard:
                self.dashboard.update(
                    timestamp=time.time(),
                    ttft=0,
                    tps=0,
                    status='failed',
                    session_id=session_id
                )

            if error_msg == "UserCancelled":
                return {"session_id": session_id, "error": "UserCancelled", **self._get_empty_metrics()}
            return {"session_id": session_id, "error": error_msg, **self._get_empty_metrics()}

        # Extract provider response
        # Note: start_time, first_token_time, end_time are monotonic timestamps
        # created_at is absolute timestamp (time.time())
        created_at = result.get("created_at", time.time())
        start_time = result["start_time"]
        first_token_time = result["first_token_time"]
        end_time = result["end_time"]
        full_response_content = result["full_response_content"]
        usage_info = result.get("usage_info")
        token_timestamps = result.get("token_timestamps")

        # Save output for preview
        self.last_output = full_response_content

        # Store in all_outputs
        self.all_outputs.append({
            "session_id": session_id,
            "prompt": prompt,
            "output": full_response_content,
            "timestamp": created_at + (end_time - start_time)  # Calculate approx absolute end time
        })

        # Calculate tokens and metrics
        prompt_tokens, completion_tokens, token_calc_method, cache_hit_tokens, api_usage = self._calculate_tokens(prompt, full_response_content, usage_info)
        ttft, tps, tpot, tpot_p95, tpot_p99, generation_time = self._calculate_metrics(start_time, first_token_time, end_time, completion_tokens, token_timestamps)
        decode_tokens_for_tps = self._decode_tokens_for_tps(completion_tokens, token_timestamps)

        # Log success with metrics
        api_p = api_usage.get("prompt_tokens")
        api_d = api_usage.get("completion_tokens")

        metrics = {
            "ttft": ttft,
            "tps": tps,
            "tpot": tpot,
            "prefill": f"{prompt_tokens} (API:{api_p})" if api_p else prompt_tokens,
            "decode": f"{completion_tokens} (API:{api_d})" if api_d else completion_tokens,
            "total_time": max(0.000001, (end_time - start_time) - self.latency_offset)
        }
        self._update_log(
            f"Session {session_id} DONE",
            level=LogLevel.SUCCESS,
            session_id=str(session_id),
            metrics=metrics
        )

        # Update dashboard - request completed successfully
        if self.dashboard:
            self.dashboard.update(
                timestamp=created_at + (end_time - start_time),
                ttft=ttft,
                tps=tps,
                status='success',
                session_id=session_id
            )

        return {
            "session_id": session_id,
            "ttft": ttft,
            "tps": tps,
            "tpot": tpot,
            "tpot_p95": tpot_p95,
            "tpot_p99": tpot_p99,
            "prefill_tokens": prompt_tokens,
            "decode_tokens": completion_tokens,
            "decode_tokens_for_tps": decode_tokens_for_tps,
            "api_prefill": api_p,
            "api_decode": api_d,
            "decode_time": generation_time,
            "total_time": max(0.000001, (end_time - start_time) - self.latency_offset),
            "start_time": start_time, # monotonic
            "end_time": end_time, # monotonic
            "created_at": created_at, # absolute
            "cache_hit_tokens": cache_hit_tokens,
            "token_calc_method": token_calc_method,
            "first_token_time": first_token_time,
            "prompt_text": prompt,
            "output_text": full_response_content,
            "error": None
        }

    def update_ui(self):
        self.progress_bar.progress(self.completed_requests / self.total_requests if self.total_requests > 0 else 0)

        if not self.results_list:
            return

        try:
            df = pd.DataFrame(self.results_list)

            if not df.empty:
                df = reorder_dataframe_columns(df)

            st.session_state.results_df = df
            with self.placeholder.container():
                st.subheader("实时Result")

                # Format for display (Same style as PageLayout)
                test_type = st.session_state.get('current_test_type')
                display_df = format_results_for_display(df, test_type)

                st.dataframe(display_df, width="stretch")

            # Update output preview if available
            if self.output_placeholder and self.last_output and self.last_output != self._last_rendered_output:
                with self.output_placeholder.container():
                    with st.expander(f"📝 最新输出预览 (Session {self.results_list[-1].get('session_id', 'Unknown')})", expanded=False):
                        st.code(self.last_output, language=None, wrap_lines=True)
                self._last_rendered_output = self.last_output

        except Exception as e:
            st.warning(f"Update UI 失败: {e}")

    def _check_control_signal(self):
        """
        Check暂停/停止信号

        Returns:
            str | None: 'pause' | 'stop' | None
        """
        # 使用全局变量检查，更可靠
        try:
            from core.providers.openai import is_pause_requested, is_stop_requested
            stop = is_stop_requested()
            pause = is_pause_requested()
            # 每次检查都打印调试信息
            if stop or pause:
                self._update_log(f"[SIGNAL] stop={stop}, pause={pause}", level=LogLevel.INFO)
            if stop:
                return 'stop'
            if pause:
                return 'pause'
        except ImportError as e:
            self._update_log(f"[SIGNAL] Import error: {e}", level=LogLevel.ERROR)

        # 后备：检查 session_state
        ss_stop = st.session_state.get('stop_requested', False)
        ss_pause = st.session_state.get('pause_requested', False)
        if ss_stop or ss_pause:
            self._update_log(f"[SIGNAL] session_state: stop={ss_stop}, pause={ss_pause}", level=LogLevel.INFO)
        if ss_stop:
            return 'stop'
        if ss_pause:
            return 'pause'
        return None

    def _save_progress(self, test_type: str, current_index: int, total_samples: int,
                       pending_prompts: list, status: str = "PAUSED"):
        """
        SaveTest进度到文件

        Args:
            test_type: Test Type
            current_index: 当前执行到Index
            total_samples: 总Sample count
            pending_prompts: 待执行 Prompt 列表
            status: Save state ('PAUSED' | 'CANCELLED')
        """
        import json
        from datetime import datetime
        from pathlib import Path

        try:
            progress_dir = Path("test_progress")
            progress_dir.mkdir(exist_ok=True)

            test_id = st.session_state.get('current_test_id')
            if not test_id:
                test_id = f"{test_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.current_test_id = test_id

            progress_data = {
                "test_id": test_id,
                "test_type": test_type,
                "status": status,
                "current_index": current_index,
                "total_samples": total_samples,
                "completed_results": self.results_list.copy(),
                "pending_prompts": pending_prompts,
                "test_config": {
                    "api_base_url": self.api_base_url,
                    "model_id": self.model_id,
                    "max_tokens": getattr(self, '_current_max_tokens', 512),
                    "concurrency": getattr(self, '_current_concurrency', 1),
                    "latency_offset": self.latency_offset,
                    "template_tokens": self.template_tokens,
                    "tokenizer_option": self.tokenizer_option,
                    "hf_tokenizer_model_id": self.hf_tokenizer_model_id,
                },
                "start_time": getattr(self, '_test_start_time', time.time()),
                "pause_time": time.time(),
            }

            progress_file = progress_dir / f"{test_id}.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2, default=str)

            # 同时保存到 session_state 的 resume_data，用于 Resume 功能
            st.session_state.resume_data = {
                'completed_results': self.results_list.copy(),
                'current_index': current_index,
                'total_samples': total_samples,
                'test_id': test_id,
                'test_type': test_type,
            }

            self._update_log(f"进度Saved到 {progress_file}", level=LogLevel.INFO)
            self._update_log(f"Resume data saved: {len(self.results_list)} results, will skip first {current_index}", level=LogLevel.INFO)
            self._update_log(f"[DEBUG] session_state.resume_data set: current_index={current_index}, results_count={len(self.results_list)}", level=LogLevel.INFO)
            return True

        except Exception as e:
            self._update_log(f"Save进度失败: {e}", level=LogLevel.ERROR)
            return False

    async def _run_concurrency_batch(self, client, prompts, max_tokens, concurrency, session_id_start):
        start_batch_time = time.monotonic()

        # Determine prompts for each request
        if isinstance(prompts, list):
            if len(prompts) != concurrency:
                # Fallback: cycle through prompts if lengths don't match, or just use first?
                # Better to error or extend. Let's extend/cycle.
                prompts = (prompts * (concurrency // len(prompts) + 1))[:concurrency]
            request_prompts = prompts
        else:
            # Single prompt string -> repeat
            request_prompts = [prompts] * concurrency

        # 预创建共享 HTTP 客户端，避免每个 task 各自创建导致串行阻塞（每个 ~0.3s）
        own_shared_client = False
        if client is None:
            import httpx
            client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(), timeout=600.0)
            own_shared_client = True

        # 创建同步屏障，让所有并发请求完成准备工作后近乎同时发送 HTTP 请求
        barrier = asyncio.Barrier(concurrency)

        try:
            tasks = [self.get_completion(client, session_id_start + i, request_prompts[i], max_tokens, barrier=barrier) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)
        finally:
            if own_shared_client:
                await client.aclose()

        end_batch_time = time.monotonic()

        max(0.001, end_batch_time - start_batch_time)

        # Calculate totals for the batch
        total_output_tokens = 0
        total_input_tokens = 0
        total_cache_hit_tokens = 0
        successful_requests = 0

        min_start_time = float('inf')
        max_end_time = 0.0
        min_first_token_time = float('inf')
        max_first_token_time = 0.0

        for res in results:
            if res and res.get("error") != "UserCancelled" and res.get("error") is None:
                total_output_tokens += res.get('decode_tokens', 0)
                total_input_tokens += res.get('prefill_tokens', 0)
                total_cache_hit_tokens += res.get('cache_hit_tokens', 0) or 0
                successful_requests += 1

                if res.get('start_time', float('inf')) < min_start_time:
                    min_start_time = res.get('start_time')
                if res.get('end_time', 0) > max_end_time:
                    max_end_time = res.get('end_time')

                ftt = res.get('first_token_time')
                if ftt:
                    if ftt < min_first_token_time:
                        min_first_token_time = ftt
                    if ftt > max_first_token_time:
                        max_first_token_time = ftt

        # Fallback if no valid times
        if min_start_time == float('inf'):
            min_start_time = start_batch_time
        if max_end_time == 0:
            max_end_time = end_batch_time

        # Apply calibration to batch total duration
        # Effectively assuming all requests started 'offset' seconds later
        batch_total_duration = max(0.001, (max_end_time - min_start_time) - self.latency_offset)

        # Output Throughput (Decode Phase Only)
        if min_first_token_time != float('inf'):
            decode_duration = max(0.001, max_end_time - min_first_token_time)
        else:
            decode_duration = batch_total_duration

        # Input Throughput (Prefill Phase Only)
        if max_first_token_time > 0:
             # Apply calibration to prefill duration too
             prefill_duration = max(0.001, (max_first_token_time - min_start_time) - self.latency_offset)
        else:
             prefill_duration = batch_total_duration

        # Calculate System Metrics (Aggregate Phase-Specific)
        # Input Throughput仅use未缓存 token 数
        uncached_input_tokens = max(0, total_input_tokens - total_cache_hit_tokens)
        system_output_throughput = total_output_tokens / decode_duration
        system_input_throughput = uncached_input_tokens / prefill_duration
        system_throughput = (total_input_tokens + total_output_tokens) / batch_total_duration
        rps = successful_requests / batch_total_duration

        for res in results:
            if res:
                res['system_output_throughput'] = system_output_throughput
                res['system_input_throughput'] = system_input_throughput
                res['system_throughput'] = system_throughput
                res['rps'] = rps

        return results

    async def _run_continuous_batch(self, client, prompt_func_or_str, max_tokens, concurrency, total_requests, session_id_start, **kwargs):
        """
        Run requests continuously using a semaphore to maintain constant concurrency.
        """
        semaphore = asyncio.Semaphore(concurrency)
        tasks = []
        results = []

        start_test_time = time.time()

        # Shared stats for real-time throughput calculation
        stats = {
            "completed_requests": 0,
            "total_output_tokens": 0,
            "total_input_tokens": 0,
            "successful_requests": 0,
            "min_start_time": float('inf'),
            "max_end_time": 0.0,
            "min_first_token_time": float('inf'), # New: Track start of decode phase
            "max_first_token_time": 0.0, # New: Track end of prefill phase
            "total_cache_hit_tokens": 0
        }

        async def worker(i):
            async with semaphore:
                # Determine prompt
                if isinstance(prompt_func_or_str, list):
                    prompt = prompt_func_or_str[i]
                elif callable(prompt_func_or_str):
                    prompt = prompt_func_or_str(i)
                else:
                    # If string, add UUID to avoid cache if needed
                    # REMOVED [Request ID] wrapper to ensure strict token calibration
                    # prompt = f"[Request ID: {uuid.uuid4()}]\n\n{prompt_func_or_str}"
                    prompt = prompt_func_or_str

                session_id = session_id_start + i

                # Check for stop signal
                if st.session_state.get('stop_requested', False):
                    return None

                req_start_time = time.time()
                try:
                    res = await self.get_completion(client, session_id, prompt, max_tokens)
                except asyncio.CancelledError:
                    return None

                req_end_time = time.time()

                # Update stats
                if res:
                    # Update time bounds
                    # Use provider timestamps if available to exclude client overhead (token counting, logging, etc.)
                    # Fallback to worker timestamps if provider didn't return them (e.g. error)
                    req_start = res.get("start_time", req_start_time)
                    req_end = res.get("end_time", req_end_time)

                    if req_start < stats["min_start_time"]:
                        stats["min_start_time"] = req_start
                    if req_end > stats["max_end_time"]:
                        stats["max_end_time"] = req_end

                    # Track Phase Bounds
                    first_token_time = res.get("first_token_time")
                    if first_token_time:
                        if first_token_time < stats["min_first_token_time"]:
                            stats["min_first_token_time"] = first_token_time
                        if first_token_time > stats["max_first_token_time"]:
                            stats["max_first_token_time"] = first_token_time

                    # Calculate cumulative metrics (approximate for real-time)
                    current_time = time.time()
                    total_elapsed = max(0.001, current_time - start_test_time)

                    # Output Throughput (Decode Phase Only)
                    # Duration = Current Time - Earliest First Token Time
                    if stats["min_first_token_time"] != float('inf'):
                        decode_elapsed = max(0.001, current_time - stats["min_first_token_time"])
                    else:
                        decode_elapsed = total_elapsed # Fallback

                    # Input Throughput (Prefill Phase Only)
                    # Duration = Latest First Token Time - Earliest Start Time
                    # Note: For continuous test, this window grows.
                    if stats["max_first_token_time"] > 0 and stats["min_start_time"] != float('inf'):
                         prefill_elapsed = max(0.001, stats["max_first_token_time"] - stats["min_start_time"])
                    else:
                         prefill_elapsed = total_elapsed # Fallback

                    if res.get("error") != "UserCancelled" and res.get("error") is None:
                        stats["total_output_tokens"] += res.get('decode_tokens', 0)
                        stats["total_input_tokens"] += res.get('prefill_tokens', 0)
                        stats["total_cache_hit_tokens"] += res.get('cache_hit_tokens', 0) or 0
                        stats["successful_requests"] += 1

                    stats["completed_requests"] += 1


                    # Calculate system throughput based on PHASE time
                    # Input Throughput仅use未缓存 token 数
                    uncached_input_tokens = max(0, stats["total_input_tokens"] - stats["total_cache_hit_tokens"])
                    res['system_output_throughput'] = stats["total_output_tokens"] / decode_elapsed
                    res['system_input_throughput'] = uncached_input_tokens / prefill_elapsed
                    res['rps'] = stats["successful_requests"] / total_elapsed # RPS is still Total Time based


                    results.append(res)

                    # Update global progress
                    self.completed_requests += 1
                    self.update_ui()

                return res

        # Launch all tasks
        tasks = [asyncio.create_task(worker(i)) for i in range(total_requests)]

        try:
            # Check for stop signal periodically or wait for all
            # But since we want immediate stop, we can rely on the worker check + cancellation
            # However, if we want to cancel *running* requests immediately when stop is pressed:

            # We need a way to monitor stop_requested while waiting.
            # Simple approach: Wait for all, but if stop_requested is set, cancel all.

            # Better approach for responsiveness:
            # Use a loop to wait for tasks, checking stop_signal.
            # Or just let the workers check before starting.
            # If we want to interrupt *in-flight* requests, we need to cancel the tasks.

            wait_task = asyncio.gather(*tasks, return_exceptions=True)

            while not wait_task.done():
                if st.session_state.get('stop_requested', False):
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    break
                await asyncio.sleep(0.1)

            await wait_task

        except asyncio.CancelledError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Recalculate final system throughput using precise request timestamps
        # This excludes framework overhead (like UI updates) that happens outside the request window
        if stats["max_end_time"] > stats["min_start_time"]:
            effective_duration = max(0.001, stats["max_end_time"] - stats["min_start_time"])

            # 1. Input Throughput: Total Input / (Max First Token - Min Start)
            # 仅use未缓存 token 数
            uncached_input_tokens = max(0, stats["total_input_tokens"] - stats["total_cache_hit_tokens"])
            if stats["max_first_token_time"] > 0:
                 prefill_dur = max(0.001, stats["max_first_token_time"] - stats["min_start_time"])
                 final_system_input_throughput = uncached_input_tokens / prefill_dur

                 # 2. Output Throughput: Total Output / (Max End - Min First Token)
                 min_ftt = stats["min_first_token_time"]
                 if min_ftt == float('inf'):
                     min_ftt = stats["min_start_time"]  # Fallback
                 decode_dur = max(0.001, stats["max_end_time"] - min_ftt)
                 final_system_output_throughput = stats["total_output_tokens"] / decode_dur
            else:
                 final_system_input_throughput = 0
                 final_system_output_throughput = 0

            # 3. System Throughput: (Input + Output) / Batch Total Duration
            final_system_throughput = (stats["total_input_tokens"] + stats["total_output_tokens"]) / effective_duration

        else:
            effective_duration = max(0.001, time.time() - start_test_time)
            final_system_throughput = stats["total_output_tokens"] / effective_duration # Fallback
            final_system_input_throughput = 0
            final_system_output_throughput = 0

        final_rps = stats["successful_requests"] / effective_duration

        for res in results:
            if res:
                res['system_throughput'] = final_system_throughput
                res['system_output_throughput'] = final_system_output_throughput
                res['system_input_throughput'] = final_system_input_throughput
                res['rps'] = final_rps

        return results

    async def _run_prefill_request(self, client, prompt, max_tokens, session_id):
        # REMOVED [Request ID] wrapper to ensure strict token calibration
        # unique_long_prompt = f"[Request ID: {uuid.uuid4()}]\n\n{prompt}"
        res = await self.get_completion(client, session_id, prompt, max_tokens)
        return res

    async def _run_long_context_request(self, client, prompt, max_tokens, session_id):
        # REMOVED [Request ID] wrapper to ensure strict token calibration from _calibrate_prompt
        # unique_long_prompt = f"[Request ID: {uuid.uuid4()}]\n\n{prompt}"
        res = await self.get_completion(client, session_id, prompt, max_tokens)
        return res

    async def run_concurrency_test(self, selected_concurrencies, rounds_per_level, max_tokens, input_tokens_target=0):
        from config.session_state import set_test_cancelled, set_test_paused

        self.total_requests = sum(c * rounds_per_level for c in selected_concurrencies)
        self._test_start_time = time.time()  # 记录Test started时间
        self._current_max_tokens = max_tokens  # Save Config用于Restore

        csv_columns = ["session_id", "concurrency", "round",
                       "ttft", "tps", "tpot", "prefill_speed",
                       "system_throughput", "system_input_throughput", "rps",
                       "prefill_tokens", "decode_tokens", "total_time", "decode_time",
                       "start_time", "end_time", "cache_hit_tokens", "token_calc_method", "input_tokens_target", "error"]
        initialize_csv(csv_columns, self.csv_file)

        # 启动DatabaseTest运行
        config = {
            "selected_concurrencies": selected_concurrencies,
            "rounds_per_level": rounds_per_level,
            "max_tokens": max_tokens,
            "input_tokens_target": input_tokens_target,
        }
        self._start_db_run("concurrency", config)

        # Checkis否isRestore模式
        is_resuming = st.session_state.get('is_resuming', False)
        resume_data = st.session_state.get('resume_data', None)
        start_session_counter = 0

        if is_resuming and resume_data:
            # LoadSavedResult
            saved_results = resume_data.get('completed_results', [])
            if saved_results:
                self.results_list = saved_results.copy()
                self.completed_requests = len(saved_results)
                start_session_counter = resume_data.get('current_index', 0)
                # 如果 current_index 不存在或为 0，使用 completed_requests 作为后备
                if start_session_counter == 0:
                    start_session_counter = self.completed_requests
                self._update_log(f"从进度Restore: Completed {self.completed_requests} requests, will skip first {start_session_counter}", level=LogLevel.INFO)
                # 清除Restore标志
                st.session_state.is_resuming = False
                st.session_state.resume_data = None

        # Generate Calibrated Prompt if target > 0
        if input_tokens_target > 0:
            self.status_text.info(f"currentlyGenerate {input_tokens_target} Token 校准 Prompt...")
            # We don't need to store a single 'calibrated_prompt' anymore as we generate per-request
            pass

        session_counter = start_session_counter

        # 1. Pre-calculate baseline token count if strictly needed for "0" target?
        # If target=0, we use the user's prompt length as the target for uniqueness generation
        base_target_tokens = input_tokens_target
        if base_target_tokens <= 0:
             # Fallback if UI passes 0: use a default reasonable length (e.g., 64)
             base_target_tokens = 64

        # No client needed - requests library creates connections per-request
        for concurrency in selected_concurrencies:
            # Check控制信号
            signal = self._check_control_signal()
            if signal:
                # Save进度
                pending_prompts = []  # ConcurrencyTest prompt is动态Generate，no法精确Restore
                status = "PAUSED" if signal == 'pause' else "CANCELLED"
                # 使用已完成的请求数作为 current_index，而不是 session_counter
                completed_count = len(self.results_list)
                self._save_progress("concurrency", completed_count, self.total_requests,
                                   pending_prompts, status)
                if signal == 'pause':
                    set_test_paused()
                    st.warning("TestPaused，进度Saved")
                else:
                    set_test_cancelled()
                    st.warning("Test已停止，进度Saved")
                return pd.DataFrame(self.results_list)

            self._current_concurrency = concurrency
            self.status_text.info(f"currently以 {concurrency} ConcurrencyRun Test...")

            for r in range(rounds_per_level):
                # Check控制信号
                signal = self._check_control_signal()
                if signal:
                    pending_prompts = []
                    status = "PAUSED" if signal == 'pause' else "CANCELLED"
                    # 使用已完成的请求数作为 current_index，而不是 session_counter
                    completed_count = len(self.results_list)
                    self._save_progress("concurrency", completed_count, self.total_requests,
                                       pending_prompts, status)
                    if signal == 'pause':
                        set_test_paused()
                        st.warning("TestPaused，进度Saved")
                    else:
                        set_test_cancelled()
                        st.warning("Test已停止，进度Saved")
                    return pd.DataFrame(self.results_list)

                # 跳过已完成的请求（Resume时）
                if session_counter < start_session_counter:
                    session_counter += concurrency
                    continue

                self.status_text.info(f"ConcurrencyTest: {concurrency} Concurrency,  {r+1}/{rounds_per_level} 轮...")

                # Generate UNIQUE prompts for this batch to avoid Cache Hits
                # We use _calibrate_prompt to generate distinct random content of the SAME target length
                # Pre-load tokenizer in main thread to avoid thread-safety issues with Streamlit
                target_tokens = self._prompt_generation_target(input_tokens_target if input_tokens_target > 0 else base_target_tokens)
                cached_tokenizer = self._get_tokenizer()  # Load tokenizer in main thread

                # Parallelize prompt generation using thread pool with pre-loaded tokenizer
                loop = asyncio.get_event_loop()
                prompt_tasks = [
                    loop.run_in_executor(None, self._calibrate_prompt, target_tokens, "", cached_tokenizer)
                    for _ in range(concurrency)
                ]
                batch_prompts = await asyncio.gather(*prompt_tasks)

                results = await self._run_concurrency_batch(
                    None,
                    batch_prompts,
                    max_tokens,
                    concurrency,
                    session_counter
                )

                session_counter += concurrency

                # Assign Round info and save
                for res in results:
                    if res and res.get("error") != "UserCancelled":
                        res['concurrency'] = concurrency
                        res['round'] = r + 1
                        self._add_result(res, csv_columns)

                self.update_ui()

        # 批量SaveResult到Database并完成运行
        self._batch_save_results_to_db()
        self._complete_db_run(success=True)

        return pd.DataFrame(self.results_list)

    async def run_prefill_test(self, token_levels, requests_per_level, max_tokens):
        from config.session_state import set_test_cancelled, set_test_paused

        self.total_requests = len(token_levels) * requests_per_level
        self._test_start_time = time.time()
        self._current_max_tokens = max_tokens

        csv_columns = ["input_tokens_target", "session_id",
                       "ttft", "tps", "tpot", "prefill_speed",
                       "system_throughput", "system_input_throughput", "rps",
                       "prefill_tokens", "decode_tokens", "total_time", "decode_time",
                       "start_time", "end_time", "cache_hit_tokens", "token_calc_method", "error"]
        initialize_csv(csv_columns, self.csv_file)

        # 启动DatabaseTest运行
        config = {
            "token_levels": token_levels,
            "requests_per_level": requests_per_level,
            "max_tokens": max_tokens,
        }
        self._start_db_run("prefill", config)

        # No client needed - requests library creates connections per-request
        for tokens_target in token_levels:
            # Check控制信号
            signal = self._check_control_signal()
            if signal:
                pending_prompts = []
                status = "PAUSED" if signal == 'pause' else "CANCELLED"
                self._save_progress("prefill", self.completed_requests, self.total_requests,
                                   pending_prompts, status)
                if signal == 'pause':
                    set_test_paused()
                    st.warning("TestPaused，进度Saved")
                else:
                    set_test_cancelled()
                    st.warning("Test已停止，进度Saved")
                return pd.DataFrame(self.results_list)

            self.status_text.info(f"currently准备 {tokens_target} (目标) Token Tip...")

            # Adjust multiplier based on tokenizer
            # Adjust multiplier based on tokenizer
            # Legacy logic removed: we trust _calibrate_prompt or tokenizer directly

            # Precision Mode for Ultra-Short Contexts
            if tokens_target < 20:
                self.status_text.info(f"currentlyTest (目标: {tokens_target}, 精细模式)...")

                for i in range(requests_per_level):
                    # Generate fresh random prompt of exact length
                    # Sync with Strict Calibration Logic
                    if tokens_target <= 32:
                         raw_prompt = self._calibrate_prompt(self._prompt_generation_target(tokens_target, PREFILL_PROMPT_OVERHEAD), suffix="")
                    else:
                         suffix_inst = "\n\n请先Statistics前文都多少字数然后尽你所能直接创作一越长越好超长篇科幻小说。"
                         raw_prompt = self._calibrate_prompt(self._prompt_generation_target(tokens_target, PREFILL_PROMPT_OVERHEAD), suffix=suffix_inst)

                    res = await self.get_completion(None, i, raw_prompt, max_tokens)

                    if res and res.get("error") != "UserCancelled":
                        res['input_tokens_target'] = tokens_target

                        # Ensure prefill_tokens is used
                        actual_prompt_tokens = res.get('prefill_tokens', 0)
                        if actual_prompt_tokens == 0 and res.get("error") is None:
                                self._update_log(f"Warning Session {res.get('session_id')}: API/Tiktoken Return 0 prompt tokens。", level=LogLevel.WARNING, session_id=str(res.get('session_id')))

                        # 仅use未缓存 token Calculate输入速度
                        cache_hit = res.get('cache_hit_tokens', 0) or 0
                        uncached_prompt_tokens = max(0, actual_prompt_tokens - cache_hit)

                        if res['ttft'] > 0:
                            res['prefill_speed'] = uncached_prompt_tokens / res['ttft']
                        else:
                            res['prefill_speed'] = 0

                        # For single request, system throughput is same as single throughput
                        # Use PHASE time: Decode Time for Output, TTFT for Input
                        # Note: _calculate_metrics now puts 'decode_time' in res
                        decode_dur = res.get('decode_time', 0.001)
                        if decode_dur <= 0:
                            decode_dur = 0.001

                        ttft_dur = res.get('ttft', 0.001)
                        if ttft_dur <= 0:
                            ttft_dur = 0.001

                        res['system_output_throughput'] = self._result_decode_tokens_for_tps(res) / decode_dur
                        res['system_input_throughput'] = uncached_prompt_tokens / ttft_dur

                        total_time_val = res.get('total_time', 0.001)
                        res['rps'] = 1 / total_time_val if total_time_val > 0 else 0

                        append_to_csv(res, csv_columns, self.csv_file)
                        self.results_list.append(res)

                    self.completed_requests += 1
                    self.update_ui()

            else:
                # Standard Mode
                # We simply request the target tokens; logic inside _get_text_for_token_count handles overhead/generation
                local_tokens_to_generate = self._prompt_generation_target(tokens_target)

                if local_tokens_to_generate <= 0:
                     local_tokens_to_generate = 1

                self.status_text.info(f"currentlyTest (目标: {tokens_target})...")

                # Pre-generate all prompts in parallel for speed
                cached_tokenizer = self._get_tokenizer()
                loop = asyncio.get_event_loop()
                prompt_tasks = [
                    loop.run_in_executor(None, self._calibrate_prompt, local_tokens_to_generate, "", cached_tokenizer)
                    for _ in range(requests_per_level)
                ]
                pregen_prompts = await asyncio.gather(*prompt_tasks)

                for i in range(requests_per_level):
                    # Generate fresh prompt for each request
                    prompt_text = pregen_prompts[i]
                    res = await self._run_prefill_request(None, prompt_text, max_tokens, i)

                    if res and res.get("error") != "UserCancelled":
                        res['input_tokens_target'] = tokens_target

                        # Ensure prefill_tokens is used
                        actual_prompt_tokens = res.get('prefill_tokens', 0)
                        if actual_prompt_tokens == 0 and res.get("error") is None:
                                self._update_log(f"Warning Session {res.get('session_id')}: API/Tiktoken Return 0 prompt tokens。", level=LogLevel.WARNING, session_id=str(res.get('session_id')))

                        # 仅use未缓存 token Calculate输入速度
                        cache_hit = res.get('cache_hit_tokens', 0) or 0
                        uncached_prompt_tokens = max(0, actual_prompt_tokens - cache_hit)

                        if res['ttft'] > 0:
                            res['prefill_speed'] = uncached_prompt_tokens / res['ttft']
                        else:
                            res['prefill_speed'] = 0

                        # For single request, system throughput is same as single throughput
                        # Use PHASE time: Decode Time for Output, TTFT for Input
                        # Note: _calculate_metrics now puts 'decode_time' in res
                        decode_dur = res.get('decode_time', 0.001)
                        if decode_dur <= 0:
                            decode_dur = 0.001

                        ttft_dur = res.get('ttft', 0.001)
                        if ttft_dur <= 0:
                            ttft_dur = 0.001

                        res['system_output_throughput'] = self._result_decode_tokens_for_tps(res) / decode_dur
                        res['system_input_throughput'] = uncached_prompt_tokens / ttft_dur

                        total_time_val = res.get('total_time', 0.001)
                        res['rps'] = 1 / total_time_val if total_time_val > 0 else 0

                        append_to_csv(res, csv_columns, self.csv_file)
                        self.results_list.append(res)

                    self.completed_requests += 1
                    self.update_ui()

        # 批量SaveResult到Database并完成运行
        self._batch_save_results_to_db()
        self._complete_db_run(success=True)

        return pd.DataFrame(self.results_list)

    async def run_segmented_prefill_test(
        self,
        segment_levels,
        requests_per_segment,
        max_tokens,
        cumulative_mode=True,
        total_rounds=1,
        per_round_unique=False,
        concurrency=1
    ):
        """
        分段累计 Prefill Test

        模拟真实场景：用户通常notwill一次性发送 60K tokens，而is分段累计发送。
        用于Test Prefix Caching 效果。

        Args:
            segment_levels: Segment levels列表，如 [2000, 8000, 20000, 40000, 60000]
            requests_per_segment: 每Segment levels发送请求数
            max_tokens: 最大Generate token 数
            cumulative_mode:
                True = Cumulative Mode：所has分段共享同一前缀，Test Prefix Caching 效果
                False = Independent Mode：Each segment has independent content as no-cache control group
            total_rounds: Total Test Rounds，整分段序列重复执行次数
            per_round_unique:
                True = 每轮重新Generatenot同 Prompt，避免跨轮Cache Hit
                False = 所has轮次共享同一 Prompt，Test缓存持久性
            concurrency: 每Segment levelsConcurrency请求数

        Test流程 (cumulative_mode=True):
            1. 先Generate最长段 base_prompt
            2. 从 base_prompt 截取not同长度作is各分段 prompt
            3. 按从短到长顺序发送，TestCache Hit效果
        """
        # 确保Segment levelsSort
        segment_levels = sorted(segment_levels)
        max_segment = max(segment_levels)

        # InitializeBaseline prefill 速度追踪（用于 TTFT 推断 cache hit）
        # key: concurrency_index, value: prefill_speed (tokens/sec)
        self._segmented_baseline_prefill_speed = {}
        self._segmented_baseline_segment = segment_levels[0]  # 最小分段作isBaseline来源

        # CalculateTotal Requests（分段数 × Requests Per Segment × 整体轮数 × Concurrency）
        self.total_requests = len(segment_levels) * requests_per_segment * total_rounds * concurrency

        csv_columns = [
            "session_id", "concurrency", "context_length_target", "round",
            "ttft", "tpot", "prefill_speed", "tps", "rps",
            "system_input_throughput", "system_output_throughput", "system_total_throughput",
            "prefill_tokens", "decode_tokens", "cache_hit_tokens", "cache_hit_source",
            "api_prefill", "api_decode",
            "effective_prefill_tokens", "effective_decode_tokens", "token_source",
            "error", "token_calc_method", "cumulative_mode"
        ]
        initialize_csv(csv_columns, self.csv_file)

        # 启动DatabaseTest运行
        config = {
            "segment_levels": segment_levels,
            "requests_per_segment": requests_per_segment,
            "max_tokens": max_tokens,
            "cumulative_mode": cumulative_mode,
            "total_rounds": total_rounds,
            "concurrency": concurrency,
        }
        self._start_db_run("segmented_prefill", config)

        # Get tokenizer
        tokenizer = self._get_tokenizer()
        if not tokenizer:
            st.error("no法Load Tokenizer，Segmented Context Testneed精确 Token 控制")
            return pd.DataFrame()

        # Generate基础 prompt（最大长度）- 每Concurrencywill话need独立 prompt
        suffix_inst = "\n\n请先Statistics前文都多少字数然后尽你所能直接创作一越长越好超长篇科幻小说。"

        # base_prompts_list: 存储每Concurrencywill话 (base_prompt, base_tokens)
        base_prompts_list = []

        # ifnotis每轮Independent Mode，in循环外Generate base_prompts
        if cumulative_mode and not per_round_unique:
            self._update_log(
                f"Cumulative Mode（共享 Prompt）：currentlyis {concurrency} Concurrencywill话Generate {max_segment} tokens 基础 Prompt...",
                level=LogLevel.INFO
            )
            for c_idx in range(concurrency):
                base_prompt = self._calibrate_prompt(self._prompt_generation_target(max_segment), suffix=suffix_inst)
                if hasattr(tokenizer, 'encode'):
                    base_tokens = tokenizer.encode(base_prompt, add_special_tokens=False)
                else:
                    base_tokens = tokenizer.encode(base_prompt)
                base_prompts_list.append((base_prompt, base_tokens))
                self._update_log(
                    f"  will话 {c_idx + 1}/{concurrency} Prompt Generate完成，实际长度: {len(base_tokens)} tokens",
                    level=LogLevel.INFO
                )
            self._update_log(f"所has {concurrency} 基础 Prompt Generate完成", level=LogLevel.SUCCESS)

        # 整体轮次循环
        for overall_round in range(total_rounds):
            if st.session_state.get('stop_requested', False):
                st.warning("Test已停止。")
                break

            self._update_log(f"开始 {overall_round + 1}/{total_rounds} 轮Test", level=LogLevel.INFO)

            # ifis每轮Independent Mode，每轮重新Generate base_prompts
            if cumulative_mode and per_round_unique:
                base_prompts_list = []
                self._update_log(
                    f"Cumulative Mode（Unique Prompt Per Round）：currentlyis {concurrency} Concurrencywill话Generate {max_segment} tokens 基础 Prompt...",
                    level=LogLevel.INFO
                )
                for c_idx in range(concurrency):
                    base_prompt = self._calibrate_prompt(self._prompt_generation_target(max_segment), suffix=suffix_inst)
                    if hasattr(tokenizer, 'encode'):
                        base_tokens = tokenizer.encode(base_prompt, add_special_tokens=False)
                    else:
                        base_tokens = tokenizer.encode(base_prompt)
                    base_prompts_list.append((base_prompt, base_tokens))
                self._update_log(
                    f"轮次 {overall_round + 1} 所has {concurrency} 基础 Prompt Generate完成",
                    level=LogLevel.SUCCESS
                )

            # 按Segment levels从小到大发送
            for seg_idx, segment_length in enumerate(segment_levels):
                if st.session_state.get('stop_requested', False):
                    st.warning("Test已停止。")
                    break

                self.status_text.info(
                    f"轮次 {overall_round + 1}/{total_rounds} - 分段 {seg_idx + 1}/{len(segment_levels)}: "
                    f"{segment_length} tokens (Concurrency: {concurrency}, Cumulative Mode: {'is' if cumulative_mode else '否'})..."
                )

                # is每Concurrencywill话Generate对应 segment_prompt
                if cumulative_mode:
                    # Cumulative Mode：每Concurrencywill话从自己 base_prompt 截取
                    segment_prompts = []
                    for c_idx in range(concurrency):
                        base_prompt, base_tokens = base_prompts_list[c_idx]
                        segment_token_length = self._prompt_generation_target(segment_length)
                        if segment_token_length >= len(base_tokens):
                            segment_prompts.append(base_prompt)
                        else:
                            if hasattr(tokenizer, 'decode'):
                                segment_prompts.append(tokenizer.decode(base_tokens[:segment_token_length]))
                            else:
                                # fallback: truncate by char estimate
                                char_ratio = len(base_prompt) / len(base_tokens)
                                segment_prompts.append(base_prompt[:int(segment_token_length * char_ratio)])
                else:
                    # Independent Mode：每分段独立Generate
                    segment_prompts = []
                    for _ in range(concurrency):
                        segment_prompt = self._calibrate_prompt(self._prompt_generation_target(segment_length), suffix=suffix_inst)
                        segment_prompts.append(segment_prompt)

                # 发送Concurrency请求
                for req_idx in range(requests_per_segment):
                    if st.session_state.get('stop_requested', False):
                        break

                    # Concurrency执行
                    tasks = []
                    for c_idx in range(concurrency):
                        segment_prompt = segment_prompts[c_idx]
                        session_id = f"R{overall_round + 1}_S{seg_idx + 1}_C{c_idx + 1}_R{req_idx + 1}"

                        task = self._run_segmented_request(
                            segment_prompt,
                            max_tokens,
                            session_id,
                            segment_length,
                            concurrency,
                            overall_round,
                            cumulative_mode,
                            seg_idx=seg_idx,
                            c_idx=c_idx
                        )
                        tasks.append(task)

                    # 执行所hasConcurrency任务
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # ProcessResult
                    for result in results:
                        if isinstance(result, Exception):
                            self._update_log(f"请求异常: {result}", level=LogLevel.ERROR)
                        elif result and result.get("error") != "UserCancelled":
                            append_to_csv(result, csv_columns, self.csv_file)
                            self.results_list.append(result)

                        self.completed_requests += 1
                        self.update_ui()

        self._update_log("分段 Prefill Test completed", level=LogLevel.SUCCESS)

        # 批量SaveResult到Database并完成运行
        self._batch_save_results_to_db()
        self._complete_db_run(success=True)

        return pd.DataFrame(self.results_list)

    async def _run_segmented_request(
        self,
        prompt,
        max_tokens,
        session_id,
        segment_length,
        concurrency,
        round_num,
        cumulative_mode,
        seg_idx=0,
        c_idx=0
    ):
        """执行单分段请求"""
        try:
            res = await self.get_completion(None, 0, prompt, max_tokens)

            if res and res.get("error") != "UserCancelled":
                # Add分段Test特定字段
                res['session_id'] = session_id
                res['concurrency'] = concurrency
                res['context_length_target'] = segment_length
                res['round'] = round_num + 1
                res['cumulative_mode'] = cumulative_mode

                # Calculate cache_hit_tokens（从 API ReturninGet）
                if 'cache_hit_tokens' not in res or res['cache_hit_tokens'] is None:
                    res['cache_hit_tokens'] = 0
                res['cache_hit_source'] = 'API' if res['cache_hit_tokens'] > 0 else 'none'

                # === 优先use API usage Data，回退到 tokenizer Statistics ===
                # api_prefill / api_decode is API Return原始 usage 字段
                # prefill_tokens / decode_tokens is _calculate_tokens based on优先级Calculate
                api_p = res.get('api_prefill')
                api_d = res.get('api_decode')
                tokenizer_p = res.get('prefill_tokens', 0)
                tokenizer_d = res.get('decode_tokens', 0)

                # 优先级: api_usage > tokenizer
                # api 值可能is None or 0 (网络波动/APINot supported)，此时回退到 tokenizer
                effective_prefill = api_p if api_p and api_p > 0 else tokenizer_p
                effective_decode = api_d if api_d and api_d > 0 else tokenizer_d
                token_source = "API" if (api_p and api_p > 0) else "Tokenizer"

                # 记录实际use token 来源
                res['effective_prefill_tokens'] = effective_prefill
                res['effective_decode_tokens'] = effective_decode
                res['token_source'] = token_source

                # Calculate prefill_speed (仅use未缓存 prefill tokens)
                # hasCache Hit时，TTFT 只反映未缓存部分Process时间，
                # therefore速度Calculate应该用 uncached tokens / TTFT
                cache_hit = res.get('cache_hit_tokens', 0) or 0
                uncached_prefill = max(0, effective_prefill - cache_hit)

                if res['ttft'] > 0:
                    res['prefill_speed'] = uncached_prefill / res['ttft']
                else:
                    res['prefill_speed'] = 0

                # 时间基准
                decode_dur = res.get('decode_time', 0.001)
                if decode_dur <= 0:
                    decode_dur = 0.001

                ttft_dur = res.get('ttft', 0.001)
                if ttft_dur <= 0:
                    ttft_dur = 0.001

                total_time = res.get('total_time', 0.001)
                if total_time <= 0:
                    total_time = 0.001

                # CalculateSystem Throughput (Input Throughput仅use未缓存 token 数)
                res['system_output_throughput'] = effective_decode / decode_dur
                res['system_input_throughput'] = uncached_prefill / ttft_dur
                res['system_total_throughput'] = (effective_prefill + effective_decode) / total_time

                # Calculate TPS (tokens per second) - 仅 decode 阶段Generate速率
                res['tps'] = effective_decode / decode_dur if decode_dur > 0 else 0

                # Calculate RPS (requests per second)
                res['rps'] = 1 / total_time if total_time > 0 else 0

                # Calculate TPOT (time per output token)
                res['tpot'] = decode_dur / effective_decode if effective_decode > 0 else 0

                # token_calc_method 记录实际useCalculate方法
                base_method = res.get('token_calc_method', self.tokenizer_option)
                res['token_calc_method'] = f"{base_method} (metrics: {token_source})"

                # === TTFT-based cache hit inference ===
                # 当 API 未on报 cache_hit 时，via对比 TTFT 与Baseline prefill 速度推断
                baseline_speeds = getattr(self, '_segmented_baseline_prefill_speed', {})
                baseline_seg = getattr(self, '_segmented_baseline_segment', None)

                if seg_idx == 0 and round_num == 0 and c_idx not in baseline_speeds:
                    # 一轮一分段首次请求作isBaseline（no缓存 cold start）
                    if res['ttft'] > 0 and effective_prefill > 0:
                        baseline_speed = effective_prefill / res['ttft']
                        baseline_speeds[c_idx] = baseline_speed
                        self._update_log(
                            f"Baseline prefill speed (C{c_idx}): {baseline_speed:.1f} t/s "
                            f"(from {effective_prefill} tokens in {res['ttft']:.4f}s)",
                            level=LogLevel.INFO
                        )

                elif cumulative_mode and res['cache_hit_tokens'] == 0 and c_idx in baseline_speeds:
                    # 非首次请求且 API 没hason报 cache_hit → 尝试 TTFT 推断
                    baseline_speed = baseline_speeds[c_idx]
                    if baseline_speed > 0 and effective_prefill > 0 and res['ttft'] > 0:
                        expected_ttft = effective_prefill / baseline_speed
                        actual_ttft = res['ttft']

                        if expected_ttft > actual_ttft * 2.0:  # 至少 2x 加速才推断（排除引擎预热假阳性）
                            # 推断Cache Hit token 数
                            # 逻辑: actual_ttft 只Process uncached 部分
                            # uncached_tokens = actual_ttft * baseline_speed
                            # cached_tokens = effective_prefill - uncached_tokens
                            uncached_tokens = actual_ttft * baseline_speed
                            inferred_cache = max(0, int(effective_prefill - uncached_tokens))
                            res['cache_hit_tokens'] = inferred_cache
                            res['cache_hit_source'] = 'TTFT_inferred'

                            cache_pct = (inferred_cache / effective_prefill * 100) if effective_prefill > 0 else 0
                            self._update_log(
                                f"Cache inferred ({session_id}): "
                                f"expected_ttft={expected_ttft:.3f}s actual={actual_ttft:.3f}s → "
                                f"~{inferred_cache} tokens cached ({cache_pct:.0f}%)",
                                level=LogLevel.INFO,
                                session_id=str(session_id)
                            )

                            # use推断缓存信息重新Calculate输入相关速度指标
                            uncached_prefill = max(0, effective_prefill - inferred_cache)
                            if res['ttft'] > 0:
                                res['prefill_speed'] = uncached_prefill / res['ttft']
                            ttft_dur = res.get('ttft', 0.001)
                            if ttft_dur <= 0:
                                ttft_dur = 0.001
                            res['system_input_throughput'] = uncached_prefill / ttft_dur

                self._update_log(
                    f"Segment {session_id}: prefill={effective_prefill}({token_source}) "
                    f"decode={effective_decode}({token_source}) "
                    f"TTFT={res['ttft']:.4f}s TPS={res['tps']:.1f} "
                    f"cache_hit={res.get('cache_hit_tokens', 0)}({res.get('cache_hit_source', 'none')})",
                    level=LogLevel.DEBUG,
                    session_id=str(session_id)
                )

            return res

        except Exception as e:
            self._update_log(f"分段请求失败: {e}", level=LogLevel.ERROR)
            return {
                'session_id': session_id,
                'concurrency': concurrency,
                'context_length_target': segment_length,
                'round': round_num + 1,
                'cumulative_mode': cumulative_mode,
                'error': str(e),
                'ttft': 0,
                'tps': 0,
                'tpot': 0,
                'prefill_speed': 0,
                'rps': 0,
                'system_input_throughput': 0,
                'system_output_throughput': 0,
                'system_total_throughput': 0,
                'prefill_tokens': 0,
                'decode_tokens': 0,
                'cache_hit_tokens': 0,
                'effective_prefill_tokens': 0,
                'effective_decode_tokens': 0,
                'token_source': 'N/A',
                'token_calc_method': self.tokenizer_option
            }

    async def run_long_context_test(self, context_lengths, rounds_per_level, max_tokens):
        self.total_requests = len(context_lengths) * rounds_per_level
        csv_columns = ["context_length_target", "round", "session_id",
                       "ttft", "tps", "tpot", "prefill_speed",
                       "system_throughput", "system_input_throughput", "system_output_throughput", "rps",
                       "prefill_tokens", "decode_tokens", "cache_hit_tokens",
                       "token_calc_method", "error"]
        initialize_csv(csv_columns, self.csv_file)

        # 启动DatabaseTest运行
        config = {
            "context_lengths": context_lengths,
            "rounds_per_level": rounds_per_level,
            "max_tokens": max_tokens,
        }
        self._start_db_run("long_context", config)

        # No client needed - requests library creates connections per-request
        for length_target in context_lengths:
            self.status_text.info(f"currently准备 {length_target} (目标) Token Tip...")

            # Adjust multiplier based on tokenizer

            # Precision Mode for Ultra-Short Contexts
            if length_target < 20:
                self.status_text.info(f"currentlyTest (目标: {length_target}, 精细模式)...")
                for r in range(rounds_per_level):
                    raw_prompt, _, _ = self._get_text_for_token_count(self._prompt_generation_target(length_target))
                    res = await self.get_completion(None, 0, raw_prompt, max_tokens)

                    if res and res.get("error") != "UserCancelled":
                        actual_prompt_tokens = res.get('prefill_tokens', 0)
                        if actual_prompt_tokens == 0 and res.get("error") is None:
                                self._update_log(f"Warning Session {res.get('session_id')}: API/Tiktoken Return 0 prompt tokens。", level=LogLevel.WARNING, session_id=str(res.get('session_id')))

                        res.update({
                            'test_type': 'long_context',
                            'context_length_target': length_target,
                            'round': r + 1
                        })

                        # 仅use未缓存 token Calculate输入速度
                        cache_hit = res.get('cache_hit_tokens', 0) or 0
                        uncached_prompt_tokens = max(0, actual_prompt_tokens - cache_hit)

                        if res['ttft'] > 0:
                            res['prefill_speed'] = uncached_prompt_tokens / res['ttft']
                        else:
                            res['prefill_speed'] = 0

                        # Use PHASE time: Decode Time for Output, TTFT for Input
                        # Note: _calculate_metrics now puts 'decode_time' in res
                        decode_dur = res.get('decode_time', 0.001)
                        if decode_dur <= 0:
                            decode_dur = 0.001

                        ttft_dur = res.get('ttft', 0.001)
                        if ttft_dur <= 0:
                            ttft_dur = 0.001


                        total_dur = res.get('total_time', 0.001)
                        if total_dur <= 0:
                            total_dur = 0.001  # Prevent division by zero

                        res['system_output_throughput'] = self._result_decode_tokens_for_tps(res) / decode_dur
                        res['system_input_throughput'] = uncached_prompt_tokens / ttft_dur
                        res['system_throughput'] = (res.get('prefill_tokens', 0) + res.get('decode_tokens', 0)) / total_dur
                        res['rps'] = 1 / total_dur

                        append_to_csv(res, csv_columns, self.csv_file)
                        self.results_list.append(res)

                    self.completed_requests += 1
                    self.update_ui()
            else:
                # Standard Mode
                local_tokens_to_generate = self._prompt_generation_target(length_target)

                if local_tokens_to_generate <= 0:
                     local_tokens_to_generate = 1

                # Logic:
                # <= 32 tokens (USER UPDATE): Random noise, no suffix
                # > 32 tokens: Random noise body + suffix instructions

                for r in range(rounds_per_level):
                    self.status_text.info(f"currentlyTest (目标: {length_target}, 轮数: {r+1}/{rounds_per_level})...")
                    # Generate unique prompt per request
                    if length_target <= 32:
                        long_prompt = self._calibrate_prompt(self._prompt_generation_target(length_target), suffix="") # PREFILL_PROMPT_OVERHEAD removed
                    else:
                        # Use adaptive suffix system
                        long_prompt = self._calibrate_prompt(self._prompt_generation_target(length_target), suffix="") # PREFILL_PROMPT_OVERHEAD removed

                    res = await self._run_long_context_request(None, long_prompt, max_tokens, 0)

                    if res and res.get("error") != "UserCancelled":

                        actual_prompt_tokens = res.get('prefill_tokens', 0)
                        if actual_prompt_tokens == 0 and res.get("error") is None:
                                self._update_log(f"Warning Session {res.get('session_id')}: API/Tiktoken Return 0 prompt tokens。", level=LogLevel.WARNING, session_id=str(res.get('session_id')))

                        res.update({
                            'test_type': 'long_context',
                            'context_length_target': length_target,
                            'round': r + 1
                        })

                        # 仅use未缓存 token Calculate输入速度
                        cache_hit = res.get('cache_hit_tokens', 0) or 0
                        uncached_prompt_tokens = max(0, actual_prompt_tokens - cache_hit)

                        if res['ttft'] > 0:
                            res['prefill_speed'] = uncached_prompt_tokens / res['ttft']
                        else:
                            res['prefill_speed'] = 0

                        # Use PHASE time: Decode Time for Output, TTFT for Input
                        # Note: _calculate_metrics now puts 'decode_time' in res
                        decode_dur = res.get('decode_time', 0.001)
                        if decode_dur <= 0:
                            decode_dur = 0.001

                        ttft_dur = res.get('ttft', 0.001)
                        if ttft_dur <= 0:
                            ttft_dur = 0.001

                        total_dur = res.get('total_time', 0.001)
                        if total_dur <= 0:
                            total_dur = 0.001  # Prevent division by zero

                        res['system_output_throughput'] = self._result_decode_tokens_for_tps(res) / decode_dur
                        res['system_input_throughput'] = uncached_prompt_tokens / ttft_dur
                        res['system_throughput'] = (res.get('prefill_tokens', 0) + res.get('decode_tokens', 0)) / total_dur
                        res['rps'] = 1 / total_dur
                        # Duplicate key set removed, kept only one assignment above

                        append_to_csv(res, csv_columns, self.csv_file)
                        self.results_list.append(res)

                    self.completed_requests += 1
                    self.update_ui()

        # 批量SaveResult到Database并完成运行
        self._batch_save_results_to_db()
        self._complete_db_run(success=True)

        return pd.DataFrame(self.results_list)

    async def run_throughput_matrix_test(self, concurrencies, context_lengths, rounds, max_tokens, enable_warmup=False):
        # Calculate total requests correctly: sum of (concurrency * rounds) for each concurrency level, repeated for each context length
        total_reqs_per_context = sum(c * rounds for c in concurrencies)
        self.total_requests = total_reqs_per_context * len(context_lengths)
        csv_columns = ["session_id", "concurrency", "context_length_target", "round",
                       "ttft", "tps", "tpot", "prefill_speed",
                       "system_output_throughput", "system_input_throughput", "rps",
                       "prefill_tokens", "decode_tokens", "total_time", "decode_time",
                       "start_time", "end_time", "cache_hit_tokens", "token_calc_method", "error"]
        initialize_csv(csv_columns, self.csv_file)

        # 启动DatabaseTest运行
        config = {
            "concurrencies": concurrencies,
            "context_lengths": context_lengths,
            "rounds": rounds,
            "max_tokens": max_tokens,
            "enable_warmup": enable_warmup,
        }
        self._start_db_run("throughput_matrix", config)

        session_counter = 0
        # No client needed - requests library creates connections per-request
        for concurrency in concurrencies:
            for length_target in context_lengths:
                if st.session_state.get('stop_requested', False):
                    st.warning("Test已停止。")
                    break

                self.status_text.info(f"currentlyTest: {concurrency} Concurrency, {length_target} Context Length...")

                # Adjust multiplier based on tokenizer
                if self.tokenizer_option == "HuggingFace Tokenizer" or self._infer_hf_model_id():
                    pass

                # Pre-generate all prompts in parallel for speed
                total_reqs_for_level = concurrency * rounds
                cached_tokenizer = self._get_tokenizer()
                target_len = self._prompt_generation_target(length_target)
                loop = asyncio.get_event_loop()
                prompt_tasks = [
                    loop.run_in_executor(None, self._calibrate_prompt, target_len, "", cached_tokenizer)
                    for _ in range(total_reqs_for_level)
                ]
                pregen_prompts = await asyncio.gather(*prompt_tasks)

                # Use continuous execution with pre-generated prompts
                results = await self._run_continuous_batch(
                    None,
                    pregen_prompts,
                    max_tokens,
                    concurrency,
                    total_reqs_for_level,
                    session_counter
                )
                session_counter += total_reqs_for_level

                for i, res in enumerate(results):
                    if res and res.get("error") != "UserCancelled":

                        actual_prompt_tokens = res.get('prefill_tokens', 0)
                        if actual_prompt_tokens == 0 and res.get("error") is None:
                                self._update_log(f"Warning Session {res.get('session_id')}: API/Tiktoken Return 0 prompt tokens。", level=LogLevel.WARNING, session_id=str(res.get('session_id')))

                        res.update({
                            'test_type': 'matrix',
                            'concurrency': concurrency,
                            'context_length_target': length_target,
                            'round': (i // concurrency) + 1
                        })

                        # 仅use未缓存 token Calculate输入速度
                        cache_hit = res.get('cache_hit_tokens', 0) or 0
                        uncached_prompt_tokens = max(0, actual_prompt_tokens - cache_hit)

                        if res['ttft'] > 0:
                            res['prefill_speed'] = uncached_prompt_tokens / res['ttft']
                        else:
                            res['prefill_speed'] = 0

                        append_to_csv(res, csv_columns, self.csv_file)
                        self.results_list.append(res)

        # 批量SaveResult到Database并完成运行
        self._batch_save_results_to_db()
        self._complete_db_run(success=True)

        return pd.DataFrame(self.results_list)

    async def run_custom_text_test(self, selected_concurrencies, rounds_per_level, base_prompt, suffix_instruction, max_tokens, avoid_cache=True):
        self.total_requests = sum(c * rounds_per_level for c in selected_concurrencies)
        csv_columns = ["session_id", "concurrency", "round",
                       "ttft", "tps", "tpot", "prefill_speed", "system_output_throughput", "system_input_throughput",
                       "prefill_tokens", "decode_tokens", "total_time", "decode_time",
                       "start_time", "end_time", "cache_hit_tokens", "token_calc_method", "error"]
        initialize_csv(csv_columns, self.csv_file)

        # 启动DatabaseTest运行
        config = {
            "selected_concurrencies": selected_concurrencies,
            "rounds_per_level": rounds_per_level,
            "base_prompt": base_prompt[:100] + "..." if len(base_prompt) > 100 else base_prompt,
            "max_tokens": max_tokens,
            "avoid_cache": avoid_cache,
        }
        self._start_db_run("custom_text", config)

        session_counter = 0
        # No client needed - requests library creates connections per-request
        for concurrency in selected_concurrencies:
            self.status_text.info(f"currently以 {concurrency} Concurrency运行Custom Text Test (总请求: {concurrency * rounds_per_level})...")

            full_prompt = f"{base_prompt}\n\n{suffix_instruction}"
            total_reqs_for_level = concurrency * rounds_per_level

            # Use continuous execution
            results = await self._run_continuous_batch(
                None,
                full_prompt,
                max_tokens,
                concurrency,
                total_reqs_for_level,
                session_counter
            )
            session_counter += total_reqs_for_level

            for i, res in enumerate(results):
                if res and res.get("error") != "UserCancelled":
                    res['concurrency'] = concurrency
                    res['round'] = (i // concurrency) + 1
                    append_to_csv(res, csv_columns, self.csv_file)
                    self.results_list.append(res)

        # 批量SaveResult到Database并完成运行
        self._batch_save_results_to_db()
        self._complete_db_run(success=True)

        return pd.DataFrame(self.results_list)

    async def run_dataset_test(self, dataset_rows, concurrency, max_tokens, rounds=1, dataset_filename="custom_dataset"):
        self.total_requests = len(dataset_rows) * rounds
        csv_columns = ["dataset_filename", "row_index", "session_id",
                       "ttft", "tps", "system_output_throughput", "system_input_throughput", "rps",
                       "prefill_tokens", "decode_tokens", "cache_hit_tokens",
                       "token_calc_method", "error", "expected_output"]
        initialize_csv(csv_columns, self.csv_file)

        # 启动DatabaseTest运行
        config = {
            "dataset_filename": dataset_filename,
            "dataset_rows_count": len(dataset_rows),
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "rounds": rounds,
        }
        self._start_db_run("dataset", config)

        # Add internal index to rows to track them
        for i, row in enumerate(dataset_rows):
            row['__index__'] = i

        session_counter = 0

        for r in range(rounds):
            # Process dataset in chunks of size 'concurrency'
            for i in range(0, len(dataset_rows), concurrency):
                batch = dataset_rows[i : i + concurrency]
                current_concurrency = len(batch)

                self.status_text.info(f"currently运行DatasetTest: 轮数 {r+1}/{rounds}, 进度 {i}/{len(dataset_rows)} (Concurrency: {current_concurrency})...")

                results = await self._run_dataset_batch(None, batch, max_tokens, session_counter)
                session_counter += current_concurrency

                for res in results:
                    if res and res.get("error") != "UserCancelled":
                        res['dataset_filename'] = dataset_filename
                        res['round'] = r + 1
                        append_to_csv(res, csv_columns, self.csv_file)
                        self.results_list.append(res)

                    self.completed_requests += 1
                    self.update_ui()

        # 批量SaveResult到Database并完成运行
        self._batch_save_results_to_db()
        self._complete_db_run(success=True)

        return pd.DataFrame(self.results_list)

    async def run_all_tests(self, concurrencies_csv, rounds_per_level_c, input_tokens_c, max_tokens_c,
                              token_levels, req_per_level_p, max_tokens_p,
                              context_lengths, rounds_per_level_l, max_tokens_l):

        session_counter = 0
        initialize_csv(self.combined_csv_columns, self.csv_file)

        # 启动DatabaseTest运行
        config = {
            "concurrencies_csv": concurrencies_csv,
            "rounds_per_level_c": rounds_per_level_c,
            "input_tokens_c": input_tokens_c,
            "max_tokens_c": max_tokens_c,
            "token_levels": token_levels,
            "req_per_level_p": req_per_level_p,
            "max_tokens_p": max_tokens_p,
            "context_lengths": context_lengths,
            "rounds_per_level_l": rounds_per_level_l,
            "max_tokens_l": max_tokens_l,
        }
        self._start_db_run("all_tests", config)

        concurrencies = [int(c.strip()) for c in concurrencies_csv.split(',') if c.strip()]

        # 1. Concurrency Test
        self.status_text.info("开始Concurrency Test...")
        # No client needed
        for concurrency in concurrencies:
            for r in range(rounds_per_level_c):
                self.status_text.info(f"ConcurrencyTest: {concurrency} Concurrency,  {r+1}/{rounds_per_level_c} 轮...")

                # Generate calibrated prompts (parallelized to avoid serial bottleneck)
                # Pre-load tokenizer in main thread and pass to threads for thread-safety
                cached_tokenizer = self._get_tokenizer()

                loop = asyncio.get_event_loop()
                prompt_tasks = [
                    loop.run_in_executor(None, self._calibrate_prompt, self._prompt_generation_target(input_tokens_c), "", cached_tokenizer)
                    for _ in range(concurrency)
                ]
                batch_prompts = await asyncio.gather(*prompt_tasks)

                results = await self._run_concurrency_batch(None, batch_prompts, max_tokens_c, concurrency, session_counter)
                session_counter += concurrency

                for res in results:
                    if res and res.get("error") != "UserCancelled":
                        res.update({'test_type': 'concurrency', 'concurrency': concurrency, 'round': r + 1})
                        append_to_csv(res, self.combined_csv_columns, self.csv_file)
                        self.results_list.append(res)

                    self.completed_requests += 1
                    self.update_ui()

        # 2. Prefill Test
        self.status_text.info("Start Prefill Stress Test...")
        # No client needed
        for tokens_target in token_levels:
            self.status_text.info(f"currently准备 {tokens_target} (目标) Token Tip...")

            local_tokens_to_generate = self._prompt_generation_target(tokens_target)

            if local_tokens_to_generate <= 0:
                st.warning(f"目标 Token {tokens_target} 太小，跳过。")
                self.completed_requests += req_per_level_p
                continue
            prompt_text, local_token_estimate, q_summary = self._get_text_for_token_count(local_tokens_to_generate)
            long_prompt = prompt_text

            status_msg = f"currentlyTest (目标: {tokens_target})..."
            if q_summary:
                status_msg += f"\n📋 {q_summary}"
            self.status_text.info(status_msg)
            for i in range(req_per_level_p):
                res = await self._run_prefill_request(None, long_prompt, max_tokens_p, i)

                if res and res.get("error") != "UserCancelled":

                    actual_prompt_tokens = res.get('prefill_tokens', 0)
                    if actual_prompt_tokens == 0 and res.get("error") is None:
                            self._update_log(f"Warning Session {res.get('session_id')}: API/Tiktoken Return 0 prompt tokens。", level=LogLevel.WARNING, session_id=str(res.get('session_id')))

                    res.update({
                        'test_type': 'prefill',
                        'input_tokens_target': tokens_target
                    })

                    # 仅use未缓存 token Calculate输入速度
                    cache_hit = res.get('cache_hit_tokens', 0) or 0
                    uncached_prompt_tokens = max(0, actual_prompt_tokens - cache_hit)

                    if res['ttft'] > 0:
                        res['prefill_speed'] = uncached_prompt_tokens / res['ttft']
                    else:
                        res['prefill_speed'] = 0

                    append_to_csv(res, self.combined_csv_columns, self.csv_file)
                    self.results_list.append(res)

                self.completed_requests += 1
                self.update_ui()

        # 3. Long Context Test
        self.status_text.info("Start Long Context Test...")
        # No client needed
        for length_target in context_lengths:
            self.status_text.info(f"currently准备 {length_target} (目标) Token Tip...")

            local_tokens_to_generate = self._prompt_generation_target(length_target)

            if local_tokens_to_generate <= 0:
                st.warning(f"目标 Token {length_target} 太小，跳过。")
                self.completed_requests += 1
                continue

            for r in range(rounds_per_level_l):
                # Generate unique prompt
                prompt_text, _, q_summary = self._get_text_for_token_count(local_tokens_to_generate, force_random=True)
                long_prompt = prompt_text

                status_msg = f"currentlyTest (目标: {length_target}, 轮数: {r+1}/{rounds_per_level_l})..."
                if q_summary:
                    status_msg += f"\n📋 {q_summary}"
                self.status_text.info(status_msg)

                res = await self._run_long_context_request(None, long_prompt, max_tokens_l, 0)

                if res and res.get("error") != "UserCancelled":

                    actual_prompt_tokens = res.get('prefill_tokens', 0)
                    if actual_prompt_tokens == 0 and res.get("error") is None:
                            self._update_log(f"Warning Session {res.get('session_id')}: API/Tiktoken Return 0 prompt tokens。", level=LogLevel.WARNING, session_id=str(res.get('session_id')))

                    res.update({
                        'test_type': 'long_context',
                        'context_length_target': length_target,
                        'round': r + 1
                    })

                    # 仅use未缓存 token Calculate输入速度
                    cache_hit = res.get('cache_hit_tokens', 0) or 0
                    uncached_prompt_tokens = max(0, actual_prompt_tokens - cache_hit)

                    if res['ttft'] > 0:
                        res['prefill_speed'] = uncached_prompt_tokens / res['ttft']
                    else:
                        res['prefill_speed'] = 0

                    append_to_csv(res, self.combined_csv_columns, self.csv_file)
                    self.results_list.append(res)

                self.completed_requests += 1
                self.update_ui()

        # 批量SaveResult到Database并完成运行
        self._batch_save_results_to_db()
        self._complete_db_run(success=True)

        return pd.DataFrame(self.results_list)

    async def _run_time_based_batch(self, client, prompt_func_or_str, max_tokens, concurrency, duration, session_id_start):
        """
        Run requests continuously for a specific duration with fixed concurrency.
        """
        start_test_time = time.time()
        end_test_time = start_test_time + duration

        # Shared counter for session IDs
        session_counter = [session_id_start]

        tasks = []
        results = []

        # Shared stats for real-time throughput calculation
        stats = {
            "completed_requests": 0,
            "total_output_tokens": 0,
            "total_input_tokens": 0,
            "successful_requests": 0,
            "min_start_time": float('inf'),
            "max_end_time": 0.0,
            "min_first_token_time": float('inf'),
            "max_first_token_time": 0.0,
            "total_cache_hit_tokens": 0
        }

        async def worker(worker_index):
            while time.time() < end_test_time:
                # Check for stop signal
                if st.session_state.get('stop_requested', False):
                    break

                # Get next session ID
                session_id = session_counter[0]
                session_counter[0] += 1

                # Determine prompt
                if isinstance(prompt_func_or_str, list):
                    prompt = prompt_func_or_str[session_id % len(prompt_func_or_str)]
                elif callable(prompt_func_or_str):
                    prompt = prompt_func_or_str(session_id)
                else:
                    prompt = prompt_func_or_str

                req_start_time = time.time()
                try:
                    res = await self.get_completion(client, session_id, prompt, max_tokens)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._update_log(f"Worker {worker_index} error: {e}", level=LogLevel.ERROR)
                    res = None

                req_end_time = time.time()

                # Update stats and results
                if res:
                    # Time bounds
                    req_start = res.get("start_time", req_start_time)
                    req_end = res.get("end_time", req_end_time)

                    if req_start < stats["min_start_time"]:
                        stats["min_start_time"] = req_start
                    if req_end > stats["max_end_time"]:
                        stats["max_end_time"] = req_end

                    first_token_time = res.get("first_token_time")
                    if first_token_time:
                        if first_token_time < stats["min_first_token_time"]:
                            stats["min_first_token_time"] = first_token_time
                        if first_token_time > stats["max_first_token_time"]:
                            stats["max_first_token_time"] = first_token_time

                    # Calculate cumulative metrics
                    current_time = time.time()
                    total_elapsed = max(0.001, current_time - start_test_time)

                    # Output Throughput
                    if stats["min_first_token_time"] != float('inf'):
                        decode_elapsed = max(0.001, current_time - stats["min_first_token_time"])
                    else:
                        decode_elapsed = total_elapsed

                    # Input Throughput
                    if stats["max_first_token_time"] > 0 and stats["min_start_time"] != float('inf'):
                         prefill_elapsed = max(0.001, stats["max_first_token_time"] - stats["min_start_time"])
                    else:
                         prefill_elapsed = total_elapsed

                    if res.get("error") != "UserCancelled" and res.get("error") is None:
                        stats["total_output_tokens"] += res.get('decode_tokens', 0)
                        stats["total_input_tokens"] += res.get('prefill_tokens', 0)
                        stats["total_cache_hit_tokens"] += res.get('cache_hit_tokens', 0) or 0
                        stats["successful_requests"] += 1

                    stats["completed_requests"] += 1

                    # Update result with system metrics
                    # Input Throughput仅use未缓存 token 数
                    uncached_input_tokens = max(0, stats["total_input_tokens"] - stats["total_cache_hit_tokens"])
                    res['system_output_throughput'] = stats["total_output_tokens"] / decode_elapsed
                    res['system_input_throughput'] = uncached_input_tokens / prefill_elapsed
                    res['rps'] = stats["successful_requests"] / total_elapsed

                    results.append(res)

                    # Update global progress
                    self.completed_requests += 1
                    if self.total_requests > 0:
                        self.update_ui()

        # Launch workers
        tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]

        await asyncio.gather(*tasks)

        return results

    async def run_stability_test(self, concurrency, duration_seconds, max_tokens, input_tokens_target=0):
        self.total_requests = 0 # Indeterminate

        csv_columns = ["session_id", "concurrency", "timestamp",
                       "ttft", "tps", "tpot", "prefill_speed",
                       "system_throughput", "system_input_throughput", "rps",
                       "prefill_tokens", "decode_tokens", "total_time", "decode_time",
                       "start_time", "end_time", "cache_hit_tokens", "token_calc_method", "input_tokens_target", "error"]
        initialize_csv(csv_columns, self.csv_file)

        # 启动DatabaseTest运行
        config = {
            "concurrency": concurrency,
            "duration_seconds": duration_seconds,
            "max_tokens": max_tokens,
            "input_tokens_target": input_tokens_target,
        }
        self._start_db_run("stability", config)

        self.status_text.info(f"currently以 {concurrency} Concurrency运行Stability Test (持续 {duration_seconds} seconds)...")

        # Generator for prompts
        if input_tokens_target > 0:
             def prompt_source(i):
                 return self._calibrate_prompt(self._prompt_generation_target(input_tokens_target), suffix="")
        else:
             def prompt_source(i):
                 return self._calibrate_prompt(self._prompt_generation_target(64), suffix="")

        results = await self._run_time_based_batch(
            None,
            prompt_source,
            max_tokens,
            concurrency,
            duration_seconds,
            session_id_start=0
        )

        for res in results:
            if res and res.get("error") != "UserCancelled":
                res['concurrency'] = concurrency
                res['timestamp'] = res.get('end_time')
                append_to_csv(res, csv_columns, self.csv_file)
                self.results_list.append(res)

        # 批量SaveResult到Database并完成运行
        self._batch_save_results_to_db()
        self._complete_db_run(success=True)

        return pd.DataFrame(self.results_list)
