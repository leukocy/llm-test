"""
Test Panels Module

Provides configuration panels for various test types, including:
- Concurrency Test
- Prefill Stress Test
- Long Context Test
- Concurrency-Context Matrix Test
- Custom Text Test
- All Tests
- Stability Test

Startup Optimization:
- BenchmarkRunner uses lazy import to reduce startup time

Bug Fixes:
- Added unique keys to all input controls to prevent accidental test triggers on parameter changes
"""

import streamlit as st

from config.session_state import set_current_test_type
from config.test_types import test_type_label
from ui.design_system import material_icon

# Lazy import BenchmarkRunner (loaded only when test is executed)
_BenchmarkRunner = None


def _get_benchmark_runner():
    """Lazy-load BenchmarkRunner class"""
    global _BenchmarkRunner
    if _BenchmarkRunner is None:
        from core.benchmark_runner import BenchmarkRunner

        _BenchmarkRunner = BenchmarkRunner
    return _BenchmarkRunner


def _is_test_active():
    """Check if test is currently active (running or paused)"""
    test_status = st.session_state.get("test_status", "Idle")
    return test_status in ("Running", "Paused")


def _get_button_disabled_state():
    """
    Get button disabled state - directly from session_state without caching.
    """
    # 直接从 session_state 读取，不做任何缓存
    return st.session_state.get("test_running", False)


def _execute_pending_test(run_test_func, test_type):
    """
    Execute pending test if there's one scheduled.
    This is called at the start of render_test_panels to execute tests
    after the state has been set and UI has been rerun.

    支持两种格式：
    1. 新测试：{'test_type': ..., 'test_func': ..., 'runner_class': ..., 'args': ...}
    2. 恢复测试：{'test_type': ..., 'is_resume': True, 'resume_data': ...}

    返回：
        bool: True 如果执行了测试（需要 rerun 来刷新 UI）
    """
    pending = st.session_state.get("_pending_test", None)
    if not pending:
        return False

    pending_test_type = set_current_test_type(
        pending.get("test_type"),
        sync_widget_key=False,
    )
    if pending_test_type != test_type:
        st.session_state["_force_test_type_selector"] = pending_test_type
        st.rerun()
        return False

    # 清除待执行标志
    st.session_state["_pending_test"] = None

    # 检查是否是恢复模式（支持两种检查方式）
    is_resume = pending.get("is_resume") or st.session_state.get("is_resuming", False)

    if is_resume:
        # Resume 模式：设置恢复标志，然后执行测试
        # 优先从 pending 获取，然后从 session_state 获取
        resume_data = pending.get("resume_data")
        if not resume_data:
            resume_data = st.session_state.get("resume_data", {})

        # 确保 resume_data 不是空字典
        if resume_data:
            st.session_state.is_resuming = True
            st.session_state.resume_data = resume_data

            # 显示恢复信息
            completed = resume_data.get("current_index", 0)
            total = resume_data.get("total_samples", 0) or resume_data.get(
                "total_requests", 0
            )
            st.info(
                f"Resuming {test_type_label(test_type)} from saved progress "
                f"({completed}/{total} completed)."
            )
        else:
            st.warning("No saved progress found, starting fresh test.")
            st.session_state.is_resuming = False

    # 执行测试（无论是新测试还是恢复测试）
    if "test_func" in pending and "runner_class" in pending:
        test_func = pending["test_func"]
        runner_class = pending["runner_class"]
        args = pending.get("args", ())

        run_test_func(test_func, runner_class, *args)

        # 测试完成后触发 rerun 以刷新 UI 状态
        st.rerun()

    return True


def render_test_panels(test_type, run_test_func):
    """
    Render all test panels

    Args:
        test_type: Currently selected test type
        run_test_func: Test execution function

    Returns:
        bool: Returns True if test was triggered, False otherwise
    """

    # 检查是否有待执行的测试（在状态更新后的渲染周期执行）
    # 如果执行了测试，函数会在内部调用 st.rerun() 刷新页面
    _execute_pending_test(run_test_func, test_type)

    # Concurrency Test
    if test_type == "concurrency":
        st.header("Concurrency Test")
        with st.expander("Test parameters", expanded=True):
            concurrencies_select = st.multiselect(
                "Select Concurrency Levels",
                [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                default=[1, 2],
                key="con_concurrencies_select",
            )
            concurrencies_custom = st.text_input(
                "Custom Concurrency (comma-separated)", key="con_custom"
            )
            rounds_per_level = st.number_input(
                "Rounds Per Level", min_value=1, value=1, key="con_rounds_per_level"
            )
            con_input_tokens = st.number_input(
                "Input Token Length (Input Calibration)",
                min_value=1,
                value=64,
                step=1,
                help="Strictly calibrate input prompt token length (auto-match instruction suffix based on length)",
                key="con_input_tokens",
            )
            max_tokens = st.number_input(
                "Max Output Tokens", min_value=1, value=512, key="con_max_tokens"
            )
            st.markdown("---")
            start_btn_con = st.button(
                "Run concurrency test",
                key="start_con_main_btn",
                type="primary",
                icon=material_icon("play_arrow"),
                disabled=_get_button_disabled_state(),
            )

        if start_btn_con:
            try:
                custom_values = [
                    int(c.strip())
                    for c in concurrencies_custom.split(",")
                    if c.strip() and c.strip().isdigit()
                ]
            except (ValueError, AttributeError):
                custom_values = []

            selected_concurrencies = sorted(set(concurrencies_select + custom_values))

            if not selected_concurrencies:
                st.error("Please enter at least one concurrency level.")
            else:
                # 保存完整的测试配置（用于 Resume）
                st.session_state.current_test_config = {
                    "concurrency_levels": selected_concurrencies,
                    "rounds": rounds_per_level,
                    "max_tokens": max_tokens,
                    "input_tokens": con_input_tokens,
                }

                # 设置运行状态并存储待执行的测试
                from config.session_state import set_test_running

                set_current_test_type(test_type, sync_widget_key=False)
                st.session_state["_pending_test"] = {
                    "test_type": test_type,
                    "test_func": _get_benchmark_runner().run_concurrency_test,
                    "runner_class": _get_benchmark_runner(),
                    "args": (
                        selected_concurrencies,
                        rounds_per_level,
                        max_tokens,
                        con_input_tokens,
                    ),
                }
                set_test_running()
                st.rerun()

    # Prefill Stress Test
    elif test_type == "prefill":
        st.header("Prefill Stress Test")
        with st.expander("Test parameters", expanded=True):
            prefill_tokens_select = st.multiselect(
                "Select Token Counts",
                [
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                    130000,
                    260000,
                    520000,
                    1000000,
                ],
                default=[4096, 8192, 16384, 32768, 65536, 130000],
                key="prefill_tokens_select",
            )
            prefill_tokens_custom = st.text_input(
                "Custom Token Counts (comma-separated)", key="prefill_tokens_custom"
            )
            requests_per_level = st.number_input(
                "Requests Per Level",
                min_value=1,
                value=1,
                key="prefill_requests_per_level",
            )

            st.markdown("---")
            prefill_isolation_mode = st.checkbox(
                "Prefill Isolation Test (max_tokens=1)",
                value=True,
                help="When checked, forces Max Output Tokens to 1 for precise prefill performance measurement.",
                key="prefill_isolation_mode",
            )
            max_tokens_prefill_input = st.number_input(
                "Max Output Tokens",
                min_value=1,
                value=512,
                disabled=prefill_isolation_mode,
                key="prefill_max_tokens",
            )
            st.markdown("---")
            start_btn_prefill = st.button(
                "Run prefill stress test",
                key="start_prefill_main_btn",
                type="primary",
                icon=material_icon("play_arrow"),
                disabled=_get_button_disabled_state(),
            )

        if start_btn_prefill:
            max_tokens_to_use = (
                1 if prefill_isolation_mode else max_tokens_prefill_input
            )

            if prefill_isolation_mode:
                st.info("Prefill isolation test activated (max_tokens=1).")

            custom_tokens = [
                int(t.strip())
                for t in prefill_tokens_custom.split(",")
                if t.strip() and t.strip().isdigit()
            ]
            token_levels = sorted(set(prefill_tokens_select + custom_tokens))

            if not token_levels:
                st.error("Please enter at least one token count.")
            else:
                from config.session_state import set_test_running

                set_current_test_type(test_type, sync_widget_key=False)
                st.session_state.current_test_config = {
                    "token_levels": token_levels,
                    "requests_per_level": requests_per_level,
                    "max_tokens": max_tokens_to_use,
                }
                st.session_state["_pending_test"] = {
                    "test_type": test_type,
                    "test_func": _get_benchmark_runner().run_prefill_test,
                    "runner_class": _get_benchmark_runner(),
                    "args": (token_levels, requests_per_level, max_tokens_to_use),
                }
                set_test_running()
                st.rerun()

    # Segmented Context Test
    elif test_type == "segmented":
        st.header("Segmented Context Test (Prefix Caching)")
        st.info(
            """
        **Test Description**: Simulates real-world scenarios where users send cumulative long text in segments (e.g., document Q&A, multi-turn conversations).

        - **Cumulative Mode**: All segments share the same prefix, testing Prefix Caching optimization
        - **Independent Mode**: Each segment has independent content, serving as a no-cache control group
        """
        )

        with st.expander("Test parameters", expanded=True):
            # Segment level selection
            segment_presets = {
                "Progressive": [2000, 8000, 20000, 40000, 60000],
                "Rapid Growth": [4000, 16000, 32000, 64000],
                "Fine-grained": [1000, 2000, 4000, 8000, 16000, 32000, 64000],
            }

            selected_preset = st.selectbox(
                "Select Preset Strategy",
                ["Custom"] + list(segment_presets.keys()),
                key="segment_preset",
            )

            if selected_preset == "Custom":
                segment_levels_input = st.text_input(
                    "Segment Levels (comma-separated, unit: tokens)",
                    "2000, 8000, 20000, 40000, 60000",
                    help="Example: 2000, 8000, 20000, 40000, 60000",
                    key="segment_levels_input",
                )
                segment_levels = sorted(
                    {
                        int(s.strip())
                        for s in segment_levels_input.split(",")
                        if s.strip() and s.strip().isdigit()
                    }
                )
            else:
                segment_levels = segment_presets[selected_preset]
                st.info(f"Segment levels: {segment_levels}")

            requests_per_segment = st.number_input(
                "Requests Per Segment",
                min_value=1,
                value=1,
                key="segment_requests_per_segment",
            )

            st.markdown("---")

            # Test mode
            cumulative_mode = st.checkbox(
                "Cumulative Mode (Prefix Caching)",
                value=True,
                help="When checked, all segments share the same prefix to test Prefix Caching effect",
                key="segment_cumulative_mode",
            )

            # Test rounds
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                total_rounds = st.number_input(
                    "Total Test Rounds",
                    min_value=1,
                    value=1,
                    key="segment_total_rounds",
                )
            with col_r2:
                per_round_unique = st.checkbox(
                    "Unique Prompt Per Round",
                    value=False,
                    key="segment_per_round_unique",
                )

            # Concurrency settings
            segment_concurrency = st.number_input(
                "Concurrency", min_value=1, value=1, key="segment_concurrency"
            )

            st.markdown("---")

            # Max tokens
            max_tokens_segment = st.number_input(
                "Max Output Tokens", min_value=1, value=512, key="segment_max_tokens"
            )

            start_btn_segment = st.button(
                "Run segmented context test",
                key="start_segment_main_btn",
                type="primary",
                icon=material_icon("play_arrow"),
                disabled=_get_button_disabled_state(),
            )

        if start_btn_segment:
            if not segment_levels:
                st.error("Please enter at least one segment level.")
            else:
                mode_str = "Cumulative Mode" if cumulative_mode else "Independent Mode"
                st.info(f"Starting Segmented Context Test ({mode_str})")
                st.info(f"Segment levels: {segment_levels}")
                st.info(
                    f"Requests per segment: {requests_per_segment}, Test rounds: {total_rounds}, Concurrency: {segment_concurrency}"
                )

                from config.session_state import set_test_running

                set_current_test_type(test_type, sync_widget_key=False)
                st.session_state.current_test_config = {
                    "segment_levels": segment_levels,
                    "requests_per_segment": requests_per_segment,
                    "max_tokens": max_tokens_segment,
                    "cumulative_mode": cumulative_mode,
                    "total_rounds": total_rounds,
                    "per_round_unique": per_round_unique,
                    "concurrency": segment_concurrency,
                }
                st.session_state["_pending_test"] = {
                    "test_type": test_type,
                    "test_func": _get_benchmark_runner().run_segmented_prefill_test,
                    "runner_class": _get_benchmark_runner(),
                    "args": (
                        segment_levels,
                        requests_per_segment,
                        max_tokens_segment,
                        cumulative_mode,
                        total_rounds,
                        per_round_unique,
                        segment_concurrency,
                    ),
                }
                set_test_running()
                st.rerun()

    # Long Context Test
    elif test_type == "long_context":
        st.header("Long Context Test")
        with st.expander("Test parameters", expanded=True):
            context_lengths_select = st.multiselect(
                "Select Context Lengths",
                [
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                    130000,
                    260000,
                    520000,
                    1000000,
                ],
                default=[4096, 8192, 16384, 32768, 65536, 130000],
                key="long_context_lengths_select",
            )
            context_lengths_custom = st.text_input(
                "Custom Context Lengths (comma-separated)",
                key="long_context_lengths_custom",
            )
            context_rounds = st.number_input(
                "Rounds Per Level", min_value=1, value=1, key="long_context_rounds"
            )
            max_tokens_long = st.number_input(
                "Max Output Tokens", min_value=1, value=512, key="long_max_tokens"
            )
            st.markdown("---")
            start_btn_long = st.button(
                "Run long context test",
                key="start_long_main_btn",
                type="primary",
                icon=material_icon("play_arrow"),
                disabled=_get_button_disabled_state(),
            )

        if start_btn_long:
            custom_lengths = [
                int(l.strip())
                for l in context_lengths_custom.split(",")
                if l.strip() and l.strip().isdigit()
            ]
            context_lengths = sorted(set(context_lengths_select + custom_lengths))

            if not context_lengths:
                st.error("Please enter at least one context length.")
            else:
                from config.session_state import set_test_running

                set_current_test_type(test_type, sync_widget_key=False)
                st.session_state.current_test_config = {
                    "context_lengths": context_lengths,
                    "rounds": context_rounds,
                    "max_tokens": max_tokens_long,
                }
                st.session_state["_pending_test"] = {
                    "test_type": test_type,
                    "test_func": _get_benchmark_runner().run_long_context_test,
                    "runner_class": _get_benchmark_runner(),
                    "args": (context_lengths, context_rounds, max_tokens_long),
                }
                set_test_running()
                st.rerun()

    # Concurrency-Context Matrix Test
    elif test_type == "matrix":
        st.header("Concurrency-Context Matrix Test")
        with st.expander("Test parameters", expanded=True):
            matrix_concurrencies_select = st.multiselect(
                "Select Concurrency Levels",
                [1, 2, 4, 8, 16, 64, 128, 256, 512, 1024],
                default=[1, 2],
                key="matrix_concurrencies_select",
            )
            matrix_concurrencies_custom = st.text_input(
                "Custom Concurrency Levels (comma-separated)", key="matrix_con_custom"
            )

            matrix_context_select = st.multiselect(
                "Select Context Lengths",
                [1024, 2048, 4096, 8192, 16384, 32768, 65536, 260000, 520000, 1000000],
                default=[1024, 4096, 16384, 65536],
                key="matrix_context_select",
            )
            matrix_context_custom = st.text_input(
                "Custom Context Lengths (comma-separated)", key="matrix_ctx_custom"
            )

            matrix_rounds = st.number_input(
                "Rounds Per Combination", min_value=1, value=1, key="matrix_rounds"
            )
            matrix_max_tokens = st.number_input(
                "Max Output Tokens", min_value=1, value=256, key="matrix_max_tokens"
            )

            enable_warmup = st.checkbox(
                "Enable Warmup",
                value=True,
                help="Run a single warmup request before each test group to trigger Prefix Caching, ensuring consistent Prefill time for concurrent requests and synchronized Decode phase.",
                key="matrix_enable_warmup",
            )
            st.markdown("---")
            start_btn_matrix = st.button(
                "Run concurrency-context matrix test",
                key="start_matrix_main_btn",
                type="primary",
                icon=material_icon("play_arrow"),
                disabled=_get_button_disabled_state(),
            )

        if start_btn_matrix:
            try:
                custom_cons = [
                    int(c.strip())
                    for c in matrix_concurrencies_custom.split(",")
                    if c.strip() and c.strip().isdigit()
                ]
                selected_concurrencies = sorted(
                    set(matrix_concurrencies_select + custom_cons)
                )

                custom_ctxs = [
                    int(l.strip())
                    for l in matrix_context_custom.split(",")
                    if l.strip() and l.strip().isdigit()
                ]
                selected_contexts = sorted(set(matrix_context_select + custom_ctxs))

                if not selected_concurrencies or not selected_contexts:
                    st.error(
                        "Please select at least one concurrency level and one context length."
                    )
                else:
                    from config.session_state import set_test_running

                    set_current_test_type(test_type, sync_widget_key=False)
                    st.session_state.current_test_config = {
                        "concurrency_levels": selected_concurrencies,
                        "context_lengths": selected_contexts,
                        "rounds": matrix_rounds,
                        "max_tokens": matrix_max_tokens,
                        "enable_warmup": enable_warmup,
                    }
                    st.session_state["_pending_test"] = {
                        "test_type": test_type,
                        "test_func": _get_benchmark_runner().run_throughput_matrix_test,
                        "runner_class": _get_benchmark_runner(),
                        "args": (
                            selected_concurrencies,
                            selected_contexts,
                            matrix_rounds,
                            matrix_max_tokens,
                            enable_warmup,
                        ),
                    }
                    set_test_running()
                    st.rerun()
            except ValueError:
                st.error(
                    "Invalid custom value format. Please use comma-separated numbers."
                )

    # Custom Text Test
    elif test_type == "custom":
        st.header("Custom Text Test")
        st.info(
            "Choose a prompt source, optionally pad to a context length, and run. "
            "No file required — pick problems from the test pool, or type your own."
        )

        # --- Prompt source ---
        source_mode = st.radio(
            "Prompt Source",
            ["Test Pool Problems", "Upload TXT File", "Manual Input"],
            index=0,
            key="custom_source_mode",
            horizontal=True,
        )

        selected_problems = None  # list of (source_id, text)
        base_prompt = ""
        base_prompt_source = "custom_text"
        uploaded_file = None

        if source_mode == "Upload TXT File":
            uploaded_file = st.file_uploader(
                "Upload TXT File (base context)",
                type=["txt"],
                key="custom_uploaded_file",
            )
        elif source_mode == "Test Pool Problems":
            try:
                from core.benchmark_runner import SUFFIX_TYPE_OPTIONS, _load_typed_pools

                typed = _load_typed_pools()
                # Type selector
                avail_types = [
                    (k, lbl) for k, lbl, _ in SUFFIX_TYPE_OPTIONS if typed.get(k)
                ]
                if not avail_types:
                    st.warning("No test-pool problems available.")
                else:
                    t_keys = [k for k, _ in avail_types]
                    t_labels = {k: f"{lbl} ({len(typed[k])})" for k, lbl in avail_types}
                    sel_type = st.selectbox(
                        "Problem Type",
                        options=t_keys,
                        format_func=lambda k: t_labels[k],
                        key="custom_problem_type",
                    )
                    problems = typed.get(sel_type, [])  # list of (source_id, text)
                    labels = [sid for sid, _ in problems]
                    chosen = st.multiselect(
                        f"Select Problems ({len(labels)} available, rotate per request)",
                        options=range(len(labels)),
                        format_func=lambda i: labels[i],
                        key="custom_selected_problems",
                        help="Multiple problems are rotated one-per-request. Single selection repeats the same problem.",
                    )
                    if chosen:
                        selected_problems = [
                            (problems[i][0], problems[i][1]) for i in chosen
                        ]
                        # Show the content of each selected problem
                        st.markdown(f"**Selected problem content ({len(chosen)}):**")
                        for i in chosen:
                            sid, text = problems[i]
                            with st.expander(
                                f"{sid}  ({len(text)} chars)", expanded=False
                            ):
                                st.text(
                                    text[:4000] + ("..." if len(text) > 4000 else "")
                                )
            except Exception as e:
                st.warning(f"Could not load test pool: {e}")
        else:  # Manual Input
            base_prompt = st.text_area(
                "Base Prompt", height=150, key="custom_manual_base"
            )

        # --- Optional suffix instruction (appended after the base/problem) ---
        suffix_instruction = st.text_area(
            "Extra Suffix Instruction (optional)",
            "",
            height=80,
            key="custom_suffix_instruction",
            help="Appended after the base/problem text. Leave empty when using test-pool problems (they are already tasks).",
        )

        # --- Context length (padding) ---
        context_length = st.number_input(
            "Context Length (tokens)",
            min_value=0,
            value=0,
            step=64,
            key="custom_context_length",
            help="0 = send prompt verbatim (no padding; may hit cache — use when you only care about output). "
            ">0 = pad each prompt with random noise to exactly this many tokens to avoid cache hits.",
        )

        # --- Concurrency (aligned with other tests) ---
        custom_concurrencies_select = st.multiselect(
            "Select Concurrency Levels",
            [1, 2, 4, 8, 16],
            default=[1, 2, 4],
            key="custom_concurrencies_select",
        )
        custom_concurrencies_csv = st.text_input(
            "Custom Concurrency (comma-separated)", key="custom_concurrencies_csv"
        )
        custom_rounds = st.number_input(
            "Rounds Per Level", min_value=1, value=1, key="custom_rounds"
        )
        custom_max_tokens = st.number_input(
            "Max Output Tokens", min_value=1, value=512, key="custom_max_tokens"
        )

        st.markdown("---")
        start_btn_custom = st.button(
            "Run custom text test",
            key="start_custom_main_btn",
            type="primary",
            icon=material_icon("play_arrow"),
            disabled=_get_button_disabled_state(),
        )

        if start_btn_custom:
            # Resolve concurrency (preset + custom)
            try:
                custom_values = [
                    int(c.strip())
                    for c in custom_concurrencies_csv.split(",")
                    if c.strip() and c.strip().isdigit()
                ]
            except (ValueError, AttributeError):
                custom_values = []
            selected_concurrencies = sorted(
                set(custom_concurrencies_select + custom_values)
            )

            # Resolve base prompt from file
            if uploaded_file is not None:
                try:
                    base_prompt = uploaded_file.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    base_prompt = ""

            # Validate
            if not selected_concurrencies:
                st.error("Please select or enter at least one concurrency level.")
            elif source_mode == "Upload TXT File" and not base_prompt:
                st.error("Please upload a non-empty TXT file.")
            elif source_mode == "Manual Input" and not base_prompt.strip():
                st.error("Please enter a non-empty base prompt.")
            elif source_mode == "Test Pool Problems" and not selected_problems:
                st.error("Please select at least one problem.")
            else:
                from config.session_state import set_test_running

                set_current_test_type(test_type, sync_widget_key=False)
                st.session_state.current_test_config = {
                    "concurrency_levels": selected_concurrencies,
                    "rounds": custom_rounds,
                    "max_tokens": custom_max_tokens,
                    "context_length": int(context_length),
                }
                st.session_state["_pending_test"] = {
                    "test_type": test_type,
                    "test_func": _get_benchmark_runner().run_custom_text_test,
                    "runner_class": _get_benchmark_runner(),
                    "args": (
                        selected_concurrencies,
                        custom_rounds,
                        base_prompt,
                        suffix_instruction,
                        custom_max_tokens,
                        True,
                        int(context_length),
                        base_prompt_source,
                        selected_problems,
                    ),
                }
                set_test_running()
                st.rerun()

    # All Tests
    elif test_type == "all":
        st.header("All Tests")
        st.warning(
            "This option will run all test types sequentially, which may take a long time."
        )

        st.subheader("Concurrency Test Configuration")
        all_con = st.text_input(
            "Concurrency Levels (comma-separated)", "1,2", key="all_con_levels"
        )
        all_con_rounds = st.number_input(
            "Concurrency Test Rounds", min_value=1, value=1, key="all_con_rounds"
        )
        all_con_input_tokens = st.number_input(
            "Input Token Length (Concurrency Phase)",
            min_value=1,
            value=64,
            step=1,
            key="all_con_input_tokens",
        )
        all_con_tokens = st.number_input(
            "Max Output Tokens (Concurrency)",
            min_value=1,
            value=512,
            key="all_con_max_tokens",
        )

        st.subheader("Prefill Test Configuration")
        all_prefill = st.text_input(
            "Token Counts (comma-separated)", "20000,40000", key="all_prefill_tokens"
        )
        all_prefill_req = st.number_input(
            "Requests Per Level", min_value=1, value=1, key="all_prefill_requests"
        )
        all_prefill_tokens = st.number_input(
            "Max Output Tokens (Prefill)",
            min_value=1,
            value=1,
            key="all_prefill_max_tokens",
        )

        st.subheader("Long Context Test Configuration")
        all_context = st.text_input(
            "Context Lengths (comma-separated)",
            "4096,8192,16384,32768,65536,130000,260000,520000,1000000",
            key="all_context_lengths",
        )
        all_context_rounds = st.number_input(
            "Rounds Per Level (Long Context)",
            min_value=1,
            value=1,
            key="all_context_rounds",
        )
        all_context_tokens = st.number_input(
            "Max Output Tokens (Long Context)",
            min_value=1,
            value=512,
            key="all_context_max_tokens",
        )

        st.markdown("---")
        start_btn_all = st.button(
            "Run all tests",
            key="start_all_main_btn",
            type="primary",
            icon=material_icon("play_arrow"),
            disabled=_get_button_disabled_state(),
        )
        if start_btn_all:
            try:
                token_levels_list = [
                    int(t.strip()) for t in all_prefill.split(",") if t.strip()
                ]
                context_lengths_list = [
                    int(l.strip()) for l in all_context.split(",") if l.strip()
                ]

                from config.session_state import set_test_running

                set_current_test_type(test_type, sync_widget_key=False)
                st.session_state.current_test_config = {
                    "concurrency_levels": all_con,
                    "concurrency_rounds": all_con_rounds,
                    "concurrency_input_tokens": all_con_input_tokens,
                    "concurrency_max_tokens": all_con_tokens,
                    "token_levels": token_levels_list,
                    "prefill_requests_per_level": all_prefill_req,
                    "prefill_max_tokens": all_prefill_tokens,
                    "context_lengths": context_lengths_list,
                    "context_rounds": all_context_rounds,
                    "context_max_tokens": all_context_tokens,
                }
                st.session_state["_pending_test"] = {
                    "test_type": test_type,
                    "test_func": _get_benchmark_runner().run_all_tests,
                    "runner_class": _get_benchmark_runner(),
                    "args": (
                        all_con,
                        all_con_rounds,
                        all_con_input_tokens,
                        all_con_tokens,
                        token_levels_list,
                        all_prefill_req,
                        all_prefill_tokens,
                        context_lengths_list,
                        all_context_rounds,
                        all_context_tokens,
                    ),
                }
                set_test_running()
                st.rerun()
            except ValueError as e:
                st.error(f"Configuration error: {e}")

    # Stability Test
    elif test_type == "stability":
        st.header("Stability Test (Time-based Stability)")
        with st.expander("Test parameters", expanded=True):
            st.info("This mode will continuously run tests for the specified duration.")

            stability_concurrency = st.number_input(
                "Concurrency",
                min_value=1,
                value=1,
                help="Number of concurrent requests",
                key="stability_concurrency",
            )

            stability_duration = st.number_input(
                "Duration (seconds)",
                min_value=10,
                value=60,
                step=10,
                help="Duration to run the test",
                key="stability_duration",
            )

            stability_max_tokens = st.number_input(
                "Max Output Tokens", min_value=1, value=512, key="stability_max_tokens"
            )
            stability_input_tokens = st.number_input(
                "Input Token Length",
                min_value=1,
                value=64,
                step=1,
                help="Strictly calibrate input prompt token length",
                key="stability_input_tokens",
            )

            st.markdown("---")
            start_btn_stability = st.button(
                "Run stability test",
                key="start_stability_main_btn",
                type="primary",
                icon=material_icon("play_arrow"),
                disabled=_get_button_disabled_state(),
            )

        if start_btn_stability:
            from config.session_state import set_test_running

            set_current_test_type(test_type, sync_widget_key=False)
            st.session_state.current_test_config = {
                "concurrency": stability_concurrency,
                "duration": stability_duration,
                "max_tokens": stability_max_tokens,
                "input_tokens": stability_input_tokens,
            }
            st.session_state["_pending_test"] = {
                "test_type": test_type,
                "test_func": _get_benchmark_runner().run_stability_test,
                "runner_class": _get_benchmark_runner(),
                "args": (
                    stability_concurrency,
                    stability_duration,
                    stability_max_tokens,
                    stability_input_tokens,
                ),
            }
            set_test_running()
            st.rerun()

    return False
