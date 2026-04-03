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
    test_status = st.session_state.get('test_status', 'Idle')
    return test_status in ('Running', 'Paused')


def _get_button_disabled_state():
    """
    Get button disabled state - directly from session_state without caching.
    """
    # 直接从 session_state 读取，不做任何缓存
    return st.session_state.get('test_running', False)


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
    pending = st.session_state.get('_pending_test', None)
    if not pending:
        return False

    pending_test_type = pending.get('test_type')
    if pending_test_type != test_type:
        return False

    # 清除待执行标志
    st.session_state._pending_test = None

    # 检查是否是恢复模式（支持两种检查方式）
    is_resume = pending.get('is_resume') or st.session_state.get('is_resuming', False)

    # 调试：显示状态
    st.write(f"[DEBUG] pending type: {pending.get('test_type')}, is_resuming from session: {st.session_state.get('is_resuming', False)}")
    st.write(f"[DEBUG] resume_data exists: {'resume_data' in st.session_state}, value: {bool(st.session_state.get('resume_data'))}")

    if is_resume:
        # Resume 模式：设置恢复标志，然后执行测试
        # 优先从 pending 获取，然后从 session_state 获取
        resume_data = pending.get('resume_data')
        if not resume_data:
            resume_data = st.session_state.get('resume_data', {})

        # 确保 resume_data 不是空字典
        if resume_data:
            st.session_state.is_resuming = True
            st.session_state.resume_data = resume_data

            # 显示恢复信息
            completed = resume_data.get('current_index', 0)
            total = resume_data.get('total_samples', 0) or resume_data.get('total_requests', 0)
            st.info(f"🔄 Resuming {test_type} from saved progress... ({completed}/{total} completed)")
        else:
            st.warning("No saved progress found, starting fresh test.")
            st.session_state.is_resuming = False

    # 执行测试（无论是新测试还是恢复测试）
    if 'test_func' in pending and 'runner_class' in pending:
        test_func = pending['test_func']
        runner_class = pending['runner_class']
        args = pending.get('args', ())

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
    if test_type == "Concurrency Test":
        st.header("⚡ Concurrency Test")
        with st.sidebar.expander("📊 Parameter Settings", expanded=True):
            concurrencies_select = st.multiselect(
                "Select Concurrency Levels",
                [1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024],
                default=[1, 2],
                key="con_concurrencies_select"
            )
            concurrencies_custom = st.text_input(
                "Custom Concurrency (comma-separated)",
                key="con_custom"
            )
            rounds_per_level = st.number_input(
                "Rounds Per Level",
                min_value=1,
                value=1,
                key="con_rounds_per_level"
            )
            con_input_tokens = st.number_input(
                "Input Token Length (Input Calibration)",
                min_value=1,
                value=64,
                step=1,
                help="Strictly calibrate input prompt token length (auto-match instruction suffix based on length)",
                key="con_input_tokens"
            )
            max_tokens = st.number_input(
                "Max Output Tokens",
                min_value=1,
                value=512,
                key="con_max_tokens"
            )
            st.markdown("---")
            start_btn_con_sidebar = st.button(
                "🚀 Start Concurrency Test (S)",
                key="start_con_sidebar_btn",
                type="primary",
                disabled=_get_button_disabled_state()
            )

        start_btn_con_main = st.button(
            "🚀 Start Concurrency Test (M)",
            key="start_con_main_btn",
            disabled=_get_button_disabled_state()
        )

        if start_btn_con_main or start_btn_con_sidebar:
            try:
                custom_values = [int(c.strip()) for c in concurrencies_custom.split(',')
                               if c.strip() and c.strip().isdigit()]
            except (ValueError, AttributeError):
                custom_values = []

            selected_concurrencies = sorted(set(concurrencies_select + custom_values))

            if not selected_concurrencies:
                st.error("Please enter at least one concurrency level.")
            else:
                # 保存完整的测试配置（用于 Resume）
                st.session_state.current_test_config = {
                    'concurrency_levels': selected_concurrencies,
                    'rounds': rounds_per_level,
                    'max_tokens': max_tokens,
                    'input_tokens': con_input_tokens
                }

                # 设置运行状态并存储待执行的测试
                from config.session_state import set_test_running
                st.session_state._pending_test = {
                    'test_type': test_type,
                    'test_func': _get_benchmark_runner().run_concurrency_test,
                    'runner_class': _get_benchmark_runner(),
                    'args': (selected_concurrencies, rounds_per_level, max_tokens, con_input_tokens)
                }
                set_test_running()
                st.rerun()

    # Prefill Stress Test
    elif test_type == "Prefill Stress Test":
        st.header("🔥 Prefill Stress Test")
        with st.sidebar.expander("📊 Parameter Settings", expanded=True):
            prefill_tokens_select = st.multiselect(
                "Select Token Counts",
                [512, 1024, 2048, 4096, 8192, 16384, 20000, 32768, 40000, 65536, 98304, 128000, 131072],
                default=[20000, 40000],
                key="prefill_tokens_select"
            )
            prefill_tokens_custom = st.text_input(
                "Custom Token Counts (comma-separated)",
                key="prefill_tokens_custom"
            )
            requests_per_level = st.number_input(
                "Requests Per Level",
                min_value=1,
                value=1,
                key="prefill_requests_per_level"
            )

            st.markdown("---")
            prefill_isolation_mode = st.checkbox(
                "Prefill Isolation Test (max_tokens=1)",
                value=True,
                help="When checked, forces Max Output Tokens to 1 for precise prefill performance measurement.",
                key="prefill_isolation_mode"
            )
            max_tokens_prefill_input = st.number_input(
                "Max Output Tokens",
                min_value=1,
                value=512,
                disabled=prefill_isolation_mode,
                key="prefill_max_tokens"
            )
            st.markdown("---")
            start_btn_prefill_sidebar = st.button(
                "🚀 Start Prefill Test (S)",
                key="start_prefill_sidebar_btn",
                type="primary",
                disabled=_get_button_disabled_state()
            )

        start_btn_prefill_main = st.button(
            "🚀 Start Prefill Stress Test (M)",
            key="start_prefill_main_btn",
            disabled=_get_button_disabled_state()
        )
        if start_btn_prefill_main or start_btn_prefill_sidebar:
            max_tokens_to_use = 1 if prefill_isolation_mode else max_tokens_prefill_input

            if prefill_isolation_mode:
                st.info("Prefill isolation test activated (max_tokens=1).")

            custom_tokens = [int(t.strip()) for t in prefill_tokens_custom.split(',')
                            if t.strip() and t.strip().isdigit()]
            token_levels = sorted(set(prefill_tokens_select + custom_tokens))

            if not token_levels:
                st.error("Please enter at least one token count.")
            else:
                from config.session_state import set_test_running
                st.session_state._pending_test = {
                    'test_type': test_type,
                    'test_func': _get_benchmark_runner().run_prefill_test,
                    'runner_class': _get_benchmark_runner(),
                    'args': (token_levels, requests_per_level, max_tokens_to_use)
                }
                set_test_running()
                st.rerun()

    # Segmented Context Test
    elif test_type == "Segmented Context Test":
        st.header("📊 Segmented Context Test (Prefix Caching)")
        st.info("""
        **Test Description**: Simulates real-world scenarios where users send cumulative long text in segments (e.g., document Q&A, multi-turn conversations).

        - **Cumulative Mode**: All segments share the same prefix, testing Prefix Caching optimization
        - **Independent Mode**: Each segment has independent content, serving as a no-cache control group
        """)

        with st.sidebar.expander("📊 Parameter Settings", expanded=True):
            # Segment level selection
            segment_presets = {
                "Progressive": [2000, 8000, 20000, 40000, 60000],
                "Rapid Growth": [4000, 16000, 32000, 64000],
                "Fine-grained": [1000, 2000, 4000, 8000, 16000, 32000, 64000]
            }

            selected_preset = st.selectbox(
                "Select Preset Strategy",
                ["Custom"] + list(segment_presets.keys()),
                key="segment_preset"
            )

            if selected_preset == "Custom":
                segment_levels_input = st.text_input(
                    "Segment Levels (comma-separated, unit: tokens)",
                    "2000, 8000, 20000, 40000, 60000",
                    help="Example: 2000, 8000, 20000, 40000, 60000",
                    key="segment_levels_input"
                )
                segment_levels = sorted(set(
                    int(s.strip()) for s in segment_levels_input.split(',')
                    if s.strip() and s.strip().isdigit()
                ))
            else:
                segment_levels = segment_presets[selected_preset]
                st.info(f"Segment levels: {segment_levels}")

            requests_per_segment = st.number_input(
                "Requests Per Segment",
                min_value=1,
                value=1,
                key="segment_requests_per_segment"
            )

            st.markdown("---")

            # Test mode
            cumulative_mode = st.checkbox(
                "Cumulative Mode (Prefix Caching)",
                value=True,
                help="When checked, all segments share the same prefix to test Prefix Caching effect",
                key="segment_cumulative_mode"
            )

            # Test rounds
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                total_rounds = st.number_input(
                    "Total Test Rounds",
                    min_value=1,
                    value=1,
                    key="segment_total_rounds"
                )
            with col_r2:
                per_round_unique = st.checkbox(
                    "Unique Prompt Per Round",
                    value=False,
                    key="segment_per_round_unique"
                )

            # Concurrency settings
            segment_concurrency = st.number_input(
                "Concurrency",
                min_value=1,
                value=1,
                key="segment_concurrency"
            )

            st.markdown("---")

            # Max tokens
            max_tokens_segment = st.number_input(
                "Max Output Tokens",
                min_value=1,
                value=512,
                key="segment_max_tokens"
            )

            start_btn_segment_sidebar = st.button(
                "🚀 Start Segmented Test (S)",
                key="start_segment_sidebar_btn",
                type="primary",
                disabled=_get_button_disabled_state()
            )

        start_btn_segment_main = st.button(
            "🚀 Start Segmented Context Test (M)",
            key="start_segment_main_btn",
            disabled=_get_button_disabled_state()
        )

        if start_btn_segment_main or start_btn_segment_sidebar:
            if not segment_levels:
                st.error("Please enter at least one segment level.")
            else:
                mode_str = "Cumulative Mode" if cumulative_mode else "Independent Mode"
                st.info(f"Starting Segmented Context Test ({mode_str})")
                st.info(f"Segment levels: {segment_levels}")
                st.info(f"Requests per segment: {requests_per_segment}, Test rounds: {total_rounds}, Concurrency: {segment_concurrency}")

                from config.session_state import set_test_running
                st.session_state._pending_test = {
                    'test_type': test_type,
                    'test_func': _get_benchmark_runner().run_segmented_prefill_test,
                    'runner_class': _get_benchmark_runner(),
                    'args': (segment_levels, requests_per_segment, max_tokens_segment,
                            cumulative_mode, total_rounds, per_round_unique, segment_concurrency)
                }
                set_test_running()
                st.rerun()

    # Long Context Test
    elif test_type == "Long Context Test":
        st.header("📏 Long Context Test")
        with st.sidebar.expander("📊 Parameter Settings", expanded=True):
            context_lengths_select = st.multiselect(
                "Select Context Lengths",
                [1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 128000, 256000],
                default=[1024, 4096, 16384, 65536, 98304, 128000],
                key="long_context_lengths_select"
            )
            context_lengths_custom = st.text_input(
                "Custom Context Lengths (comma-separated)",
                key="long_context_lengths_custom"
            )
            context_rounds = st.number_input(
                "Rounds Per Level",
                min_value=1,
                value=1,
                key="long_context_rounds"
            )
            max_tokens_long = st.number_input(
                "Max Output Tokens",
                min_value=1,
                value=512,
                key="long_max_tokens"
            )
            st.markdown("---")
            start_btn_long_sidebar = st.button(
                "🚀 Start Long Context Test (S)",
                key="start_long_sidebar_btn",
                type="primary",
                disabled=_get_button_disabled_state()
            )

        start_btn_long_main = st.button(
            "🚀 Start Long Context Test (M)",
            key="start_long_main_btn",
            disabled=_get_button_disabled_state()
        )
        if start_btn_long_main or start_btn_long_sidebar:
            custom_lengths = [int(l.strip()) for l in context_lengths_custom.split(',')
                             if l.strip() and l.strip().isdigit()]
            context_lengths = sorted(set(context_lengths_select + custom_lengths))

            if not context_lengths:
                st.error("Please enter at least one context length.")
            else:
                from config.session_state import set_test_running
                st.session_state._pending_test = {
                    'test_type': test_type,
                    'test_func': _get_benchmark_runner().run_long_context_test,
                    'runner_class': _get_benchmark_runner(),
                    'args': (context_lengths, context_rounds, max_tokens_long)
                }
                set_test_running()
                st.rerun()

    # Concurrency-Context Matrix Test
    elif test_type == "Concurrency-Context Matrix Test":
        st.header("🔬 Concurrency-Context Matrix Test")
        with st.sidebar.expander("📊 Parameter Settings", expanded=True):
            matrix_concurrencies_select = st.multiselect(
                "Select Concurrency Levels",
                [1, 2, 4, 8, 12, 16, 64, 128, 256, 512, 1024],
                default=[1, 2],
                key="matrix_concurrencies_select"
            )
            matrix_concurrencies_custom = st.text_input(
                "Custom Concurrency Levels (comma-separated)",
                key="matrix_con_custom"
            )

            matrix_context_select = st.multiselect(
                "Select Context Lengths",
                [1024, 2048, 4096, 8192, 16384, 32768, 65536, 256000],
                default=[1024, 4096, 16384, 65536],
                key="matrix_context_select"
            )
            matrix_context_custom = st.text_input(
                "Custom Context Lengths (comma-separated)",
                key="matrix_ctx_custom"
            )

            matrix_rounds = st.number_input(
                "Rounds Per Combination",
                min_value=1,
                value=1,
                key="matrix_rounds"
            )
            matrix_max_tokens = st.number_input(
                "Max Output Tokens",
                min_value=1,
                value=256,
                key="matrix_max_tokens"
            )

            enable_warmup = st.checkbox(
                "Enable Warmup",
                value=True,
                help="Run a single warmup request before each test group to trigger Prefix Caching, ensuring consistent Prefill time for concurrent requests and synchronized Decode phase.",
                key="matrix_enable_warmup"
            )
            st.markdown("---")
            start_btn_matrix_sidebar = st.button(
                "🚀 Start Matrix Test (S)",
                key="start_matrix_sidebar_btn",
                type="primary",
                disabled=_get_button_disabled_state()
            )

        start_btn_matrix_main = st.button(
            "🚀 Start Concurrency-Context Matrix Test (M)",
            key="start_matrix_main_btn",
            disabled=_get_button_disabled_state()
        )
        if start_btn_matrix_main or start_btn_matrix_sidebar:
            try:
                custom_cons = [int(c.strip()) for c in matrix_concurrencies_custom.split(',')
                              if c.strip() and c.strip().isdigit()]
                selected_concurrencies = sorted(set(matrix_concurrencies_select + custom_cons))

                custom_ctxs = [int(l.strip()) for l in matrix_context_custom.split(',')
                              if l.strip() and l.strip().isdigit()]
                selected_contexts = sorted(set(matrix_context_select + custom_ctxs))

                if not selected_concurrencies or not selected_contexts:
                    st.error("Please select at least one concurrency level and one context length.")
                else:
                    from config.session_state import set_test_running
                    st.session_state._pending_test = {
                        'test_type': test_type,
                        'test_func': _get_benchmark_runner().run_throughput_matrix_test,
                        'runner_class': _get_benchmark_runner(),
                        'args': (selected_concurrencies, selected_contexts,
                                matrix_rounds, matrix_max_tokens, enable_warmup)
                    }
                    set_test_running()
                    st.rerun()
            except ValueError:
                st.error("Invalid custom value format. Please use comma-separated numbers.")

    # Custom Text Test
    elif test_type == "Custom Text Test":
        st.header("📄 Custom Text Test")
        st.info("Upload a TXT file as prompt context and add custom instructions. The system will automatically inject random noise to avoid caching.")

        uploaded_file = st.file_uploader(
            "Upload TXT File (Context)",
            type=["txt"],
            key="custom_uploaded_file"
        )

        suffix_instruction = st.text_area(
            "Custom Suffix Instruction",
            "Please summarize the above content.",
            height=100,
            key="custom_suffix_instruction"
        )

        custom_concurrencies_select = st.multiselect(
            "Select Concurrency Levels",
            [1, 2, 4, 6, 8, 16],
            default=[1, 2, 4],
            key="custom_concurrencies_select"
        )
        custom_rounds = st.number_input(
            "Rounds Per Level",
            min_value=1,
            value=1,
            key="custom_rounds"
        )
        custom_max_tokens = st.number_input(
            "Max Output Tokens",
            min_value=1,
            value=512,
            key="custom_max_tokens"
        )

        avoid_cache = st.checkbox(
            "Avoid Caching (inject random noise)",
            value=True,
            disabled=True,
            help="Forced on to meet test requirements",
            key="custom_avoid_cache"
        )

        st.sidebar.markdown("---")
        start_btn_custom_sidebar = st.sidebar.button(
            "🚀 Start Custom Test (S)",
            key="start_custom_sidebar_btn",
            type="primary",
            disabled=_get_button_disabled_state()
        )

        start_btn_custom_main = st.button(
            "🚀 Start Custom Text Test (M)",
            key="start_custom_main_btn",
            disabled=_get_button_disabled_state()
        )
        if start_btn_custom_main or start_btn_custom_sidebar:
            if not uploaded_file:
                st.error("Please upload a TXT file.")
            elif not custom_concurrencies_select:
                st.error("Please select at least one concurrency level.")
            else:
                try:
                    base_prompt = uploaded_file.read().decode("utf-8")
                    from config.session_state import set_test_running
                    st.session_state._pending_test = {
                        'test_type': test_type,
                        'test_func': _get_benchmark_runner().run_custom_text_test,
                        'runner_class': _get_benchmark_runner(),
                        'args': (custom_concurrencies_select, custom_rounds,
                                base_prompt, suffix_instruction, custom_max_tokens, avoid_cache)
                    }
                    set_test_running()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to read file: {e}")

    # All Tests
    elif test_type == "All Tests":
        st.header("🎯 All Tests")
        st.warning("This option will run all test types sequentially, which may take a long time.")

        st.subheader("Concurrency Test Configuration")
        all_con = st.text_input(
            "Concurrency Levels (comma-separated)",
            "1,2",
            key="all_con_levels"
        )
        all_con_rounds = st.number_input(
            "Concurrency Test Rounds",
            min_value=1,
            value=1,
            key="all_con_rounds"
        )
        all_con_input_tokens = st.number_input(
            "Input Token Length (Concurrency Phase)",
            min_value=1,
            value=64,
            step=1,
            key="all_con_input_tokens"
        )
        all_con_tokens = st.number_input(
            "Max Output Tokens (Concurrency)",
            min_value=1,
            value=512,
            key="all_con_max_tokens"
        )

        st.subheader("Prefill Test Configuration")
        all_prefill = st.text_input(
            "Token Counts (comma-separated)",
            "20000,40000",
            key="all_prefill_tokens"
        )
        all_prefill_req = st.number_input(
            "Requests Per Level",
            min_value=1,
            value=1,
            key="all_prefill_requests"
        )
        all_prefill_tokens = st.number_input(
            "Max Output Tokens (Prefill)",
            min_value=1,
            value=1,
            key="all_prefill_max_tokens"
        )

        st.subheader("Long Context Test Configuration")
        all_context = st.text_input(
            "Context Lengths (comma-separated)",
            "1024,4096,16384,65536,128000",
            key="all_context_lengths"
        )
        all_context_rounds = st.number_input(
            "Rounds Per Level (Long Context)",
            min_value=1,
            value=1,
            key="all_context_rounds"
        )
        all_context_tokens = st.number_input(
            "Max Output Tokens (Long Context)",
            min_value=1,
            value=512,
            key="all_context_max_tokens"
        )

        st.sidebar.markdown("---")
        start_btn_all_sidebar = st.sidebar.button(
            "🚀 Start All Tests (S)",
            key="start_all_sidebar_btn",
            type="primary",
            disabled=_get_button_disabled_state()
        )

        start_btn_all_main = st.button(
            "🚀 Start All Tests (M)",
            key="start_all_main_btn",
            disabled=_get_button_disabled_state()
        )
        if start_btn_all_main or start_btn_all_sidebar:
            try:
                token_levels_list = [int(t.strip()) for t in all_prefill.split(',') if t.strip()]
                context_lengths_list = [int(l.strip()) for l in all_context.split(',') if l.strip()]

                from config.session_state import set_test_running
                st.session_state._pending_test = {
                    'test_type': test_type,
                    'test_func': _get_benchmark_runner().run_all_tests,
                    'runner_class': _get_benchmark_runner(),
                    'args': (all_con, all_con_rounds, all_con_input_tokens, all_con_tokens,
                            token_levels_list, all_prefill_req, all_prefill_tokens,
                            context_lengths_list, all_context_rounds, all_context_tokens)
                }
                set_test_running()
                st.rerun()
            except ValueError as e:
                st.error(f"Configuration error: {e}")

    # Stability Test
    elif test_type == "Stability Test":
        st.header("⏱️ Stability Test (Time-based Stability)")
        with st.sidebar.expander("📊 Parameter Settings", expanded=True):
            st.info("This mode will continuously run tests for the specified duration.")

            stability_concurrency = st.number_input(
                "Concurrency",
                min_value=1,
                value=1,
                help="Number of concurrent requests",
                key="stability_concurrency"
            )

            stability_duration = st.number_input(
                "Duration (seconds)",
                min_value=10,
                value=60,
                step=10,
                help="Duration to run the test",
                key="stability_duration"
            )

            stability_max_tokens = st.number_input(
                "Max Output Tokens",
                min_value=1,
                value=512,
                key="stability_max_tokens"
            )
            stability_input_tokens = st.number_input(
                "Input Token Length",
                min_value=1,
                value=64,
                step=1,
                help="Strictly calibrate input prompt token length",
                key="stability_input_tokens"
            )

            st.markdown("---")
            start_btn_stability_sidebar = st.button(
                "🚀 Start Stability Test (S)",
                key="start_stability_sidebar_btn",
                type="primary",
                disabled=_get_button_disabled_state()
            )

        start_btn_stability_main = st.button(
            "🚀 Start Stability Test (M)",
            key="start_stability_main_btn",
            disabled=_get_button_disabled_state()
        )

        if start_btn_stability_main or start_btn_stability_sidebar:
            from config.session_state import set_test_running
            st.session_state._pending_test = {
                'test_type': test_type,
                'test_func': _get_benchmark_runner().run_stability_test,
                'runner_class': _get_benchmark_runner(),
                'args': (stability_concurrency, stability_duration, stability_max_tokens, stability_input_tokens)
            }
            set_test_running()
            st.rerun()

    return False
