"""
Test Execution Module

Provides test execution workflow, including:
- Test state management
- Progress display
- Real-time logging
- System info capture
- Test result storage
"""

import asyncio
import os
import sys
import time
import traceback

import pandas as pd
import streamlit as st

from utils.helpers import reorder_dataframe_columns


class TestExecutor:
    """Test executor class"""

    def __init__(self, config):
        """
        Initialize test executor

        Args:
            config: Configuration dict containing api_base_url, model_id, api_key, etc.
        """
        self.config = config
        self.test_running = False

    def run_test(self, test_function, runner_class, test_type, *args):
        """
        Execute test and manage status

        Args:
            test_function: Test function
            runner_class: Test runner class
            test_type: Test type name
            *args: Parameters passed to test function

        Returns:
            pd.DataFrame: Test results DataFrame
        """
        from config.session_state import set_test_running, set_test_completed, clear_control_flags

        # 强制重置停止标志 - 确保开始新测试时状态干净
        try:
            from core.providers.openai import set_stop_requested
            set_stop_requested(False)
        except ImportError:
            pass

        # 保存当前测试类型和配置（用于 Pause/Resume）
        st.session_state.current_test_type = test_type

        # Check if in resume mode
        is_resuming = st.session_state.get('is_resuming', False)
        resume_data = st.session_state.get('resume_data', None)

        if is_resuming and resume_data:
            st.info(f"🔄 Resuming test from saved progress: {resume_data.get('test_type', 'Unknown')}")
            # Restore completed results
            preloaded_results = resume_data.get('completed_results', [])
        else:
            preloaded_results = []

        # 注意：test_running 状态已经在 test_panels.py 中设置
        # 这里只初始化结果数据
        st.session_state.results_df = pd.DataFrame()
        st.session_state.logger = None
        df = pd.DataFrame()

        # Create UI components
        progress_bar = st.progress(0)
        status_text = st.empty()

        log_container = st.empty()
        with log_container.container():
            with st.expander("📋 Real-time logging (Live Request Log)", expanded=True):
                log_placeholder = st.empty()
                log_placeholder.info("ℹ️ Log window initialized... waiting for test to start")

        result_container = st.empty()
        placeholder = result_container

        output_container = st.empty()
        output_placeholder = output_container.empty()
        output_placeholder.info("📝 Output preview area ready, waiting for first request to complete...")

        try:
            # Create filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            test_type_str = test_type.replace(" ", "_").replace("-", "_")
            model_dir = os.path.join("raw_data", self.config['model_id'])
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            csv_filename = os.path.join(model_dir,
                f"benchmark_results_{self.config['model_id']}_{test_type_str}_{timestamp}.csv")
            st.session_state.current_csv_file = csv_filename

            # Initialize runner
            runner_instance = runner_class(
                placeholder, progress_bar, status_text,
                self.config['api_base_url'],
                self.config['model_id'],
                self.config['tokenizer_option'],
                csv_filename,
                self.config['api_key'],
                log_placeholder,
                self.config['provider'],
                dashboard=None,
                output_placeholder=output_placeholder,
                hf_tokenizer_model_id=self.config.get('hf_tokenizer_model_id'),
                latency_offset=st.session_state.latency_offset,
                random_seed=st.session_state.get('random_seed')
            )

            # 存储 runner 实例到 session_state，以便 Stop 按钮可以访问结果
            st.session_state._current_runner_instance = runner_instance

            # Capture system info
            self._capture_system_info(runner_instance)

            # Calculate total requests
            self._calculate_total_requests(test_function, runner_instance, args)

            # Execute test
            start_time = time.time()
            df = self._execute_test_async(test_function, runner_instance, args)
            end_time = time.time()
            duration_sec = end_time - start_time
            st.session_state.test_duration = duration_sec

            # Capture test configuration
            self._capture_test_config(test_function, test_type, args, timestamp)

            # Cleanup UI
            log_container.empty()
            output_container.empty()
            result_container.empty()

            st.success(f"Test completed! Duration: {duration_sec:.2f}s. Results saved to {csv_filename}")

        except asyncio.CancelledError:
            from config.session_state import set_test_cancelled
            set_test_cancelled()
            st.warning("Test stopped by user.")
            df = pd.DataFrame(runner_instance.results_list) if 'runner_instance' in locals() else pd.DataFrame()
        except Exception as e:
            st.error(f"Error occurred during test execution: {e}")
            st.code(traceback.format_exc())
            # 确保异常情况下也清理状态
            from config.session_state import set_test_completed
            set_test_completed()
        finally:
            # Reorder Result DataFrame columns for UI and download consistency
            if not df.empty:
                df = reorder_dataframe_columns(df)
            st.session_state.results_df = df

            # 检查当前状态，只有在没有被取消的情况下才标记为完成
            test_status = st.session_state.get('test_status', 'Idle')

            # Clear resume flags（在任何情况下都清除）
            was_resuming = st.session_state.get('is_resuming', False)
            st.session_state.is_resuming = False
            st.session_state.resume_data = None

            if test_status == 'Running':
                # 正常完成，设置状态
                set_test_completed()

            # 确保 test_running 标志被清除（无论成功、取消还是异常）
            st.session_state.test_running = False

            # 清除暂停状态（如果是 Resume 完成的情况）
            if was_resuming:
                st.session_state.test_paused = False

            if 'runner_instance' in locals():
                st.session_state.logger = runner_instance.logger
                if hasattr(runner_instance, 'all_outputs'):
                    st.session_state.test_outputs = runner_instance.all_outputs

            # 清除 runner 实例引用
            if '_current_runner_instance' in st.session_state:
                del st.session_state._current_runner_instance

        return df

    def _capture_system_info(self, runner_instance):
        """Capture system information"""
        try:
            sys_info = runner_instance.get_system_info()

            # Merge user custom system info
            custom_sys_info = st.session_state.get('custom_sys_info', {})
            for key in ['processor', 'mainboard', 'memory', 'gpu', 'system', 'engine_name']:
                custom_value = custom_sys_info.get(key, '')
                if custom_value:
                    sys_info[key] = custom_value

            # Fallback values
            if not sys_info.get('model_name') or sys_info.get('model_name') == 'Unknown':
                sys_info['model_name'] = self.config['model_id']
            # engine_name has no fallback — provider is not the engine
            # Engine refers to inference backend (e.g. vLLM, SGLang, TRT-LLM), users can fill in manually via sidebar

            st.session_state.system_info = sys_info
        except Exception:
            st.session_state.system_info = {}

    def _calculate_total_requests(self, test_function, runner_instance, args):
        """Calculate total requests"""
        func_name = test_function.__name__

        if func_name == 'run_concurrency_test':
            selected_concurrencies, rounds_per_level, *_ = args
            runner_instance.total_requests = sum(c * rounds_per_level for c in selected_concurrencies)
        elif func_name == 'run_prefill_test':
            token_levels, requests_per_level, _ = args
            runner_instance.total_requests = len(token_levels) * requests_per_level
        elif func_name == 'run_long_context_test':
            context_lengths, rounds, _ = args
            runner_instance.total_requests = len(context_lengths) * rounds
        elif func_name == 'run_throughput_matrix_test':
            selected_concurrencies, context_lengths, rounds_per_level, *_ = args
            total_reqs = sum(c * len(context_lengths) * rounds_per_level for c in selected_concurrencies)
            runner_instance.total_requests = total_reqs
        elif func_name == 'run_segmented_prefill_test':
            segment_levels, requests_per_segment, _max_tokens, *rest = args
            cumulative_mode = rest[0] if len(rest) > 0 else True
            total_rounds = rest[1] if len(rest) > 1 else 1
            _per_round_unique = rest[2] if len(rest) > 2 else False
            concurrency = rest[3] if len(rest) > 3 else 1
            runner_instance.total_requests = len(segment_levels) * requests_per_segment * total_rounds

    def _execute_test_async(self, test_function, runner_instance, args):
        """Execute test asynchronously with stop signal monitoring"""
        import threading
        from core.providers.openai import is_stop_requested

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        import sniffio
        token = sniffio.current_async_library_cvar.set("asyncio")

        stop_event = threading.Event()
        main_task = None

        def monitor_stop_signal():
            while not stop_event.is_set():
                if is_stop_requested():
                    if main_task and not main_task.done():
                        loop.call_soon_threadsafe(main_task.cancel)
                    break
                stop_event.wait(0.05)

        try:
            main_task = loop.create_task(test_function(runner_instance, *args))
            monitor_thread = threading.Thread(target=monitor_stop_signal, daemon=True)
            monitor_thread.start()
            result = loop.run_until_complete(main_task)
            return result

        except asyncio.CancelledError:
            raise
        finally:
            stop_event.set()
            sniffio.current_async_library_cvar.reset(token)

            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                if pending:
                    try:
                        loop.run_until_complete(
                            asyncio.wait(pending, timeout=0.5, return_when=asyncio.ALL_COMPLETED)
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                loop.close()
            except Exception:
                pass
            asyncio.set_event_loop(None)

    def _capture_test_config(self, test_function, test_type, args, timestamp):
        """Capture test configuration"""
        config_summary = {
            "Test Type": test_type,
            "Model ID": self.config['model_id'],
            "Provider": self.config['provider'],
            "Timestamp": timestamp
        }

        func_name = test_function.__name__

        if func_name == 'run_concurrency_test':
            selected_concurrencies, rounds_per_level, *others = args
            input_target = others[-1] if others else "N/A"
            config_summary["Input Tokens"] = str(input_target)
            config_summary["Concurrency Levels"] = str(selected_concurrencies)
            config_summary["Rounds per Level"] = str(rounds_per_level)
            if len(others) >= 1:
                config_summary["Max Tokens"] = str(others[0])

        elif func_name == 'run_prefill_test':
            token_levels, requests_per_level, _ = args
            config_summary["Token Levels"] = str(token_levels)
            config_summary["Requests per Level"] = str(requests_per_level)

        elif func_name == 'run_long_context_test':
            context_lengths, rounds, _ = args
            config_summary["Context Lengths"] = str(context_lengths)
            config_summary["Rounds"] = str(rounds)

        elif func_name == 'run_throughput_matrix_test':
            selected_concurrencies, context_lengths, rounds_per_level, *_ = args
            config_summary["Concurrency Levels"] = str(selected_concurrencies)
            config_summary["Context Lengths"] = str(context_lengths)
            config_summary["Rounds"] = str(rounds_per_level)

        elif func_name == 'run_segmented_prefill_test':
            segment_levels, requests_per_segment, max_tokens, *rest = args
            cumulative_mode = rest[0] if len(rest) > 0 else True
            total_rounds = rest[1] if len(rest) > 1 else 1
            per_round_unique = rest[2] if len(rest) > 2 else False
            concurrency = rest[3] if len(rest) > 3 else 1
            config_summary["Segment Levels"] = str(segment_levels)
            config_summary["Requests per Segment"] = str(requests_per_segment)
            config_summary["Max Tokens"] = str(max_tokens)
            config_summary["Cumulative Mode"] = "Yes (Prefix Caching)" if cumulative_mode else "No (Independent)"
            config_summary["Total Rounds"] = str(total_rounds)
            config_summary["Per Round Unique"] = str(per_round_unique)
            config_summary["Concurrency"] = str(concurrency)

        st.session_state.test_config = config_summary


# Backward compatibility function interface
def create_test_executor(config):
    """Create test executor instance"""
    return TestExecutor(config)
