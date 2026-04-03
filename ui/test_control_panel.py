"""
Unified Test Control Panel Module

Provides centralized test control interface:
- Unified test start button
- Test progress display
- Test status monitoring
- Test control operations (start/stop/resume)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from config.session_state import init_session_state


# ============================================================================
# Test status definitions
# ============================================================================

class TestStatus:
    """Test status constants"""
    IDLE = "Idle"
    RUNNING = "Running"
    PAUSED = "Paused"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


# ============================================================================
# Test Configuration Class
# ============================================================================

class TestConfig:
    """Test configuration data class"""

    def __init__(
        self,
        test_type: str,
        api_base_url: str,
        model_id: str,
        api_key: str,
        concurrency: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.0,
        thinking_enabled: bool = False,
        thinking_budget: int = 0,
        reasoning_effort: str = "medium",
        context_length: int = 4096,
        **kwargs
    ):
        self.test_type = test_type
        self.api_base_url = api_base_url
        self.model_id = model_id
        self.api_key = api_key
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort
        self.context_length = context_length
        self.extra_params = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_type": self.test_type,
            "api_base_url": self.api_base_url,
            "model_id": self.model_id,
            "api_key": self.api_key,
            "concurrency": self.concurrency,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking_enabled": self.thinking_enabled,
            "thinking_budget": self.thinking_budget,
            "reasoning_effort": self.reasoning_effort,
            "context_length": self.context_length,
            **self.extra_params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestConfig":
        """Create from dictionary"""
        extra_params = {k: v for k, v in data.items() if k not in [
            "test_type", "api_base_url", "model_id", "api_key",
            "concurrency", "max_tokens", "temperature",
            "thinking_enabled", "thinking_budget", "reasoning_effort",
            "context_length"
        ]}
        return cls(**{**data, **extra_params})

    @classmethod
    def from_session_state(cls, test_type: str) -> "TestConfig":
        """Create configuration from session_state"""
        return cls(
            test_type=test_type,
            api_base_url=st.session_state.get('current_api_base', ''),
            model_id=st.session_state.get('current_model_id', ''),
            api_key=st.session_state.get('current_api_key', ''),
            concurrency=st.session_state.get('current_concurrency', 1),
            max_tokens=st.session_state.get('current_max_tokens', 512),
            temperature=st.session_state.get('current_temperature', 0.0),
            thinking_enabled=st.session_state.get('thinking_enabled', False),
            thinking_budget=st.session_state.get('thinking_budget', 0),
            reasoning_effort=st.session_state.get('reasoning_effort', 'medium'),
            context_length=st.session_state.get('current_context_length', 4096)
        )


# ============================================================================
# Test Progress Class
# ============================================================================

class TestProgress:
    """Test progress tracking class"""

    def __init__(
        self,
        test_id: str,
        test_type: str,
        total_samples: int = 0,
        completed_samples: int = 0,
        failed_samples: int = 0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        status: str = TestStatus.IDLE
    ):
        self.test_id = test_id
        self.test_type = test_type
        self.total_samples = total_samples
        self.completed_samples = completed_samples
        self.failed_samples = failed_samples
        self.start_time = start_time
        self.end_time = end_time
        self.status = status
        self.error_message: Optional[str] = None
        self.results: List[Dict[str, Any]] = []

    @property
    def progress_percentage(self) -> float:
        """Get progress percentage"""
        if self.total_samples == 0:
            return 0.0
        return (self.completed_samples / self.total_samples) * 100

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time (s)"""
        if not self.start_time:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining time (s)"""
        if self.completed_samples == 0 or self.total_samples == 0:
            return None
        rate = self.elapsed_time / self.completed_samples
        remaining = self.total_samples - self.completed_samples
        return rate * remaining

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "test_type": self.test_type,
            "total_samples": self.total_samples,
            "completed_samples": self.completed_samples,
            "failed_samples": self.failed_samples,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "error_message": self.error_message,
            "results_count": len(self.results)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestProgress":
        """Create from dictionary"""
        progress = cls(
            test_id=data["test_id"],
            test_type=data["test_type"],
            total_samples=data.get("total_samples", 0),
            completed_samples=data.get("completed_samples", 0),
            failed_samples=data.get("failed_samples", 0),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            status=data.get("status", TestStatus.IDLE)
        )
        progress.error_message = data.get("error_message")
        return progress


# ============================================================================
# Progress Persistence Manager
# ============================================================================

class ProgressManager:
    """Test progress persistence manager"""

    def __init__(self, progress_dir: str = "test_progress"):
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(exist_ok=True)

    def save_progress(self, progress: TestProgress) -> bool:
        """Save test progress to file"""
        try:
            progress_file = self.progress_dir / f"{progress.test_id}.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save progress: {e}")
            return False

    def load_progress(self, test_id: str) -> Optional[TestProgress]:
        """Load test progress from file"""
        try:
            progress_file = self.progress_dir / f"{test_id}.json"
            if not progress_file.exists():
                return None
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TestProgress.from_dict(data)
        except Exception as e:
            st.error(f"Failed to load progress: {e}")
            return None

    def list_saved_progress(self) -> List[Dict[str, Any]]:
        """List all saved progress"""
        progress_list = []
        for progress_file in self.progress_dir.glob("*.json"):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                progress_list.append({
                    "test_id": data.get("test_id", progress_file.stem),
                    "test_type": data.get("test_type", "Unknown"),
                    "status": data.get("status", TestStatus.IDLE),
                    "progress": f"{data.get('completed_samples', 0)}/{data.get('total_samples', 0)}",
                    "file_time": datetime.fromtimestamp(progress_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception:
                continue
        return sorted(progress_list, key=lambda x: x["file_time"], reverse=True)

    def delete_progress(self, test_id: str) -> bool:
        """Delete saved progress"""
        try:
            progress_file = self.progress_dir / f"{test_id}.json"
            if progress_file.exists():
                progress_file.unlink()
            return True
        except Exception as e:
            st.error(f"Failed to delete progress: {e}")
            return False

    def clear_old_progress(self, days: int = 7) -> int:
        """Clear old progress files"""
        import time as time_module
        cutoff_time = time_module.time() - (days * 86400)
        count = 0
        for progress_file in self.progress_dir.glob("*.json"):
            if progress_file.stat().st_mtime < cutoff_time:
                try:
                    progress_file.unlink()
                    count += 1
                except Exception:
                    pass
        return count


# Global progress manager instance
progress_manager = ProgressManager()


# ============================================================================
# Unified Test Control Panel Components
# ============================================================================

def render_test_control_panel():
    """
    Render unified test control panel

    简化版控制面板，只提供：
    - Stop 按钮
    - Pause/Resume 按钮
    - 状态显示
    """
    from config.session_state import (
        is_test_running, is_test_paused, request_pause, request_stop
    )

    # Import abort functions
    try:
        from core.providers.openai import set_stop_requested, set_pause_requested as set_global_pause, is_pause_requested
    except ImportError:
        def set_stop_requested(value): pass
        def set_global_pause(value): pass
        def is_pause_requested(): return False

    try:
        from core.providers.gemini import abort_all_clients as abort_gemini_clients
    except ImportError:
        def abort_gemini_clients(): pass

    st.markdown("---")

    # Get current status - 直接使用 test_running 标志
    test_running = st.session_state.get('test_running', False)
    test_paused = st.session_state.get('test_paused', False)

    # 创建两列布局
    col_status, col_buttons = st.columns([1, 2])

    with col_status:
        st.subheader("📊 Status")

        # Current status display
        current_status = st.session_state.get("test_status", TestStatus.IDLE)

        # Status color mapping
        status_colors = {
            TestStatus.IDLE: "🔵",
            TestStatus.RUNNING: "🟢",
            TestStatus.PAUSED: "🟡",
            TestStatus.COMPLETED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.CANCELLED: "⏹️"
        }

        st.metric(
            "Status",
            f"{status_colors.get(current_status, '⚪')} {current_status}"
        )

        # Progress display(ifhas)
        if "current_progress" in st.session_state:
            progress = st.session_state.current_progress

            st.markdown("**Progress:**")
            progress_bar = st.progress(0)

            if progress.total_samples > 0:
                pct = progress.progress_percentage / 100
                progress_bar.progress(pct)

                st.caption(f"{progress.completed_samples}/{progress.total_samples} "
                          f"({progress.progress_percentage:.1f}%)")

            # Time information
            if progress.start_time:
                elapsed = progress.elapsed_time
                st.caption(f"⏱️ Elapsed: {format_time(elapsed)}")

                remaining = progress.estimated_remaining_time
                if remaining:
                    st.caption(f"⏳ Est. remaining: {format_time(remaining)}")

    with col_buttons:
        st.subheader("🎮 Test Control")

        # Control buttons - 只有 Pause, Resume, Stop
        col_pause, col_continue, col_stop = st.columns(3)

        with col_pause:
            pause_button = st.button(
                "⏸️ Pause",
                disabled=not test_running,
                key="control_pause_btn",
                use_container_width=True
            )
            if pause_button:
                # 1. 设置暂停标志（benchmark_runner 会检测这个标志）
                request_pause()
                # 同时设置全局暂停标志（更可靠）
                set_global_pause(True)

                # 2. 停止正在进行的请求（和 Stop 一样）
                set_stop_requested(True)
                abort_gemini_clients()

                # 3. 保存当前结果和测试配置（用于 Resume）
                if '_current_runner_instance' in st.session_state:
                    runner = st.session_state._current_runner_instance
                    if runner:
                        # 保存结果
                        if hasattr(runner, 'results_list') and runner.results_list:
                            import pandas as pd
                            from utils.helpers import reorder_dataframe_columns
                            df = pd.DataFrame(runner.results_list)
                            if not df.empty:
                                df = reorder_dataframe_columns(df)
                                st.session_state.results_df = df

                        # 保存完整的测试配置（用于 Resume）
                        # 从 runner 实例获取配置
                        test_config = {}

                        # 通用配置
                        if hasattr(runner, '_current_max_tokens'):
                            test_config['max_tokens'] = runner._current_max_tokens
                        if hasattr(runner, 'total_requests'):
                            test_config['total_requests'] = runner.total_requests

                        # 从 _current_test_config 获取更详细的配置
                        current_config = st.session_state.get('current_test_config', {})
                        if current_config:
                            test_config.update(current_config)

                        st.session_state.current_test_config = test_config

                        # 保存 resume_data（已完成的请求）
                        completed_count = len(runner.results_list) if hasattr(runner, 'results_list') else 0
                        st.session_state.resume_data = {
                            'completed_results': runner.results_list.copy() if hasattr(runner, 'results_list') else [],
                            'current_index': completed_count,  # 使用已完成数量作为跳过索引
                            'total_samples': getattr(runner, 'total_requests', 0)
                        }
                        st.toast(f"Paused: {completed_count} requests completed, will resume from index {completed_count}", icon="⏸️")

                # 4. 更新状态为 Paused
                st.session_state.test_running = False
                st.session_state.test_paused = True
                st.session_state.test_status = "Paused"

                st.toast("Test paused.", icon="⏸️")
                st.rerun()

        with col_continue:
            continue_button = st.button(
                "▶️ Resume",
                disabled=not test_paused,
                key="control_continue_btn",
                use_container_width=True
            )
            if continue_button:
                # Resume: 从暂停处继续测试
                from config.session_state import set_test_running

                # 重置停止和暂停标志
                set_stop_requested(False)
                set_global_pause(False)
                st.session_state.stop_requested = False
                st.session_state.pause_requested = False

                # 设置运行状态
                set_test_running()

                # 设置恢复标志和恢复数据
                st.session_state.is_resuming = True
                # resume_data 应该在 Pause 时已经保存了
                # 确保它存在
                if 'resume_data' not in st.session_state:
                    st.warning("No saved progress found. Cannot resume.")
                    return

                # 从 session_state 获取保存的测试配置
                test_type = st.session_state.get('current_test_type', '')
                saved_config = st.session_state.get('current_test_config', {})

                if test_type and saved_config:
                    # 构造 _pending_test 来重新触发测试
                    # 注意：benchmark_runner 会检测 is_resuming 标志并跳过已完成的请求
                    if test_type == "Concurrency Test":
                        from core.benchmark_runner import BenchmarkRunner
                        st.session_state._pending_test = {
                            'test_type': test_type,
                            'test_func': BenchmarkRunner.run_concurrency_test,
                            'runner_class': BenchmarkRunner,
                            'args': (
                                saved_config.get('concurrency_levels', [1, 2]),
                                saved_config.get('rounds', 1),
                                saved_config.get('max_tokens', 512),
                                saved_config.get('input_tokens', 64)
                            )
                        }
                    elif test_type == "Prefill Stress Test":
                        from core.benchmark_runner import BenchmarkRunner
                        st.session_state._pending_test = {
                            'test_type': test_type,
                            'test_func': BenchmarkRunner.run_prefill_test,
                            'runner_class': BenchmarkRunner,
                            'args': (
                                saved_config.get('token_levels', [20000, 40000]),
                                saved_config.get('requests_per_level', 1),
                                saved_config.get('max_tokens', 1)
                            )
                        }
                    elif test_type == "Long Context Test":
                        from core.benchmark_runner import BenchmarkRunner
                        st.session_state._pending_test = {
                            'test_type': test_type,
                            'test_func': BenchmarkRunner.run_long_context_test,
                            'runner_class': BenchmarkRunner,
                            'args': (
                                saved_config.get('context_lengths', [1024, 4096]),
                                saved_config.get('rounds', 1),
                                saved_config.get('max_tokens', 512)
                            )
                        }
                    elif test_type == "Segmented Context Test":
                        from core.benchmark_runner import BenchmarkRunner
                        st.session_state._pending_test = {
                            'test_type': test_type,
                            'test_func': BenchmarkRunner.run_segmented_prefill_test,
                            'runner_class': BenchmarkRunner,
                            'args': (
                                saved_config.get('segment_levels', [2000, 8000]),
                                saved_config.get('requests_per_segment', 1),
                                saved_config.get('max_tokens', 512),
                                saved_config.get('cumulative_mode', True),
                                saved_config.get('total_rounds', 1),
                                saved_config.get('per_round_unique', False),
                                saved_config.get('concurrency', 1)
                            )
                        }
                    elif test_type == "Concurrency-Context Matrix Test":
                        from core.benchmark_runner import BenchmarkRunner
                        st.session_state._pending_test = {
                            'test_type': test_type,
                            'test_func': BenchmarkRunner.run_throughput_matrix_test,
                            'runner_class': BenchmarkRunner,
                            'args': (
                                saved_config.get('concurrency_levels', [1, 2]),
                                saved_config.get('context_lengths', [1024, 4096]),
                                saved_config.get('rounds', 1),
                                saved_config.get('max_tokens', 256),
                                saved_config.get('enable_warmup', True)
                            )
                        }
                    else:
                        st.warning(f"Resume not supported for: {test_type}")

                st.toast("Resuming test...", icon="▶️")
                st.rerun()

        with col_stop:
            stop_button = st.button(
                "⏹️ Stop",
                disabled=not (test_running or test_paused),
                key="control_stop_btn",
                type="secondary",
                use_container_width=True
            )
            if stop_button:
                # 1. 先设置全局停止标志
                request_stop()
                set_stop_requested(True)
                abort_gemini_clients()

                # 2. 尝试从当前 runner 实例获取结果（如果有）
                if '_current_runner_instance' in st.session_state:
                    runner = st.session_state._current_runner_instance
                    if runner and hasattr(runner, 'results_list') and runner.results_list:
                        import pandas as pd
                        from utils.helpers import reorder_dataframe_columns
                        df = pd.DataFrame(runner.results_list)
                        if not df.empty:
                            df = reorder_dataframe_columns(df)
                            st.session_state.results_df = df

                # 3. 立即更新 session_state 状态
                st.session_state.test_running = False
                st.session_state.test_paused = False
                st.session_state.test_status = "Cancelled"
                st.session_state.stop_requested = False
                st.session_state.pause_requested = False

                # 4. 清除待执行的测试和 runner 实例
                if '_pending_test' in st.session_state:
                    del st.session_state._pending_test
                if '_current_runner_instance' in st.session_state:
                    del st.session_state._current_runner_instance

                st.toast("Test stopped.", icon="⏹️")
                st.rerun()

    return {
        "pause": pause_button if test_running else False,
        "continue": continue_button,
        "stop": stop_button
    }


def render_progress_history():
    """Render progress history panel"""
    with st.expander("📜 Test History", expanded=False):
        saved_progress_list = progress_manager.list_saved_progress()

        if not saved_progress_list:
            st.info("No saved test progress")
            return

        for item in saved_progress_list:
            with st.container():
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.write(f"**{item['test_type']}**")

                with col2:
                    st.write(item['status'])

                with col3:
                    st.caption(item['progress'])

                with col4:
                    if st.button("🗑️", key=f"del_{item['test_id']}", help="Delete this progress"):
                        if progress_manager.delete_progress(item['test_id']):
                            st.rerun()

                st.caption(f"Save time: {item['file_time']}")
                st.markdown("---")


def render_resumable_tests():
    """
    Render resumable test panel

    Display all paused or cancelled tests, allowing users to resume
    """
    saved_progress_list = progress_manager.list_saved_progress()

    # Filter resumable tests
    resumable = [p for p in saved_progress_list
                 if p['status'] in [TestStatus.PAUSED, TestStatus.CANCELLED]]

    if not resumable:
        return None

    with st.expander("🔄 Resumable Tests", expanded=True):
        st.caption("The following tests can be resumed:")

        selected_test_id = None

        for prog in resumable:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

            with col1:
                st.write(f"**{prog['test_type']}**")
                st.caption(f"Save: {prog['file_time']}")

            with col2:
                status_icon = "🟡" if prog['status'] == TestStatus.PAUSED else "⏹️"
                st.write(f"{status_icon} {prog['status']}")

            with col3:
                st.write(prog['progress'])

            with col4:
                if st.button("▶️ Restore", key=f"resume_{prog['test_id']}", use_container_width=True):
                    selected_test_id = prog['test_id']

        return selected_test_id


def format_time(s: float) -> str:
    """Format time display"""
    if s < 60:
        return f"{s:.1f}s"
    elif s < 3600:
        min = s / 60
        return f"{min:.1f}min"
    else:
        hr = s / 3600
        return f"{hr:.1f}hr"


# ============================================================================
# Helper Functions
# ============================================================================

def save_current_progress(progress: TestProgress) -> bool:
    """Save current test progress"""
    st.session_state.current_progress = progress
    return progress_manager.save_progress(progress)


def load_saved_progress(test_id: str) -> Optional[TestProgress]:
    """Load saved test progress"""
    progress = progress_manager.load_progress(test_id)
    if progress:
        st.session_state.current_progress = progress
        st.session_state.current_test_id = test_id
    return progress


def clear_current_progress():
    """Clear current progress"""
    if "current_progress" in st.session_state:
        del st.session_state.current_progress
    if "current_test_id" in st.session_state:
        del st.session_state.current_test_id


def get_test_config(test_type: str) -> TestConfig:
    """Get test configuration for specified type"""
    return TestConfig.from_session_state(test_type)


def load_resume_data(test_id: str) -> Optional[Dict[str, Any]]:
    """
    Load resume data

    Args:
        test_id: Test ID

    Returns:
        Dictionary containing resume data, returns None if load failed
    """
    try:
        progress_file = Path("test_progress") / f"{test_id}.json"
        if not progress_file.exists():
            return None

        with open(progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Set resume flag in session_state
        st.session_state.is_resuming = True
        st.session_state.resume_data = data
        st.session_state.current_test_id = test_id

        return data

    except Exception as e:
        st.error(f"Failed to load resume data: {e}")
        return None


def prepare_resume_from_progress(progress_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare resume parameters from progress data

    Args:
        progress_data: Progress data dictionary

    Returns:
        Resume parameters dictionary
    """
    return {
        "test_id": progress_data.get("test_id"),
        "test_type": progress_data.get("test_type"),
        "completed_results": progress_data.get("completed_results", []),
        "pending_prompts": progress_data.get("pending_prompts", []),
        "current_index": progress_data.get("current_index", 0),
        "total_samples": progress_data.get("total_samples", 0),
        "test_config": progress_data.get("test_config", {}),
    }
