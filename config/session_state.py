"""
Streamlit Session State Management Module

Centralized management of all Streamlit session_state variable initialization and access.
"""

import time
import pandas as pd
import streamlit as st


def init_session_state():
    """
    Initialize all session_state variables

    Should be called once at application startup to ensure all necessary state variables are initialized.
    """
    # Test control state
    if 'test_running' not in st.session_state:
        st.session_state.test_running = False
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False
    if 'pause_requested' not in st.session_state:
        st.session_state.pause_requested = False
    if 'test_paused' not in st.session_state:
        st.session_state.test_paused = False
    if 'test_status' not in st.session_state:
        st.session_state.test_status = "Idle"  # Idle | Running | Paused | Cancelled | Completed

    # Resume test related state
    if 'is_resuming' not in st.session_state:
        st.session_state.is_resuming = False
    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = None
    if 'current_test_id' not in st.session_state:
        st.session_state.current_test_id = None

    # Result data state
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if 'report' not in st.session_state:
        st.session_state.report = ""
    if 'logger' not in st.session_state:
        st.session_state.logger = None

    # Test configuration state
    if 'test_duration' not in st.session_state:
        st.session_state.test_duration = 0.0
    if 'test_config' not in st.session_state:
        st.session_state.test_config = {}
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = None  # None means no seed (random behavior)

    # Latency calibration state
    if 'latency_offset' not in st.session_state:
        st.session_state.latency_offset = 0.0  # Default: No calibration

    # Data file state
    if 'current_csv_file' not in st.session_state:
        st.session_state.current_csv_file = ""
    if 'current_log_file' not in st.session_state:
        st.session_state.current_log_file = ""

    # System info state
    if 'system_info' not in st.session_state:
        st.session_state.system_info = {}

    # Model related state
    if 'fetched_models' not in st.session_state:
        st.session_state.fetched_models = []
    if 'custom_sys_info' not in st.session_state:
        st.session_state.custom_sys_info = {}

    # Quality test state
    if 'quality_results' not in st.session_state:
        st.session_state.quality_results = {}
    if 'quality_test_running' not in st.session_state:
        st.session_state.quality_test_running = False

    # A/B comparison state
    if 'ab_result' not in st.session_state:
        st.session_state.ab_result = {}
    if 'ab_comparison_running' not in st.session_state:
        st.session_state.ab_comparison_running = False

    # Test output state
    if 'test_outputs' not in st.session_state:
        st.session_state.test_outputs = []


def get_state(key, default=None):
    """
    Get session_state value

    Args:
        key: State key name
        default: Default value (returned when key doesn't exist)

    Returns:
        State value, or default
    """
    return st.session_state.get(key, default)


def set_state(key, value):
    """
    Set session_state value

    Args:
        key: State key name
        value: Value to set
    """
    st.session_state[key] = value


def reset_test_state():
    """Reset test-related state"""
    st.session_state.test_running = False
    st.session_state.stop_requested = False
    st.session_state.pause_requested = False
    st.session_state.test_paused = False
    st.session_state.test_status = "Idle"
    st.session_state.test_duration = 0.0


def reset_results():
    """Reset test results"""
    st.session_state.results_df = pd.DataFrame()
    st.session_state.report = ""
    st.session_state.test_outputs = []


def is_test_running():
    """Check if test is running"""
    return st.session_state.get('test_running', False)


def is_stop_requested():
    """Check if stop is requested"""
    return st.session_state.get('stop_requested', False)


def is_pause_requested():
    """Check if pause is requested"""
    return st.session_state.get('pause_requested', False)


def is_test_paused():
    """Check if test is paused"""
    return st.session_state.get('test_paused', False)


def request_stop():
    """Request to stop test"""
    st.session_state.stop_requested = True


def request_pause():
    """Request to pause test"""
    st.session_state.pause_requested = True


def set_test_running():
    """Set test to running state"""
    st.session_state.test_running = True
    st.session_state.test_paused = False
    st.session_state.test_status = "Running"
    st.session_state.stop_requested = False
    st.session_state.pause_requested = False
    # 强制刷新时间戳
    st.session_state._test_running_timestamp = time.time()
    # 同时重置全局停止标志
    try:
        from core.providers.openai import set_stop_requested
        set_stop_requested(False)
    except ImportError:
        pass


def set_test_paused():
    """Set test to paused state"""
    st.session_state.test_running = False
    st.session_state.test_paused = True
    st.session_state.test_status = "Paused"
    st.session_state.pause_requested = False


def set_test_cancelled():
    """Set test to cancelled state"""
    st.session_state.test_running = False
    st.session_state.test_paused = False
    st.session_state.test_status = "Cancelled"
    st.session_state.stop_requested = False
    st.session_state.pause_requested = False
    # 关键：强制刷新 session_state 确保状态立即生效
    st.session_state._state['_test_running_timestamp'] = time.time()
    # 同时重置全局停止标志
    try:
        from core.providers.openai import set_stop_requested
        set_stop_requested(False)
    except ImportError:
        pass


def set_test_completed():
    """Set test to completed state"""
    st.session_state.test_running = False
    st.session_state.test_paused = False
    st.session_state.test_status = "Completed"
    st.session_state.stop_requested = False
    st.session_state.pause_requested = False
    # 强制刷新时间戳
    st.session_state._test_running_timestamp = time.time()


def clear_control_flags():
    """Clear all control flags"""
    st.session_state.stop_requested = False
    st.session_state.pause_requested = False
