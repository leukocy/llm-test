"""
Streamlit Session State Management Module

Centralized management of all Streamlit session_state variable initialization and access.
"""

import os
import time

import pandas as pd
import streamlit as st

DEFAULT_TEST_TYPE = "Concurrency Test"


def _touch_test_running_timestamp():
    """Refresh the test state timestamp using the public session_state API."""
    st.session_state['_test_running_timestamp'] = time.time()

_TEST_TYPE_ALIASES = {
    "concurrency": "Concurrency Test",
    "concurrency_test": "Concurrency Test",
    "prefill": "Prefill Stress Test",
    "prefill_test": "Prefill Stress Test",
    "prefill_stress": "Prefill Stress Test",
    "prefill_stress_test": "Prefill Stress Test",
    "segmented": "Segmented Context Test",
    "segmented_context": "Segmented Context Test",
    "segmented_context_test": "Segmented Context Test",
    "segmented_prefill": "Segmented Context Test",
    "long_context": "Long Context Test",
    "long_context_test": "Long Context Test",
    "matrix": "Concurrency-Context Matrix Test",
    "throughput_matrix": "Concurrency-Context Matrix Test",
    "concurrency_context_matrix": "Concurrency-Context Matrix Test",
    "concurrency_context_matrix_test": "Concurrency-Context Matrix Test",
    "custom": "Custom Text Test",
    "custom_text": "Custom Text Test",
    "custom_text_test": "Custom Text Test",
    "all": "All Tests",
    "all_tests": "All Tests",
    "stability": "Stability Test",
    "stability_test": "Stability Test",
}


def _test_type_key(value):
    """Return a stable comparison key for a test type value."""
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("/", " ")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _match_allowed_test_type(label, allowed_options=None):
    """Prefer the exact option object from the active selectbox options."""
    if not allowed_options:
        return label

    if label in allowed_options:
        return label

    target_key = _test_type_key(label)
    for option in allowed_options:
        option_key = _test_type_key(option)
        if option_key == target_key or option_key.endswith(target_key):
            return option

    label_lower = str(label).lower()
    for option in allowed_options:
        option_lower = str(option).lower()
        if "batch" in label_lower and "batch test" in option_lower:
            return option
        if "quality" in label_lower and "model quality test" in option_lower:
            return option
        if "comparison" in label_lower and "a/b model comparison" in option_lower:
            return option
        if "advanced" in label_lower and "advanced evaluation" in option_lower:
            return option

    return DEFAULT_TEST_TYPE if DEFAULT_TEST_TYPE in allowed_options else allowed_options[0]


def normalize_test_type(value, allowed_options=None):
    """
    Normalize legacy/internal test type names to the UI label used by widgets.
    """
    key = _test_type_key(value)
    label = _TEST_TYPE_ALIASES.get(key)

    if not label:
        text = str(value or "")
        lower_text = text.lower()
        if "batch test" in lower_text:
            label = "Batch Test"
        elif "model quality test" in lower_text:
            label = "Model Quality Test"
        elif "a/b model comparison" in lower_text:
            label = "A/B Model Comparison"
        elif "advanced evaluation" in lower_text:
            label = "Advanced Evaluation"
        elif text:
            label = text
        else:
            label = DEFAULT_TEST_TYPE

    return _match_allowed_test_type(label, allowed_options)


def set_current_test_type(value, allowed_options=None, sync_widget_key=False):
    """
    Persist the active test type in app state.
    """
    selected = normalize_test_type(value, allowed_options)
    st.session_state.current_test_type = selected
    if sync_widget_key:
        version = st.session_state.get('_test_type_selector_version', 0)
        st.session_state[f'test_type_selector_widget_{version}'] = selected
    return selected


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
    if 'template_tokens' not in st.session_state:
        st.session_state.template_tokens = 0  # Default: no provider/model template overhead

    # PD separation toggle: skip first token for TPS/TPOT
    if 'skip_first_token_for_tps' not in st.session_state:
        st.session_state.skip_first_token_for_tps = False

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

    # Test navigation state. Streamlit reruns on every interaction, so keep the
    # selectbox key and the app-level current value in lockstep.
    selected_test_type = (
        st.session_state.get('current_test_type')
        or st.session_state.get('test_type_selector')
        or DEFAULT_TEST_TYPE
    )
    set_current_test_type(selected_test_type)


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
    st.session_state.test_running = False


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
    _touch_test_running_timestamp()
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
    _touch_test_running_timestamp()
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
    _touch_test_running_timestamp()


def clear_control_flags():
    """Clear all control flags"""
    st.session_state.stop_requested = False
    st.session_state.pause_requested = False
