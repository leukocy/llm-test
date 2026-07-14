"""
LLM Performance Benchmark Platform - V2 Streamlined Edition

This is the refactored version using modular components:
- config.session_state - Session state management
- ui.sidebar - Sidebar configuration
- ui.test_panels - Test panels
- ui.test_runner - Test execution
- ui.page_layout - Page layout

Startup optimization:
- Lazy import of non-essential modules to improve startup speed
"""

import os
import warnings

# Environment configuration
os.environ["STREAMLIT_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")

import streamlit as st

# === Core modules (immediate import) ===
from config.session_state import init_session_state, is_test_running

# === Lazy import modules (on-demand loading) ===
# The following modules are only imported when needed, reducing startup time
_lazy_modules = {}


def _get_advanced_panels():
    """Lazy load advanced panels module"""
    if "advanced_panels" not in _lazy_modules:
        from ui import advanced_panels

        _lazy_modules["advanced_panels"] = advanced_panels
    return _lazy_modules["advanced_panels"]


def _get_batch_test():
    """Lazy load batch test module"""
    if "batch_test" not in _lazy_modules:
        from ui import batch_test

        _lazy_modules["batch_test"] = batch_test
    return _lazy_modules["batch_test"]


def _get_test_panels():
    """Lazy load test panels module"""
    if "test_panels" not in _lazy_modules:
        from ui import test_panels

        _lazy_modules["test_panels"] = test_panels
    return _lazy_modules["test_panels"]


def _get_test_runner():
    """Lazy load test runner module"""
    if "test_runner" not in _lazy_modules:
        from ui import test_runner

        _lazy_modules["test_runner"] = test_runner
    return _lazy_modules["test_runner"]


def _get_custom_config():
    """Lazy load custom config module"""
    if "custom_config" not in _lazy_modules:
        from utils import custom_config

        _lazy_modules["custom_config"] = custom_config
    return _lazy_modules["custom_config"]


def _get_preset_manager():
    """Lazy load preset manager module"""
    if "preset_manager" not in _lazy_modules:
        from utils import preset_manager

        _lazy_modules["preset_manager"] = preset_manager
    return _lazy_modules["preset_manager"]


def _get_test_config_manager():
    """Lazy load test config manager module"""
    if "test_config_manager" not in _lazy_modules:
        from utils import test_config_manager

        _lazy_modules["test_config_manager"] = test_config_manager
    return _lazy_modules["test_config_manager"]


# === Page configuration ===
st.set_page_config(page_title="LLM Benchmark Platform", layout="wide")


# === UI module imports (lightweight, immediate import) ===
from ui.design_system import apply_design_system, material_icon, render_application_header
from ui.onboarding import (
    init_onboarding_state,
    render_compact_welcome_banner,
    render_onboarding_guide,
    render_onboarding_trigger,
)
from ui.sidebar import render_sidebar, render_sidebar_bottom


# === Test execution function (backward compatible interface) ===
def run_test(test_function, runner_class, *args):
    """
    Backward compatible wrapper function for test execution

    Args:
        test_function: Test function
        runner_class: Test runner class
        *args: Test parameters
    """
    # Get current configuration
    config = {
        "provider": st.session_state.get("current_provider", ""),
        "api_base_url": st.session_state.get("current_api_base", ""),
        "model_id": st.session_state.get("current_model_id", ""),
        "api_key": st.session_state.get("current_api_key", ""),
        "tokenizer_option": st.session_state.get("current_tokenizer", ""),
        "hf_tokenizer_model_id": st.session_state.get("current_hf_tokenizer", ""),
        "template_tokens": st.session_state.get("template_tokens", 0),
    }

    # Lazy load TestExecutor
    test_runner = _get_test_runner()
    executor = test_runner.TestExecutor(config)
    test_type = st.session_state.get("current_test_type", "Unknown Test")

    # Execute test — 启动后台线程，立即返回 _TestRunHandle（不阻塞主脚本线程）
    return executor.run_test(test_function, runner_class, test_type, *args)


# === Main program ===
def main():
    """Main program entry"""

    apply_design_system()

    # Hardware discovery launches several system commands. Start it while the
    # user is configuring a test instead of serializing it before request #1.
    from core.system_info import warm_system_info_cache

    warm_system_info_cache()

    # 1. Initialize session state
    init_session_state()

    # 2. Initialize built-in presets (lazy load)
    test_config_manager = _get_test_config_manager()
    test_config_manager.init_builtin_presets()

    # 3. Initialize onboarding state
    init_onboarding_state()

    # 4. Render sidebar and get configuration
    sidebar_config = render_sidebar()

    # The adaptive KV probe can issue /metrics and /models requests. Cache it
    # per endpoint while the user is still on the configuration screen.
    from core.engine_metrics import warm_kv_capacity_cache

    warm_kv_capacity_cache(sidebar_config["api_base_url"])

    # 5. Save configuration to session_state
    st.session_state.current_provider = sidebar_config["provider"]
    st.session_state.current_api_base = sidebar_config["api_base_url"]
    st.session_state.current_model_id = sidebar_config["model_id"]
    st.session_state.current_api_key = sidebar_config["api_key"]
    st.session_state.current_tokenizer = sidebar_config["tokenizer_option"]
    st.session_state.current_hf_tokenizer = sidebar_config["hf_tokenizer_model_id"]
    st.session_state.template_tokens = sidebar_config.get("template_tokens", 0)
    st.session_state.current_test_type = sidebar_config["test_type"]

    # 6. Render the active benchmark context
    test_type = sidebar_config["test_type"]
    render_application_header(
        provider=sidebar_config["provider"],
        model_id=sidebar_config["model_id"],
        test_type=test_type,
    )

    # 7. Render onboarding content below the stable application header
    render_compact_welcome_banner()
    render_onboarding_guide()

    # 8. Handle test types

    if test_type == "quality":
        advanced_panels = _get_advanced_panels()
        advanced_panels.render_quality_test_panel(sidebar_config, run_test)
    elif test_type == "comparison":
        advanced_panels = _get_advanced_panels()
        advanced_panels.render_ab_comparison_panel(sidebar_config)
    elif test_type == "advanced":
        advanced_panels = _get_advanced_panels()
        advanced_panels.render_advanced_eval_panel(sidebar_config)
    elif test_type == "batch":
        batch_test = _get_batch_test()
        batch_test.render_batch_test_main()
    elif test_type == "data_warehouse":
        from ui.warehouse_browser import render_warehouse_browser

        render_warehouse_browser()
    elif test_type == "environment":
        from ui.env_overview import render_env_overview

        render_env_overview(
            sidebar_config.get("api_base_url", ""),
            sidebar_config.get("model_id", ""),
        )
    else:
        # 9. Render test control panel (provides Stop button)
        from ui.test_control_panel import render_test_control_panel

        render_test_control_panel()

        # 10. Render normal test panels (includes test config and start button)
        test_panels = _get_test_panels()
        test_panels.render_test_panels(test_type, run_test)

        # 11. If test completed, show results
        if not is_test_running() and not st.session_state.results_df.empty:
            from ui.page_layout import PageLayout

            PageLayout.render(test_type)

    # 12. Render sidebar bottom config management (ensure after all test panels)
    render_sidebar_bottom()

    # 13. Render onboarding trigger (sidebar bottom)
    render_onboarding_trigger()


# === Preset management (sidebar bottom) ===
def render_preset_management():
    """Render preset management interface"""
    st.sidebar.markdown("---")

    preset_manager = _get_preset_manager()

    with st.sidebar.expander("Config Presets", expanded=False):
        preset_names = ["<New Config>"] + preset_manager.list_presets()
        selected_preset_name = st.selectbox(
            "Select Preset", options=preset_names, key="preset_selector"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Load",
                disabled=(selected_preset_name == "<New Config>"),
                width="stretch",
                icon=material_icon("download"),
            ):
                preset_data = preset_manager.load_preset(selected_preset_name)
                if preset_data:
                    st.success(f"Loaded: {selected_preset_name}")
                else:
                    st.error("Load failed")

        with col2:
            if st.button(
                "Delete",
                disabled=(selected_preset_name == "<New Config>"),
                width="stretch",
                icon=material_icon("delete"),
            ):
                if preset_manager.delete_preset(selected_preset_name):
                    st.success(f"Deleted: {selected_preset_name}")
                else:
                    st.error("Delete failed")

        st.markdown("---")
        new_preset_name = st.text_input("Preset Name", key="new_preset_name")
        if st.button("Save Preset", width="stretch"):
            if new_preset_name and new_preset_name.strip():
                st.session_state.save_preset_requested = new_preset_name
            else:
                st.error("Please enter a preset name")


# === Custom config management (sidebar bottom) ===
def render_custom_config():
    """Render custom config management interface"""
    st.sidebar.markdown("---")

    custom_config = _get_custom_config()

    with st.sidebar.expander("Custom Config Management"):
        tab1, tab2 = st.tabs(["Providers", "Models"])

        with tab1:
            st.markdown("**Add Custom Provider**")
            custom_provider_name = st.text_input("Provider Name", key="custom_provider_name")
            custom_provider_url = st.text_input("API Base URL", key="custom_provider_url")

            if st.button(
                "Add Provider",
                key="add_provider",
                icon=material_icon("add"),
            ):
                if custom_provider_name and custom_provider_url:
                    if custom_config.add_custom_provider(custom_provider_name, custom_provider_url):
                        st.success(f"Added: {custom_provider_name}")
                    else:
                        st.error("Add failed")

        with tab2:
            st.markdown("**Add Custom Model**")
            custom_model_name = st.text_input("Model Name", key="custom_model_name")

            if st.button(
                "Add Model",
                key="add_model",
                icon=material_icon("add"),
            ):
                if custom_model_name and custom_model_name.strip():
                    if custom_config.add_custom_model(custom_model_name.strip()):
                        st.success(f"Added: {custom_model_name}")
                    else:
                        st.warning("Model already exists")


# === Start application ===
if __name__ == "__main__":
    main()
