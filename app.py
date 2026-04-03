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
    if 'advanced_panels' not in _lazy_modules:
        from ui import advanced_panels
        _lazy_modules['advanced_panels'] = advanced_panels
    return _lazy_modules['advanced_panels']

def _get_batch_test():
    """Lazy load batch test module"""
    if 'batch_test' not in _lazy_modules:
        from ui import batch_test
        _lazy_modules['batch_test'] = batch_test
    return _lazy_modules['batch_test']

def _get_test_panels():
    """Lazy load test panels module"""
    if 'test_panels' not in _lazy_modules:
        from ui import test_panels
        _lazy_modules['test_panels'] = test_panels
    return _lazy_modules['test_panels']

def _get_test_runner():
    """Lazy load test runner module"""
    if 'test_runner' not in _lazy_modules:
        from ui import test_runner
        _lazy_modules['test_runner'] = test_runner
    return _lazy_modules['test_runner']

def _get_custom_config():
    """Lazy load custom config module"""
    if 'custom_config' not in _lazy_modules:
        from utils import custom_config
        _lazy_modules['custom_config'] = custom_config
    return _lazy_modules['custom_config']

def _get_preset_manager():
    """Lazy load preset manager module"""
    if 'preset_manager' not in _lazy_modules:
        from utils import preset_manager
        _lazy_modules['preset_manager'] = preset_manager
    return _lazy_modules['preset_manager']

def _get_test_config_manager():
    """Lazy load test config manager module"""
    if 'test_config_manager' not in _lazy_modules:
        from utils import test_config_manager
        _lazy_modules['test_config_manager'] = test_config_manager
    return _lazy_modules['test_config_manager']

# === Page configuration ===
st.set_page_config(page_title="LLM Benchmark Platform V2", layout="wide")


# === CSS styles ===
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e6e9ef;
}
[data-testid="main"] {
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 0 15px rgba(0,0,0,0.03);
    margin: 1rem;
    padding: 1.5rem;
}
h1 {
    color: #1a1a1a;
    font-weight: 600;
    border-bottom: 2px solid #f0f2f6;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# === UI module imports (lightweight, immediate import) ===
from ui.onboarding import (
    init_onboarding_state,
    render_onboarding_modal,
    render_onboarding_trigger,
    show_onboarding
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
        'provider': st.session_state.get('current_provider', ''),
        'api_base_url': st.session_state.get('current_api_base', ''),
        'model_id': st.session_state.get('current_model_id', ''),
        'api_key': st.session_state.get('current_api_key', ''),
        'tokenizer_option': st.session_state.get('current_tokenizer', ''),
        'hf_tokenizer_model_id': st.session_state.get('current_hf_tokenizer', ''),
    }

    # Lazy load TestExecutor
    test_runner = _get_test_runner()
    executor = test_runner.TestExecutor(config)
    test_type = st.session_state.get('current_test_type', 'Unknown Test')

    # Execute test
    return executor.run_test(test_function, runner_class, test_type, *args)


# === Main program ===
def main():
    """Main program entry"""

    # 1. Initialize session state
    init_session_state()

    # 2. Initialize built-in presets (lazy load)
    test_config_manager = _get_test_config_manager()
    test_config_manager.init_builtin_presets()

    # 3. Initialize onboarding state
    init_onboarding_state()

    # 4. Render onboarding (if needed)
    if show_onboarding():
        render_onboarding_modal()

    # 5. Render sidebar and get configuration
    sidebar_config = render_sidebar()

    # 6. Save configuration to session_state
    st.session_state.current_provider = sidebar_config['provider']
    st.session_state.current_api_base = sidebar_config['api_base_url']
    st.session_state.current_model_id = sidebar_config['model_id']
    st.session_state.current_api_key = sidebar_config['api_key']
    st.session_state.current_tokenizer = sidebar_config['tokenizer_option']
    st.session_state.current_hf_tokenizer = sidebar_config['hf_tokenizer_model_id']
    st.session_state.current_test_type = sidebar_config['test_type']

    # 7. Handle test types
    test_type = sidebar_config['test_type']

    if test_type == "📝 Model Quality Test":
        advanced_panels = _get_advanced_panels()
        advanced_panels.render_quality_test_panel(sidebar_config, run_test)
    elif test_type == "🔄 A/B Model Comparison":
        advanced_panels = _get_advanced_panels()
        advanced_panels.render_ab_comparison_panel(sidebar_config)
    elif test_type == "🔬 Advanced Evaluation":
        advanced_panels = _get_advanced_panels()
        advanced_panels.render_advanced_eval_panel(sidebar_config)
    elif test_type == "📦 Batch Test":
        batch_test = _get_batch_test()
        batch_test.render_batch_test_main()
    else:
        # 8. Render test control panel (provides Stop button)
        from ui.test_control_panel import render_test_control_panel
        control_actions = render_test_control_panel()

        # 9. Render normal test panels (includes test config and start button)
        test_panels = _get_test_panels()
        test_panels.render_test_panels(
            test_type,
            run_test
        )

        # 10. If test completed, show results
        if not is_test_running() and not st.session_state.results_df.empty:
            from ui.page_layout import PageLayout
            PageLayout.render(test_type)

    # 10. Render sidebar bottom config management (ensure after all test panels)
    render_sidebar_bottom()

    # 11. Render onboarding trigger (sidebar bottom)
    render_onboarding_trigger()


# === Preset management (sidebar bottom) ===
def render_preset_management():
    """Render preset management interface"""
    st.sidebar.markdown("---")

    preset_manager = _get_preset_manager()

    with st.sidebar.expander("⚙️ Config Presets", expanded=False):
        preset_names = ["<New Config>"] + preset_manager.list_presets()
        selected_preset_name = st.selectbox(
            "Select Preset",
            options=preset_names,
            key="preset_selector"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load", disabled=(selected_preset_name == "<New Config>"), width='stretch'):
                preset_data = preset_manager.load_preset(selected_preset_name)
                if preset_data:
                    st.success(f"Loaded: {selected_preset_name}")
                else:
                    st.error("Load failed")

        with col2:
            if st.button("🗑️ Delete", disabled=(selected_preset_name == "<New Config>"), width='stretch'):
                if preset_manager.delete_preset(selected_preset_name):
                    st.success(f"Deleted: {selected_preset_name}")
                else:
                    st.error("Delete failed")

        st.markdown("---")
        new_preset_name = st.text_input("Preset Name", key="new_preset_name")
        if st.button("Save Preset", width='stretch'):
            if new_preset_name and new_preset_name.strip():
                st.session_state.save_preset_requested = new_preset_name
            else:
                st.error("Please enter a preset name")


# === Custom config management (sidebar bottom) ===
def render_custom_config():
    """Render custom config management interface"""
    st.sidebar.markdown("---")

    custom_config = _get_custom_config()

    with st.sidebar.expander("⚙️ Custom Config Management"):
        tab1, tab2 = st.tabs(["Providers", "Models"])

        with tab1:
            st.markdown("**Add Custom Provider**")
            custom_provider_name = st.text_input("Provider Name", key="custom_provider_name")
            custom_provider_url = st.text_input("API Base URL", key="custom_provider_url")

            if st.button("➕ Add Provider", key="add_provider"):
                if custom_provider_name and custom_provider_url:
                    if custom_config.add_custom_provider(custom_provider_name, custom_provider_url):
                        st.success(f"Added: {custom_provider_name}")
                    else:
                        st.error("Add failed")

        with tab2:
            st.markdown("**Add Custom Model**")
            custom_model_name = st.text_input("Model Name", key="custom_model_name")

            if st.button("➕ Add Model", key="add_model"):
                if custom_model_name and custom_model_name.strip():
                    if custom_config.add_custom_model(custom_model_name.strip()):
                        st.success(f"Added: {custom_model_name}")
                    else:
                        st.warning("Model already exists")


# === Start application ===
if __name__ == "__main__":
    main()
