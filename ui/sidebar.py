"""
Sidebar Configuration Module

Provides the Streamlit sidebar configuration interface, including:
- API provider selection
- Model configuration
- Tokenizer configuration
- System calibration
- Report info configuration
- Test type selection
"""

import time

import pandas as pd
import streamlit as st

from config.session_state import set_current_test_type
from config.settings import HF_MODEL_MAPPING
from core.error_messages import ErrorMessages, get_error_info
from core.tokenizer_utils import get_cached_tokenizer
from utils.custom_config import get_all_models, get_all_providers


def _auto_map_tokenizer(model_id: str) -> str:
    """
    Auto-map a model ID to the corresponding HuggingFace tokenizer.

    Args:
        model_id: The model ID to map

    Returns:
        The mapped tokenizer path, or the first available preset if no match
    """
    model_mapping = HF_MODEL_MAPPING
    hf_model_presets = list(model_mapping.values())

    current_model_lower = model_id.lower()

    # Find matching HF model (longer keys first so specific matches beat generic ones)
    for mapping_key, hf_id in sorted(model_mapping.items(), key=lambda x: len(x[0]), reverse=True):
        if mapping_key.lower() in current_model_lower:
            return hf_id

    # Return first preset if no match
    return hf_model_presets[0] if hf_model_presets else ""


def _on_model_change():
    """
    Callback when model selection changes - auto-update tokenizer mapping.
    """
    model_id = st.session_state.get('model_id_selector', '')
    if model_id:
        mapped_tokenizer = _auto_map_tokenizer(model_id)
        if mapped_tokenizer:
            st.session_state.hf_model_selector = mapped_tokenizer


def fetch_models(api_base, api_key):
    """
    Fetch model list from API

    Args:
        api_base: API base URL
        api_key: API key

    Returns:
        List of models
    """
    try:
        import requests
        url = api_base.rstrip('/')
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        response = requests.get(f'{url}/models', headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'data' in data:
            return [model['id'] for model in data['data']]
        return []
    except requests.exceptions.Timeout as e:
        error_info = get_error_info(e, context=f"API URL: {api_base}", language="en")
        error_msg = ErrorMessages.format_for_display(error_info, language="en")
        st.error(error_msg)
        return []
    except requests.exceptions.ConnectionError as e:
        error_info = get_error_info(e, context=f"API URL: {api_base}", language="en")
        error_msg = ErrorMessages.format_for_display(error_info, language="en")
        st.error(error_msg)
        return []
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
            error_info = get_error_info(e, context="API Key Verification", language="en")
        elif status_code == 429:
            error_info = get_error_info(e, context="Fetching Model List", language="en")
        elif status_code >= 500:
            error_info = get_error_info(e, context=f"API URL: {api_base}", language="en")
        else:
            error_info = get_error_info(e, language="en")

        error_msg = ErrorMessages.format_for_display(error_info, language="en")
        st.error(error_msg)
        return []
    except Exception as e:
        error_info = get_error_info(e, context=f"API URL: {api_base}", language="en")
        error_msg = ErrorMessages.format_for_display(error_info, language="en")
        st.error(error_msg)
        return []


def _render_tokenizer_status_panel():
    """Render tokenizer availability status panel with download buttons."""
    from core.tokenizer_utils import ensure_tokenizer_available, list_registered_tokenizers

    tokenizers = list_registered_tokenizers()
    missing = [t for t in tokenizers if not t["available"]]

    with st.expander("Tokenizer Status", expanded=False):
        # Show status for all registered tokenizers
        for t in tokenizers:
            icon = "✅" if t["available"] else "❌"
            size_str = f"{t['size_mb']:.1f} MB" if t["available"] else "—"
            st.markdown(f"{icon} **{t['name']}** — {size_str}")

        # Download buttons for missing tokenizers
        if missing:
            st.markdown("---")
            st.caption(f"{len(missing)} tokenizer(s) need download")

            if st.button("Download All Missing Tokenizers", key="download_all_tokenizers", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                total = len(missing)
                ok = 0
                for i, t in enumerate(missing):
                    status.info(f"Downloading {t['name']} from {t['hf_repo_id']}... ({i+1}/{total})")
                    local = ensure_tokenizer_available(t["local_path"])
                    if local:
                        ok += 1
                    else:
                        st.warning(f"Failed to download {t['name']}")
                    progress.progress((i + 1) / total)
                if ok == total:
                    st.success(f"All {total} tokenizer(s) downloaded!")
                elif ok > 0:
                    st.warning(f"Downloaded {ok}/{total}")
                else:
                    st.error("Download failed for all tokenizers")
                st.rerun()

            for t in missing:
                if st.button(f"Download {t['name']}", key=f"dl_tok_{t['name']}"):
                    with st.spinner(f"Downloading {t['name']}..."):
                        local = ensure_tokenizer_available(t["local_path"])
                    if local:
                        st.success(f"Downloaded {t['name']}")
                        st.rerun()
                    else:
                        st.error(f"Failed to download {t['name']} from {t['hf_repo_id']}")


def render_sidebar():
    """
    Render the complete sidebar configuration interface

    Returns:
        dict: Dictionary containing all configuration items
            {
                'provider': str,
                'api_base_url': str,
                'model_id': str,
                'api_key': str,
                'tokenizer_option': str,
                'hf_tokenizer_model_id': str,
                'test_type': str,
                'latency_offset': float,
                'template_tokens': int,
                'custom_sys_info': dict
            }
    """
    config = {}

    pending_preset = st.session_state.get('_pending_preset_apply')
    if pending_preset:
        st.session_state['_pending_preset_apply'] = None
        try:
            from utils.test_config_manager import apply_preset
            if apply_preset(pending_preset):
                st.session_state['_applied_preset_name'] = pending_preset
        except Exception as e:
            st.session_state['_preset_apply_error'] = str(e)

    # Import stop flag control function
    try:
        from core.providers.openai import set_stop_requested
    except ImportError:
        def set_stop_requested(value): pass

    # Import Gemini abort function
    try:
        from core.providers.gemini import abort_all_clients as abort_gemini_clients
    except ImportError:
        def abort_gemini_clients(): pass

    # Import pause/stop control functions
    from config.session_state import is_test_paused, is_test_running

    # Only show status indicator in sidebar, control buttons are in test_control_panel
    is_running = is_test_running()
    is_paused = is_test_paused()
    test_status = st.session_state.get('test_status', 'Idle')

    # Auto-fix: if test_status is not Running/Paused but flags are still True, fix them
    if test_status not in ('Running', 'Paused') and (is_running or is_paused):
        st.session_state.test_running = False
        st.session_state.test_paused = False
        # 不调用 st.rerun()，让当前渲染周期正常完成，避免全页白屏刷新

    # Show status indicator only
    if is_running:
        st.sidebar.info("Test running...")
    elif is_paused:
        st.sidebar.warning("Test paused")

    # Model configuration
    with st.sidebar.expander("Model Configuration", expanded=True):
        # Get all providers (built-in + custom)
        all_providers = get_all_providers()

        selected_provider = st.selectbox(
            "Select API Provider",
            options=list(all_providers.keys()),
            key="provider_selector"
        )

        default_api_base_url = all_providers[selected_provider]
        api_base_url = st.text_input("API Base URL", default_api_base_url)

        if "Non-Compatible" in selected_provider:
            st.warning(f"**Special Handling**: {selected_provider} uses non-OpenAI standard API. Dedicated adapter enabled.")

        api_key = st.text_input("API Key (Optional)", "", type="password")

        # Dynamically fetch model list
        if 'fetched_models' not in st.session_state:
            st.session_state.fetched_models = []

        col_model_1, col_model_2 = st.columns([3, 1])
        with col_model_2:
            if st.button("🔄", key="fetch_models_btn", help="Fetch model list from API"):
                with st.spinner("Fetching..."):
                    models = fetch_models(api_base_url, api_key)
                    if models:
                        st.session_state.fetched_models = models
                        if models and st.session_state.get('model_id_selector') not in models:
                            st.session_state.model_id_selector = models[0]
                            # Auto-map tokenizer for the first model
                            _on_model_change()
                        st.success(f"Fetched {len(models)} models")

        with col_model_1:
            options = st.session_state.fetched_models or get_all_models()

            if not isinstance(options, list):
                options = list(options)

            model_id_selected = st.selectbox(
                "Model ID (Quick Select)",
                options=options,
                key="model_id_selector",
                on_change=_on_model_change
            )

        model_id_custom = st.text_input(
            "Custom Model ID (Overrides above)",
            "",
            help="If entered here, this ID will be used instead."
        )

        model_id = model_id_custom if model_id_custom else model_id_selected

        st.markdown("---")

        # System calibration
        st.markdown("##### ⚡ System Calibration")

        # Probe button section - rendered FIRST so its callback can update state before widget is created
        col_cal_1, col_cal_2 = st.columns([2, 1])
        with col_cal_2:
            st.write("")
            st.write("")
            if st.button("Probe", key="latency_probe_btn", help="Auto-measure current network RTT and fill in the field"):
                with st.spinner("Measuring..."):
                    t_starts = []
                    for _ in range(3):
                        t0 = time.time()
                        fetch_models(api_base_url, api_key)
                        t_starts.append(time.time() - t0)

                    measured_rtt = min(t_starts)
                    st.session_state["latency_offset_input"] = measured_rtt
                    st.toast(f"Calibrated latency: {measured_rtt:.3f}s", icon="⚡")
                    # 无需 rerun，Streamlit 自动检测 widget key 值变化

        # Number input section
        with col_cal_1:
            # Ensure default value exists for the widget key
            if "latency_offset_input" not in st.session_state:
                st.session_state.latency_offset_input = 0.0

            latency_offset = st.number_input(
                "Global Latency Offset (s)",
                min_value=0.0,
                max_value=5.0,
                step=0.01,
                format="%.3f",
                key="latency_offset_input",
                help="This value is automatically subtracted from each TTFT calculation to exclude network latency and system overhead. Use the Probe button on the right to auto-detect."
            )
            # Sync to session_state.latency_offset for other components to read
            st.session_state.latency_offset = latency_offset

        if "template_tokens_input" not in st.session_state:
            st.session_state.template_tokens_input = int(st.session_state.get("template_tokens", 0) or 0)

        template_tokens = st.number_input(
            "Template Token Overhead",
            min_value=0,
            step=1,
            key="template_tokens_input",
            help="Extra tokens added by the engine/model chat template. This value is subtracted from generated prompt length targets."
        )
        st.session_state.template_tokens = int(template_tokens)

        # PD separation toggle
        st.checkbox(
            "Skip First Token for TPS",
            key="skip_first_token_for_tps",
            help="For PD-separated providers: treat the 2nd token as generation start for TPS/TPOT, so decode speed is measured accurately. TTFT is unaffected."
        )

        # Random seed (for reproducibility)
        st.markdown("##### 🎲 Random Seed")
        col_seed_1, col_seed_2 = st.columns([3, 1])
        with col_seed_1:
            seed_enabled = st.checkbox(
                "Enable Fixed Seed",
                value=False,
                key="random_seed_enabled",
                help="When enabled, random operations during tests (e.g., prompt generation) will use a fixed seed for reproducibility"
            )
        with col_seed_2:
            if seed_enabled:
                random_seed = st.number_input(
                    "Seed Value",
                    min_value=0,
                    max_value=2**31 - 1,
                    value=42,
                    step=1,
                    key="random_seed_value",
                    help="Fixed seed value. Same seed + same config = reproducible results"
                )
                st.session_state.random_seed = random_seed
            else:
                st.session_state.random_seed = None

        tokenizer_option = st.selectbox(
            "Token Counting Method",
            ["HuggingFace Tokenizer", "API (usage field)", "Character Count (Fallback)"],
            key="tokenizer_option_selector",
            help="Recommend 'API (usage field)' for the most accurate results. For non-OpenAI models, choose 'HuggingFace Tokenizer' and specify the model ID."
        )

        # Tokenizer status panel (when HF tokenizer is selected)
        if tokenizer_option == "HuggingFace Tokenizer":
            _render_tokenizer_status_panel()

        hf_tokenizer_model_id = ""
        if tokenizer_option == "HuggingFace Tokenizer":
            model_mapping = HF_MODEL_MAPPING
            hf_model_presets = list(model_mapping.values())
            if "Custom" not in hf_model_presets:
                hf_model_presets.append("Custom")

            current_model_lower = model_id.lower()

            # Calculate default matching HF model (longer keys first for specificity)
            default_hf_model = hf_model_presets[0]
            for mapping_key, hf_id in sorted(model_mapping.items(), key=lambda x: len(x[0]), reverse=True):
                if mapping_key.lower() in current_model_lower:
                    default_hf_model = hf_id
                    break

            # Get the index of the default model for auto-selection
            default_index = 0
            if default_hf_model in hf_model_presets:
                default_index = hf_model_presets.index(default_hf_model)

            selected_hf_preset = st.selectbox(
                "Select HuggingFace Model ID",
                hf_model_presets,
                index=default_index,
                key="hf_model_selector",
                help="Auto-recommends matching tokenizer based on current model"
            )

            if selected_hf_preset == "Custom":
                hf_tokenizer_model_id = st.text_input(
                    "Enter Custom Model ID",
                    value="",
                    key="hf_custom_model_id",
                    help="Enter the HuggingFace model ID, e.g., 'deepseek-ai/deepseek-llm-7b-chat' or a local path."
                )
            else:
                hf_tokenizer_model_id = selected_hf_preset

        # Token counter
        with st.sidebar.expander("🧮 Token Counter", expanded=False):
            st.caption(f"Current Tokenizer: {hf_tokenizer_model_id if hf_tokenizer_model_id else 'Not Selected'}")

            calc_text = st.text_area("Enter text to count tokens", height=100, key="token_calc_input")

            if calc_text:
                if not hf_tokenizer_model_id:
                    st.warning("Please select a model in the HuggingFace Tokenizer section above first.")
                else:
                    try:
                        with st.spinner("Loading Tokenizer..."):
                            calc_tokenizer = get_cached_tokenizer(hf_tokenizer_model_id)

                        if calc_tokenizer:
                            tokens = calc_tokenizer.encode(calc_text, add_special_tokens=False)
                            count = len(tokens)
                            st.info(f"Token Count: **{count}**")
                        else:
                            st.error("Tokenizer load failed")
                    except Exception as e:
                        st.error(f"Calculation error: {e}")

    st.sidebar.markdown("---")

    # Test configuration
    st.sidebar.header("🧪 Test Configuration")

    # Check optional modules
    import importlib
    try:
        importlib.import_module("core.quality_evaluator")
        quality_available = True
    except ImportError:
        quality_available = False
    try:
        importlib.import_module("core.reasoning_evaluator")
        enhanced_available = True
    except ImportError:
        enhanced_available = False

    # Build test type list
    _test_types = [
        "Concurrency Test",
        "Prefill Stress Test",
        "Segmented Context Test",
        "Long Context Test",
        "Concurrency-Context Matrix Test",
        "Custom Text Test",
        "All Tests",
        "Stability Test",
        "📦 Batch Test"
    ]
    if quality_available:
        _test_types.append("📝 Model Quality Test")
        _test_types.append("🔄 A/B Model Comparison")
    if enhanced_available:
        _test_types.append("🔬 Advanced Evaluation")

    forced_test_type = st.session_state.get("_force_test_type_selector")
    if forced_test_type:
        st.session_state["_force_test_type_selector"] = None
        st.session_state["_test_type_selector_version"] = (
            st.session_state.get("_test_type_selector_version", 0) + 1
        )

    selector_version = st.session_state.get("_test_type_selector_version", 0)
    test_type_widget_key = f"test_type_selector_widget_{selector_version}"

    selected_test_type = set_current_test_type(
        forced_test_type
        or st.session_state.get(test_type_widget_key)
        or st.session_state.get("current_test_type"),
        _test_types,
    )

    test_type = st.sidebar.selectbox(
        "Select Test Type",
        _test_types,
        index=_test_types.index(selected_test_type),
        key=test_type_widget_key,
        disabled=is_running or is_paused,
    )
    set_current_test_type(test_type, _test_types, sync_widget_key=False)

    # Return configuration dict
    config.update({
        'provider': selected_provider,
        'api_base_url': api_base_url,
        'model_id': model_id,
        'api_key': api_key,
        'tokenizer_option': tokenizer_option,
        'hf_tokenizer_model_id': hf_tokenizer_model_id,
        'test_type': test_type,
        'latency_offset': st.session_state.latency_offset,
        'template_tokens': st.session_state.template_tokens,
        'custom_sys_info': st.session_state.custom_sys_info
    })

    return config


def render_sidebar_bottom():
    """
    Render the sidebar bottom configuration management interface.
    Should be called after all other sidebar content has been rendered.
    """
    st.sidebar.markdown("---")

    # Saved benchmark results
    with st.sidebar.expander("Historical Results Management", expanded=False):
        try:
            from core.results.history_service import (
                delete_saved_result,
                list_saved_results,
                restore_result_file_to_session,
            )

            saved_results = list_saved_results()
            if saved_results:
                total_size_kb = sum(item.get("size_kb", 0) for item in saved_results)
                st.caption(f"{len(saved_results)} saved result(s), {total_size_kb:.1f} KB")

                def _format_saved_result(item):
                    modified = time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(item.get("modified_time", 0)),
                    )
                    model = item.get("model_id") or "Unknown model"
                    test_type = item.get("test_type") or "Unknown test"
                    return f"{modified} | {model} | {test_type}"

                options = {_format_saved_result(item): item["path"] for item in saved_results}
                selected_saved = st.selectbox(
                    "Select Saved Result",
                    list(options.keys()),
                    key="saved_result_select",
                )
                selected_item = next(
                    item for item in saved_results if item["path"] == options[selected_saved]
                )

                st.caption(
                    f"File: {selected_item['display_name']} | "
                    f"Provider: {selected_item.get('provider', 'Unknown')}"
                )

                load_col, delete_col = st.columns(2)
                with load_col:
                    if st.button(
                        "Load",
                        key="load_saved_result_btn",
                        use_container_width=True,
                        help="Load this saved CSV and regenerate statistics, charts, and conclusions.",
                    ):
                        if restore_result_file_to_session(st.session_state, options[selected_saved]):
                            st.session_state["_force_test_type_selector"] = st.session_state.get(
                                "current_test_type"
                            )
                            st.success("Loaded. Regenerating charts and conclusions...")
                            st.rerun()
                        else:
                            st.error("Failed to load saved result.")
                with delete_col:
                    confirm_delete = st.checkbox(
                        "Confirm delete",
                        key=f"confirm_delete_saved_{selected_item['path']}",
                    )
                    if st.button(
                        "Delete",
                        key="delete_saved_result_btn",
                        use_container_width=True,
                        disabled=not confirm_delete,
                        help="Delete the selected CSV and its metadata.",
                    ):
                        if delete_saved_result(options[selected_saved]):
                            if st.session_state.get("current_csv_file") == options[selected_saved]:
                                st.session_state.results_df = pd.DataFrame()
                                st.session_state.report = ""
                                st.session_state.restored_from_csv = False
                                st.session_state.restored_result_context = {}
                            st.success("Deleted saved result.")
                            st.rerun()
                        else:
                            st.error("Failed to delete saved result.")
            else:
                st.info("No saved result CSV found in raw_data.")
        except Exception as e:
            st.warning(f"Saved result loader unavailable: {e}")

    # Config presets
    with st.sidebar.expander("📁 Config Presets", expanded=False):
        try:
            from utils.test_config_manager import config_manager

            # List all presets
            all_presets = config_manager.list_presets()

            if all_presets:
                preset_names = [p["name"] for p in all_presets]
                selected_preset = st.selectbox("Select Preset", [""] + preset_names, key="preset_select")

                applied_preset = st.session_state.get('_applied_preset_name')
                if applied_preset:
                    st.success(f"Applied: {applied_preset}")
                    st.session_state['_applied_preset_name'] = None

                preset_error = st.session_state.get('_preset_apply_error')
                if preset_error:
                    st.error(f"Load failed: {preset_error}")
                    st.session_state['_preset_apply_error'] = None

                if selected_preset:
                    col_load, col_del = st.columns(2)

                    with col_load:
                        if st.button("📥 Apply", key="preset_load_btn", use_container_width=True):
                            st.session_state['_pending_preset_apply'] = selected_preset
                            st.rerun()

                    with col_del:
                        if st.button("🗑️", key="preset_del_btn", help="Delete preset", use_container_width=True):
                            if config_manager.delete_preset(selected_preset):
                                st.success(f"Deleted: {selected_preset}")
                                st.rerun()
            else:
                st.info("No saved presets")
        except Exception as e:
            st.warning(f"Preset management unavailable: {e}")

    # Save current config as preset
    with st.sidebar.expander("💾 Save Current Config", expanded=False):
        preset_name = st.text_input("Preset Name", key="save_preset_name", placeholder="e.g.: My Quick Test")
        preset_desc = st.text_input("Description (Optional)", key="save_preset_desc", placeholder="Describe config purpose...")

        if st.button("💾 Save as Preset", key="save_preset_btn", use_container_width=True):
            if preset_name:
                try:
                    from utils.test_config_manager import ConfigPreset, get_current_config

                    current_config = get_current_config()
                    current_config.update({
                        'api_base_url': st.session_state.get('current_api_base', ''),
                        'model_id': st.session_state.get('current_model_id', ''),
                        'concurrency': st.session_state.get('current_concurrency', 1)
                    })

                    preset = ConfigPreset(
                        name=preset_name,
                        description=preset_desc,
                        config=current_config
                    )

                    if config_manager.save_preset(preset):
                        st.success(f"Saved preset: {preset_name}")
                except Exception as e:
                    st.error(f"Save failed: {e}")
            else:
                st.warning("Please enter a preset name")

    # Report info configuration
    with st.sidebar.expander("🖥️ Report Info Config", expanded=False):
        st.caption("Manually enter hardware info here, which will be displayed in generated charts.\n(Leave empty to hide; defaults to empty for API tests)")

        if 'custom_sys_info' not in st.session_state:
            st.session_state.custom_sys_info = {}

        c_proc = st.text_input("Processor", value=st.session_state.custom_sys_info.get('processor', ''), key="bottom_proc")
        c_mb = st.text_input("Mainboard", value=st.session_state.custom_sys_info.get('mainboard', ''), key="bottom_mb")
        c_mem = st.text_input("Memory", value=st.session_state.custom_sys_info.get('memory', ''), key="bottom_mem")
        c_gpu = st.text_input("GPU", value=st.session_state.custom_sys_info.get('gpu', ''), key="bottom_gpu")
        c_sys = st.text_input("System", value=st.session_state.custom_sys_info.get('system', ''), key="bottom_sys")
        c_engine = st.text_input("Engine", value=st.session_state.custom_sys_info.get('engine_name', ''), help="Inference engine name, e.g., vLLM, SGLang, Ollama, etc.", key="bottom_engine")

        st.session_state.custom_sys_info.update({
            'processor': c_proc,
            'mainboard': c_mb,
            'memory': c_mem,
            'gpu': c_gpu,
            'system': c_sys,
            'engine_name': c_engine
        })

    # 数据仓库输入面板：模型规格 / 服务配置 / 测试元数据（详见 ui/warehouse_panels.py）
    try:
        from ui.warehouse_panels import (
            render_engine_runtime_panel,
            render_model_spec_panel,
            render_serving_config_panel,
            render_test_metadata_panel,
        )
        render_model_spec_panel(st.session_state.get('model_id_selector', ''))
        render_serving_config_panel()
        render_test_metadata_panel()
        render_engine_runtime_panel(st.session_state.get('api_base_url_input', ''))
    except Exception:
        # 面板渲染失败不应阻塞 sidebar
        pass
