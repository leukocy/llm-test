"""
Advanced Test Panels Module

Contains quality testing, A/B comparison, advanced evaluation features

Startup Optimization:
- Evaluation modules use lazy import to reduce startup time
"""

import asyncio
import os

import pandas as pd
import streamlit as st

# ============================================================================
# Lazy Import Modules (loaded on demand)
# ============================================================================

_quality_eval_module = None
_enhanced_eval_module = None


def _get_quality_eval_module():
    """Lazy load quality assessment module"""
    global _quality_eval_module
    if _quality_eval_module is None:
        try:
            from core import quality_evaluator
            from evaluators import (  # type: ignore[attr-defined]
                EVALUATOR_REGISTRY,
                GSM8KEvaluator,
                MMLUEvaluator,
            )
            from ui import quality_reports

            _quality_eval_module = {
                "QualityEvaluator": quality_evaluator.QualityEvaluator,
                "QualityTestConfig": quality_evaluator.QualityTestConfig,
                "EVALUATOR_REGISTRY": EVALUATOR_REGISTRY,
                "GSM8KEvaluator": GSM8KEvaluator,
                "MMLUEvaluator": MMLUEvaluator,
                "generate_quality_summary": quality_reports.generate_quality_summary,
                "render_quality_report": quality_reports.render_quality_report,
            }
        except ImportError:
            pass
    return _quality_eval_module


def _get_enhanced_eval_module():
    """Lazy load enhanced evaluation module"""
    global _enhanced_eval_module
    if _enhanced_eval_module is None:
        try:
            from core import model_comparator, reasoning_evaluator, smart_answer_parser

            _enhanced_eval_module = {
                "ModelComparator": model_comparator.ModelComparator,
                "ModelConfig": model_comparator.ModelConfig,
                "ReasoningQualityEvaluator": reasoning_evaluator.ReasoningQualityEvaluator,
                "AnswerType": smart_answer_parser.AnswerType,
                "SmartAnswerParser": smart_answer_parser.SmartAnswerParser,
            }
        except ImportError:
            pass
    return _enhanced_eval_module


def is_quality_eval_available():
    """Check if quality assessment module is available"""
    return _get_quality_eval_module() is not None


def is_enhanced_eval_available():
    """Check if enhanced evaluation module is available"""
    return _get_enhanced_eval_module() is not None


# Compatibility variables (lazy detection)
QUALITY_EVAL_AVAILABLE = None  # Will be detected on first access
ENHANCED_EVAL_AVAILABLE = None  # Will be detected on first access


def _check_quality_eval():
    """Lazy detect quality assessment availability"""
    global QUALITY_EVAL_AVAILABLE
    if QUALITY_EVAL_AVAILABLE is None:
        QUALITY_EVAL_AVAILABLE = _get_quality_eval_module() is not None
    return QUALITY_EVAL_AVAILABLE


def _check_enhanced_eval():
    """Lazy detect enhanced evaluation availability"""
    global ENHANCED_EVAL_AVAILABLE
    if ENHANCED_EVAL_AVAILABLE is None:
        ENHANCED_EVAL_AVAILABLE = _get_enhanced_eval_module() is not None
    return ENHANCED_EVAL_AVAILABLE


# Mapping from dataset key to display name
_DATASET_DISPLAY_NAMES = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "humaneval": "HumanEval",
    "ceval": "C-Eval",
    "gpqa": "GPQA",
    "arc": "ARC-Challenge",
    "truthfulqa": "TruthfulQA",
    "winogrande": "WinoGrande",
    "hellaswag": "HellaSwag",
    "mbpp": "MBPP",
    "aime2025": "AIME 2025",
    "aime2026": "AIME 2026",
    "longbench": "LongBench",
    "arena_hard": "Arena Hard",
    "global_piqa": "PIQA",
    "swebench_lite": "SWE-Bench Lite",
    "needle_haystack": "Needle Haystack",
    "custom_needle": "Custom Needle",
}


def _render_dataset_status_panel(available_datasets):
    """Render dataset availability status with download buttons."""
    # Datasets that generate synthetic data (no download needed)
    _SYNTHETIC_DATASETS = {"needle_haystack", "custom_needle"}

    # Only show datasets registered in DatasetManager
    try:
        from core.dataset_manager import get_manager

        manager = get_manager()
        managed_datasets = [
            d for d in available_datasets if d in manager.configs and d not in _SYNTHETIC_DATASETS
        ]
    except Exception:
        return

    if not managed_datasets:
        return

    with st.expander("Dataset Status", expanded=False):
        # Check status for each dataset
        status_data = []
        missing_datasets = []

        for ds_name in managed_datasets:
            is_avail = manager.is_available(ds_name)
            display_name = _DATASET_DISPLAY_NAMES.get(ds_name, ds_name)

            size_str = "—"
            if is_avail:
                info = manager.get_info(ds_name)
                size_mb = info.get("total_size_mb", 0)
                if size_mb > 0:
                    size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.1f} GB"

            status_data.append(
                {
                    "Dataset": display_name,
                    "Key": ds_name,
                    "Available": is_avail,
                    "Size": size_str,
                }
            )
            if not is_avail:
                missing_datasets.append(ds_name)

        # Display status table
        for row in status_data:
            status = "Available" if row["Available"] else "Missing"
            st.markdown(f"**{row['Dataset']}** — {status} — {row['Size']}")

        # Download buttons for missing datasets
        if missing_datasets:
            st.markdown("---")
            st.caption(f"{len(missing_datasets)} dataset(s) need download")

            # Download All button
            if st.button("Download All Missing", key="download_all_datasets", type="primary"):
                _download_datasets(manager, missing_datasets)

            # Per-dataset download buttons
            for ds_name in missing_datasets:
                display_name = _DATASET_DISPLAY_NAMES.get(ds_name, ds_name)
                if st.button(f"Download {display_name}", key=f"dl_{ds_name}"):
                    _download_datasets(manager, [ds_name])


def _download_datasets(manager, datasets):
    """Download datasets with progress feedback."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(datasets)
    success_count = 0

    for i, ds_name in enumerate(datasets):
        display_name = _DATASET_DISPLAY_NAMES.get(ds_name, ds_name)
        status_text.info(f"Downloading {display_name}... ({i+1}/{total})")

        def progress_cb(p, msg):
            progress_bar.progress((i + p) / total, f"{display_name}: {msg}")

        if manager.download(ds_name, progress_callback=progress_cb):
            success_count += 1
        else:
            st.warning(f"Failed to download {display_name}")

    progress_bar.progress(1.0)
    if success_count == total:
        st.success(f"All {total} dataset(s) downloaded!")
    elif success_count > 0:
        st.warning(f"Downloaded {success_count}/{total} dataset(s)")
    else:
        st.error("Download failed for all datasets")

    status_text.empty()
    st.rerun()


def render_quality_test_panel(config, run_test_func):
    """
    Render quality test panel

    Args:
        config: Sidebar configuration dictionary
        run_test_func: Test execution function

    Returns:
        bool: Whether test was triggered
    """
    st.header("Model Quality Test")
    st.info("Use public benchmarks (MMLU, GSM8K, etc.) to evaluate model reasoning and knowledge.")

    # Lazy detect and load module
    if not _check_quality_eval():
        st.error(
            "Quality assessment module failed to load. Please check if the evaluators directory is complete."
        )
        return False

    quality_module = _get_quality_eval_module()

    # Initialize quality test status
    if "quality_results" not in st.session_state:
        st.session_state.quality_results = {}
    if "quality_test_running" not in st.session_state:
        st.session_state.quality_test_running = False

    available_datasets = list(quality_module["EVALUATOR_REGISTRY"].keys())

    with st.sidebar.expander("Quality Test Parameters", expanded=True):
        # Dataset status panel
        _render_dataset_status_panel(available_datasets)

        # Dataset selection
        st.markdown("**Select test datasets**")

        # Define dataset layout: (registry_key, display_name, default, help_text)
        _BASIC_DATASETS = [
            ("mmlu", "MMLU", True, "General knowledge (57 subjects)"),
            ("gsm8k", "GSM8K", True, "Elementary math reasoning"),
            ("math500", "MATH-500", False, "Advanced math competitions"),
            ("humaneval", "HumanEval", False, "Python code generation"),
            ("ceval", "C-Eval", False, "Chinese knowledge evaluation"),
        ]
        _REASONING_DATASETS = [
            ("gpqa", "GPQA", False, "Graduate-level scientific reasoning"),
            ("arc", "ARC-Challenge", False, "Scientific commonsense reasoning"),
            ("truthfulqa", "TruthfulQA", False, "Truthfulness testing"),
            ("winogrande", "WinoGrande", False, "Coreference resolution"),
            ("hellaswag", "HellaSwag", False, "Commonsense reasoning"),
        ]
        _CODE_EXTRA_DATASETS = [
            ("mbpp", "MBPP", False, "Python code generation (extended)"),
            ("aime2025", "AIME 2025", False, "Competition math (hard)"),
            (
                "aime2026",
                "AIME 2026",
                False,
                "Competition math (hard) - official GLM-5.2",
            ),
            ("longbench", "LongBench", False, "Long context understanding"),
            ("arena_hard", "Arena Hard", False, "Hard prompts evaluation"),
            ("global_piqa", "PIQA", False, "Physical reasoning"),
        ]
        _SPECIAL_DATASETS = [
            ("swebench_lite", "SWE-Bench Lite", False, "Software engineering"),
            ("needle_haystack", "Needle Haystack", False, "Long context retrieval"),
            ("custom_needle", "Custom Needle", False, "Custom retrieval tests"),
        ]

        selected_datasets = []

        def _render_dataset_group(group_key, datasets_list):
            cols = st.columns(3)
            for i, (key, name, default, help_text) in enumerate(datasets_list):
                if key in available_datasets:
                    with cols[i % 3]:
                        if st.checkbox(name, value=default, key=f"ds_{key}", help=help_text):
                            selected_datasets.append(key)

        # Basic Datasets
        _render_dataset_group("basic", _BASIC_DATASETS)

        # Advanced Datasets
        st.caption("Reasoning test sets")
        _render_dataset_group("reasoning", _REASONING_DATASETS)

        st.caption("Code & extended test sets")
        _render_dataset_group("code_extra", _CODE_EXTRA_DATASETS)

        st.caption("Special test sets")
        _render_dataset_group("special", _SPECIAL_DATASETS)

        # Evaluation Parameters
        st.markdown("---")
        st.markdown("**Evaluation Parameters**")

        model_type_label = st.selectbox(
            "Model Type",
            ["Standard Model", "Reasoning Model (CoT)", "Code Model"],
            index=0,
            key="quality_model_type",
        )

        model_type = "standard"
        if "Reasoning" in model_type_label:
            model_type = "thinking"
        elif "Code" in model_type_label:
            model_type = "code"

        # Sampling Mode
        sampling_mode = st.radio(
            "Sampling Mode",
            [
                "Quick sampling (100 questions/dataset)",
                "Medium sampling (500 questions)",
                "Custom sampling",
                "Full test (all)",
            ],
            key="quality_sampling_mode",
            horizontal=True,
        )

        # Dataset sample count mapping
        DATASET_SAMPLE_COUNTS = {
            "arc": 1167,
            "gpqa": 195,
            "gsm8k": 1314,
            "hellaswag": 10037,
            "humaneval": 164,
            "longbench": 4150,
            "math500": 496,
            "mbpp": 497,
            "mmlu": 14037,
            "needle_haystack": 30,
            "swebench_lite": 300,
            "truthfulqa": 812,
            "winogrande": 1262,
            "aime2025": 30,
            "aime2026": 30,
            "arena_hard": 500,
            "global_piqa": 2000,
            "custom_needle": 30,
            "ceval": 13948,
        }

        if sampling_mode == "Quick sampling (100 questions/dataset)":
            max_samples = 100
        elif sampling_mode == "Medium sampling (500 questions)":
            max_samples = 500
        elif sampling_mode == "Custom sampling":
            if selected_datasets:
                min_dataset_size = min(
                    DATASET_SAMPLE_COUNTS.get(ds, 10000) for ds in selected_datasets
                )
                max_allowed = min(min_dataset_size, 10000)
                st.caption(f"Min sample count for selected datasets: {min_dataset_size}")
            else:
                max_allowed = 10000
            max_samples = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=max_allowed,
                value=min(200, max_allowed),
                step=50,
                key="custom_sample_count",
            )
        else:
            max_samples = None

        # Few-shot Set
        default_shots = 0 if model_type == "thinking" else 5
        num_shots = st.slider("Few-shot examples", 0, 10, default_shots, key="quality_num_shots")

        # Detailed parameters
        with st.expander("Detailed Parameter Settings", expanded=True):
            quality_temperature = st.number_input(
                "Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1
            )
            quality_max_tokens = st.number_input(
                "Max Output Tokens",
                min_value=1,
                max_value=131072,
                value=8192,
                step=1024,
            )
            quality_concurrency = st.slider(
                "Concurrent requests",
                1,
                32,
                4,
                help="Recommended: Local model 1-2, API model 4-10",
            )

            st.markdown("---")
            quality_use_llm_judge = st.checkbox(
                "AI Judge (secondary verification)",
                value=False,
                key="quality_use_llm_judge",
                help="Use LLM to re-evaluate uncertain answers, reducing misjudgments",
            )
            quality_use_cache = st.checkbox(
                "Response caching",
                value=True,
                key="quality_use_cache",
                help="Cache API responses to avoid duplicate calls (7-day TTL)",
            )

            # Thinking model parameters (only relevant for reasoning models)
            if model_type == "thinking":
                st.markdown("---")
                st.markdown("**Reasoning Model Settings**")
                quality_thinking_budget = st.number_input(
                    "Thinking Token Budget",
                    min_value=256,
                    max_value=131072,
                    value=4096,
                    step=512,
                    key="quality_thinking_budget",
                    help="Maximum tokens for the model's internal reasoning chain",
                )
                quality_reasoning_effort = st.selectbox(
                    "Reasoning Effort",
                    ["low", "medium", "high"],
                    index=1,
                    key="quality_reasoning_effort",
                    help="Controls reasoning depth: low=fast, high=thorough",
                )
            else:
                quality_thinking_budget = 4096
                quality_reasoning_effort = "medium"

        # Start button
        start_quality_btn = st.button(
            "Start Quality Assessment",
            key="start_quality_btn",
            type="primary",
            disabled=st.session_state.quality_test_running or not selected_datasets,
        )

    # Main area
    if not selected_datasets:
        st.warning("Please select at least one test dataset in the left panel.")
        return False

    if start_quality_btn and selected_datasets:
        return _run_quality_test(
            config,
            selected_datasets,
            max_samples,
            num_shots,
            model_type,
            quality_temperature,
            quality_max_tokens,
            quality_concurrency,
            quality_use_llm_judge,
            quality_use_cache,
            quality_thinking_budget,
            quality_reasoning_effort,
        )

    # Display results
    if st.session_state.quality_results:
        st.markdown("---")
        quality_module["render_quality_report"](
            st.session_state.quality_results, config["model_id"], show_details=True
        )

    return False


def _run_quality_test(
    config,
    selected_datasets,
    max_samples,
    num_shots,
    model_type,
    temperature,
    max_tokens,
    concurrency,
    use_llm_judge=False,
    use_cache=True,
    thinking_budget=4096,
    reasoning_effort="medium",
):
    """Run quality test"""
    # Pre-check: ensure datasets are available
    _SYNTHETIC_DATASETS = {"needle_haystack", "custom_needle"}
    try:
        from core.dataset_manager import get_manager

        manager = get_manager()
        missing = [
            ds
            for ds in selected_datasets
            if ds not in _SYNTHETIC_DATASETS
            and ds in manager.configs
            and not manager.is_available(ds)
        ]
        if missing:
            missing_names = [_DATASET_DISPLAY_NAMES.get(d, d) for d in missing]
            st.warning(f"Missing datasets: {', '.join(missing_names)}. Attempting auto-download...")
            for ds_name in missing:
                display_name = _DATASET_DISPLAY_NAMES.get(ds_name, ds_name)
                with st.spinner(f"Downloading {display_name}..."):
                    ok = manager.download(ds_name)
                if ok:
                    st.success(f"Downloaded {display_name}")
                else:
                    st.error(f"Failed to download {display_name}. Skipping.")
            # Re-check and filter out still-missing datasets
            still_missing = [ds for ds in missing if not manager.is_available(ds)]
            if still_missing:
                selected_datasets = [ds for ds in selected_datasets if ds not in still_missing]
                if not selected_datasets:
                    st.error("No datasets available after download attempts.")
                    return False
    except Exception:
        pass  # DatasetManager not available, proceed without pre-check

    st.session_state.quality_test_running = True
    st.session_state.quality_results = {}

    # Get lazy-loaded module
    quality_module = _get_quality_eval_module()
    if not quality_module:
        st.error("Quality assessment module not available")
        return False

    # Display progress UI
    st.markdown("---")
    st.markdown("### Test in progress...")

    progress_col1, progress_col2 = st.columns([3, 1])
    with progress_col1:
        progress_bar = st.progress(0)
    with progress_col2:
        progress_text = st.empty()
        progress_text.markdown("**0%**")

    status_text = st.empty()
    status_text.info(f"Initializing quality assessment... Datasets: {', '.join(selected_datasets)}")

    try:
        # Build configuration
        from core.thinking_params import get_intelligent_preset

        thinking_enabled = model_type == "thinking"

        # For thinking models, try intelligent preset first, fall back to UI values
        if thinking_enabled:
            try:
                preset = get_intelligent_preset(config["api_base_url"], config["model_id"])
                # Use UI values as override, preset as starting point
                final_budget = thinking_budget or preset.get("thinking_budget", 4096)
                final_effort = reasoning_effort or preset.get("reasoning_effort", "medium")
            except Exception:
                final_budget = thinking_budget
                final_effort = reasoning_effort
        else:
            final_budget = thinking_budget
            final_effort = reasoning_effort

        quality_config = quality_module["QualityTestConfig"](
            datasets=selected_datasets,
            num_shots=num_shots,
            max_samples=max_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency=concurrency,
            model_type=model_type,
            thinking_enabled=thinking_enabled,
            thinking_budget=final_budget,
            reasoning_effort=final_effort,
            dataset_overrides={},
            use_llm_judge=use_llm_judge,
            use_cache=use_cache,
        )

        # Create evaluator
        evaluator = quality_module["QualityEvaluator"](
            api_base_url=config["api_base_url"],
            model_id=config["model_id"],
            api_key=config["api_key"],
            provider=config["provider"],
            output_dir="quality_results",
        )

        # Register evaluator classes
        for ds_name, evaluator_class in quality_module["EVALUATOR_REGISTRY"].items():
            evaluator.register_evaluator(ds_name, evaluator_class)

        # Progress callback
        def progress_callback(current, total, message):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            progress_text.markdown(f"**{progress*100:.1f}%**")
            status_text.info(message)

        # Run evaluation
        async def run_quality_eval():
            return await evaluator.run_evaluation(quality_config, progress_callback)

        # Create new event loop (fix Python 3.10+ non-main thread issue)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        import sniffio

        token = sniffio.current_async_library_cvar.set("asyncio")

        try:
            results = loop.run_until_complete(run_quality_eval())
        finally:
            sniffio.current_async_library_cvar.reset(token)
            loop.close()
            asyncio.set_event_loop(None)  # Cleanup event loop reference

        st.session_state.quality_results = results

        progress_bar.progress(1.0)
        progress_text.markdown("**100%**")

        total_samples = sum(r.total_samples for r in results.values())
        total_correct = sum(r.correct_samples for r in results.values())
        final_accuracy = total_correct / total_samples if total_samples > 0 else 0

        status_text.success(
            f"Quality assessment complete! Evaluated {len(results)} datasets, "
            f"{total_correct}/{total_samples} correct ({final_accuracy*100:.1f}%)"
        )

    except Exception as e:
        st.error(f"Quality assessment error: {e}")
    finally:
        st.session_state.quality_test_running = False

    return True


def render_ab_comparison_panel(config):
    """Render A/B model comparison panel"""
    st.header("A/B Model Comparison")
    st.info("Compare two models on the same test set with statistical significance analysis.")

    if not _check_enhanced_eval():
        st.error("A/B comparison module failed to load.")
        return False

    enhanced_module = _get_enhanced_eval_module()
    quality_module = _get_quality_eval_module()

    if "ab_comparison_running" not in st.session_state:
        st.session_state.ab_comparison_running = False
    if "ab_result" not in st.session_state:
        st.session_state.ab_result = None

    col_models, col_config = st.columns([1, 1])

    with col_models:
        st.subheader("Model Configuration")

        st.markdown("##### Model A (sidebar config)")
        st.success(f"ID: `{config['model_id']}`\n\nProvider: `{config['provider']}`")

        st.markdown("---")

        st.markdown("##### Model B (comparison target)")

        model_b_id = st.text_input("Model ID", key="ab_model_b_id")
        model_b_provider = st.selectbox(
            "Provider",
            ["openai", "anthropic", "gemini", "siliconflow", "together", "deepseek"],
            key="ab_model_b_provider",
        )
        model_b_base = st.text_input(
            "API Base URL", key="ab_model_b_base", value="https://api.openai.com/v1"
        )
        model_b_key = st.text_input("API Key", key="ab_model_b_key", type="password")

    with col_config:
        st.subheader("Test Configuration")

        available_datasets = (
            list(quality_module["EVALUATOR_REGISTRY"].keys()) if quality_module else []
        )
        st.markdown("**Select test datasets**")

        selected_datasets_ab = []
        dataset_cols = st.columns(2)
        for i, ds_name in enumerate(available_datasets):
            with dataset_cols[i % 2]:
                if st.checkbox(ds_name, key=f"ab_ds_{ds_name}"):
                    selected_datasets_ab.append(ds_name)

        ab_max_samples = st.number_input(
            "Samples per dataset", min_value=10, value=50, step=10, key="ab_max_samples"
        )
        ab_num_shots = st.number_input(
            "Number of few-shots", min_value=0, value=0, key="ab_num_shots"
        )

        start_ab_btn = st.button(
            "Start Comparison",
            type="primary",
            disabled=st.session_state.ab_comparison_running
            or not selected_datasets_ab
            or not model_b_id,
        )

        # Run comparison
        if start_ab_btn and selected_datasets_ab and model_b_id:
            return _run_ab_comparison(
                config,
                model_b_id,
                model_b_provider,
                model_b_base,
                model_b_key,
                selected_datasets_ab,
                ab_max_samples,
                ab_num_shots,
            )

        # Display results
        if st.session_state.ab_result:
            _display_ab_comparison_results(st.session_state.ab_result)

    return False


def _run_ab_comparison(
    config, model_b_id, provider, base_url, api_key, datasets, max_samples, num_shots
):
    """Run A/B comparison test"""
    st.session_state.ab_comparison_running = True
    st.session_state.ab_result = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        enhanced_module = _get_enhanced_eval_module()
        if not enhanced_module:
            st.error("Enhanced evaluation module not available")
            return False

        ModelComparator = enhanced_module["ModelComparator"]
        ModelConfig = enhanced_module["ModelConfig"]

        comparator = ModelComparator(output_dir=os.path.join("quality_results", "comparisons"))

        # Configure Model A
        config_a = ModelConfig(
            model_id=config["model_id"],
            api_base_url=config["api_base_url"],
            api_key=config["api_key"],
            provider=config["provider"],
            label=f"{config['model_id']} (A)",
        )
        comparator.add_model("model_a", config_a)

        # Configure Model B
        config_b = ModelConfig(
            model_id=model_b_id,
            api_base_url=base_url,
            api_key=api_key,
            provider=provider,
            label=f"{model_b_id} (B)",
        )
        comparator.add_model("model_b", config_b)

        def update_progress(p, msg):
            progress_bar.progress(p)
            status_text.info(msg)

        import asyncio

        result = asyncio.run(
            comparator.run_comparison(
                datasets=datasets,
                max_samples=max_samples,
                num_shots=num_shots,
                progress_callback=update_progress,
            )
        )

        st.session_state.ab_result = result
        st.success("Comparison complete!")

    except Exception as e:
        st.error(f"Comparison failed: {e}")
    finally:
        st.session_state.ab_comparison_running = False

    return True


def _display_ab_comparison_results(result):
    """Display A/B comparison results"""
    st.markdown("---")
    st.header("Comparison Report")

    # Summary table
    st.subheader("1. Overall Performance")

    summary_data = []
    for ds_name, ds_res in result.datasets.items():
        row = {"Dataset": ds_name}

        acc_a = ds_res.accuracies.get("model_a", 0)
        acc_b = ds_res.accuracies.get("model_b", 0)

        row[f"{result.model_labels['model_a']}"] = f"{acc_a:.1%}"
        row[f"{result.model_labels['model_b']}"] = f"{acc_b:.1%}"

        diff = acc_b - acc_a
        row["Difference (B-A)"] = f"{diff:+.1%}"

        if "model_a_vs_model_b" in ds_res.statistical_tests:
            test_res = ds_res.statistical_tests["model_a_vs_model_b"]
            sig = "Significant" if test_res.get("significant") else "Not significant"
            row["Statistical Significance"] = f"{sig} (p={test_res.get('p_value', 1.0):.3f})"
        else:
            row["Statistical Significance"] = "N/A"

        summary_data.append(row)

    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)


def render_advanced_eval_panel(config):
    """Render advanced evaluation analysis panel"""
    st.header("Advanced Evaluation Analysis")
    st.info(
        "Provides advanced evaluation tools: consistency testing, robustness testing, smart answer parser demo."
    )

    if not _check_enhanced_eval():
        st.error(
            "Enhanced evaluation module failed to load. Please check if core/ directory contains required modules."
        )
        return False

    enhanced_module = _get_enhanced_eval_module()

    adv_tabs = st.tabs(["Smart Parser Demo", "Consistency Test", "Robustness Test"])

    with adv_tabs[0]:
        st.subheader("Smart Answer Parser Demo")
        st.markdown(
            """
        Smart answer parser uses a layered strategy to extract answers:
        1. **Pattern matching** (fast) - Matches standard formats like `\\boxed{}`, `####`, `ANSWER: X`
        2. **Heuristic rules** - Pattern matching like "answer is X", "答案是X"
        3. **Symbolic comparison** (math) - SymPy equivalence for math expressions
        """
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            demo_response = st.text_area(
                "Enter model response",
                value="Let me solve this step by step.\n\n15 + 27 = 42\n\nTherefore, the answer is \\boxed{42}.",
                height=150,
                key="smart_parse_input",
            )

        with col2:
            demo_dataset_type = st.selectbox(
                "Dataset Type",
                [
                    "auto",
                    "mmlu",
                    "gsm8k",
                    "math500",
                    "humaneval",
                    "gpqa",
                    "truthfulqa",
                    "longbench",
                ],
                key="smart_parse_dataset_type",
                help="Select the dataset type to use the appropriate parser. 'auto' tries all parsers.",
            )
            demo_correct = st.text_input(
                "Correct answer (optional)", "42", key="smart_parse_correct"
            )

        if st.button("Parse Answer", key="smart_parse_btn"):
            try:
                from evaluators.answer_parser import MathAnswerParser, get_parser_for_dataset

                parser = get_parser_for_dataset(demo_dataset_type)
                extracted = parser.parse(demo_response)

                # Determine parse method name
                parser_name = type(parser).__name__

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Extracted Answer", extracted or "N/A")
                with col2:
                    st.metric("Parser Type", parser_name)
                with col3:
                    st.metric("Dataset", demo_dataset_type)

                # Show comparison result if correct answer provided
                if demo_correct and extracted:
                    is_match = False
                    if isinstance(parser, MathAnswerParser):
                        is_match = MathAnswerParser.check_answer(extracted, demo_correct)
                    else:
                        from evaluators.answer_parser import TextAnswerParser

                        is_match = TextAnswerParser.check_answer(extracted, demo_correct)

                    if is_match:
                        st.success("Answer matches!")
                    else:
                        st.error(
                            f"Answer does not match. Expected: {demo_correct}, Got: {extracted}"
                        )

            except Exception as e:
                st.error(f"Parse failed: {e}")

    with adv_tabs[1]:
        st.subheader("Consistency Test")
        st.info(
            "Consistency testing queries the same question multiple times to evaluate model answer stability."
        )

        col1, col2 = st.columns(2)
        with col1:
            cons_runs = st.slider("Tests per question", 2, 10, 5, key="cons_runs")
            cons_threshold = st.slider("Stability threshold", 0.5, 1.0, 0.8, key="cons_threshold")
        with col2:
            st.text_input("Test question", "What is 15 + 27?", key="cons_question")
            st.text_input("Correct answer", "42", key="cons_answer")

        st.info(
            f"Will run question {cons_runs} times, consistency rate ≥ {cons_threshold*100:.0f}% considered stable"
        )

        if st.button("Start Consistency Test", key="cons_start_btn"):
            st.info(
                "Consistency testing requires API connection. Please configure API in the left panel first."
            )

    with adv_tabs[2]:
        st.subheader("Robustness Test")
        st.info("Robustness testing evaluates model sensitivity to input perturbations.")

        from core.robustness_tester import PerturbationType, TextPerturber

        perturber = TextPerturber()

        st.markdown("#### Perturbation Demo")

        robust_input = st.text_input(
            "Enter original question",
            "If you have 1000 apples and buy 500 more, how many do you have in total?",
            key="robust_input",
        )

        robust_ptype = st.selectbox(
            "Perturbation Type", [p.value for p in PerturbationType], key="robust_ptype"
        )

        if st.button("Apply Perturbation", key="robust_perturb_btn"):
            result = perturber.perturb(robust_input, PerturbationType(robust_ptype))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original question:**")
                st.code(result.original_question)
            with col2:
                st.markdown("**Perturbed question:**")
                st.code(result.perturbed_question)

            st.info(f"Perturbation details: {result.perturbation_details}")

    return False
