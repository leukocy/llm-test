"""
Onboarding System Module

Provides first-time use help and guidance:
- Compact welcome banner (non-blocking)
- Inline step-by-step tutorial
- Quick Start Guide
- FAQ
"""

import json
import os

import streamlit as st

from ui.design_system import material_icon

# ----------------------------------------------------------------------------
# Onboarding State Management
# ----------------------------------------------------------------------------

_ONBOARDING_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".onboarding_state"
)

def _load_onboarding_file():
    """Load persistent onboarding state from disk."""
    try:
        with open(_ONBOARDING_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_onboarding_file(data):
    """Save persistent onboarding state to disk."""
    with open(_ONBOARDING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


class OnboardingState:
    """Onboarding state class."""

    STEPS = [
        "welcome",
        "api_config",
        "test_selection",
        "run_test",
        "results",
        "complete",
    ]

    def __init__(self):
        self.current_step = 0
        self.show_onboarding = True

    def next_step(self):
        if self.current_step < len(self.STEPS) - 1:
            self.current_step += 1

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1

    def skip(self):
        self.show_onboarding = False

    @property
    def current_step_name(self) -> str:
        return self.STEPS[self.current_step]

    @property
    def progress(self) -> float:
        return (self.current_step + 1) / len(self.STEPS)


def init_onboarding_state():
    """Initialize onboarding state (session + persistent file)."""
    file_state = _load_onboarding_file()
    dismissed = file_state.get("dismissed", False)

    if "onboarding" not in st.session_state:
        state = OnboardingState()
        if dismissed:
            state.show_onboarding = False
        st.session_state.onboarding = state

    state = st.session_state.onboarding
    # Do not override when user actively reopened the guide via sidebar
    if not st.session_state.get("show_onboarding_guide", False):
        if st.session_state.get("onboarding_completed", False) or dismissed:
            state.show_onboarding = False

    return state


def _dismiss_onboarding(permanently: bool = True):
    """Dismiss onboarding. Optionally writes persistent file."""
    state = init_onboarding_state()
    state.show_onboarding = False
    st.session_state.onboarding_completed = True
    st.session_state.show_onboarding_guide = False
    if permanently:
        _save_onboarding_file({"dismissed": True, "completed": True})


# ============================================================================
# Onboarding Content
# ============================================================================

ONBOARDING_CONTENT = {
    "welcome": {
        "title": "Welcome to the LLM Benchmark Platform",
        "content": """
        ### This is a professional LLM performance testing tool

        **Main Features:**

        - **Concurrency Test** - Test model performance at different concurrency levels
        - **Prefill Stress Test** - Evaluate model long input processing capability
        - **Long Context Test** - Test model performance in long context scenarios
        - **Matrix Test** - Multi-dimensional comprehensive model evaluation
        - **Custom Test** - Test using custom text
        - **All Tests** - Run all test types with one click

        **Test Metrics:**

        - **TTFT** (Time To First Token) - First token latency
        - **TPS** (Tokens Per Second) - Generation speed
        - **Token usage** - Input/output token statistics
        - **Total duration** - Complete request duration
        """,
        "tips": [
            "Tip: You can reopen this guide anytime from the sidebar",
            "Tip: Test results are automatically saved and can be viewed in History"
        ]
    },

    "api_config": {
        "title": "Configure the API connection",
        "content": """
        ### Step 1:Configure your API connection

        **In the left sidebar:**

        1. **Select API Provider**
           - OpenAI
           - Gemini
           - or other compatible providers

        2. **Fill in API information**
           - API Base URL: API endpoint address
           - Model ID: Model name to test
           - API Key: Your API key

        3. **Configure Tokenizer** (optional)
           - Select HuggingFace Tokenizer for precise counting
           - or use API-returned token counts

        **Security tips:**
        - API Key is only stored in the local session and is never uploaded
        - Use a restricted API Key for testing when possible
        """,
        "tips": [
            "Tip: Fetch the model list to validate the API connection",
            "Tip: Common configurations can be saved as presets for future use"
        ]
    },

    "test_selection": {
        "title": "Select a test type",
        "content": """
        ### Step 2:Choose the test to run

        **Test Type Description:**

        | Test Type | Description | Use Case |
        |---------|------|---------|
        | Concurrency Test | Test performance at different concurrency levels | Evaluate concurrency processing capability |
        | Prefill Stress Test | Test long input processing capability | Evaluate prefill performance |
        | Long Context Test | Test long context processing | Evaluate long text comprehension |
        | Matrix Test | Concurrency + context combination test | Comprehensive performance evaluation |
        | Custom Test | Use custom text | Specific scenario test |

        **Parameter Configuration:**
        - Concurrency: Number of simultaneous requests
        - Max Tokens: Maximum generated tokens
        - Temperature: controls output randomness
        - Thinking mode: enable model reasoning
        """,
        "tips": [
            "Tip: First-time users should start with the 'Quick Test' preset",
            "Tip: Stress tests consume more API quota"
        ]
    },

    "run_test": {
        "title": "Run a test",
        "content": """
        ### Step 3:Start Test

        **In the unified test control panel:**

        1. **Confirm configuration** - Review configuration summary to ensure correct parameters
        2. **Start Test** - Click the primary run button
        3. **Monitor progress** - View test progress and status in real-time

        **Test control:**
        - **Stop** - Interrupt the current test
        - **Resume** - Continue from saved progress

        **Real-time Monitoring:**
        - Progress bar shows current test progress
        - Real-time logs show status of each request
        - Dynamic charts display performance metrics
        """,
        "tips": [
            "Tip: You can stop the test without losing completed results",
            "Tip: Run a small test first to validate the configuration"
        ]
    },

    "results": {
        "title": "View results",
        "content": """
        ### Step 4:Analyze Test Results

        **Results Display:**

        1. **Data Table**
           - Detailed test results data
           - Supports sorting and filtering
           - Can be exported as CSV file

        2. **Visualization Charts**
           - Performance trend charts
           - Concurrency performance comparison
           - Token usage distribution

        3. **Test Report**
           - Automatically generates performance analysis reports
           - Includes key metrics interpretation
           - Can be exported as Markdown

        4. **Result Comparison**
           - Compare different model performances
           - Compare different configuration effects
        """,
        "tips": [
            "Tip: Use result comparison to compare multiple tests",
            "Tip: History is automatically saved in the project directory"
        ]
    },

    "complete": {
        "title": "Setup complete",
        "content": """
        ### You have completed the onboarding!

        **You have learned:**

        - Configure the API connection
        - Select a test type
        - Run a test
        - View and analyze results

        **Next:**

        - Try different test configurations
        - Save common configurations as presets
        - Compare model performance
        - Review detailed results

        **Need help?**

        - Click the sidebar help icon to reopen this guide anytime
        - Check the project README for more information
        """,
        "tips": [
            "You are ready to start benchmarking.",
            "Tip: Saving common configurations can significantly improve testing efficiency"
        ]
    }
}


# ============================================================================
# Onboarding UI Components
# ============================================================================

def _show_previous_onboarding_step():
    """Move onboarding to the previous step."""
    init_onboarding_state().prev_step()


def _show_next_onboarding_step():
    """Move onboarding to the next step."""
    init_onboarding_state().next_step()


def _complete_onboarding():
    """Mark onboarding as completed (persistent)."""
    _dismiss_onboarding(permanently=True)


def _skip_onboarding():
    """Skip current onboarding guide (non-persistent)."""
    state = init_onboarding_state()
    state.show_onboarding = False
    st.session_state.show_onboarding_guide = False


def _reset_onboarding():
    """Re-open onboarding guide at step 0."""
    state = init_onboarding_state()
    state.show_onboarding = True
    state.current_step = 0
    st.session_state.show_onboarding_guide = True


def render_compact_welcome_banner():
    """Render a compact non-blocking welcome banner.

    Returns True if the banner was rendered (i.e. onboarding is not dismissed).
    """
    state = init_onboarding_state()
    if not state.show_onboarding:
        return False

    with st.container():
        st.markdown(
            """
            <div class="welcome-banner">
                <h4>Welcome to LLM Benchmark Platform</h4>
                <p>A professional LLM performance &amp; quality testing tool. Not familiar? View the guide or get started directly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button(
                "Quick guide",
                use_container_width=True,
                key="btn_open_guide",
                icon=material_icon("menu_book"),
            ):
                st.session_state.show_onboarding_guide = True
                st.rerun()
        with c2:
            if st.button(
                "Dismiss",
                use_container_width=True,
                key="btn_dismiss_guide",
                icon=material_icon("close"),
            ):
                _dismiss_onboarding(permanently=True)
                st.rerun()

    return True


def render_onboarding_guide():
    """Render inline onboarding guide (not a blocking modal)."""
    if not st.session_state.get("show_onboarding_guide", False):
        return

    state = init_onboarding_state()
    if not state.show_onboarding:
        return

    with st.container(border=True):
        # Header with progress
        st.progress(state.progress, text=f"Step {state.current_step + 1} / {len(state.STEPS)}")

        # Current step content
        step_name = state.current_step_name
        content = ONBOARDING_CONTENT.get(step_name, ONBOARDING_CONTENT["welcome"])

        st.markdown(f"#### {content['title']}")
        st.markdown(content["content"])

        if content.get("tips"):
            st.info("\n".join(content["tips"]))

        # Navigation buttons
        prev_col, next_col, skip_col = st.columns(3)
        with prev_col:
            if state.current_step > 0:
                st.button(
                    "Previous",
                    on_click=_show_previous_onboarding_step,
                    key="onb_prev",
                    icon=material_icon("arrow_back"),
                )
        with next_col:
            if state.current_step < len(OnboardingState.STEPS) - 1:
                st.button(
                    "Next",
                    type="primary",
                    on_click=_show_next_onboarding_step,
                    key="onb_next",
                    icon=material_icon("arrow_forward"),
                    icon_position="right",
                )
            else:
                st.button(
                    "Done",
                    type="primary",
                    on_click=_complete_onboarding,
                    key="onb_done",
                    icon=material_icon("check"),
                )
        with skip_col:
            st.button("Skip", on_click=_skip_onboarding, key="onb_skip")


def render_onboarding_modal():
    """Legacy modal — now routes to inline guide. Kept for backward compat."""
    render_onboarding_guide()


def render_onboarding_trigger():
    """Render onboarding trigger in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        if st.button(
            "Help and guide",
            use_container_width=True,
            key="sidebar_help",
            icon=material_icon("help"),
        ):
            _reset_onboarding()
            st.rerun()


def render_quick_reference():
    """Render quick reference card"""
    with st.container(border=True):
        st.subheader("Quick start")
        st.markdown(
            "1. Configure the API connection in the sidebar.\n"
            "2. Select a test type.\n"
            "3. Configure the workload and use the primary run action.\n"
            "4. Review the results and generated report."
        )


def render_feature_highlights():
    """Render feature highlights"""
    with st.expander("Feature highlights", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Precise testing**
            - Supports 7 test types
            - Multi-dimensional performance metrics
            - Reproducible test environment
            """)

            st.markdown("""
            **Deep analysis**
            - Real-time progress monitoring
            - Visualization chart display
            - Automatic report generation
            """)

        with col2:
            st.markdown("""
            **Smart management**
            - Configuration preset saving
            - Test progress persistence
            - History query
            """)

            st.markdown("""
            **Flexible configuration**
            - Multi API provider support
            - Custom test parameters
            - Thinking mode testing
            """)


def render_faqs():
    """Render FAQ"""
    with st.expander("FAQ", expanded=False):
        faqs = [
            {
                "q": "How to choose the right concurrency?",
                "a": """
                Recommended to start with low concurrency:
                - **Quick test**: Concurrency 1-2
                - **Standard test**: Concurrency 4-8
                - **Stress test**: Concurrency 16+

                Note: High concurrency consumes more API quota.
                """
            },
            {
                "q": "What to do when tests fail?",
                "a": """
                Common causes and solutions:
                - **401 Error**: Check if API Key is correct
                - **429 Error**: Lower concurrency or retry later
                - **ConnectError**: Check network and API URL
                - **Timeout**: Increase max_tokens or lower concurrency
                """
            },
            {
                "q": "How to compare different model performances?",
                "a": """
                Use the result comparison feature:
                1. Run tests on multiple models
                2. Select results to compare on the comparison page
                3. View comparison charts and reports
                """
            },
            {
                "q": "Where is Test data saved?",
                "a": """
                Test results are saved in:
                - `benchmark_results*.csv` - Test data
                - `test_progress/` - Test progress
                - `test_presets/` - Configuration presets

                These files are in the project directory and can be safely backed up or deleted.
                """
            }
        ]

        for i, faq in enumerate(faqs):
            with st.expander(f"Q: {faq['q']}", expanded=False):
                st.markdown(f"A: {faq['a']}")


# ============================================================================
# Helper Functions
# ============================================================================

def show_onboarding():
    """Display onboarding (if not completed)"""
    state = init_onboarding_state()
    return state.show_onboarding


def reset_onboarding():
    """Reset onboarding state (including persistent file)."""
    if "onboarding" in st.session_state:
        del st.session_state.onboarding
    st.session_state.onboarding_completed = False
    st.session_state.show_onboarding_guide = False
    try:
        os.remove(_ONBOARDING_FILE)
    except FileNotFoundError:
        pass


def skip_onboarding():
    """Skip Onboarding"""
    _skip_onboarding()


def is_onboarding_completed() -> bool:
    """Check if onboarding is completed"""
    return st.session_state.get("onboarding_completed", False)
