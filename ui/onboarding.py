"""
Onboarding System Module

Provides first-time use help and guidance:
- Feature Introduction
- Step-by-step Tutorial
- Quick Start Guide
- FAQ
"""

from typing import Callable, Optional

import streamlit as st


# ============================================================================
# Onboarding State Management
# ============================================================================

class OnboardingState:
    """Onboarding state class"""

    # Onboarding step definitions
    STEPS = [
        "welcome",
        "api_config",
        "test_selection",
        "run_test",
        "results",
        "complete"
    ]

    def __init__(self):
        self.current_step = 0
        self.show_onboarding = True

    def next_step(self):
        """Next"""
        if self.current_step < len(self.STEPS) - 1:
            self.current_step += 1

    def prev_step(self):
        """Previous"""
        if self.current_step > 0:
            self.current_step -= 1

    def skip(self):
        """Skip Onboarding"""
        self.show_onboarding = False

    @property
    def current_step_name(self) -> str:
        """Get current step name"""
        return self.STEPS[self.current_step]

    @property
    def progress(self) -> float:
        """Get onboarding progress"""
        return (self.current_step + 1) / len(self.STEPS)


def init_onboarding_state():
    """Initialize onboarding state"""
    if "onboarding" not in st.session_state:
        st.session_state.onboarding = OnboardingState()

    # Check if onboarding is completed
    if st.session_state.get("onboarding_completed", False):
        st.session_state.onboarding.show_onboarding = False

    return st.session_state.onboarding


# ============================================================================
# Onboarding Content
# ============================================================================

ONBOARDING_CONTENT = {
    "welcome": {
        "title": "👋 Welcome to the LLM Benchmark Platform",
        "content": """
        ### This is a professional LLM performance testing tool

        **Main Features:**

        - ⚡ **Concurrency Test** - Test model performance at different concurrency levels
        - 🔥 **Prefill Stress Test** - Evaluate model long input processing capability
        - 📏 **Long Context Test** - Test model performance in long context scenarios
        - 🔬 **Matrix Test** - Multi-dimensional comprehensive model evaluation
        - 📄 **Custom Test** - Test using custom text
        - 🎯 **All Tests** - Run all test types with one click

        **Test Metrics:**

        - 🚀 **TTFT** (Time To First Token) - First token latency
        - ⚡ **TPS** (Tokens Per Second) - Generation speed
        - 💾 **Token usage** - Input/output token statistics
        - ⏱️ **Total duration** - Complete request duration
        """,
        "tips": [
            "💡 Tip: You can reopen this guide anytime from the sidebar",
            "💡 Tip: Test results are automatically saved and can be viewed in History"
        ]
    },

    "api_config": {
        "title": "🔧 Configure API Connect",
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
        - 🔒 API Key is only stored in local session, never uploaded
        - 🔒 Recommend using a restricted API Key for testing
        """,
        "tips": [
            "💡 Tip: Click 'Get Model List' to validate API connection",
            "💡 Tip: Common configurations can be saved as presets for future use"
        ]
    },

    "test_selection": {
        "title": "🎯 Select Test Type",
        "content": """
        ### Step 2:Choose the test to run

        **Test Type Description:**

        | Test Type | Description | Use Case |
        |---------|------|---------|
        | ⚡ Concurrency Test | Test performance at different concurrency levels | Evaluate concurrency processing capability |
        | 🔥 Prefill Stress Test | Test long input processing capability | Evaluate prefill performance |
        | 📏 Long Context Test | Test long context processing | Evaluate long text comprehension |
        | 🔬 Matrix Test | Concurrency + context combination test | Comprehensive performance evaluation |
        | 📄 Custom Test | Use custom text | Specific scenario test |

        **Parameter Configuration:**
        - Concurrency: Number of simultaneous requests
        - Max Tokens: Maximum generated tokens
        - Temperature: controls output randomness
        - Thinking mode: enable model reasoning
        """,
        "tips": [
            "💡 Tip: First-time users should start with the 'Quick Test' preset",
            "💡 Tip: Stress tests consume more API quota"
        ]
    },

    "run_test": {
        "title": "🚀 Run Test",
        "content": """
        ### Step 3:Start Test

        **In the unified test control panel:**

        1. **Confirm configuration** - Review configuration summary to ensure correct parameters
        2. **Start Test** - Click the '🚀 Start Test' button
        3. **Monitor progress** - View test progress and status in real-time

        **Test control:**
        - ⏹️ Stop - Interrupt current test
        - ▶️ Resume - Continue from where it was stopped

        **Real-time Monitoring:**
        - Progress bar shows current test progress
        - Real-time logs show status of each request
        - Dynamic charts display performance metrics
        """,
        "tips": [
            "💡 Tip: You can stop the test anytime without losing saved results",
            "💡 Tip: Recommend running a small test first to validate configuration"
        ]
    },

    "results": {
        "title": "📊 View Results",
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
            "💡 Tip: Use result comparison to visually compare multiple tests",
            "💡 Tip: History is automatically saved in the project directory"
        ]
    },

    "complete": {
        "title": "🎉 Congratulations!",
        "content": """
        ### You have completed the onboarding!

        **You have learned:**

        - ✅ Configure API Connect
        - ✅ Select Test Type
        - ✅ Run Test
        - ✅ View and analyze results

        **Next:**

        - 🚀 Try different test configurations
        - 📁 Save your common configurations as presets
        - 📊 Compare different model performances
        - 🔬 Deep dive into test results

        **Need help?**

        - Click the sidebar help icon to reopen this guide anytime
        - Check the project README for more information
        """,
        "tips": [
            "🎉 Happy testing!",
            "💡 Tip: Saving common configurations can significantly improve testing efficiency"
        ]
    }
}


# ============================================================================
# Onboarding UI Components
# ============================================================================

def render_onboarding_modal():
    """Render onboarding modal"""
    state = init_onboarding_state()

    if not state.show_onboarding:
        return

    # Create modal
    with st.container():
        st.markdown("---")

        # Progress bar
        st.progress(state.progress)

        # Current step content
        step_name = state.current_step_name
        content = ONBOARDING_CONTENT.get(step_name, ONBOARDING_CONTENT["welcome"])

        # Title and content
        st.markdown(f"### {content['title']}")
        st.markdown(content['content'])

        # Tip
        if content.get('tips'):
            st.info("\n".join(content['tips']))

        # Navigation buttons
        col_prev, col_next, col_skip = st.columns(3)

        with col_prev:
            if state.current_step > 0:
                if st.button("⬅️ Previous"):
                    state.prev_step()
                    st.rerun()

        with col_next:
            if state.current_step < len(OnboardingState.STEPS) - 1:
                if st.button("Next ➡️", type="primary"):
                    state.next_step()
                    st.rerun()
            else:
                if st.button("Done ✅", type="primary"):
                    state.show_onboarding = False
                    st.session_state.onboarding_completed = True
                    st.rerun()

        with col_skip:
            if st.button("Skip Onboarding ⏭️"):
                state.skip()
                st.rerun()

        st.markdown("---")


def render_onboarding_trigger():
    """Render onboarding trigger (sidebar)"""
    with st.sidebar:
        st.markdown("---")

        # Display trigger button
        if st.button("❓ Help/Guide", use_container_width=True):
            # Reset onboarding state
            if "onboarding" in st.session_state:
                st.session_state.onboarding.show_onboarding = True
                st.session_state.onboarding.current_step = 0
            st.rerun()


def render_quick_reference():
    """Render quick reference card"""
    st.markdown("""
    <style>
    .quick-ref {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        margin-bottom: 20px;
    }
    .quick-ref h3 {
        color: white;
        margin-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="quick-ref">
            <h3>🚀 Quick Start</h3>
            <p><strong>1.</strong> Configure API on the left</p>
            <p><strong>2.</strong> Select Test Type</p>
            <p><strong>3.</strong> Click 'Start Test'</p>
            <p><strong>4.</strong> View and analyze results</p>
            <p style="margin-bottom: 0;">👆 need <a href="#" onclick="Streamlit.setComponentValue('onboarding_trigger', true)">detailed help</a>?</p>
        </div>
        """, unsafe_allow_html=True)


def render_feature_highlights():
    """Render feature highlights"""
    with st.expander("✨ Feature Highlights", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🎯 Precise Testing**
            - Supports 7 test types
            - Multi-dimensional performance metrics
            - Reproducible test environment
            """)

            st.markdown("""
            **📊 Deep Analysis**
            - Real-time progress monitoring
            - Visualization chart display
            - Automatic report generation
            """)

        with col2:
            st.markdown("""
            **💾 Smart Management**
            - Configuration preset saving
            - Test progress persistence
            - History query
            """)

            st.markdown("""
            **🔧 Flexible Configuration**
            - Multi API provider support
            - Custom test parameters
            - Thinking mode testing
            """)


def render_faqs():
    """Render FAQ"""
    with st.expander("❓ FAQ", expanded=False):
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
    """Reset onboarding state"""
    if "onboarding" in st.session_state:
        del st.session_state.onboarding
    st.session_state.onboarding_completed = False


def skip_onboarding():
    """Skip Onboarding"""
    state = init_onboarding_state()
    state.skip()


def is_onboarding_completed() -> bool:
    """Check if onboarding is completed"""
    return st.session_state.get("onboarding_completed", False)
