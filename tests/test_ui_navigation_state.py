"""
Regression tests for Streamlit test-type navigation state.
"""

import pytest


@pytest.fixture
def clean_streamlit_state():
    import streamlit as st

    st.session_state._data.clear()
    return st.session_state


def test_init_session_state_normalizes_legacy_prefill_test_type(clean_streamlit_state):
    clean_streamlit_state.current_test_type = "prefill"

    from config import session_state

    session_state.init_session_state()

    assert clean_streamlit_state._data["current_test_type"] == "prefill"
    assert "test_type_selector" not in clean_streamlit_state._data


def test_apply_prefill_preset_updates_selectbox_key(
    clean_streamlit_state, tmp_path, monkeypatch
):
    from utils import test_config_manager

    manager = test_config_manager.TestConfigManager(str(tmp_path))
    manager.save_preset(
        test_config_manager.ConfigPreset(
            name="Prefill",
            description="Prefill preset",
            config={
                "test_type": "prefill",
                "max_tokens": 1,
            },
        )
    )
    monkeypatch.setattr(test_config_manager, "config_manager", manager)

    assert test_config_manager.apply_preset("Prefill")

    assert clean_streamlit_state._data["current_test_type"] == "prefill"
    assert clean_streamlit_state._data["_force_test_type_selector"] == "prefill"
    assert clean_streamlit_state._data["prefill_isolation_mode"] is True


def test_pending_prefill_forces_selector_back_before_rerun(
    clean_streamlit_state, monkeypatch
):
    from ui import test_panels

    class RerunCalled(Exception):
        pass

    clean_streamlit_state["_pending_test"] = {
        "test_type": "prefill",
        "test_func": object(),
        "runner_class": object(),
        "args": (),
    }

    def fake_rerun():
        raise RerunCalled

    monkeypatch.setattr(test_panels.st, "rerun", fake_rerun)
    clean_streamlit_state.test_type_selector = "concurrency"

    with pytest.raises(RerunCalled):
        test_panels._execute_pending_test(lambda *args: None, "concurrency")

    assert clean_streamlit_state._data["current_test_type"] == "prefill"
    assert clean_streamlit_state._data["test_type_selector"] == "concurrency"
    assert clean_streamlit_state._data["_force_test_type_selector"] == "prefill"
    assert clean_streamlit_state._data["_pending_test"]["test_type"] == "prefill"


def test_can_update_current_test_type_without_rewriting_widget_key(
    clean_streamlit_state,
):
    from config import session_state

    clean_streamlit_state.test_type_selector = "prefill"

    selected = session_state.set_current_test_type(
        "concurrency",
        sync_widget_key=False,
    )

    assert selected == "concurrency"
    assert clean_streamlit_state._data["current_test_type"] == "concurrency"
    assert clean_streamlit_state._data["test_type_selector"] == "prefill"


def test_forced_test_type_updates_selector_before_widget_creation(
    clean_streamlit_state,
):
    from config import session_state

    clean_streamlit_state["test_type_selector_widget_0"] = "concurrency"
    clean_streamlit_state["_force_test_type_selector"] = "prefill"

    forced_test_type = clean_streamlit_state.get("_force_test_type_selector")
    clean_streamlit_state["_force_test_type_selector"] = None
    clean_streamlit_state["_test_type_selector_version"] = (
        clean_streamlit_state.get("_test_type_selector_version", 0) + 1
    )
    widget_key = (
        f"test_type_selector_widget_"
        f"{clean_streamlit_state.get('_test_type_selector_version', 0)}"
    )
    selected = session_state.set_current_test_type(
        forced_test_type
        or clean_streamlit_state.get(widget_key)
        or clean_streamlit_state.get("current_test_type"),
        ["concurrency", "prefill"],
        sync_widget_key=True,
    )

    assert selected == "prefill"
    assert clean_streamlit_state._data["current_test_type"] == "prefill"
    assert clean_streamlit_state._data["test_type_selector_widget_1"] == "prefill"
    assert clean_streamlit_state._data["_force_test_type_selector"] is None
