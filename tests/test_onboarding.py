"""Tests for onboarding state transitions."""

import pytest

from ui import onboarding


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture(autouse=True)
def reset_onboarding_state():
    onboarding.st.session_state._data.clear()
    yield
    onboarding.st.session_state._data.clear()


def test_skip_onboarding_updates_state_without_interrupting_render(monkeypatch):
    state = onboarding.init_onboarding_state()
    onboarding.st.session_state.show_onboarding_guide = True

    def button(label, *args, **kwargs):
        if label.startswith("Skip"):
            on_click = kwargs.get("on_click")
            if on_click is not None:
                on_click()
                return False
            return True
        return False

    def rerun():
        raise AssertionError("Skip onboarding must not interrupt the current render cycle")

    monkeypatch.setattr(onboarding.st, "container", lambda *args, **kwargs: _Context())
    monkeypatch.setattr(onboarding.st, "columns", lambda count, *args, **kwargs: [_Context() for _ in range(count)])
    monkeypatch.setattr(onboarding.st, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(onboarding.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(onboarding.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(onboarding.st, "button", button)
    monkeypatch.setattr(onboarding.st, "rerun", rerun)

    onboarding.render_onboarding_modal()

    assert state.show_onboarding is False
