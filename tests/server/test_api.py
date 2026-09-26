"""Configuration and admission tests for the replacement control API."""

import json
from pathlib import Path

import pytest

from server.settings import Settings
from server.specs import JobSubmission


def test_settings_fail_closed_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = tmp_path / "endpoints.json"
    config.write_text(
        json.dumps(
            [
                {
                    "id": "lab",
                    "label": "Lab",
                    "provider": "OpenAI",
                    "api_base_url": "http://127.0.0.1:9010/v1",
                    "model_id": "org/model",
                    "api_key_env": "LAB_KEY",  # pragma: allowlist secret
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_TEST_ENDPOINTS_FILE", str(config))
    monkeypatch.delenv("LLM_TEST_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="LLM_TEST_API_TOKEN"):
        Settings.from_env()
    monkeypatch.setenv("LLM_TEST_API_TOKEN", "a" * 40)
    assert Settings.from_env().endpoints["lab"].model_id == "org/model"


def test_settings_reject_path_escape_in_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = tmp_path / "endpoints.json"
    config.write_text(
        json.dumps(
            [
                {
                    "id": "lab",
                    "label": "Lab",
                    "provider": "OpenAI",
                    "api_base_url": "https://api.example.com/v1",
                    "model_id": "../escape",
                    "api_key_env": "LAB_KEY",  # pragma: allowlist secret
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_TEST_ENDPOINTS_FILE", str(config))
    monkeypatch.setenv("LLM_TEST_API_TOKEN", "a" * 40)
    with pytest.raises(ValueError, match="Invalid model ID"):
        Settings.from_env()


def test_run_spec_rejects_unbounded_and_unknown_parameters():
    with pytest.raises(ValueError):
        JobSubmission(
            endpoint_id="lab",
            test_type="concurrency",
            parameters={
                "selected_concurrencies": [128],
                "rounds_per_level": 20,
                "max_tokens": 1024,
            },
        )
    with pytest.raises(ValueError):
        JobSubmission(
            endpoint_id="lab",
            test_type="prefill",
            parameters={
                "token_levels": [100],
                "requests_per_level": 1,
                "max_tokens": 10,
                "api_key": "leak",  # pragma: allowlist secret
            },
        )
