"""Configuration and admission tests for the replacement control API."""

import json
from pathlib import Path

import pytest

from server.settings import Settings
from server.specs import JobSubmission, measurement_plan


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


def test_settings_allow_empty_file_for_ui_onboarding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = tmp_path / "endpoints.json"
    config.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("LLM_TEST_ENDPOINTS_FILE", str(config))
    monkeypatch.setenv("LLM_TEST_API_TOKEN", "a" * 40)
    assert Settings.from_env().endpoints == {}


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
                "selected_concurrencies": [1025],
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


def test_measurement_plan_counts_warmup_separately_and_enforces_total_budget():
    body = JobSubmission(
        endpoint_id="lab",
        test_type="concurrency",
        parameters={
            "selected_concurrencies": [1, 4],
            "rounds_per_level": 3,
            "warmup_rounds_per_level": 1,
            "max_tokens": 32,
        },
    )
    plan = measurement_plan(body.test_type, body.parameters)
    assert plan["measured_requests"] == 15
    assert plan["warmup_requests"] == 5
    assert [cell["measured_requests"] for cell in plan["cells"]] == [3, 12]

    with pytest.raises(ValueError, match="100000 requests"):
        JobSubmission(
            endpoint_id="lab",
            test_type="matrix",
            parameters={
                "concurrencies": [1024],
                "context_lengths": [512, 1024, 2048, 4096, 8192],
                "rounds": 20,
                "max_tokens": 32,
                "enable_warmup": True,
            },
        )

    with pytest.raises(ValueError, match="unique"):
        JobSubmission(
            endpoint_id="lab",
            test_type="prefill",
            parameters={"token_levels": [512, 512], "requests_per_level": 3, "max_tokens": 32},
        )
