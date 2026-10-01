"""Legacy files convert once into bounded native plans; never reuse old execution paths."""

import pytest

from server.preset_conversion import LegacyPresetInput, convert_legacy_preset
from server.preset_templates import builtin_templates
from tests.server.test_data_api import auth, env  # noqa: F401


def legacy(kind="Concurrency Test", **config):
    return {
        "name": "Legacy",
        "description": "kept",
        "tags": ["math"],
        "created_at": "2026-05-28T12:00:00",
        "config": {
            "test_type": kind,
            "concurrency": 4,
            "max_tokens": 512,
            "temperature": 0,
            "thinking_enabled": False,
            "context_length": 4096,
            **config,
        },
    }


@pytest.mark.parametrize(
    "kind",
    [
        "Concurrency Test",
        "Prefill Stress Test",
        "Long Context Test",
        "Concurrency-Context Matrix Test",
        "Stability Test",
        "Segmented Context Test",
    ],
)
def test_original_labels_map_without_fallback(kind):
    result = convert_legacy_preset(LegacyPresetInput.model_validate(legacy(kind)))
    assert result["conversion"]["ready"] and result["preset"]["tags"] == ["math"]
    assert result["conversion"]["source_created_at"] == "2026-05-28T12:00:00"
    if result["preset"]["test_type"] == "concurrency":
        assert result["preset"]["parameters"]["selected_concurrencies"] == [4]
        assert result["preset"]["parameters"]["input_tokens_target"] == 64
    if result["preset"]["test_type"] == "matrix":
        assert result["preset"]["parameters"]["context_lengths"] == [4096]


def test_all_original_builtin_files_convert_with_explicit_defaults():
    for template in builtin_templates():
        result = convert_legacy_preset(
            LegacyPresetInput(
                name=template["name"], tags=template["tags"], config=template["source_config"]
            )
        )
        assert result["conversion"]["ready"]
        assert result["preset"]["parameters"] == template["parameters"]
        for key, value in template["source_config"].items():
            if key in {"temperature", "thinking_enabled", "thinking_budget", "reasoning_effort"}:
                assert result["preset"]["run_config"][key] == value


def test_credentials_excluded_before_fingerprint_and_response():
    plain = legacy()
    secret = legacy(
        API_KEY="synthetic-sensitive",  # pragma: allowlist secret
        api_base_url="http://credential@localhost",
    )
    safe = convert_legacy_preset(LegacyPresetInput.model_validate(plain))
    result = convert_legacy_preset(LegacyPresetInput.model_validate(secret))
    assert "synthetic-sensitive" not in str(result) and "credential@" not in str(result)
    assert (
        result["conversion"]["sanitized_config_sha256"]
        == safe["conversion"]["sanitized_config_sha256"]
    )
    assert set(result["conversion"]["excluded_fields"]) == {"API_KEY", "api_base_url"}


@pytest.mark.parametrize("data", [legacy("unknown"), legacy("All Tests"), legacy(extra_field=2)])
def test_unknown_fields_types_and_compound_tests_not_silently_changed(data):
    with pytest.raises(ValueError):
        convert_legacy_preset(LegacyPresetInput.model_validate(data))


def test_missing_custom_text_is_repairable_not_fabricated():
    result = convert_legacy_preset(LegacyPresetInput.model_validate(legacy("Custom Text Test")))
    assert not result["conversion"]["ready"] and any(
        row["field"] == "base_prompt" for row in result["conversion"]["errors"]
    )
    assert "base_prompt" not in result["preset"]["parameters"]


def test_convert_is_authenticated_readonly_and_import_is_explicit(env):  # noqa: F811
    client, _, _ = env
    path = "/api/v1/presets/convert"
    assert client.post(path, json=legacy()).status_code == 401
    response = client.post(path, json=legacy(), headers=auth())
    assert response.status_code == 200
    assert client.get("/api/v1/presets", headers=auth()).json()["items"] == []
    body = {key: response.json()[key] for key in ("format", "version", "preset")}
    body["preset"]["endpoint_id"] = "lab"
    saved = client.post("/api/v1/presets/import", json=body, headers=auth())
    assert saved.status_code == 201
    assert saved.json()["parameters"]["selected_concurrencies"] == [4]
    assert saved.json()["source_metadata"]["created_at"] == "2026-05-28T12:00:00"
    exported = client.get(
        "/api/v1/presets/" + saved.json()["preset_id"] + "/export", headers=auth()
    ).json()
    assert exported["preset"]["source_metadata"] == saved.json()["source_metadata"]
    assert client.get("/api/v1/jobs", headers=auth()).json()["items"] == []
