"""Template source fidelity is checked against the checked-in first-commit inventory."""

import json
from pathlib import Path

from server.preset_templates import BASELINE, builtin_templates
from tests.server.test_data_api import auth, env  # noqa: F401


def test_templates_preserve_every_original_source_value():
    inventory_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "ui_parity"
        / "first_commit_dynamic_choices.json"
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["baseline_commit"] == BASELINE[:7]
    expected = inventory["builtin_presets"]
    templates = builtin_templates()
    assert len(templates) == len(expected) == 8
    for template, source in zip(templates, expected, strict=True):
        assert template["name"] == source["name"]
        assert template["source_config"] == source["config"]
        assert template["tags"] == source["tags"]
        assert template["source_description"] == source["description"]
    templates[0]["source_config"]["concurrency"] = 999
    assert builtin_templates()[0]["source_config"]["concurrency"] == 1


def test_templates_are_readonly_authenticated_and_native_plans_are_admitted(env):  # noqa: F811
    client, _, _ = env
    path = "/api/v1/presets/templates"
    assert client.get(path).status_code == 401
    result = client.get(path, headers=auth())
    assert result.status_code == 200
    for template in result.json()["items"]:
        body = {
            "endpoint_id": "lab",
            "test_type": template["test_type"],
            "parameters": template["parameters"],
            "run_config": template["run_config"],
        }
        planned = client.post("/api/v1/jobs/plan", json=body, headers=auth())
        assert planned.status_code == 200 and planned.json()["total_requests"] > 0
    assert client.get("/api/v1/presets", headers=auth()).json()["items"] == []
    assert client.get("/api/v1/jobs", headers=auth()).json()["items"] == []
    prefill = next(template for template in result.json()["items"] if template["id"] == "prefill")
    assert prefill["source_config"]["concurrency"] == 4 and any(
        "串行" in note for note in prefill["notes"]
    )
