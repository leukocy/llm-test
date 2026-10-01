"""Template source fidelity is checked against the immutable original, not a duplicate fixture."""

import ast
import subprocess  # nosec B404

from server.preset_templates import BASELINE, builtin_templates
from tests.server.test_data_api import auth, env  # noqa: F401


def test_templates_preserve_every_original_source_value():
    original = subprocess.run(
        ["git", "show", f"{BASELINE}:utils/test_config_manager.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout  # nosec B603 B607
    tree = ast.parse(original)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_builtin_presets"
    )
    returned = next(node for node in function.body if isinstance(node, ast.Return))
    expected = [
        {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
        for call in returned.value.elts
    ]
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
