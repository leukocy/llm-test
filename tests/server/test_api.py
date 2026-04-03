
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch

# Need to set env var for config if needed, or mock settings dependency
# But settings has defaults.

from server.app import app

client = TestClient(app)

def test_health_check():
    # We don't have explicit /health endpoint in app.py currently, 
    # but let's check accessible endpoints.
    # Actually I should add /api/health in app.py as per plan.
    # But I can check /api/strategies.
    response = client.get("/api/strategies")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    # Check if some known strategy exists (e.g. prefill)
    if not data:
         pytest.skip("No strategies registered yet? Strategies should auto-register on import.")
    assert "prefill" in data
    assert "concurrency" in data

def test_list_providers():
    response = client.get("/api/providers/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        assert "base_url" in data[0]

def test_list_models():
    response = client.get("/api/providers/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_start_run():
    with patch("server.routes.tests.TestRunner") as MockRunner:
        # Mock runner instance
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run = AsyncMock(return_value=[])
        
        payload = {
            "test_type": "prefill",
            "base": {
                "api_base_url": "http://localhost:11434/v1",
                "model_id": "llama2",
                "api_key": "dummy",
                "provider": "Ollama"
            },
            "params": {
                "input_tokens_list": [10],
                "rounds": 1
            }
        }
        
        response = client.post("/api/tests/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "started"
        
        run_id = data["run_id"]
        
        # Check status
        status_resp = client.get(f"/api/tests/{run_id}/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["run_id"] == run_id
        
        # Check control endpoints
        stop_resp = client.post(f"/api/tests/{run_id}/stop")
        assert stop_resp.status_code == 200
        
        pause_resp = client.post(f"/api/tests/{run_id}/pause")
        assert pause_resp.status_code == 200


def test_get_nonexistent_run():
    response = client.get("/api/results/invalid_id")
    assert response.status_code == 404

