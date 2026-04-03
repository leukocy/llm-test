
import pytest
from fastapi.testclient import TestClient
from server.app import app
from server.state import active_runs

client = TestClient(app)

@pytest.fixture
def sample_run_data():
    run_id = "test_run_123"
    results = [
        {"session_id": 1, "concurrency": 1, "ttft": 0.1, "tps": 50.0, "tpot": 0.02, "error": None, "start_time": 100, "end_time": 101, "prefill_tokens": 10, "decode_tokens": 50},
        {"session_id": 2, "concurrency": 1, "ttft": 0.12, "tps": 48.0, "tpot": 0.021, "error": None, "start_time": 100, "end_time": 101, "prefill_tokens": 10, "decode_tokens": 50},
        {"session_id": 3, "concurrency": 2, "ttft": 0.2, "tps": 45.0, "tpot": 0.022, "error": None, "start_time": 101, "end_time": 102, "prefill_tokens": 10, "decode_tokens": 50},
        {"session_id": 4, "concurrency": 2, "ttft": 0.22, "tps": 43.0, "tpot": 0.023, "error": None, "start_time": 101, "end_time": 102, "prefill_tokens": 10, "decode_tokens": 50},
    ]
    config = {
        "test_type": "concurrency",
        "base": {"model_id": "gpt-4", "provider": "openai"},
        "params": {"concurrencies": [1, 2]}
    }
    
    active_runs[run_id] = {
        "status": "completed",
        "results": results,
        "config": config,
        "event_bus": None
    }
    yield run_id
    # Cleanup
    if run_id in active_runs:
        del active_runs[run_id]

def test_get_summary(sample_run_data):
    run_id = sample_run_data
    response = client.get(f"/api/reports/{run_id}/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 4
    assert data["successful_requests"] == 4
    assert data["duration_seconds"] == 2.0 # 102 - 100
    assert data["avg_ttft"] == pytest.approx(0.16)

def test_get_analysis(sample_run_data):
    run_id = sample_run_data
    response = client.get(f"/api/reports/{run_id}/analysis")
    assert response.status_code == 200
    data = response.json()
    
    # Verify ReportData structure
    assert data["test_type"] == "concurrency"
    assert len(data["sections"]) >= 1
    section = data["sections"][0]
    
    # Check charts
    assert len(section["charts"]) > 0
    chart = section["charts"][0]
    assert chart["type"] == "line"
    assert len(chart["series"]) > 0
    
    # Check stats
    assert len(section["stats"]) > 0
    stat_labels = [s["label"] for s in section["stats"]]
    assert "Duration" in stat_labels
    assert "TTFT @ C1" in stat_labels

def test_get_summary_not_found():
    response = client.get("/api/reports/nonexistent/summary")
    assert response.status_code == 404
