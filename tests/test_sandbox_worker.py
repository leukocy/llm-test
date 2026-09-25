"""Boundary checks for generated-code execution."""

import os
import threading

import httpx
import pytest

from sandbox_worker import SandboxServer, container_command, execute_in_container


def test_container_policy_has_no_network_or_host_mounts():
    command = container_command("llm-test-eval-test", 128, "python:3.12-slim")
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--user=65534:65534" in command
    assert not any(arg.startswith(("--volume", "--mount", "-v")) for arg in command)


def test_worker_rejects_requests_without_its_token(monkeypatch):
    monkeypatch.setattr(
        "sandbox_worker.execute_in_container",
        lambda *args: pytest.fail("unauthorized request reached the container runtime"),
    )
    server = SandboxServer(("127.0.0.1", 0), "x" * 32, "python:3.12-slim")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(trust_env=False, timeout=3) as client:
            response = client.post(
                f"http://127.0.0.1:{server.server_port}/execute",
                json={"code": "print(1)", "timeout_seconds": 2, "mem_limit_mb": 128},
            )
        assert response.status_code == 401
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.integration
def test_real_container_cannot_read_parent_environment_or_file(tmp_path, monkeypatch):
    if os.environ.get("RUN_SANDBOX_INTEGRATION") != "1":
        pytest.skip("Set RUN_SANDBOX_INTEGRATION=1 on a Docker host")
    marker = tmp_path / "host-only-canary.txt"
    marker.write_text("FILE_CANARY_OK", encoding="utf-8")
    monkeypatch.setenv("LLM_TEST_SANDBOX_CANARY", "ENV_CANARY_OK")

    code = f"""
import os
print('ENV:', os.environ.get('LLM_TEST_SANDBOX_CANARY'))
print('FILE:', os.path.exists({str(marker)!r}))
"""
    success, error, output = execute_in_container(code, 3, 128, "python:3.12-slim")

    assert success is True, error
    assert output == "ENV: None\nFILE: False\n"


@pytest.mark.integration
def test_real_container_enforces_timeout():
    if os.environ.get("RUN_SANDBOX_INTEGRATION") != "1":
        pytest.skip("Set RUN_SANDBOX_INTEGRATION=1 on a Docker host")
    success, error, output = execute_in_container("while True: pass", 1, 128, "python:3.12-slim")
    assert success is False
    assert "TimeoutError" in (error or "")
    assert output is None


@pytest.mark.integration
def test_real_container_has_no_external_network():
    if os.environ.get("RUN_SANDBOX_INTEGRATION") != "1":
        pytest.skip("Set RUN_SANDBOX_INTEGRATION=1 on a Docker host")
    code = """
import socket
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(('1.1.1.1', 80))
except OSError:
    print('NETWORK_BLOCKED')
else:
    print('NETWORK_CONNECTED')
finally:
    sock.close()
"""
    success, error, output = execute_in_container(code, 3, 128, "python:3.12-slim")
    assert success is True, error
    assert output == "NETWORK_BLOCKED\n"
