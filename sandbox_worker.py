"""Private execution broker. Only this service may access the Docker socket.

The HTTP handler never evaluates a submission. It creates a short-lived container
without network access, host mounts, inherited credentials, or writable rootfs.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import os
import secrets
import selectors
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, cast

from core.sandbox_child import CHILD_TEMPLATE
from core.sandbox_identity import VERSION, identity_digest, validate_identity

MAX_CODE_BYTES = 256_000
MAX_REQUEST_BYTES = 300_000
MAX_OUTPUT_BYTES = 1_000_000
MAX_ERROR_BYTES = 32_000
MAX_CONCURRENT = 4


class WorkerInfrastructureError(RuntimeError):
    """The container runtime could not start or finish a submission."""


def container_command(name: str, memory_mb: int, image: str) -> list[str]:
    """Build the fixed security policy; request data never becomes CLI options."""
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--interactive",
        "--name",
        name,
        "--network=none",
        "--ipc=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        f"--memory={memory_mb}m",
        f"--memory-swap={memory_mb}m",
        "--cpus=1",
        "--ulimit=nofile=64:64",
        "--log-driver=none",
        "--user=65534:65534",
        "--workdir=/tmp",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=16m",
        image,
        "python",
        "-I",
        "-c",
        CHILD_TEMPLATE,
    ]


def _collect_process(
    process: subprocess.Popen[bytes], code: bytes, deadline: float
) -> tuple[int, bytes, bytes]:
    """Drain pipes continuously while retaining only bounded output."""
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    offset = 0
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("sandbox execution deadline exceeded")
            for key, _events in selector.select(min(remaining, 0.2)):
                stream = cast(BinaryIO, key.fileobj)
                if key.data == "stdin":
                    try:
                        count = os.write(stream.fileno(), code[offset : offset + 65536])
                        offset += count
                    except (BrokenPipeError, OSError):
                        offset = len(code)
                    if offset >= len(code):
                        selector.unregister(stream)
                        stream.close()
                    continue

                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                target = stdout if key.data == "stdout" else stderr
                limit = MAX_OUTPUT_BYTES if key.data == "stdout" else MAX_ERROR_BYTES
                if len(target) < limit:
                    target.extend(chunk[: limit - len(target)])
        remaining = max(0.0, deadline - time.monotonic())
        try:
            return process.wait(timeout=remaining), bytes(stdout), bytes(stderr)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("sandbox execution deadline exceeded") from exc


def execute_in_container(
    code: str, timeout_seconds: float, memory_mb: int, image: str
) -> tuple[bool, str | None, str | None]:
    """Execute code with a container boundary and deterministic resource limits."""
    encoded = code.encode("utf-8")
    if not encoded or len(encoded) > MAX_CODE_BYTES:
        return False, "Submission must contain 1–256000 UTF-8 bytes", None
    if not 1 <= timeout_seconds <= 30 or not 64 <= memory_mb <= 1024:
        return False, "Sandbox limits are out of range", None

    name = f"llm-test-eval-{secrets.token_hex(12)}"
    command = container_command(name, memory_mb, image)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
    except OSError as exc:
        raise WorkerInfrastructureError("Docker CLI could not start") from exc

    try:
        try:
            return_code, stdout, stderr = _collect_process(
                process, encoded, time.monotonic() + timeout_seconds + 5
            )
        except TimeoutError:
            return False, f"TimeoutError: execution exceeded {timeout_seconds}s", None
        if return_code in {125, 126, 127}:
            raise WorkerInfrastructureError("Docker could not start the sandbox container")
        output = stdout.decode("utf-8", errors="replace") or None
        if return_code == 0:
            return True, None, output
        error_lines = stderr.decode("utf-8", errors="replace").strip().splitlines()
        error = error_lines[-1] if error_lines else f"exit code {return_code}"
        return False, error[:2000], output
    finally:
        if process.poll() is None:
            try:
                subprocess.run(
                    ["docker", "rm", "--force", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream and not stream.closed:
                stream.close()


class SandboxServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], token: str, image: str):
        super().__init__(address, SandboxHandler)
        self.token = token
        self.image = image
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT)
        self.identity_lock = threading.Lock()
        self.pinned_image: str | None = None

    def execution_identity(self) -> dict[str, Any]:
        """Resolve a tag once, then execute only that immutable image ID."""
        try:
            with self.identity_lock:
                if self.pinned_image is None:
                    result = subprocess.run(
                        ["docker", "image", "inspect", self.image, "--format", "{{.Id}}"],
                        capture_output=True,
                        timeout=5,
                        check=True,
                        text=True,
                    )
                    self.pinned_image = result.stdout.strip()
            runtime_result = subprocess.run(
                ["docker", "info", "--format", "{{json .}}"],
                capture_output=True,
                timeout=5,
                check=True,
                text=True,
            )
            info = json.loads(runtime_result.stdout)
            runtime = {
                key: info[key]
                for key in ("ServerVersion", "KernelVersion", "Architecture", "OperatingSystem")
            }
            cpu_text = Path("/proc/cpuinfo").read_text()
            cpu_models = sorted(
                {
                    line.strip()
                    for line in cpu_text.splitlines()
                    if line.startswith(("model name", "Hardware", "CPU implementer", "CPU part"))
                }
            )
            if not cpu_models:
                raise ValueError("CPU identity unavailable")
            evidence = {
                "version": VERSION,
                "image_id": self.pinned_image,
                "runtime": runtime,
                "cpu_sha256": hashlib.sha256("\n".join(cpu_models).encode()).hexdigest(),
                "policy_sha256": hashlib.sha256(
                    (
                        inspect.getsource(container_command)
                        + CHILD_TEMPLATE
                        + Path(__file__).read_text()
                    ).encode()
                ).hexdigest(),
            }
            return validate_identity({**evidence, "sha256": identity_digest(evidence)})
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            raise WorkerInfrastructureError("Sandbox environment identity unavailable") from exc


class SandboxHandler(BaseHTTPRequestHandler):
    server: SandboxServer

    def _reply(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {self.server.token}")

    def do_GET(self) -> None:
        if self.path != "/health":
            self._reply(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        elif not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized"})
        else:
            try:
                identity = self.server.execution_identity()
                self._reply(HTTPStatus.OK, {"status": "ok", "sandbox_identity": identity})
            except WorkerInfrastructureError:
                self._reply(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "Sandbox environment identity unavailable"},
                )

    def do_POST(self) -> None:
        if self.path != "/execute":
            self._reply(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        if not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, {"error": "Unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_REQUEST_BYTES:
            self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Invalid request size"})
            return
        self.connection.settimeout(15)
        try:
            request = json.loads(self.rfile.read(length))
            code = request["code"]
            timeout = float(request["timeout_seconds"])
            memory = int(request["mem_limit_mb"])
            if not isinstance(code, str) or not 1 <= timeout <= 30 or not 64 <= memory <= 1024:
                raise ValueError("Invalid sandbox request")
        except (KeyError, TypeError, ValueError, TimeoutError, json.JSONDecodeError):
            self._reply(HTTPStatus.BAD_REQUEST, {"error": "Invalid sandbox request"})
            return
        if not self.server.slots.acquire(blocking=False):
            self._reply(HTTPStatus.TOO_MANY_REQUESTS, {"error": "Sandbox is busy"})
            return
        try:
            identity = self.server.execution_identity()
            expected = request.get("sandbox_contract")
            if expected is not None and expected != identity["sha256"]:
                self._reply(HTTPStatus.CONFLICT, {"error": "Sandbox environment changed"})
                return
            success, error, output = execute_in_container(
                code, timeout, memory, identity["image_id"]
            )
            self._reply(
                HTTPStatus.OK,
                {
                    "success": success,
                    "error": error,
                    "output": output,
                    "sandbox_identity": identity,
                },
            )
        except WorkerInfrastructureError:
            self._reply(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Sandbox runtime unavailable"})
        finally:
            self.server.slots.release()

    def log_message(self, format_string: str, *args: Any) -> None:
        # The default handler logs client data; retain only the HTTP status line.
        if args:
            print(f"sandbox-worker: {args[-1]}", flush=True)


def main() -> None:
    token = os.environ.get("LLM_TEST_SANDBOX_TOKEN", "")
    image = os.environ.get("LLM_TEST_SANDBOX_IMAGE", "python:3.12-slim")
    if len(token) < 32:
        raise SystemExit("LLM_TEST_SANDBOX_TOKEN must contain at least 32 characters")
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Sandbox image is unavailable: {image}") from exc
    port = int(os.environ.get("LLM_TEST_SANDBOX_PORT", "8765"))
    server = SandboxServer(("0.0.0.0", port), token, image)
    print(f"sandbox-worker listening on {port}", flush=True)
    server.serve_forever(poll_interval=0.2)


if __name__ == "__main__":
    main()
