"""
Security tests for the llm-test platform.

Run with: pytest tests/test_security.py -v

This test suite verifies that security fixes are working correctly.
"""

import ipaddress
import os
import tempfile

import httpx
import pytest

from core.dataset_loader import DatasetLoader
from core.providers.factory import _validate_base_url
from core.safe_executor import (
    SafeExecutionError,
    SandboxUnavailableError,
    require_sandbox_available,
    safe_eval_math,
    safe_exec_code,
)
from core.url_validator import (
    SSRFError,
    is_safe_url,
    parse_trusted_private_networks,
    validate_and_normalize_url,
)


class TestSafeCodeExecution:
    """Test safe code execution."""

    def test_safe_math_expression(self):
        """Test safe math evaluation."""
        # Basic arithmetic
        result = safe_eval_math("2 + 2")
        assert result == 4.0

        # Math functions
        result = safe_eval_math("math.sqrt(16)")
        assert result == 4.0

        result = safe_eval_math("math.pi")
        assert abs(result - 3.14159) < 0.0001

        # Complex expressions
        result = safe_eval_math("2 + 2 * math.sqrt(16)")
        assert result == 10.0

    def test_blocked_malicious_code(self):
        """Test that malicious code is blocked."""
        # Import attempts should be blocked
        with pytest.raises(SafeExecutionError):
            safe_eval_math("__import__('os').system('ls')")

        with pytest.raises(SafeExecutionError):
            safe_eval_math("eval('1+1')")

        # Attribute access should be blocked
        with pytest.raises(SafeExecutionError):
            safe_eval_math("().__class__")

    @pytest.mark.parametrize(
        "expression",
        ["2**1000000000", "math.factorial(10000000)", "2**64**64", "1+" * 200 + "1"],
    )
    def test_math_evaluation_rejects_resource_exhaustion(self, expression):
        with pytest.raises(SafeExecutionError):
            safe_eval_math(expression)

    def test_code_execution_requires_isolated_worker(self, monkeypatch):
        monkeypatch.delenv("LLM_TEST_SANDBOX_URL", raising=False)
        monkeypatch.delenv("LLM_TEST_SANDBOX_TOKEN", raising=False)
        with pytest.raises(SandboxUnavailableError, match="Sandbox unavailable"):
            safe_exec_code("print('hello')")

    def test_code_is_sent_to_worker_without_local_execution(self, monkeypatch):
        monkeypatch.setenv("LLM_TEST_SANDBOX_URL", "http://sandbox-worker:8765")
        monkeypatch.setenv("LLM_TEST_SANDBOX_TOKEN", "t" * 40)
        calls = []

        def fake_request(method, url, token, timeout, **kwargs):
            calls.append((method, url, token, timeout, kwargs))
            return httpx.Response(200, json={"success": True, "error": None, "output": "ok"})

        monkeypatch.setattr("core.safe_executor._worker_request", fake_request)
        assert safe_exec_code("print('hello')") == (True, None, "ok")
        assert calls[0][0:3] == (
            "POST",
            "http://sandbox-worker:8765/execute",
            "t" * 40,
        )
        assert calls[0][4]["json"]["code"] == "print('hello')"

    def test_unhealthy_worker_blocks_code_benchmark(self, monkeypatch):
        monkeypatch.setenv("LLM_TEST_SANDBOX_URL", "http://sandbox-worker:8765")
        monkeypatch.setenv("LLM_TEST_SANDBOX_TOKEN", "t" * 40)
        monkeypatch.setattr(
            "core.safe_executor._worker_request", lambda *args, **kwargs: httpx.Response(503)
        )
        with pytest.raises(SandboxUnavailableError, match="health check"):
            require_sandbox_available()


class TestSSRFProtection:
    """Test SSRF protection."""

    def test_blocks_private_ips(self):
        """Test that private IPs are blocked."""
        # Class C private network
        is_safe, error = is_safe_url("http://192.168.1.1/api")
        assert is_safe is False
        assert "not allowed" in error.lower() or "private" in error.lower()

        # Class A private network
        is_safe, error = is_safe_url("http://10.0.0.1/api")
        assert is_safe is False

        # Class B private network
        is_safe, error = is_safe_url("http://172.16.0.1/api")
        assert is_safe is False

    def test_blocks_loopback(self):
        """Test that loopback addresses are blocked."""
        is_safe, error = is_safe_url("http://127.0.0.1/api")
        assert is_safe is False

        is_safe, error = is_safe_url("http://127.0.0.2/api")
        assert is_safe is False

    @pytest.mark.parametrize("host", ["2130706433", "0x7f000001", "127.1", "0177.0.0.1"])
    def test_blocks_noncanonical_loopback_aliases(self, host):
        is_safe, error = is_safe_url(f"http://{host}/v1", allow_private=True)
        assert is_safe is False
        assert "Non-canonical IP" in error

    def test_unknown_provider_requires_admin_trust(self):
        is_safe, error = is_safe_url("https://unconfigured.example/v1")
        assert is_safe is False
        assert "not trusted" in error

    def test_trusted_host_must_not_resolve_to_private_without_opt_in(self, monkeypatch):
        monkeypatch.setattr(
            "core.url_validator.socket.getaddrinfo",
            lambda *args, **kwargs: [(None, None, None, None, ("10.1.2.3", 443))],
        )
        url = "https://trusted.example/v1"
        is_safe, error = is_safe_url(url, custom_safe_domains={"trusted.example"})
        assert is_safe is False
        assert "private/internal" in error

        is_safe, error = is_safe_url(
            url, allow_private=True, custom_safe_domains={"trusted.example"}
        )
        assert is_safe is True
        assert error is None

    def test_allows_safe_domains(self):
        """Test that known safe domains are allowed."""
        is_safe, error = is_safe_url("https://api.openai.com/v1")
        assert is_safe is True
        assert error is None

        is_safe, error = is_safe_url("https://api.deepseek.com/v1")
        assert is_safe is True

        is_safe, error = is_safe_url("https://generativelanguage.googleapis.com")
        assert is_safe is True

    def test_private_ip_requires_explicit_trust_and_opt_in(self):
        url = "http://192.168.1.1/api"
        is_safe, error = is_safe_url(url, allow_private=True)
        assert is_safe is False
        assert "explicitly trusted" in error

        is_safe, error = is_safe_url(url, allow_private=True, custom_safe_domains={"192.168.1.1"})
        assert is_safe is True

    def test_trusted_private_cidr_is_scoped_and_requires_opt_in(self):
        networks = parse_trusted_private_networks("192.168.199.0/24")
        assert networks == {ipaddress.ip_network("192.168.199.0/24")}

        allowed, error = is_safe_url(
            "http://192.168.199.179:10814/v1",
            allow_private=True,
            trusted_private_networks=networks,
        )
        assert allowed is True and error is None

        allowed, _ = is_safe_url(
            "http://192.168.200.1:10814/v1",
            allow_private=True,
            trusted_private_networks=networks,
        )
        assert allowed is False

        allowed, _ = is_safe_url(
            "http://192.168.199.179:10814/v1", trusted_private_networks=networks
        )
        assert allowed is False

    @pytest.mark.parametrize("value", ["0.0.0.0/0", "169.254.0.0/16", "192.168.1.3/24"])
    def test_rejects_unsafe_or_noncanonical_trusted_networks(self, value):
        with pytest.raises(SSRFError):
            parse_trusted_private_networks(value)

    def test_factory_uses_trusted_private_network_configuration(self, monkeypatch):
        monkeypatch.setenv("LLM_TEST_ALLOW_PRIVATE_ENDPOINTS", "1")
        monkeypatch.setenv("LLM_TEST_TRUSTED_API_NETWORKS", "192.168.199.0/24")
        assert _validate_base_url("http://192.168.199.129:10814/v1") == (
            "http://192.168.199.129:10814/v1"
        )
        with pytest.raises(SSRFError):
            _validate_base_url("http://192.168.200.1:10814/v1")

    def test_blocks_invalid_protocols(self):
        """Test that non-http protocols are blocked."""
        is_safe, error = is_safe_url("file:///etc/passwd")
        assert is_safe is False
        assert "protocol" in error.lower()

        is_safe, error = is_safe_url("ftp://example.com")
        assert is_safe is False

        is_safe, error = is_safe_url("javascript:alert(1)")
        assert is_safe is False

    def test_validate_and_normalize_url(self):
        """Test URL validation and normalization."""
        # Safe URL should work
        url = validate_and_normalize_url("https://api.openai.com/v1")
        assert "api.openai.com" in url

        # Unsafe URL should raise error
        with pytest.raises(SSRFError):
            validate_and_normalize_url("http://192.168.1.1/api")

        # HTTPS enforcement
        with pytest.raises(SSRFError):
            validate_and_normalize_url("http://api.openai.com/v1", require_https=True)


class TestPathTraversal:
    """Test path traversal protection."""

    def test_blocks_path_traversal(self):
        """Test that path traversal attempts are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DatasetLoader(tmpdir)

            with pytest.raises(ValueError, match="Path traversal"):
                loader._get_file_path("../../../etc/passwd")

            with pytest.raises(ValueError, match="Path traversal"):
                loader._get_file_path("../../config/settings.py")

            with pytest.raises(ValueError, match="Path traversal"):
                loader._get_file_path("..\\..\\windows\\system32")

    def test_allows_normal_files(self):
        """Test that normal file access works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = os.path.join(tmpdir, "test.csv")
            with open(test_file, "w") as f:
                f.write("test")

            loader = DatasetLoader(tmpdir)
            path = loader._get_file_path("test.csv")

            assert path == test_file

    def test_blocks_absolute_paths(self):
        """Test that absolute paths outside datasets are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DatasetLoader(tmpdir)

            # Absolute path to different location
            with pytest.raises(ValueError, match="Path traversal|outside"):
                loader._get_file_path("/etc/passwd")

            with pytest.raises(ValueError, match="Path traversal|outside"):
                loader._get_file_path("C:\\Windows\\System32\\config")


class TestLogSanitization:
    """Test log sanitization."""

    def test_removes_newlines(self):
        """Test that newlines are removed."""
        from utils.log_sanitizer import sanitize_log_message

        result = sanitize_log_message("test\nmessage")
        assert "\n" not in result
        assert "\\n" in result or result == "test message"

    def test_removes_ansi_codes(self):
        """Test that ANSI codes are removed."""
        from utils.log_sanitizer import sanitize_log_message

        result = sanitize_log_message("\x1b[31mRed text\x1b[0m")
        assert "\x1b" not in result

    def test_truncates_long_messages(self):
        """Test that long messages are truncated."""
        from utils.log_sanitizer import sanitize_log_message

        long_message = "a" * 20000
        result = sanitize_log_message(long_message)
        assert len(result) <= 10000 + len("... (truncated)")

    def test_sanitizes_api_keys(self):
        """Test that API keys are redacted."""
        from utils.log_sanitizer import sanitize_api_key

        # Test with proper length API key (20+ chars)
        result = sanitize_api_key("API key: sk-1234567890abcdefghijklmnopqr")
        assert "sk-1234567890abcdefghijklmnopqr" not in result
        assert "sk-[REDACTED]" in result

        # Test with Gemini key
        result = sanitize_api_key("Key: AIza1234567890abcdefghijklmnopqrstuvwxyz")
        assert "AIza1234567890abcdefghijklmnopqrstuvwxyz" not in result
        assert "AIza[REDACTED]" in result


class TestRateLimiter:
    """Test rate limiting."""

    def test_rate_limiting(self):
        """Test that rate limiter works."""
        import time

        from core.rate_limiter import RateLimiter

        limiter = RateLimiter(rate=10, burst=10)

        # Should allow first request immediately
        start = time.time()
        assert limiter.acquire() is True
        elapsed = time.time() - start
        assert elapsed < 0.1  # Should be nearly instant

        # Drain the bucket
        for _ in range(9):
            limiter.acquire()

        # Next request should be rate limited
        start = time.time()
        assert limiter.acquire() is True
        elapsed = time.time() - start
        assert elapsed >= 0.05  # Should have waited

    def test_non_blocking_acquire(self):
        """Test non-blocking acquire."""
        from core.rate_limiter import RateLimiter

        limiter = RateLimiter(rate=1, burst=1)

        # First request should succeed
        assert limiter.acquire(blocking=False) is True

        # Second request should fail (would block)
        assert limiter.acquire(blocking=False, timeout=0.01) is False


class TestProviderSSRFValidation:
    """安全审查 #4 回归: provider 工厂接入 URL 校验。"""

    def test_blocks_file_protocol(self):
        from core.providers.factory import get_provider
        from core.url_validator import SSRFError

        with pytest.raises(SSRFError):
            get_provider("Custom", "file:///etc/passwd", "k", "m")

    def test_blocks_metadata_ip(self):
        from core.providers.factory import get_provider
        from core.url_validator import SSRFError

        with pytest.raises(SSRFError):
            get_provider("Custom", "http://169.254.169.254/latest", "k", "m")

    def test_allows_public_https(self):
        from core.providers.factory import get_provider

        provider = get_provider("DeepSeek", "https://api.deepseek.com/v1", "k", "m")
        assert provider.api_base_url.startswith("https://")

    def test_allows_local_inference_endpoints(self):
        from core.providers.factory import get_provider

        provider = get_provider("llama.cpp", "http://127.0.0.1:8080/v1", "k", "m")
        assert provider is not None


class TestCheckpointIntegrity:
    """安全审查 #7 回归: 检查点 HMAC + 名称清洗。"""

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):

        monkeypatch.chdir(tmp_path)
        from core.response_cache import ResponseCache

        cache = ResponseCache(cache_dir="cache")
        cache.save_checkpoint({"done": [1, 2]}, "ckpt")
        assert cache.load_checkpoint("ckpt")["state"] == {"done": [1, 2]}

    def test_tampered_payload_rejected(self, tmp_path, monkeypatch):
        import gzip
        import pickle

        monkeypatch.chdir(tmp_path)
        from core.response_cache import ResponseCache

        cache = ResponseCache(cache_dir="cache")
        path = cache.save_checkpoint({"a": 1}, "ckpt")
        with open(path, "rb") as f:
            raw = f.read()
        mac_line, _, compressed = raw.partition(b"\n")
        inner = pickle.dumps({"a": 999})
        with open(path, "wb") as f:
            f.write(mac_line + b"\n" + gzip.compress(inner))

        assert cache.load_checkpoint("ckpt") is None

    def test_traversal_name_contained(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from core.response_cache import ResponseCache

        cache = ResponseCache(cache_dir="cache")
        cache.save_checkpoint({}, "../../evil")
        names = [p.name for p in cache.checkpoint_dir.glob("*.pkl.gz")]
        assert names == ["evil.pkl.gz"]  # 收敛在 checkpoint_dir 内


class TestSQLIdentifierValidation:
    """安全审查 #8 回归: SQL 标识符/WHERE/ORDER BY 校验。"""

    def test_rejects_table_injection(self):
        import pytest

        from core.database.connection import _validate_identifier

        with pytest.raises(ValueError):
            _validate_identifier("t; DROP TABLE t--")

    def test_rejects_where_injection(self):
        from core.database.connection import _validate_where

        with pytest.raises(ValueError):
            _validate_where("1=1; DROP TABLE t")
        with pytest.raises(ValueError):
            _validate_where("1=1 -- comment")

    def test_rejects_order_by_injection(self):
        from core.database.connection import _validate_order_by

        with pytest.raises(ValueError):
            _validate_order_by("id; DROP TABLE t")
        # 合法形式放行
        assert _validate_order_by("id DESC")
        assert _validate_order_by("created_at ASC, id DESC")

    def test_repository_crud_still_works(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from core.database import connection as db_conn
        from core.repositories import base as repo_base

        # TestResultRepository 默认绑定模块级单例 db; 让单例指向 tmp 库
        db_conn.Database._instance = None
        database = db_conn.Database("data/t.db")
        monkeypatch.setattr(repo_base, "db", database, raising=False)
        try:
            from core.repositories.test_result import TestResultRepository

            repo = TestResultRepository()
            assert repo.find_all(limit=5) == []
        finally:
            if db_conn.Database._instance is not None:
                db_conn.Database._instance.close_all()


class TestLogSanitizationWiring:
    """安全审查 #9 回归: get_logger 输出脱敏。"""

    def test_logger_redacts_api_keys(self):
        import io

        from utils.get_logger import get_logger

        buf = io.StringIO()
        logger = get_logger("test.security.wiring")
        for handler in logger.handlers:
            handler.stream = buf
        logger.error("failed with sk-1234567890abcdefghijklmnopqr")
        out = buf.getvalue()
        assert "sk-1234567890abcdefghijklmnopqr" not in out
        assert "sk-[REDACTED]" in out

    def test_request_logger_masks_gemini_header(self):
        from core.request_logger import RequestLogger

        logger = RequestLogger(log_dir="/tmp/sec_test_logs", enabled=False)
        masked = logger._mask_headers({"x-goog-api-key": "AIzaXXX", "Accept": "*/*"})
        assert masked["x-goog-api-key"] == "*****"
        assert masked["Accept"] == "*/*"


class TestLongBenchSafeEval:
    """安全审查 #5 回归: 数据集字段不再 eval。"""

    def test_list_string_answer(self):
        from evaluators.longbench_evaluator import LongBenchEvaluator

        ev = LongBenchEvaluator.__new__(LongBenchEvaluator)
        assert ev.check_answer("paris", '["Paris", "city"]') is True
        assert ev.check_answer("nope", '["Paris"]') is False

    def test_malicious_dataset_field_rejected(self):
        from evaluators.longbench_evaluator import LongBenchEvaluator

        ev = LongBenchEvaluator.__new__(LongBenchEvaluator)
        # 恶意"列表"不会被执行, 走回退当普通字符串包含匹配
        assert ev.check_answer("x", '__import__("os").system("id")') is False


class TestDockerExecGate:
    """安全审查 #6 回归: docker exec 默认关闭。"""

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.setenv("ENGINE_CAPTURE_DOCKER_EXEC", "0")
        from core.engine_capture import _query_container_runtime_versions

        assert _query_container_runtime_versions("some-container") == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
