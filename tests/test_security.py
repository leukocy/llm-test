"""
Security tests for the llm-test platform.

Run with: pytest tests/test_security.py -v

This test suite verifies that security fixes are working correctly.
"""

import os
import tempfile
import pytest
from core.safe_executor import safe_eval_math, safe_exec_code, SafeExecutionError
from core.url_validator import is_safe_url, validate_and_normalize_url, SSRFError
from core.dataset_loader import DatasetLoader


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

    def test_safe_code_execution(self):
        """Test safe code execution."""
        code = """
def add(a, b):
    return a + b
"""
        test_code = "assert add(2, 3) == 5"

        success, error, output = safe_exec_code(code, test_code)
        assert success is True
        assert error is None

    def test_blocked_imports(self):
        """Test that imports are blocked."""
        code = "import os\nprint(os.getcwd())"

        success, error, output = safe_exec_code(code)
        assert success is False
        assert "not allowed" in error.lower()

    def test_syntax_error_handling(self):
        """Test that syntax errors are handled gracefully."""
        code = "def foo(\n"  # Invalid syntax

        success, error, output = safe_exec_code(code)
        assert success is False
        assert "SyntaxError" in error

    def test_assertion_failure(self):
        """Test that assertion failures are caught."""
        code = "def foo(): return 42"
        test_code = "assert foo() == 999"

        success, error, output = safe_exec_code(code, test_code)
        assert success is False
        assert "AssertionError" in error


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

    def test_allows_safe_domains(self):
        """Test that known safe domains are allowed."""
        is_safe, error = is_safe_url("https://api.openai.com/v1")
        assert is_safe is True
        assert error is None

        is_safe, error = is_safe_url("https://api.deepseek.com/v1")
        assert is_safe is True

        is_safe, error = is_safe_url("https://generativelanguage.googleapis.com")
        assert is_safe is True

    def test_allows_private_when_flagged(self):
        """Test that private IPs are allowed when flag is set."""
        is_safe, error = is_safe_url("http://192.168.1.1/api", allow_private=True)
        assert is_safe is True

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
            with open(test_file, 'w') as f:
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
        assert "\\n" in result or "test message" == result

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
        from core.rate_limiter import RateLimiter
        import time

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
