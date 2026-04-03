"""
Rate limiting utilities for API calls.

Uses token bucket algorithm to prevent API abuse and rate limit errors.
"""

import time
from collections import deque
from threading import Lock
from typing import Optional


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    The token bucket algorithm allows bursts of requests up to a maximum,
    then refills at a steady rate. This provides both rate limiting and
    burst handling.

    Usage:
        limiter = RateLimiter(rate=10, burst=20)

        # Acquire a token (blocks if rate limited)
        limiter.acquire()

        # Or use as context manager
        with limiter:
            # Make API call
            pass

        # Non-blocking acquire
        if limiter.acquire(blocking=False):
            # Make API call
            pass
    """

    def __init__(self, rate: float, burst: Optional[int] = None):
        """
        Initialize rate limiter.

        Args:
            rate: Maximum requests per second
            burst: Maximum burst size (defaults to rate)
        """
        self.rate = rate
        self.burst = int(burst or rate)
        self.tokens = self.burst
        self.last_update = time.monotonic()
        self._lock = Lock()

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token from the rate limiter.

        Args:
            blocking: Whether to block until a token is available
            timeout: Maximum time to wait (None = infinite)

        Returns:
            True if token acquired, False if timeout
        """
        start_wait_time = time.monotonic()

        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = max(0, now - self.last_update)
                self.last_update = now

                # Refill bucket
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

                if not blocking:
                    return False

                # Calculate wait time needed for 1 token
                wait_time = (1 - self.tokens) / self.rate

                # Check timeout
                if timeout is not None:
                    # If we've already waited too long or the next wait will push us over
                    elapsed_total = now - start_wait_time
                    if elapsed_total + wait_time > timeout:
                        return False

            # Sleep outside lock to allow other threads to run
            # When we wake up, we must loop back and re-check/re-calculate tokens
            # because another thread might have stolen the token we waited for.
            time.sleep(wait_time)

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False

    def reset(self):
        """Reset the token bucket to full capacity."""
        with self._lock:
            self.tokens = self.burst
            self.last_update = time.monotonic()


# Global rate limiter instance for API calls
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter(rate: float = 10.0, burst: Optional[int] = None) -> RateLimiter:
    """
    Get or create the global rate limiter.

    Args:
        rate: Requests per second (default: 10)
        burst: Burst size

    Returns:
        RateLimiter instance
    """
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RateLimiter(rate, burst)
    return _global_limiter


def reset_global_limiter():
    """Reset the global rate limiter."""
    global _global_limiter
    if _global_limiter is not None:
        _global_limiter.reset()


class PerEndpointLimiter:
    """
    Rate limiter with separate buckets per endpoint.

    Useful when you want to rate limit different API endpoints
    independently.

    Usage:
        limiter = PerEndpointLimiter(default_rate=10)

        # Rate limit /chat/completions
        limiter.acquire("/chat/completions")

        # Different limit for /embeddings
        limiter.set_rate("/embeddings", 50)
    """

    def __init__(self, default_rate: float = 10.0, default_burst: Optional[int] = None):
        """
        Initialize per-endpoint limiter.

        Args:
            default_rate: Default rate for endpoints
            default_burst: Default burst size
        """
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.limiters: dict[str, RateLimiter] = {}
        self._lock = Lock()

    def get_limiter(self, endpoint: str) -> RateLimiter:
        """Get or create limiter for an endpoint."""
        with self._lock:
            if endpoint not in self.limiters:
                self.limiters[endpoint] = RateLimiter(
                    self.default_rate,
                    self.default_burst
                )
            return self.limiters[endpoint]

    def acquire(self, endpoint: str, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire a token for a specific endpoint."""
        limiter = self.get_limiter(endpoint)
        return limiter.acquire(blocking=blocking, timeout=timeout)

    def set_rate(self, endpoint: str, rate: float, burst: Optional[int] = None):
        """Set the rate for a specific endpoint."""
        with self._lock:
            if endpoint in self.limiters:
                # Update existing limiter
                limiter = self.limiters[endpoint]
                limiter.rate = rate
                limiter.burst = int(burst or rate)
            else:
                # Create new limiter with custom rate
                self.limiters[endpoint] = RateLimiter(rate, burst)

    def reset(self, endpoint: str = None):
        """
        Reset limiter(s).

        Args:
            endpoint: Specific endpoint to reset, or None to reset all
        """
        with self._lock:
            if endpoint:
                if endpoint in self.limiters:
                    self.limiters[endpoint].reset()
            else:
                for limiter in self.limiters.values():
                    limiter.reset()
