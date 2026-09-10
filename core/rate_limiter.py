"""
Rate limiting utilities for API calls.

Uses token bucket algorithm to prevent API abuse and rate limit errors.
"""

import time
from threading import Lock


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

    def __init__(self, rate: float, burst: int | None = None):
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

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
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
_global_limiter: RateLimiter | None = None






