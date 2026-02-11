"""Rate limiting with token bucket algorithm."""

import asyncio
import time
from typing import Optional

from src.rag_system.utils.logger import get_logger
from src.rag_system.utils.exceptions import APIRateLimitError

logger = get_logger(__name__)


class TokenBucket:
    """Thread-safe and async-safe token bucket for rate limiting."""

    def __init__(self, capacity: float, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (burst size).
            refill_rate: Tokens added per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket (async).

        Args:
            tokens: Number of tokens to acquire.
            timeout: Maximum time to wait for tokens.

        Returns:
            bool: True if tokens were acquired, False if timed out.
        """
        async with self._lock:
            start_time = time.monotonic()
            while True:
                await self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        return False

                # Calculate time to wait for next refill
                wait_time = (tokens - self.tokens) / self.refill_rate
                await asyncio.sleep(min(wait_time, 0.1))

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens without blocking (non-async).

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            bool: True if tokens were acquired, False otherwise.
        """
        self._refill_sync()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

    def _refill_sync(self) -> None:
        """Refill tokens (sync version)."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now


class AsyncRateLimiter:
    """Async rate limiter for API calls."""

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20,
    ):
        """
        Initialize async rate limiter.

        Args:
            requests_per_second: Target rate (requests/sec).
            burst_size: Maximum burst size.
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.bucket = TokenBucket(
            capacity=burst_size,
            refill_rate=requests_per_second,
        )
        logger.info(
            f"Rate limiter initialized",
            requests_per_second=requests_per_second,
            burst_size=burst_size,
        )

    async def acquire(self, timeout: float = 300.0) -> None:
        """
        Acquire a token with backpressure.

        Args:
            timeout: Maximum time to wait (seconds).

        Raises:
            APIRateLimitError: If token cannot be acquired within timeout.
        """
        acquired = await self.bucket.acquire(tokens=1.0, timeout=timeout)
        if not acquired:
            raise APIRateLimitError(
                "Rate limit exceeded, could not acquire token within timeout",
                retry_after=int(timeout),
            )

    def reset(self) -> None:
        """Reset the token bucket to full capacity."""
        self.bucket.tokens = self.bucket.capacity
        self.bucket.last_refill = time.monotonic()
        logger.info("Rate limiter reset")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return False

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            dict: Stats including current tokens and refill rate.
        """
        return {
            "current_tokens": round(self.bucket.tokens, 2),
            "capacity": self.bucket.capacity,
            "refill_rate": self.requests_per_second,
        }
