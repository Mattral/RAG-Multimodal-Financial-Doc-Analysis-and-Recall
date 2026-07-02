"""Production retry policy: exponential backoff with full jitter, tenacity integration.

Strategy: Full Jitter (AWS-recommended) — sleep = random(0, min(cap, base * 2^attempt))
Reference: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
"""
from __future__ import annotations

import asyncio
import functools
import random
import time
from typing import Any, Awaitable, Callable, Optional, Tuple, Type, TypeVar

import structlog

from src.rag_system.utils.exceptions import (
    APIRateLimitError,
    APITimeoutError,
    MaxRetriesExceededError,
    RetryableError,
)

logger = structlog.get_logger(__name__)
T = TypeVar("T")

_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


class RetryPolicy:
    """Full-jitter exponential backoff retry policy.

    Args:
        max_attempts: Total attempts including first try.
        base_delay: Base delay in seconds.
        cap_delay: Maximum delay ceiling in seconds.
        backoff_factor: Multiplier per attempt (default 2 → doubles each time).
        retryable_exceptions: Exception types that trigger retry.
        on_retry: Optional async callback(attempt, exception, delay).
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        cap_delay: float = 60.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        on_retry: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.cap_delay = cap_delay
        self.backoff_factor = backoff_factor
        self.retryable_exceptions = retryable_exceptions or (
            RetryableError, APIRateLimitError, APITimeoutError,
            asyncio.TimeoutError, ConnectionError, OSError,
        )
        self.on_retry = on_retry

    def get_delay(self, attempt: int) -> float:
        """Full jitter: sleep = random(0, min(cap, base * factor^attempt))."""
        ceiling = min(self.cap_delay, self.base_delay * (self.backoff_factor ** attempt))
        return random.uniform(0, ceiling)

    async def execute(
        self,
        coro_func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute async callable with retry.

        Raises:
            MaxRetriesExceededError: After all attempts are exhausted.
            Original exception: For non-retryable errors.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            try:
                return await coro_func(*args, **kwargs)

            except self.retryable_exceptions as exc:
                last_exc = exc
                if attempt >= self.max_attempts - 1:
                    break

                # Honour Retry-After header from rate limit errors
                if isinstance(exc, APIRateLimitError):
                    delay = float(exc.details.get("retry_after_seconds", self.get_delay(attempt)))
                else:
                    delay = self.get_delay(attempt)

                logger.warning(
                    "retry_triggered",
                    func=getattr(coro_func, "__name__", "unknown"),
                    attempt=attempt + 1,
                    max_attempts=self.max_attempts,
                    delay_s=round(delay, 2),
                    error=str(exc)[:120],
                )

                if self.on_retry:
                    await self.on_retry(attempt, exc, delay)

                await asyncio.sleep(delay)

            except Exception as exc:
                # Non-retryable — propagate immediately
                logger.error(
                    "non_retryable_error",
                    func=getattr(coro_func, "__name__", "unknown"),
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                )
                raise

        raise MaxRetriesExceededError(
            f"Exhausted {self.max_attempts} attempts",
            retry_count=self.max_attempts,
            details={"last_error": str(last_exc)[:200]},
        )

    def execute_sync(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute synchronous callable with retry."""
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.retryable_exceptions as exc:
                last_exc = exc
                if attempt >= self.max_attempts - 1:
                    break
                delay = self.get_delay(attempt)
                logger.warning("sync_retry_triggered", attempt=attempt + 1, delay_s=round(delay, 2))
                time.sleep(delay)
            except Exception:
                raise

        raise MaxRetriesExceededError(
            f"Exhausted {self.max_attempts} attempts",
            retry_count=self.max_attempts,
            details={"last_error": str(last_exc)[:200]},
        )


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    cap_delay: float = 60.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that wraps an async function with retry logic.

    Usage::

        @with_retry(max_attempts=3, base_delay=0.5)
        async def call_openai(...):
            ...
    """
    policy = RetryPolicy(max_attempts=max_attempts, base_delay=base_delay, cap_delay=cap_delay)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await policy.execute(func, *args, **kwargs)
        return wrapper
    return decorator


# ── Default policies ──────────────────────────────────────────────────────────

DEFAULT_POLICY = RetryPolicy(max_attempts=3, base_delay=1.0, cap_delay=30.0)
VISION_POLICY = RetryPolicy(max_attempts=3, base_delay=2.0, cap_delay=60.0)
EMBEDDING_POLICY = RetryPolicy(max_attempts=5, base_delay=0.5, cap_delay=20.0)
STRICT_POLICY = RetryPolicy(max_attempts=1)  # No retries for guardrail-sensitive paths
