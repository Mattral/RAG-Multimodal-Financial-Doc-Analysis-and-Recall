"""Retry logic with exponential backoff and jitter."""

import asyncio
import random
from typing import Awaitable, Callable, Optional, TypeVar, Any
import time

from src.rag_system.utils.logger import get_logger
from src.rag_system.utils.exceptions import RetryableError, APIRateLimitError

T = TypeVar("T")
logger = get_logger(__name__)


class RetryPolicy:
    """Exponential backoff retry policy with jitter."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        backoff_factor: float = 2.0,
        jitter_factor: float = 0.1,
    ):
        """
        Initialize retry policy.

        Args:
            max_attempts: Maximum number of retry attempts.
            base_delay_seconds: Initial delay between retries.
            max_delay_seconds: Maximum delay between retries.
            backoff_factor: Factor to multiply delay by after each retry.
            jitter_factor: Factor for random jitter (0.0 to 1.0).
        """
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.backoff_factor = backoff_factor
        self.jitter_factor = jitter_factor

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number with exponential backoff and jitter.

        Args:
            attempt: The current attempt number (0-indexed).

        Returns:
            float: Delay in seconds.
        """
        delay = self.base_delay_seconds * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay_seconds)
        jitter = random.uniform(0, delay * self.jitter_factor)
        return delay + jitter

    async def execute_async(
        self,
        coro_func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute an async function with retry logic.

        Args:
            coro_func: Async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            T: Result from the function.

        Raises:
            RetryableError: If all retry attempts are exhausted.
        """
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                logger.debug(
                    f"Executing {coro_func.__name__}, attempt {attempt + 1}/{self.max_attempts}"
                )
                result = await coro_func(*args, **kwargs)
                return result

            except APIRateLimitError as e:
                last_exception = e
                retry_after = e.details.get("retry_after_seconds", self.get_delay(attempt))
                if attempt < self.max_attempts - 1:
                    logger.warning(
                        f"Rate limited on {coro_func.__name__}, retrying after {retry_after}s",
                        retry_after=retry_after,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(retry_after)
                else:
                    logger.error(
                        f"Exhausted retries for {coro_func.__name__} due to rate limit"
                    )
                    raise

            except (asyncio.TimeoutError, ConnectionError, OSError) as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = self.get_delay(attempt)
                    logger.warning(
                        f"Transient error on {coro_func.__name__}, retrying after {delay}s",
                        error_type=type(e).__name__,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Exhausted retries for {coro_func.__name__} due to transient error",
                        error_type=type(e).__name__,
                    )
                    raise

            except Exception as e:
                last_exception = e
                logger.error(
                    f"Non-retryable error on {coro_func.__name__}",
                    error_type=type(e).__name__,
                )
                raise

        # Should not reach here, but just in case
        raise RetryableError(
            f"Failed after {self.max_attempts} attempts",
            details={"function": coro_func.__name__, "last_error": str(last_exception)},
        )

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a sync function with retry logic.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            T: Result from the function.

        Raises:
            RetryableError: If all retry attempts are exhausted.
        """
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                logger.debug(
                    f"Executing {func.__name__}, attempt {attempt + 1}/{self.max_attempts}"
                )
                result = func(*args, **kwargs)
                return result

            except (ConnectionError, OSError, TimeoutError) as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = self.get_delay(attempt)
                    logger.warning(
                        f"Transient error on {func.__name__}, retrying after {delay}s",
                        error_type=type(e).__name__,
                        attempt=attempt + 1,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Exhausted retries for {func.__name__} due to transient error",
                        error_type=type(e).__name__,
                    )
                    raise

            except Exception as e:
                last_exception = e
                logger.error(
                    f"Non-retryable error on {func.__name__}",
                    error_type=type(e).__name__,
                )
                raise

        # Should not reach here, but just in case
        raise RetryableError(
            f"Failed after {self.max_attempts} attempts",
            details={"function": func.__name__, "last_error": str(last_exception)},
        )
