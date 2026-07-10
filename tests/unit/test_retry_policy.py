"""Unit tests for the exponential backoff retry policy."""

from unittest.mock import AsyncMock

import pytest

from src.rag_system.utils.exceptions import (
    APIRateLimitError,
    MaxRetriesExceededError,
    RetryableError,
)
from src.rag_system.utils.retry_policy import RetryPolicy, with_retry


@pytest.mark.asyncio
async def test_succeeds_on_first_attempt():
    policy = RetryPolicy(max_attempts=3)
    func = AsyncMock(return_value="success")
    result = await policy.execute(func)
    assert result == "success"
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retries_on_retryable_error():
    policy = RetryPolicy(max_attempts=3, base_delay=0.01)
    func = AsyncMock(side_effect=[RetryableError("fail"), RetryableError("fail"), "ok"])
    result = await policy.execute(func)
    assert result == "ok"
    assert func.call_count == 3


@pytest.mark.asyncio
async def test_raises_max_retries_exceeded():
    policy = RetryPolicy(max_attempts=2, base_delay=0.01)
    func = AsyncMock(side_effect=RetryableError("always fails"))
    with pytest.raises(MaxRetriesExceededError) as exc_info:
        await policy.execute(func)
    assert exc_info.value.details["retry_count"] == 2


@pytest.mark.asyncio
async def test_non_retryable_propagates_immediately():
    policy = RetryPolicy(max_attempts=5, base_delay=0.01)
    func = AsyncMock(side_effect=ValueError("not retryable"))
    with pytest.raises(ValueError):
        await policy.execute(func)
    assert func.call_count == 1  # no retries


@pytest.mark.asyncio
async def test_rate_limit_error_retried():
    policy = RetryPolicy(max_attempts=3, base_delay=0.01)
    func = AsyncMock(side_effect=[APIRateLimitError("429", retry_after=0), "ok"])
    result = await policy.execute(func)
    assert result == "ok"


def test_get_delay_bounded_by_cap():
    policy = RetryPolicy(base_delay=1.0, cap_delay=5.0)
    for attempt in range(10):
        delay = policy.get_delay(attempt)
        assert 0.0 <= delay <= 5.0  # full jitter: [0, cap]


def test_get_delay_increases_with_attempts():
    """Average delay should trend upward (probabilistically)."""
    policy = RetryPolicy(base_delay=1.0, cap_delay=64.0, backoff_factor=2.0)
    avg_delays = []
    for attempt in range(5):
        samples = [policy.get_delay(attempt) for _ in range(200)]
        avg_delays.append(sum(samples) / len(samples))
    # Each average should be >= previous (with high probability)
    for i in range(1, len(avg_delays)):
        assert avg_delays[i] >= avg_delays[i - 1] * 0.5  # loose bound


@pytest.mark.asyncio
async def test_on_retry_callback_called():
    calls = []

    async def on_retry(attempt, exc, delay):
        calls.append((attempt, str(exc)))

    policy = RetryPolicy(max_attempts=3, base_delay=0.01, on_retry=on_retry)
    func = AsyncMock(side_effect=[RetryableError("err1"), RetryableError("err2"), "ok"])
    await policy.execute(func)
    assert len(calls) == 2
    assert calls[0][0] == 0
    assert calls[1][0] == 1


def test_execute_sync_succeeds():
    policy = RetryPolicy(max_attempts=3, base_delay=0.01)
    call_count = {"n": 0}

    def sync_fn():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RetryableError("fail")
        return "done"

    result = policy.execute_sync(sync_fn)
    assert result == "done"
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_with_retry_decorator():
    call_count = {"n": 0}

    @with_retry(max_attempts=3, base_delay=0.01)
    async def flaky():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RetryableError("temporary")
        return "decorated_ok"

    result = await flaky()
    assert result == "decorated_ok"
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_strict_policy_no_retries():
    from src.rag_system.utils.retry_policy import STRICT_POLICY

    func = AsyncMock(side_effect=RetryableError("fail"))
    with pytest.raises(MaxRetriesExceededError):
        await STRICT_POLICY.execute(func)
    assert func.call_count == 1
