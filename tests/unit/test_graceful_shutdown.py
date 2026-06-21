"""Unit tests for graceful shutdown / connection draining (ShutdownState).

Verifies the mechanism that lets the API drain in-flight requests on
SIGTERM instead of dropping them mid-flight during a rolling deploy or
HPA scale-down:
  1. begin_shutdown() flips is_shutting_down immediately (no delay).
  2. wait_for_drain() blocks until in_flight_requests reaches zero.
  3. wait_for_drain() respects the bounded timeout and does not hang forever
     if a request never completes (e.g. a stuck connection).
  4. The readiness probe reports 503/"shutting_down" the instant shutdown
     begins, independent of pipeline health or remaining in-flight count.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from src.rag_system.api.app import ShutdownState


# ── ShutdownState core behavior ───────────────────────────────────────────────

class TestShutdownStateBasics:
    def test_starts_not_shutting_down(self):
        state = ShutdownState()
        assert state.is_shutting_down is False
        assert state.in_flight_requests == 0

    @pytest.mark.asyncio
    async def test_begin_shutdown_flips_flag_immediately(self):
        state = ShutdownState(drain_timeout_seconds=5.0)
        assert state.is_shutting_down is False
        await state.begin_shutdown()
        assert state.is_shutting_down is True

    @pytest.mark.asyncio
    async def test_request_started_increments_counter(self):
        state = ShutdownState()
        await state.request_started()
        assert state.in_flight_requests == 1
        await state.request_started()
        assert state.in_flight_requests == 2

    @pytest.mark.asyncio
    async def test_request_finished_decrements_counter(self):
        state = ShutdownState()
        await state.request_started()
        await state.request_started()
        await state.request_finished()
        assert state.in_flight_requests == 1

    @pytest.mark.asyncio
    async def test_request_finished_never_goes_negative(self):
        """Defensive: finishing more requests than started should clamp at 0,
        not go negative and corrupt the drain-completion check."""
        state = ShutdownState()
        await state.request_finished()
        await state.request_finished()
        assert state.in_flight_requests == 0

    @pytest.mark.asyncio
    async def test_concurrent_increments_are_consistent(self):
        """The internal lock must prevent lost updates under concurrency."""
        state = ShutdownState()
        await asyncio.gather(*[state.request_started() for _ in range(50)])
        assert state.in_flight_requests == 50
        await asyncio.gather(*[state.request_finished() for _ in range(50)])
        assert state.in_flight_requests == 0


# ── Drain behavior ─────────────────────────────────────────────────────────────

class TestShutdownStateDrain:
    @pytest.mark.asyncio
    async def test_drain_returns_immediately_with_zero_in_flight(self):
        state = ShutdownState(drain_timeout_seconds=5.0)
        start = time.monotonic()
        await state.wait_for_drain()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # should not wait the full timeout when already drained

    @pytest.mark.asyncio
    async def test_drain_waits_for_in_flight_to_reach_zero(self):
        state = ShutdownState(drain_timeout_seconds=5.0)
        await state.request_started()

        async def finish_after_delay():
            await asyncio.sleep(0.3)
            await state.request_finished()

        start = time.monotonic()
        await asyncio.gather(state.wait_for_drain(), finish_after_delay())
        elapsed = time.monotonic() - start

        assert state.in_flight_requests == 0
        # Should have waited roughly until the request finished, not the full 5s timeout
        assert 0.25 < elapsed < 2.0

    @pytest.mark.asyncio
    async def test_drain_respects_timeout_when_request_never_finishes(self):
        """A stuck/never-completing request must not hang shutdown forever —
        the drain must bail out after drain_timeout_seconds regardless."""
        state = ShutdownState(drain_timeout_seconds=0.3)
        await state.request_started()  # never finished

        start = time.monotonic()
        await state.wait_for_drain()
        elapsed = time.monotonic() - start

        # Bounded by the timeout, with reasonable slack for the poll interval
        assert elapsed < 1.0
        # The request is still "in flight" from the counter's perspective —
        # wait_for_drain does not force-clear it, it just stops waiting.
        assert state.in_flight_requests == 1

    @pytest.mark.asyncio
    async def test_drain_with_multiple_requests_finishing_at_different_times(self):
        state = ShutdownState(drain_timeout_seconds=5.0)
        await state.request_started()
        await state.request_started()
        await state.request_started()

        async def finish_one(delay: float):
            await asyncio.sleep(delay)
            await state.request_finished()

        await asyncio.gather(
            state.wait_for_drain(),
            finish_one(0.1),
            finish_one(0.2),
            finish_one(0.3),
        )
        assert state.in_flight_requests == 0


# ── Integration with FastAPI app (readiness probe + middleware) ─────────────

class TestReadinessProbeDuringShutdown:
    @pytest.mark.asyncio
    async def test_readyz_reports_shutting_down_immediately(self):
        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-test")
        os.environ.setdefault("ENVIRONMENT", "testing")

        from fastapi.testclient import TestClient
        from src.rag_system.api.app import create_app

        app = create_app()
        app.state.shutdown = ShutdownState(drain_timeout_seconds=5.0)
        app.state.pipeline = None  # pipeline state shouldn't matter once shutting down

        client = TestClient(app)

        # Before shutdown: not-ready because pipeline is None, but NOT
        # because of shutdown state.
        resp_before = client.get("/readyz")
        assert resp_before.status_code == 503
        assert resp_before.json().get("status") != "shutting_down"

        await app.state.shutdown.begin_shutdown()

        resp_after = client.get("/readyz")
        assert resp_after.status_code == 503
        assert resp_after.json()["status"] == "shutting_down"

    @pytest.mark.asyncio
    async def test_readyz_reports_in_flight_count_during_shutdown(self):
        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-test")
        os.environ.setdefault("ENVIRONMENT", "testing")

        from fastapi.testclient import TestClient
        from src.rag_system.api.app import create_app

        app = create_app()
        state = ShutdownState(drain_timeout_seconds=5.0)
        app.state.shutdown = state
        app.state.pipeline = None

        await state.request_started()
        await state.request_started()
        await state.begin_shutdown()

        client = TestClient(app)
        resp = client.get("/readyz")
        assert resp.status_code == 503
        assert resp.json()["in_flight_requests"] == 2

    def test_health_and_healthz_unaffected_by_shutdown(self):
        """Liveness probes should keep reporting alive during the drain
        window — only readiness should flip, so Kubernetes doesn't think
        the container itself crashed (which could trigger a restart loop)."""
        import os
        import asyncio as _asyncio
        os.environ.setdefault("OPENAI_API_KEY", "sk-test")
        os.environ.setdefault("ENVIRONMENT", "testing")

        from fastapi.testclient import TestClient
        from src.rag_system.api.app import create_app

        app = create_app()
        state = ShutdownState()
        app.state.shutdown = state
        app.state.pipeline = None
        _asyncio.get_event_loop().run_until_complete(state.begin_shutdown())

        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert client.get("/healthz").status_code == 200
