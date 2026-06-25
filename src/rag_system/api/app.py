"""FastAPI application factory — v2.0 enterprise edition.

Routers: health, ingest, query, documents, tenants, feedback
Middleware: CORS, API-Key auth, request timing, in-flight request tracking
Lifecycle: async pipeline warmup on startup, graceful shutdown with connection
draining (flip readiness false immediately on SIGTERM, then wait up to
SHUTDOWN_DRAIN_TIMEOUT_SECONDS for in-flight requests to finish before exit).

Why this matters operationally: a naive shutdown handler that just logs and
exits can drop in-flight queries/ingests when Kubernetes sends SIGTERM during
a rolling deploy or HPA scale-down. The pattern here follows standard
zero-downtime deployment practice: (1) readiness probe flips unhealthy the
instant shutdown begins, so the Service/Ingress stops routing new traffic to
this pod within one probe interval, (2) the process keeps serving requests
already in flight, (3) after a bounded drain window OR all requests complete
(whichever is first), the process exits cleanly.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.rag_system.api.middleware.auth import APIKeyMiddleware
from src.rag_system.api.routers import (
    documents,
    feedback,
    health,
    ingest,
    query,
    tenants,
)
from src.rag_system.config import get_config

logger = structlog.get_logger(__name__)
_pipeline = None


class ShutdownState:
    """Shared mutable state for graceful shutdown / connection draining.

    Exposed on app.state so the readiness probe and the in-flight-request
    middleware can both observe and update it without global singletons.
    """

    def __init__(self, drain_timeout_seconds: float = 30.0) -> None:
        self.is_shutting_down: bool = False
        self.in_flight_requests: int = 0
        self.drain_timeout_seconds = drain_timeout_seconds
        self._lock = asyncio.Lock()

    async def request_started(self) -> None:
        async with self._lock:
            self.in_flight_requests += 1

    async def request_finished(self) -> None:
        async with self._lock:
            self.in_flight_requests = max(0, self.in_flight_requests - 1)

    async def begin_shutdown(self) -> None:
        self.is_shutting_down = True
        logger.info(
            "graceful_shutdown_begin",
            in_flight_requests=self.in_flight_requests,
            drain_timeout_s=self.drain_timeout_seconds,
        )

    async def wait_for_drain(self) -> None:
        """Block until in-flight requests reach zero or the timeout elapses."""
        deadline = time.monotonic() + self.drain_timeout_seconds
        poll_interval = 0.1
        while self.in_flight_requests > 0 and time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)
        remaining = self.in_flight_requests
        if remaining > 0:
            logger.warning(
                "graceful_shutdown_drain_timeout_exceeded",
                requests_still_in_flight=remaining,
                drain_timeout_s=self.drain_timeout_seconds,
            )
        else:
            logger.info("graceful_shutdown_drain_complete")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup: warm pipeline. Shutdown: drain in-flight requests before exit."""
    global _pipeline
    cfg = get_config()
    app.state.shutdown = ShutdownState(
        drain_timeout_seconds=cfg.observability_config.shutdown_drain_timeout_seconds
    )

    logger.info("api_startup_begin")
    if hasattr(app.state, "pipeline"):
        # Pipeline attribute was pre-set (e.g. a test fixture injecting a
        # mock, or explicitly None to simulate degraded mode) before the
        # TestClient context manager triggered this lifespan. Respect it —
        # skip real construction so the test's intent survives startup.
        _pipeline = app.state.pipeline
        logger.info("api_startup_using_preset_pipeline", is_none=_pipeline is None)
    else:
        try:
            from src.rag_system.pipeline import create_pipeline
            _pipeline = await create_pipeline()
            app.state.pipeline = _pipeline
            logger.info("api_startup_complete")
        except Exception as exc:
            logger.error("api_startup_failed", error=str(exc))
            # Start in degraded mode — health check will report not_ready
            app.state.pipeline = None

    yield

    # ── Graceful shutdown ────────────────────────────────────────────────────
    # uvicorn invokes this on SIGTERM/SIGINT after it stops accepting new
    # connections at the socket level; we additionally flip readiness so any
    # load balancer polling /readyz (not just the socket) routes away first.
    await app.state.shutdown.begin_shutdown()
    await app.state.shutdown.wait_for_drain()
    logger.info("api_shutdown_complete")


def create_app() -> FastAPI:
    """Application factory — called by uvicorn with --factory."""
    cfg = get_config()

    app = FastAPI(
        title="RAG Financial Multimodal API",
        description=(
            "Enterprise-grade multimodal RAG for financial document analysis.\n\n"
            "Features: PDF ingestion, GPT-4o/Gemini/Qwen2-VL vision, hybrid retrieval, "
            "numeric guardrails, PII redaction, multi-tenancy, full OTel observability."
        ),
        version="2.0.0",
        docs_url="/docs" if not cfg.is_production else None,
        redoc_url="/redoc" if not cfg.is_production else None,
        openapi_tags=[
            {"name": "Health", "description": "Liveness and readiness probes"},
            {"name": "Ingest", "description": "PDF ingestion endpoints"},
            {"name": "Query", "description": "RAG query endpoints"},
            {"name": "Documents", "description": "Document management and GDPR deletion"},
            {"name": "Tenants", "description": "Multi-tenant management"},
            {"name": "Feedback", "description": "Human-in-the-loop quality feedback"},
        ],
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not cfg.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APIKeyMiddleware)

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response

    @app.middleware("http")
    async def track_in_flight_requests(request: Request, call_next) -> Response:
        """Track in-flight requests so graceful shutdown can drain correctly.

        Health probe paths are excluded from tracking — they're cheap,
        high-frequency, and shouldn't factor into "are we done draining".
        """
        shutdown_state: ShutdownState = request.app.state.shutdown
        if request.url.path in ("/health", "/healthz", "/readyz"):
            return await call_next(request)

        await shutdown_state.request_started()
        try:
            return await call_next(request)
        finally:
            await shutdown_state.request_finished()

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": str(exc)[:200]},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router, tags=["Health"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(tenants.router, prefix="/api/v1", tags=["Tenants"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])

    return app
