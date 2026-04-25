"""FastAPI application factory — v2.0 enterprise edition.

Routers: health, ingest, query, documents, tenants, feedback
Middleware: CORS, API-Key auth, request timing
Lifecycle: async pipeline warmup on startup, graceful shutdown
"""
from __future__ import annotations

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup: warm pipeline. Shutdown: log cleanly."""
    global _pipeline
    logger.info("api_startup_begin")
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
    logger.info("api_shutdown")


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
