"""FastAPI application factory with middleware, routers, and lifecycle management."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import structlog

from src.rag_system.api.routers import ingest, query, health, tenants
from src.rag_system.api.middleware.auth import APIKeyMiddleware
from src.rag_system.config import get_config
from src.rag_system.pipeline import create_pipeline

logger = structlog.get_logger(__name__)
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup and shutdown lifecycle."""
    global _pipeline
    logger.info("api_startup_begin")
    try:
        _pipeline = await create_pipeline()
        app.state.pipeline = _pipeline
        logger.info("api_startup_complete")
    except Exception as exc:
        logger.error("api_startup_failed", error=str(exc))
    yield
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(
        title="RAG Financial Multimodal API",
        description="Enterprise-grade multimodal RAG for financial document analysis",
        version="2.0.0",
        docs_url="/docs" if not cfg.is_production else None,
        redoc_url="/redoc" if not cfg.is_production else None,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not cfg.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Key auth (skip in dev if key not set)
    app.add_middleware(APIKeyMiddleware)

    # Request timing middleware
    @app.middleware("http")
    async def add_timing(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})

    # Routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(tenants.router, prefix="/api/v1", tags=["Tenants"])

    return app
