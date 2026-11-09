"""API Key authentication middleware with hashed key storage."""

from __future__ import annotations

import hashlib
import hmac
import os

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from filing_qa.config import get_config

logger = structlog.get_logger(__name__)

_EXEMPT_PATHS = {"/health", "/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json"}


def unauthenticated_access_reason() -> str | None:
    """Why requests may pass without a key, or None when a key is required.

    With ``RAG_API_MASTER_KEY`` set, a key is always required. Without it, requests
    are rejected unless ``ENVIRONMENT=development`` is set explicitly (process env or
    .env; the config default does not count) or ``AUTH_DISABLED=true``.
    """
    if os.environ.get("RAG_API_MASTER_KEY", ""):
        return None
    if os.environ.get("AUTH_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        return "AUTH_DISABLED=true"
    cfg = get_config()
    if cfg.environment == "development" and "environment" in cfg.model_fields_set:
        return "ENVIRONMENT=development"
    return None


def log_auth_startup_status() -> None:
    """Log once at startup how the API authenticates requests."""
    reason = unauthenticated_access_reason()
    if reason:
        logger.warning(
            "api_auth_disabled",
            reason=reason,
            detail="RAG_API_MASTER_KEY is not set; requests are accepted without a key",
        )
    elif not os.environ.get("RAG_API_MASTER_KEY", ""):
        logger.error(
            "api_auth_unconfigured",
            detail="RAG_API_MASTER_KEY is not set; all API requests will be rejected (401)",
        )


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates X-API-Key header against hashed keys in environment."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path in _EXEMPT_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        master_key = os.environ.get("RAG_API_MASTER_KEY", "")
        if not master_key:
            # Fail closed: without a configured key, only an explicit development or
            # AUTH_DISABLED setting lets requests through.
            if unauthenticated_access_reason():
                return await call_next(request)
            return JSONResponse(
                status_code=401, content={"error": "API authentication is not configured"}
            )

        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            return JSONResponse(status_code=401, content={"error": "Missing X-API-Key header"})

        # Hash both (equal-length inputs), then compare in constant time.
        provided_hash = hashlib.sha256(api_key.encode()).digest()
        expected_hash = hashlib.sha256(master_key.encode()).digest()
        if not hmac.compare_digest(provided_hash, expected_hash):
            logger.warning(
                "invalid_api_key",
                path=path,
                ip=request.client.host if request.client else "unknown",
            )
            return JSONResponse(status_code=403, content={"error": "Invalid API key"})

        # Attach tenant from header or default
        request.state.tenant_id = request.headers.get("X-Tenant-ID", "default")
        return await call_next(request)
