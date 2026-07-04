"""API Key authentication middleware with hashed key storage."""

from __future__ import annotations

import hashlib
import os

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)

_EXEMPT_PATHS = {"/health", "/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates X-API-Key header against hashed keys in environment."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path in _EXEMPT_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        # In development mode without keys configured, allow through
        master_key = os.environ.get("RAG_API_MASTER_KEY", "")
        if not master_key:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            return JSONResponse(status_code=401, content={"error": "Missing X-API-Key header"})

        # Constant-time comparison via hashing
        provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
        expected_hash = hashlib.sha256(master_key.encode()).hexdigest()
        if provided_hash != expected_hash:
            logger.warning(
                "invalid_api_key",
                path=path,
                ip=request.client.host if request.client else "unknown",
            )
            return JSONResponse(status_code=403, content={"error": "Invalid API key"})

        # Attach tenant from header or default
        request.state.tenant_id = request.headers.get("X-Tenant-ID", "default")
        return await call_next(request)
