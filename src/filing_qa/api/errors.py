"""Error responses that never echo exception text to the client.

Failures are logged server-side with the exception and a request id; the client gets a
stable ``code``, a fixed ``message`` and the same ``request_id`` to quote in a report.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict

import structlog
from fastapi import Request
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

# Stable codes and the only messages clients ever see for them.
MESSAGES: Dict[str, str] = {
    "internal_error": "Internal server error.",
    "query_failed": "The query could not be completed.",
    "ingest_failed": "The document could not be ingested.",
    "document_list_failed": "Documents could not be listed.",
    "document_delete_failed": "The document could not be deleted.",
    "health_check_failed": "Health check failed.",
}


def request_id(request: Request) -> str:
    """The caller's X-Request-ID if well-formed, else a new id; stable per request."""
    existing = getattr(request.state, "request_id", None)
    if existing:
        return str(existing)
    header = request.headers.get("X-Request-ID", "")
    rid = header if _REQUEST_ID_RE.match(header) else uuid.uuid4().hex
    request.state.request_id = rid
    return rid


def error_body(request: Request, code: str, event: str, exc: BaseException) -> Dict[str, Any]:
    """Log ``exc`` server-side and return the safe body for ``code``."""
    rid = request_id(request)
    logger.error(
        event,
        code=code,
        request_id=rid,
        path=request.url.path,
        error_type=type(exc).__name__,
        error=str(exc)[:500],
        exc_info=exc,
    )
    return {"code": code, "message": MESSAGES[code], "request_id": rid}


def error_response(
    request: Request, code: str, event: str, exc: BaseException, status_code: int = 500
) -> JSONResponse:
    body = error_body(request, code, event, exc)
    return JSONResponse(
        status_code=status_code, content=body, headers={"X-Request-ID": body["request_id"]}
    )
