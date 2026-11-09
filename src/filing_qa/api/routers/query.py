"""Query API router."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()
logger = structlog.get_logger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class QueryResponse(BaseModel):
    status: str
    query: str
    answer: Optional[str]
    sources: list
    guardrails: Dict[str, Any]
    metrics: Dict[str, Any]
    tenant_id: str


@router.post("/query", response_model=QueryResponse, summary="Query ingested documents")
async def query_documents(request: Request, body: QueryRequest):
    """Query the RAG pipeline with a financial question."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})

    effective_tenant = body.tenant_id or getattr(request.state, "tenant_id", "default")

    try:
        result = await pipeline.query(
            query_text=body.query,
            tenant_id=effective_tenant,
            top_k=body.top_k,
            filters=body.filters,
        )
        return result
    except Exception as exc:
        logger.error("query_endpoint_failed", error=str(exc))
        return JSONResponse(status_code=500, content={"error": str(exc)})


def _sse(event: str, data: Any) -> str:
    """One server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post(
    "/query/stream",
    summary="Query with a streamed answer (server-sent events)",
    response_class=StreamingResponse,
)
async def query_documents_stream(request: Request, body: QueryRequest):
    """Same request body as ``POST /query``; the response is ``text/event-stream``.

    Events, in order:

    * ``sources``: ``{"sources": [...], "analysis": {...}}`` once retrieval is done
    * ``token``: ``{"text": "..."}`` for each piece of the answer as it is generated
    * ``done``: the same JSON object ``POST /query`` returns (answer, sources,
      guardrails, metrics). Guardrails run on the complete answer, so treat
      streamed tokens as provisional until ``done`` reports ``guardrails``.
    * ``error``: ``{"error": "..."}`` if the pipeline fails mid-stream.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})

    effective_tenant = body.tenant_id or getattr(request.state, "tenant_id", "default")

    async def events() -> AsyncIterator[str]:
        try:
            async for event in pipeline.query_stream(
                query_text=body.query,
                tenant_id=effective_tenant,
                top_k=body.top_k,
                filters=body.filters,
            ):
                name = event.get("event", "message")
                if name == "done":
                    yield _sse("done", event.get("result", {}))
                else:
                    yield _sse(name, {k: v for k, v in event.items() if k != "event"})
        except Exception as exc:
            logger.error("query_stream_failed", error=str(exc))
            yield _sse("error", {"error": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Disable proxy buffering (e.g. nginx) so tokens reach the client as they arrive.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
