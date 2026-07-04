"""Query API router."""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
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
