"""Document management endpoints: list, version history, delete (GDPR/CCPA).

Guideline §3.1: 'GDPR/CCPA-style data subject access/deletion (delete by tenant/doc),
consent/versioning, data residency controls.'
"""
from __future__ import annotations

from typing import Optional

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/documents", summary="List all ingested documents for a tenant")
async def list_documents(request: Request, tenant_id: Optional[str] = None):
    """Return all documents and their current version info for a tenant."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})
    effective_tenant = tenant_id or getattr(request.state, "tenant_id", "default")
    try:
        docs = await pipeline.list_documents(tenant_id=effective_tenant)
        return {"tenant_id": effective_tenant, "total": len(docs), "documents": docs}
    except Exception as exc:
        logger.error("list_documents_failed", error=str(exc))
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.delete("/documents/{doc_id:path}", summary="GDPR/CCPA soft-delete a document")
async def delete_document(request: Request, doc_id: str, tenant_id: Optional[str] = None):
    """Soft-delete a document for GDPR/CCPA compliance. Logged to audit trail."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})
    effective_tenant = tenant_id or getattr(request.state, "tenant_id", "default")
    try:
        result = await pipeline.delete_document(doc_id, tenant_id=effective_tenant)
        return result
    except Exception as exc:
        logger.error("delete_document_failed", doc_id=doc_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": str(exc)})
