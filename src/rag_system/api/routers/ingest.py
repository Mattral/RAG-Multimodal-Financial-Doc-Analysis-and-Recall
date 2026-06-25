"""Ingest API router."""
from __future__ import annotations

import tempfile
from typing import Optional

import structlog
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()
logger = structlog.get_logger(__name__)

@router.post("/ingest", summary="Ingest a financial document")
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="PDF file to ingest"),
    tenant_id: Optional[str] = Form(None),
    process_vision: bool = Form(True),
):
    """Upload and ingest a PDF financial document."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})

    effective_tenant = tenant_id or getattr(request.state, "tenant_id", "default")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = await pipeline.ingest(
            file_paths=[tmp_path],
            tenant_id=effective_tenant,
            process_vision=process_vision,
        )
        result["original_filename"] = file.filename
        return result
    except Exception as exc:
        logger.error("ingest_endpoint_failed", error=str(exc))
        return JSONResponse(status_code=500, content={"error": str(exc)})
