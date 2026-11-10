"""Ingest API router."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from filing_qa.api.errors import error_response

router = APIRouter()
logger = structlog.get_logger(__name__)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]")


def _safe_filename(name: Optional[str]) -> str:
    """The upload's base name, usable as a file name and as the citation source."""
    base = Path((name or "").replace("\\", "/")).name
    base = _UNSAFE_CHARS.sub("_", base).strip(" .")[:200]
    if not base:
        base = "document"
    return base if base.lower().endswith(".pdf") else f"{base}.pdf"


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
        # The parser cites a document by its file name, so the upload is saved under
        # its own (sanitized) name inside a private temp dir, which is removed after.
        with tempfile.TemporaryDirectory(prefix="filing-qa-ingest-") as tmp_dir:
            tmp_path = Path(tmp_dir) / _safe_filename(file.filename)
            tmp_path.write_bytes(contents)
            result = await pipeline.ingest(
                file_paths=[str(tmp_path)],
                tenant_id=effective_tenant,
                process_vision=process_vision,
            )
        result["original_filename"] = file.filename
        return result
    except Exception as exc:
        return error_response(request, "ingest_failed", "ingest_endpoint_failed", exc)
