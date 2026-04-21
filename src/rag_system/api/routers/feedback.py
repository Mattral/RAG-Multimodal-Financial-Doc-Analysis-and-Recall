"""Human-in-the-loop feedback endpoint.

Guideline §3: 'Human-in-the-Loop / Feedback Loop: thumbs up/down + free-text
feedback stored with query_id, used later for fine-tuning reranker or reward model.'
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

router = APIRouter()
logger = structlog.get_logger(__name__)

_FEEDBACK_PATH = Path("./data/feedback.jsonl")


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Query ID from the /query response")
    query_text: str = Field(..., max_length=2000)
    answer_text: str = Field(..., max_length=10000)
    rating: Literal["thumbs_up", "thumbs_down", "neutral"] = Field(
        ..., description="Quick rating"
    )
    comment: Optional[str] = Field(None, max_length=2000, description="Optional free-text")
    tenant_id: Optional[str] = None
    model_used: Optional[str] = None
    latency_ms: Optional[float] = None
    sources: Optional[list] = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    message: str


@router.post("/feedback", response_model=FeedbackResponse, summary="Submit answer feedback")
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Submit thumbs-up/down + optional comment on a generated answer.

    Feedback is stored in `data/feedback.jsonl` and can be used for:
    - Fine-tuning reranker with preference pairs
    - Reward model training
    - Quality monitoring dashboards
    - Golden dataset expansion (high-quality rated answers)
    """
    import uuid

    feedback_id = str(uuid.uuid4())[:8]
    tenant_id = body.tenant_id or getattr(request.state, "tenant_id", "default")

    record = {
        "feedback_id": feedback_id,
        "query_id": body.query_id,
        "tenant_id": tenant_id,
        "rating": body.rating,
        "comment": body.comment,
        "query_text": body.query_text,
        "answer_preview": body.answer_text[:500],
        "model_used": body.model_used,
        "latency_ms": body.latency_ms,
        "num_sources": len(body.sources or []),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_FEEDBACK_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.error("feedback_write_failed", error=str(exc))
        return JSONResponse(status_code=500, content={"error": "Failed to save feedback"})

    logger.info(
        "feedback_received",
        feedback_id=feedback_id,
        rating=body.rating,
        tenant_id=tenant_id,
    )

    return FeedbackResponse(
        status="saved",
        feedback_id=feedback_id,
        message=f"Thank you for your feedback ({body.rating}). "
                f"It will be used to improve answer quality.",
    )


@router.get("/feedback/summary", summary="Get feedback summary for a tenant")
async def feedback_summary(request: Request, limit: int = 100):
    """Return recent feedback records for monitoring dashboards."""
    tenant_id = getattr(request.state, "tenant_id", "default")
    records = []
    try:
        if _FEEDBACK_PATH.exists():
            with open(_FEEDBACK_PATH) as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        if r.get("tenant_id") == tenant_id:
                            records.append(r)
                    except Exception:
                        continue
    except Exception as exc:
        logger.error("feedback_read_failed", error=str(exc))

    records = records[-limit:]
    thumbs_up = sum(1 for r in records if r.get("rating") == "thumbs_up")
    thumbs_down = sum(1 for r in records if r.get("rating") == "thumbs_down")
    total = len(records)

    return {
        "tenant_id": tenant_id,
        "total_feedback": total,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "satisfaction_rate": thumbs_up / total if total > 0 else None,
        "recent_records": records[-10:],
    }
