"""Tenant management endpoints (RBAC stub)."""
from typing import Optional

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
logger = structlog.get_logger(__name__)

class TenantCreate(BaseModel):
    tenant_id: str
    display_name: Optional[str] = None
    queries_per_day: int = 1000
    tokens_per_month: int = 10_000_000

@router.post("/tenants", summary="Register a new tenant")
async def create_tenant(body: TenantCreate):
    logger.info("tenant_created", tenant_id=body.tenant_id)
    return {"status": "created", "tenant_id": body.tenant_id, "config": body.model_dump()}

@router.get("/tenants/{tenant_id}/usage", summary="Get tenant usage summary")
async def get_usage(tenant_id: str):
    from src.rag_system.utils.cost_tracker import get_cost_tracker
    tracker = get_cost_tracker()
    summary = tracker.get_tenant_summary(tenant_id)
    if not summary:
        return {"tenant_id": tenant_id, "total_cost_usd": 0.0, "total_tokens": 0, "query_count": 0}
    return {"tenant_id": tenant_id, "total_cost_usd": summary.total_cost_usd,
            "total_tokens": summary.total_tokens, "query_count": summary.query_count}
