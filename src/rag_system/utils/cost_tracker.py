"""Per-query, per-tenant cost tracking for LLM and embedding calls.

Pricing data is approximate and should be updated as providers change rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# USD per 1M tokens (as of mid-2025)
_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"prompt": 5.0, "completion": 15.0},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4-vision-preview": {"prompt": 10.0, "completion": 30.0},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "text-embedding-3-small": {"prompt": 0.02, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.13, "completion": 0.0},
    "text-embedding-ada-002": {"prompt": 0.10, "completion": 0.0},
}


@dataclass
class CostRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "unknown"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        pricing = _PRICING.get(self.model, {"prompt": 0.0, "completion": 0.0})
        return (
            self.prompt_tokens * pricing["prompt"]
            + self.completion_tokens * pricing["completion"]
        ) / 1_000_000


@dataclass
class TenantCostAccumulator:
    """Accumulates cost per tenant for quota enforcement."""

    tenant_id: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    query_count: int = 0
    records: list = field(default_factory=list)

    def add(self, record: CostRecord) -> None:
        self.total_cost_usd += record.cost_usd
        self.total_tokens += record.total_tokens
        self.query_count += 1
        self.records.append(record)


class CostTracker:
    """Thread-safe cost tracker with per-tenant aggregation."""

    def __init__(self) -> None:
        self._tenants: Dict[str, TenantCostAccumulator] = {}

    def record(
        self,
        tenant_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
    ) -> CostRecord:
        rec = CostRecord(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )
        if tenant_id not in self._tenants:
            self._tenants[tenant_id] = TenantCostAccumulator(tenant_id=tenant_id)
        self._tenants[tenant_id].add(rec)
        return rec

    def get_tenant_summary(self, tenant_id: str) -> Optional[TenantCostAccumulator]:
        return self._tenants.get(tenant_id)

    def check_quota(self, tenant_id: str, monthly_token_limit: int) -> bool:
        """Return False if tenant has exceeded monthly token quota."""
        acc = self._tenants.get(tenant_id)
        if acc is None:
            return True
        return acc.total_tokens <= monthly_token_limit


# Global singleton
_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    return _tracker
