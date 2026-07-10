"""Unit tests for src/rag_system/utils/cost_tracker.py.

Property-based math correctness (monotonic cost, non-negative cost) is
already covered in test_property_based.py. This file covers:
  - The fixed pricing table values
  - TenantCostAccumulator bookkeeping
  - check_quota() boundary behaviour
  - The Prometheus telemetry gauge wiring added alongside the SLO
    burn-rate alerting rules (scripts/alerting/slo-burn-rate.yml), which
    depend on rag_tenant_monthly_tokens_used / rag_tenant_monthly_token_quota
    actually being published on every check_quota() call.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.rag_system.utils.cost_tracker import (
    _PRICING,
    CostRecord,
    CostTracker,
    TenantCostAccumulator,
    get_cost_tracker,
)

# ── CostRecord pricing ─────────────────────────────────────────────────────────


class TestCostRecordPricing:
    def test_gpt4o_pricing(self):
        rec = CostRecord(prompt_tokens=1_000_000, completion_tokens=1_000_000, model="gpt-4o")
        assert rec.cost_usd == pytest.approx(5.0 + 15.0)

    def test_gpt4o_mini_cheaper_than_gpt4o(self):
        mini = CostRecord(prompt_tokens=1000, completion_tokens=1000, model="gpt-4o-mini")
        full = CostRecord(prompt_tokens=1000, completion_tokens=1000, model="gpt-4o")
        assert mini.cost_usd < full.cost_usd

    def test_embedding_model_has_zero_completion_cost(self):
        rec = CostRecord(prompt_tokens=1000, completion_tokens=500, model="text-embedding-3-small")
        # Embeddings have no "completion" concept; pricing table sets it to 0
        prompt_only = CostRecord(
            prompt_tokens=1000, completion_tokens=0, model="text-embedding-3-small"
        )
        assert rec.cost_usd == prompt_only.cost_usd

    def test_unknown_model_defaults_to_zero_cost(self):
        rec = CostRecord(
            prompt_tokens=1_000_000, completion_tokens=1_000_000, model="some-future-model-xyz"
        )
        assert rec.cost_usd == 0.0

    def test_total_tokens_sums_prompt_and_completion(self):
        rec = CostRecord(prompt_tokens=300, completion_tokens=120, model="gpt-4o-mini")
        assert rec.total_tokens == 420

    def test_zero_tokens_zero_cost(self):
        rec = CostRecord(prompt_tokens=0, completion_tokens=0, model="gpt-4o")
        assert rec.cost_usd == 0.0

    def test_pricing_table_has_no_negative_rates(self):
        for _model, rates in _PRICING.items():
            assert rates["prompt"] >= 0.0
            assert rates["completion"] >= 0.0


# ── TenantCostAccumulator ──────────────────────────────────────────────────────


class TestTenantCostAccumulator:
    def test_starts_empty(self):
        acc = TenantCostAccumulator(tenant_id="acme")
        assert acc.total_cost_usd == 0.0
        assert acc.total_tokens == 0
        assert acc.query_count == 0

    def test_add_accumulates(self):
        acc = TenantCostAccumulator(tenant_id="acme")
        acc.add(CostRecord(prompt_tokens=100, completion_tokens=50, model="gpt-4o-mini"))
        acc.add(CostRecord(prompt_tokens=200, completion_tokens=80, model="gpt-4o-mini"))
        assert acc.query_count == 2
        assert acc.total_tokens == 430
        assert len(acc.records) == 2

    def test_add_preserves_record_history(self):
        acc = TenantCostAccumulator(tenant_id="acme")
        rec = CostRecord(prompt_tokens=10, completion_tokens=5, model="gpt-4o")
        acc.add(rec)
        assert acc.records[0] is rec


# ── CostTracker ────────────────────────────────────────────────────────────────


class TestCostTrackerRecord:
    def test_record_creates_new_tenant(self):
        tracker = CostTracker()
        tracker.record("new_tenant", "gpt-4o-mini", prompt_tokens=100, completion_tokens=20)
        summary = tracker.get_tenant_summary("new_tenant")
        assert summary is not None
        assert summary.query_count == 1

    def test_record_returns_cost_record(self):
        tracker = CostTracker()
        rec = tracker.record("t1", "gpt-4o-mini", prompt_tokens=100, completion_tokens=20)
        assert isinstance(rec, CostRecord)
        assert rec.model == "gpt-4o-mini"

    def test_unknown_tenant_summary_is_none(self):
        tracker = CostTracker()
        assert tracker.get_tenant_summary("never_seen") is None

    def test_completion_tokens_default_zero(self):
        tracker = CostTracker()
        rec = tracker.record("t1", "gpt-4o-mini", prompt_tokens=100)
        assert rec.completion_tokens == 0

    def test_separate_tenants_isolated(self):
        tracker = CostTracker()
        tracker.record("tenant_a", "gpt-4o-mini", prompt_tokens=1000)
        tracker.record("tenant_b", "gpt-4o-mini", prompt_tokens=5000)
        assert tracker.get_tenant_summary("tenant_a").total_tokens == 1000
        assert tracker.get_tenant_summary("tenant_b").total_tokens == 5000


class TestCostTrackerQuota:
    def test_unknown_tenant_passes_quota(self):
        tracker = CostTracker()
        assert tracker.check_quota("brand_new_tenant", monthly_token_limit=1000) is True

    def test_under_quota_passes(self):
        tracker = CostTracker()
        tracker.record("t1", "gpt-4o-mini", prompt_tokens=500)
        assert tracker.check_quota("t1", monthly_token_limit=1000) is True

    def test_exactly_at_quota_passes(self):
        tracker = CostTracker()
        tracker.record("t1", "gpt-4o-mini", prompt_tokens=1000)
        assert tracker.check_quota("t1", monthly_token_limit=1000) is True

    def test_over_quota_fails(self):
        tracker = CostTracker()
        tracker.record("t1", "gpt-4o-mini", prompt_tokens=1500)
        assert tracker.check_quota("t1", monthly_token_limit=1000) is False

    def test_accumulates_across_multiple_records_before_failing(self):
        tracker = CostTracker()
        tracker.record("t1", "gpt-4o-mini", prompt_tokens=400)
        assert tracker.check_quota("t1", monthly_token_limit=1000) is True
        tracker.record("t1", "gpt-4o-mini", prompt_tokens=400)
        assert tracker.check_quota("t1", monthly_token_limit=1000) is True
        tracker.record("t1", "gpt-4o-mini", prompt_tokens=400)
        assert tracker.check_quota("t1", monthly_token_limit=1000) is False


class TestCostTrackerTelemetryWiring:
    """check_quota() must publish Prometheus gauges on every call so the
    RAGTenantQuotaNearExhaustion alert (scripts/alerting/slo-burn-rate.yml)
    always observes current values rather than stale or absent series."""

    def test_check_quota_calls_record_tenant_quota(self):
        tracker = CostTracker()
        tracker.record("t1", "gpt-4o-mini", prompt_tokens=300)

        with patch("src.rag_system.utils.telemetry.record_tenant_quota") as mock_record:
            tracker.check_quota("t1", monthly_token_limit=1000)
            mock_record.assert_called_once_with("t1", 300, 1000)

    def test_check_quota_publishes_zero_for_unknown_tenant(self):
        tracker = CostTracker()
        with patch("src.rag_system.utils.telemetry.record_tenant_quota") as mock_record:
            tracker.check_quota("never_seen", monthly_token_limit=5000)
            mock_record.assert_called_once_with("never_seen", 0, 5000)

    def test_check_quota_does_not_raise_if_telemetry_missing(self):
        """If telemetry import fails for any reason, quota checks must still
        work — observability is a side-effect, never a hard dependency."""
        tracker = CostTracker()
        tracker.record("t1", "gpt-4o-mini", prompt_tokens=100)

        with patch.dict("sys.modules", {"src.rag_system.utils.telemetry": None}):
            # Forcing the module to None makes the lazy `from ... import`
            # inside check_quota() raise ImportError; the method must catch
            # it and still return the correct quota result.
            result = tracker.check_quota("t1", monthly_token_limit=1000)
            assert result is True


# ── Global singleton ────────────────────────────────────────────────────────────


class TestGetCostTrackerSingleton:
    def test_returns_same_instance(self):
        t1 = get_cost_tracker()
        t2 = get_cost_tracker()
        assert t1 is t2

    def test_singleton_state_persists_across_calls(self):
        tracker = get_cost_tracker()
        tracker.record("singleton_test_tenant", "gpt-4o-mini", prompt_tokens=42)
        tracker_again = get_cost_tracker()
        summary = tracker_again.get_tenant_summary("singleton_test_tenant")
        assert summary is not None
        assert summary.total_tokens == 42
