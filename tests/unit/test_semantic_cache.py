"""Unit tests for the semantic query cache.

Covers the previously-stub-only CacheConfig.semantic_cache_enabled /
semantic_cache_threshold flags, which now have a real implementation that
caches full query→answer pairs by embedding cosine similarity.
"""
from __future__ import annotations

import pytest

from src.rag_system.utils.semantic_cache import (
    SemanticQueryCache,
    _cosine_similarity,
    build_semantic_cache,
)

# ── Cosine similarity helper ───────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_similarity_one(self):
        v = [0.1, 0.2, 0.3, 0.4]
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_similarity_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_similarity_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_empty_vectors_returns_zero(self):
        assert _cosine_similarity([], []) == 0.0
        assert _cosine_similarity([1.0], []) == 0.0

    def test_mismatched_length_returns_zero(self):
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_magnitude_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_near_duplicate_vectors_high_similarity(self):
        a = [1.0, 2.0, 3.0]
        b = [1.01, 2.01, 2.99]
        assert _cosine_similarity(a, b) > 0.99


# ── SemanticQueryCache (in-memory fallback path — no real Redis needed) ──────

@pytest.fixture
def cache():
    """Cache pointed at an unreachable Redis URL so it always exercises the
    in-process memory fallback path deterministically in unit tests."""
    return SemanticQueryCache(
        redis_url="redis://localhost:1",  # deliberately unreachable
        threshold=0.90,
        ttl_seconds=60,
        max_entries_per_tenant=5,
    )


class TestSemanticQueryCacheMiss:
    @pytest.mark.asyncio
    async def test_empty_cache_is_miss(self, cache):
        result = await cache.get([0.1, 0.2, 0.3], tenant_id="t1")
        assert result.hit is False
        assert result.answer_payload is None

    @pytest.mark.asyncio
    async def test_dissimilar_query_is_miss(self, cache):
        await cache.set(
            query_text="What was Q3 revenue?",
            query_embedding=[1.0, 0.0, 0.0],
            answer_payload={"answer": "Revenue was $23.35B."},
            tenant_id="t1",
        )
        # Orthogonal embedding — should not match
        result = await cache.get([0.0, 1.0, 0.0], tenant_id="t1")
        assert result.hit is False


class TestSemanticQueryCacheHit:
    @pytest.mark.asyncio
    async def test_identical_embedding_is_hit(self, cache):
        await cache.set(
            query_text="What was Q3 revenue?",
            query_embedding=[0.5, 0.5, 0.5],
            answer_payload={"answer": "Revenue was $23.35B."},
            tenant_id="t1",
        )
        result = await cache.get([0.5, 0.5, 0.5], tenant_id="t1")
        assert result.hit is True
        assert result.answer_payload == {"answer": "Revenue was $23.35B."}
        assert result.similarity == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_near_duplicate_paraphrase_is_hit(self, cache):
        # Simulates "What was Q3 revenue?" vs "What was revenue in Q3?"
        await cache.set(
            query_text="What was Q3 revenue?",
            query_embedding=[1.0, 2.0, 3.0, 4.0],
            answer_payload={"answer": "Revenue was $23.35B."},
            tenant_id="t1",
        )
        result = await cache.get([1.01, 2.01, 2.99, 4.02], tenant_id="t1")
        assert result.hit is True
        assert result.matched_query == "What was Q3 revenue?"

    @pytest.mark.asyncio
    async def test_below_threshold_is_miss(self):
        cache = SemanticQueryCache(redis_url="redis://localhost:1", threshold=0.999)
        await cache.set(
            query_text="What was Q3 revenue?",
            query_embedding=[1.0, 2.0, 3.0],
            answer_payload={"answer": "x"},
            tenant_id="t1",
        )
        # Similar but not similar enough for a threshold this strict
        result = await cache.get([1.1, 2.1, 2.9], tenant_id="t1")
        assert result.hit is False

    @pytest.mark.asyncio
    async def test_returns_best_match_among_multiple_entries(self, cache):
        await cache.set("Query A", [1.0, 0.0, 0.0], {"answer": "A"}, tenant_id="t1")
        await cache.set("Query B", [0.0, 1.0, 0.0], {"answer": "B"}, tenant_id="t1")
        await cache.set("Query C", [0.0, 0.0, 1.0], {"answer": "C"}, tenant_id="t1")

        result = await cache.get([0.95, 0.05, 0.0], tenant_id="t1")
        assert result.hit is True
        assert result.matched_query == "Query A"


class TestSemanticQueryCacheTenantIsolation:
    @pytest.mark.asyncio
    async def test_separate_tenants_do_not_share_cache(self, cache):
        await cache.set(
            query_text="What was revenue?",
            query_embedding=[1.0, 0.0],
            answer_payload={"answer": "tenant_a_answer"},
            tenant_id="tenant_a",
        )
        result_a = await cache.get([1.0, 0.0], tenant_id="tenant_a")
        result_b = await cache.get([1.0, 0.0], tenant_id="tenant_b")

        assert result_a.hit is True
        assert result_b.hit is False


class TestSemanticQueryCacheEviction:
    @pytest.mark.asyncio
    async def test_evicts_oldest_beyond_max_entries(self, cache):
        # cache fixture has max_entries_per_tenant=5
        for i in range(8):
            await cache.set(
                query_text=f"Query {i}",
                query_embedding=[float(i), 0.0, 0.0],
                answer_payload={"answer": f"answer {i}"},
                tenant_id="t1",
            )
        entries = await cache._load_entries("t1")
        assert len(entries) == 5
        # The oldest (Query 0, Query 1, Query 2) should have been evicted
        remaining_texts = {e.query_text for e in entries}
        assert "Query 0" not in remaining_texts
        assert "Query 7" in remaining_texts


class TestSemanticQueryCacheInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_tenant_clears_entries(self, cache):
        await cache.set("Q", [1.0, 0.0], {"answer": "x"}, tenant_id="t1")
        assert (await cache.get([1.0, 0.0], tenant_id="t1")).hit is True

        await cache.invalidate_tenant("t1")

        assert (await cache.get([1.0, 0.0], tenant_id="t1")).hit is False

    @pytest.mark.asyncio
    async def test_invalidate_does_not_affect_other_tenants(self, cache):
        await cache.set("Q", [1.0, 0.0], {"answer": "x"}, tenant_id="t1")
        await cache.set("Q", [1.0, 0.0], {"answer": "y"}, tenant_id="t2")

        await cache.invalidate_tenant("t1")

        assert (await cache.get([1.0, 0.0], tenant_id="t1")).hit is False
        assert (await cache.get([1.0, 0.0], tenant_id="t2")).hit is True


class TestSemanticQueryCacheStats:
    @pytest.mark.asyncio
    async def test_stats_empty_cache(self, cache):
        stats = await cache.stats("t1")
        assert stats["cached_entries"] == 0
        assert stats["total_hits"] == 0

    @pytest.mark.asyncio
    async def test_stats_tracks_hit_count(self, cache):
        await cache.set("Q", [1.0, 0.0], {"answer": "x"}, tenant_id="t1")
        await cache.get([1.0, 0.0], tenant_id="t1")
        await cache.get([1.0, 0.0], tenant_id="t1")

        stats = await cache.stats("t1")
        assert stats["cached_entries"] == 1
        assert stats["total_hits"] == 2

    @pytest.mark.asyncio
    async def test_stats_reports_threshold(self, cache):
        stats = await cache.stats("t1")
        assert stats["threshold"] == 0.90


# ── build_semantic_cache factory ──────────────────────────────────────────────

class TestBuildSemanticCacheFactory:
    def test_returns_none_when_cache_disabled(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            enabled=False, semantic_cache_enabled=True,
            redis_url="redis://x", semantic_cache_threshold=0.9,
            query_cache_ttl_seconds=60,
        )
        assert build_semantic_cache(cfg) is None

    def test_returns_none_when_semantic_cache_disabled(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            enabled=True, semantic_cache_enabled=False,
            redis_url="redis://x", semantic_cache_threshold=0.9,
            query_cache_ttl_seconds=60,
        )
        assert build_semantic_cache(cfg) is None

    def test_returns_cache_instance_when_enabled(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            enabled=True, semantic_cache_enabled=True,
            redis_url="redis://localhost:6379/0", semantic_cache_threshold=0.92,
            query_cache_ttl_seconds=3600,
        )
        result = build_semantic_cache(cfg)
        assert isinstance(result, SemanticQueryCache)
        assert result._threshold == 0.92

    def test_reads_from_global_config_when_none_passed(self, monkeypatch):
        monkeypatch.setenv("CACHE_CONFIG__SEMANTIC_CACHE_ENABLED", "true")
        monkeypatch.setenv("CACHE_CONFIG__ENABLED", "true")
        from src.rag_system.config import reset_config
        reset_config()
        result = build_semantic_cache()
        assert isinstance(result, SemanticQueryCache)
        reset_config()
