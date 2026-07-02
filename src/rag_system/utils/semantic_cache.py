"""Semantic query cache — caches full query→answer pairs by embedding similarity.

This is distinct from the embedding cache in components/embedder/__init__.py
(RedisEmbeddingCache), which caches individual chunk embeddings by exact
content hash. The semantic cache here caches entire generated answers and
returns a cache hit whenever a *new* query is semantically close enough to
a *previously answered* query — even if the wording differs.

Example: "What was Q3 revenue?" and "What was the revenue in the third
quarter?" embed to nearly the same vector. The second query can reuse the
first query's full answer (skipping retrieval + generation entirely),
eliminating the LLM cost and most of the latency for that request.

Backend: Redis sorted-set per tenant, storing up to `max_entries_per_tenant`
recent (query_embedding, answer_payload) pairs. On lookup, we brute-force
cosine similarity against the cached set — fine at the scale this is meant
for (hundreds to low thousands of distinct cached queries per tenant); for
very large caches, swap in a proper ANN index (e.g. a small DeepLake/FAISS
side-index) behind the same two methods (`get`, `set`).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


@dataclass
class SemanticCacheEntry:
    """A single cached (query, answer) pair with its embedding."""
    query_text: str
    embedding: List[float]
    answer_payload: Dict[str, Any]
    cached_at: float
    hit_count: int = 0


@dataclass
class SemanticCacheResult:
    """Result of a semantic cache lookup."""
    hit: bool
    answer_payload: Optional[Dict[str, Any]] = None
    matched_query: Optional[str] = None
    similarity: float = 0.0


class SemanticQueryCache:
    """Embedding-similarity cache for full query→answer pairs.

    Usage::

        cache = SemanticQueryCache(redis_url=cfg.redis_url, threshold=0.92)

        result = await cache.get(query_embedding, tenant_id="acme")
        if result.hit:
            return result.answer_payload  # skip retrieval + generation entirely

        # ... run normal retrieval + generation ...
        await cache.set(query_text, query_embedding, answer_payload, tenant_id="acme")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        threshold: float = 0.92,
        ttl_seconds: int = 3600,
        max_entries_per_tenant: int = 500,
    ) -> None:
        self._url = redis_url
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries_per_tenant
        self._client = None
        # In-memory fallback used automatically if Redis is unavailable,
        # so the cache degrades gracefully instead of breaking the pipeline.
        self._memory_store: Dict[str, List[SemanticCacheEntry]] = {}

    def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(self._url, decode_responses=True)
            except Exception as exc:
                logger.warning(
                    "semantic_cache_redis_unavailable",
                    detail="Falling back to in-process memory cache",
                    error=str(exc),
                )
        return self._client

    def _redis_key(self, tenant_id: str) -> str:
        return f"semantic_cache:{tenant_id}"

    async def get(
        self,
        query_embedding: List[float],
        tenant_id: str = "default",
    ) -> SemanticCacheResult:
        """Look up the most similar cached query above the similarity threshold."""
        entries = await self._load_entries(tenant_id)
        if not entries:
            return SemanticCacheResult(hit=False)

        best_score = -1.0
        best_entry: Optional[SemanticCacheEntry] = None
        for entry in entries:
            score = _cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self._threshold:
            best_entry.hit_count += 1
            await self._persist_entries(tenant_id, entries)
            logger.info(
                "semantic_cache_hit",
                tenant_id=tenant_id,
                similarity=round(best_score, 4),
                matched_query=best_entry.query_text[:80],
            )
            return SemanticCacheResult(
                hit=True,
                answer_payload=best_entry.answer_payload,
                matched_query=best_entry.query_text,
                similarity=best_score,
            )

        return SemanticCacheResult(hit=False, similarity=max(best_score, 0.0))

    async def set(
        self,
        query_text: str,
        query_embedding: List[float],
        answer_payload: Dict[str, Any],
        tenant_id: str = "default",
    ) -> None:
        """Store a new query→answer pair in the cache."""
        entries = await self._load_entries(tenant_id)
        entries.append(SemanticCacheEntry(
            query_text=query_text,
            embedding=query_embedding,
            answer_payload=answer_payload,
            cached_at=time.time(),
        ))
        # Evict oldest entries beyond the per-tenant cap
        if len(entries) > self._max_entries:
            entries = sorted(entries, key=lambda e: e.cached_at)[-self._max_entries:]
        await self._persist_entries(tenant_id, entries)

    async def invalidate_tenant(self, tenant_id: str) -> None:
        """Clear all cached entries for a tenant (e.g. after document re-ingest)."""
        client = self._get_client()
        if client:
            try:
                await client.delete(self._redis_key(tenant_id))
            except Exception as exc:
                logger.warning("semantic_cache_invalidate_failed", error=str(exc))
        self._memory_store.pop(tenant_id, None)
        logger.info("semantic_cache_invalidated", tenant_id=tenant_id)

    async def stats(self, tenant_id: str = "default") -> Dict[str, Any]:
        entries = await self._load_entries(tenant_id)
        total_hits = sum(e.hit_count for e in entries)
        return {
            "tenant_id": tenant_id,
            "cached_entries": len(entries),
            "total_hits": total_hits,
            "threshold": self._threshold,
            "backend": "redis" if self._get_client() else "memory",
        }

    # ── Internal persistence ──────────────────────────────────────────────────

    async def _load_entries(self, tenant_id: str) -> List[SemanticCacheEntry]:
        client = self._get_client()
        if client:
            try:
                raw = await client.get(self._redis_key(tenant_id))
                if raw:
                    data = json.loads(raw)
                    return [
                        SemanticCacheEntry(
                            query_text=d["query_text"],
                            embedding=d["embedding"],
                            answer_payload=d["answer_payload"],
                            cached_at=d["cached_at"],
                            hit_count=d.get("hit_count", 0),
                        )
                        for d in data
                    ]
                return []
            except Exception as exc:
                logger.warning("semantic_cache_load_failed", error=str(exc))
                # Fall through to memory store below
        return list(self._memory_store.get(tenant_id, []))

    async def _persist_entries(self, tenant_id: str, entries: List[SemanticCacheEntry]) -> None:
        client = self._get_client()
        if client:
            try:
                payload = json.dumps([
                    {
                        "query_text": e.query_text,
                        "embedding": e.embedding,
                        "answer_payload": e.answer_payload,
                        "cached_at": e.cached_at,
                        "hit_count": e.hit_count,
                    }
                    for e in entries
                ])
                await client.setex(self._redis_key(tenant_id), self._ttl, payload)
                return
            except Exception as exc:
                logger.warning("semantic_cache_persist_failed", error=str(exc))
        # Fall back to in-process memory (no TTL enforcement, capped by max_entries)
        self._memory_store[tenant_id] = entries


def build_semantic_cache(config: Optional[Any] = None) -> Optional[SemanticQueryCache]:
    """Factory: build a SemanticQueryCache from CacheConfig, or None if disabled."""
    if config is None:
        from src.rag_system.config import get_config
        config = get_config().cache_config

    if not config.enabled or not config.semantic_cache_enabled:
        return None

    return SemanticQueryCache(
        redis_url=config.redis_url,
        threshold=config.semantic_cache_threshold,
        ttl_seconds=config.query_cache_ttl_seconds,
    )
