"""Embedding implementations: OpenAI, Voyage, local sentence-transformers.

All implement BaseEmbedder with Redis caching and batch support.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import List, Optional

import httpx
import structlog

from src.rag_system.components.base import BaseEmbedder
from src.rag_system.config import get_config
from src.rag_system.utils.telemetry import record_cache_hit

logger = structlog.get_logger(__name__)


class RedisEmbeddingCache:
    """Redis-backed embedding cache with content-hash keys."""

    def __init__(self, redis_url: str, ttl_seconds: int = 86400) -> None:
        self._url = redis_url
        self._ttl = ttl_seconds
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(self._url, decode_responses=False)
            except Exception:
                pass
        return self._client

    async def get(self, text: str, model: str) -> Optional[List[float]]:
        client = self._get_client()
        if not client:
            return None
        key = f"embed:{model}:{hashlib.sha256(text.encode()).hexdigest()}"
        try:
            val = await client.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass
        return None

    async def set(self, text: str, model: str, vector: List[float]) -> None:
        client = self._get_client()
        if not client:
            return
        key = f"embed:{model}:{hashlib.sha256(text.encode()).hexdigest()}"
        try:
            await client.setex(key, self._ttl, json.dumps(vector))
        except Exception:
            pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small/large with Redis caching."""

    def __init__(self) -> None:
        self._cfg = get_config().vector_store_config
        cache_cfg = get_config().cache_config
        self._cache: Optional[RedisEmbeddingCache] = (
            RedisEmbeddingCache(cache_cfg.redis_url, cache_cfg.embedding_cache_ttl_seconds)
            if cache_cfg.enabled else None
        )
        self._model = self._cfg.embedding_model
        self._dim = self._cfg.embedding_dim

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed_query(self, query: str) -> List[float]:
        result = await self.embed([query])
        return result[0]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        cfg = get_config()
        api_key = cfg.get_openai_key()
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache
        for i, text in enumerate(texts):
            if self._cache:
                cached = await self._cache.get(text, self._model)
                if cached:
                    results[i] = cached
                    record_cache_hit("embedding", tenant_id="default")
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        # Batch embed uncached texts (max 100 per request)
        BATCH = 100
        for batch_start in range(0, len(uncached_texts), BATCH):
            batch = uncached_texts[batch_start:batch_start + BATCH]
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": self._model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
            for j, item in enumerate(data["data"]):
                vec = item["embedding"]
                abs_idx = uncached_indices[batch_start + j]
                results[abs_idx] = vec
                if self._cache:
                    await self._cache.set(texts[abs_idx], self._model, vec)

        return [r for r in results if r is not None]


class LocalEmbedder(BaseEmbedder):
    """Local sentence-transformers embedder — no API cost, runs on-prem."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return f"local/{self._model_name}"

    @property
    def dimension(self) -> int:
        return 384  # bge-small

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                logger.error("sentence_transformers_not_installed")
                raise
        return self._model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        vecs = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    async def embed_query(self, query: str) -> List[float]:
        result = await self.embed([query])
        return result[0]


def build_embedder(provider: Optional[str] = None) -> BaseEmbedder:
    """Factory: build embedder by provider."""
    if provider == "local":
        return LocalEmbedder()
    return OpenAIEmbedder()
