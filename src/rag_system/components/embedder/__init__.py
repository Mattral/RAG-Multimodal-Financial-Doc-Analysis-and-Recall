"""Embedding implementations: OpenAI, Voyage, local sentence-transformers.

All implement BaseEmbedder with Redis caching and batch support.
"""
from __future__ import annotations

import asyncio
import contextlib
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
        with contextlib.suppress(Exception):
            await client.setex(key, self._ttl, json.dumps(vector))


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
        batch_size = 100
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start:batch_start + batch_size]
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


class VoyageEmbedder(BaseEmbedder):
    """Voyage AI embedder — voyage-3 / voyage-finance-2 (finance-domain-tuned).

    Voyage publishes a model specifically trained on financial text
    (voyage-finance-2), which often outperforms general-purpose embeddings
    on 10-K/10-Q retrieval. Set VOYAGE_API_KEY in .env.
    """

    def __init__(self, model: str = "voyage-finance-2") -> None:
        self._model = model
        cache_cfg = get_config().cache_config
        self._cache: Optional[RedisEmbeddingCache] = (
            RedisEmbeddingCache(cache_cfg.redis_url, cache_cfg.embedding_cache_ttl_seconds)
            if cache_cfg.enabled else None
        )

    @property
    def name(self) -> str:
        return f"voyage/{self._model}"

    @property
    def dimension(self) -> int:
        return 1024  # voyage-finance-2 / voyage-3 default

    def _get_api_key(self) -> str:
        cfg = get_config()
        if not cfg.voyage_api_key:
            from src.rag_system.utils.exceptions import ConfigurationError
            raise ConfigurationError(
                "VOYAGE_API_KEY not set — required for embedding provider 'voyage'",
                config_key="VOYAGE_API_KEY",
            )
        return cfg.voyage_api_key.get_secret_value()

    async def embed_query(self, query: str) -> List[float]:
        result = await self.embed([query])
        return result[0]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        api_key = self._get_api_key()
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            if self._cache:
                cached = await self._cache.get(text, self._model)
                if cached:
                    results[i] = cached
                    record_cache_hit("embedding", tenant_id="default")
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        batch_size = 128
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start:batch_start + batch_size]
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.voyageai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": self._model, "input": batch, "input_type": "document"},
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


class CohereEmbedder(BaseEmbedder):
    """Cohere embed-v3 embedder — strong multilingual + retrieval quality."""

    def __init__(self, model: str = "embed-english-v3.0") -> None:
        self._model = model
        cache_cfg = get_config().cache_config
        self._cache: Optional[RedisEmbeddingCache] = (
            RedisEmbeddingCache(cache_cfg.redis_url, cache_cfg.embedding_cache_ttl_seconds)
            if cache_cfg.enabled else None
        )

    @property
    def name(self) -> str:
        return f"cohere/{self._model}"

    @property
    def dimension(self) -> int:
        return 1024

    def _get_api_key(self) -> str:
        cfg = get_config()
        if not cfg.cohere_api_key:
            from src.rag_system.utils.exceptions import ConfigurationError
            raise ConfigurationError(
                "COHERE_API_KEY not set — required for embedding provider 'cohere'",
                config_key="COHERE_API_KEY",
            )
        return cfg.cohere_api_key.get_secret_value()

    async def embed_query(self, query: str) -> List[float]:
        vecs = await self._embed_raw([query], input_type="search_query")
        return vecs[0]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            if self._cache:
                cached = await self._cache.get(text, self._model)
                if cached:
                    results[i] = cached
                    record_cache_hit("embedding", tenant_id="default")
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        if uncached_texts:
            vecs = await self._embed_raw(uncached_texts, input_type="search_document")
            for j, vec in enumerate(vecs):
                abs_idx = uncached_indices[j]
                results[abs_idx] = vec
                if self._cache:
                    await self._cache.set(texts[abs_idx], self._model, vec)

        return [r for r in results if r is not None]

    async def _embed_raw(self, texts: List[str], input_type: str) -> List[List[float]]:
        api_key = self._get_api_key()
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.cohere.com/v1/embed",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": self._model, "texts": texts, "input_type": input_type},
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]


def build_embedder(provider: Optional[str] = None) -> BaseEmbedder:
    """Factory: build embedder by provider name.

    If provider is not passed, infers from VECTOR_STORE_CONFIG__EMBEDDING_MODEL
    or falls back to OpenAI. This is the single switch that lets users choose
    a fully local/open-source embedder with zero pipeline code changes.
    """
    cfg = get_config().vector_store_config
    name = (provider or "").lower()

    # Infer provider from a configured embedding_model name if not explicit
    if not name:
        model_name = cfg.embedding_model.lower()
        if "voyage" in model_name:
            name = "voyage"
        elif "cohere" in model_name or "embed-english" in model_name or "embed-multilingual" in model_name:
            name = "cohere"
        elif "bge" in model_name or "local" in model_name or "sentence-transformers" in model_name:
            name = "local"
        else:
            name = "openai"

    providers = {
        "openai": OpenAIEmbedder,
        "local": LocalEmbedder,
        "voyage": VoyageEmbedder,
        "cohere": CohereEmbedder,
    }
    embedder_cls = providers.get(name)
    if embedder_cls is None:
        logger.warning(
            "unknown_embedder_provider", provider=name, fallback="openai",
            available=sorted(providers.keys()),
        )
        embedder_cls = OpenAIEmbedder

    logger.info("embedder_provider_selected", provider=name, resolved_class=embedder_cls.__name__)
    return embedder_cls()
