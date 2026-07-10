"""Reranker implementations: cross-encoder (local) and Cohere API."""

from __future__ import annotations

from typing import List, Optional

import structlog

from src.rag_system.components.base import BaseReranker, RetrievedChunk

logger = structlog.get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker using sentence-transformers.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality).
    Swap for cross-encoder/ms-marco-electra-base for higher quality.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None  # Lazy-loaded

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
                logger.info("cross_encoder_loaded", model=self._model_name)
            except ImportError:
                logger.warning(
                    "sentence_transformers_not_installed",
                    detail="pip install sentence-transformers",
                )
                return None
        return self._model

    @property
    def name(self) -> str:
        return f"cross_encoder/{self._model_name}"

    async def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: int = 5,
    ) -> List[RetrievedChunk]:
        import asyncio

        if not chunks:
            return []

        model = self._get_model()
        if model is None:
            return chunks[:top_n]

        pairs = [[query, chunk.text] for chunk in chunks]
        scores = await asyncio.to_thread(model.predict, pairs)

        scored = sorted(zip(scores, chunks, strict=True), key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(**{**chunk.model_dump(), "score": float(score)})
            for score, chunk in scored[:top_n]
        ]


class CohereReranker(BaseReranker):
    """Cohere Rerank API-based reranker (high quality, cloud)."""

    def __init__(self, model: str = "rerank-english-v3.0") -> None:
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import cohere

                from src.rag_system.config import get_config

                api_key = get_config().cohere_api_key
                if api_key:
                    self._client = cohere.Client(api_key.get_secret_value())
            except Exception as exc:
                logger.warning("cohere_client_init_failed", error=str(exc))
        return self._client

    @property
    def name(self) -> str:
        return f"cohere/{self._model}"

    async def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: int = 5,
    ) -> List[RetrievedChunk]:
        import asyncio

        client = self._get_client()
        if not client:
            return chunks[:top_n]

        docs = [chunk.text for chunk in chunks]
        response = await asyncio.to_thread(
            client.rerank,
            query=query,
            documents=docs,
            top_n=top_n,
            model=self._model,
        )

        reranked = []
        for result in response.results:
            chunk = chunks[result.index]
            reranked.append(
                RetrievedChunk(**{**chunk.model_dump(), "score": result.relevance_score})
            )
        return reranked


class NoOpReranker(BaseReranker):
    """Pass-through reranker (disabled/testing)."""

    @property
    def name(self) -> str:
        return "noop"

    async def rerank(
        self, query: str, chunks: List[RetrievedChunk], top_n: int = 5
    ) -> List[RetrievedChunk]:
        return chunks[:top_n]


def build_reranker(provider: str, model: Optional[str] = None) -> BaseReranker:
    """Factory: build reranker by provider name."""
    if provider == "cross_encoder":
        return CrossEncoderReranker(model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    elif provider == "cohere":
        return CohereReranker(model or "rerank-english-v3.0")
    elif provider == "none":
        return NoOpReranker()
    else:
        logger.warning("unknown_reranker_provider", provider=provider, fallback="cross_encoder")
        return CrossEncoderReranker()
