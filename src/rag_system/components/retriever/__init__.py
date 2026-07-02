"""Hybrid retriever: dense vector + BM25 keyword search fused via Reciprocal Rank Fusion.

This is the primary retrieval strategy for financial documents where
exact numeric matches and keyword overlap are important alongside
semantic similarity.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import structlog

from src.rag_system.components.base import (
    BaseEmbedder,
    BaseReranker,
    BaseRetriever,
    BaseVectorStore,
    RetrievedChunk,
)
from src.rag_system.config import get_config
from src.rag_system.utils.telemetry import async_trace_span

logger = structlog.get_logger(__name__)


def _reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievedChunk]],
    k: int = 60,
) -> List[RetrievedChunk]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Score for chunk c = Σ 1 / (k + rank_i(c))
    """
    scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            # Use chunk text hash as unique key
            key = f"{chunk.source_document}_{chunk.page_number}_{hash(chunk.text)}"
            scores[key] += 1.0 / (k + rank)
            if key not in chunk_map:
                chunk_map[key] = chunk

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [
        RetrievedChunk(
            **{**chunk_map[k].model_dump(), "score": scores[k]}
        )
        for k in sorted_keys
    ]


class BM25Index:
    """Lightweight in-memory BM25 index for keyword search.

    For production, swap with a Tantivy / Elasticsearch / OpenSearch backend.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: List[RetrievedChunk] = []
        self._tf: List[Dict[str, float]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0
        self._token_re = re.compile(r"\b\w+\b", re.I)

    def _tokenise(self, text: str) -> List[str]:
        return [t.lower() for t in self._token_re.findall(text)]

    def build(self, chunks: List[RetrievedChunk]) -> None:
        """Build index from a list of chunks."""
        self._docs = chunks
        self._tf = []
        self._df = defaultdict(int)
        total_len = 0

        for chunk in chunks:
            tokens = self._tokenise(chunk.text)
            total_len += len(tokens)
            freq: Dict[str, float] = defaultdict(float)
            for t in tokens:
                freq[t] += 1
            self._tf.append(dict(freq))
            for term in set(tokens):
                self._df[term] += 1

        n = len(chunks)
        self._avgdl = total_len / n if n > 0 else 1.0

    def search(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        """BM25 search; returns ranked chunks."""
        if not self._docs:
            return []

        import math

        n = len(self._docs)
        q_tokens = self._tokenise(query)
        scores = []

        for i, (_chunk, tf) in enumerate(zip(self._docs, self._tf, strict=True)):
            dl = sum(tf.values())
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                f = tf[term]
                df = self._df.get(term, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                score += idf * tf_norm
            scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            c = self._docs[idx]
            results.append(RetrievedChunk(**{**c.model_dump(), "score": score}))
        return results


class HybridRetriever(BaseRetriever):
    """Combines dense vector search and BM25, fused with RRF.

    Optional reranker stage follows.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        reranker: Optional[BaseReranker] = None,
        bm25_index: Optional[BM25Index] = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._bm25 = bm25_index
        self._cfg = get_config().retriever_config

    @property
    def name(self) -> str:
        return "hybrid_rrf_retriever"

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        async with async_trace_span(
            "retrieval",
            {"tenant_id": tenant_id or "default", "strategy": self._cfg.strategy},
        ):
            query_vec = await self._embedder.embed_query(query)

            # Dense retrieval
            dense_task = self._vector_store.search(
                query_vector=query_vec,
                top_k=self._cfg.top_k_dense,
                filters=filters,
                tenant_id=tenant_id,
            )

            # BM25 retrieval (async-wrapped sync)
            if self._bm25 is not None:
                bm25_task = asyncio.to_thread(
                    self._bm25.search, query, self._cfg.top_k_bm25
                )
                dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)
                fused = _reciprocal_rank_fusion(
                    [dense_results, bm25_results], k=self._cfg.rrf_k
                )
            else:
                dense_results = await dense_task
                fused = dense_results

            candidates = fused[: self._cfg.top_k_final * 3]  # over-fetch for reranker

            # Reranking
            if self._reranker and self._cfg.enable_reranker:
                candidates = await self._reranker.rerank(
                    query, candidates, top_n=self._cfg.top_k_final
                )
            else:
                candidates = candidates[: self._cfg.top_k_final]

            logger.info(
                "hybrid_retrieval_complete",
                tenant_id=tenant_id,
                num_results=len(candidates),
            )
            return candidates
