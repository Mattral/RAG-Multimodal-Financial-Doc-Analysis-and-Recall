"""Hybrid retriever: dense vector + BM25 keyword search fused via Reciprocal Rank Fusion.

This is the primary retrieval strategy for financial documents where
exact numeric matches and keyword overlap are important alongside
semantic similarity.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
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
    weights: Optional[List[float]] = None,
) -> List[RetrievedChunk]:
    """Merge multiple ranked lists using (optionally weighted) Reciprocal Rank Fusion.

    Score for chunk c = Σ w_i / (k + rank_i(c)), with every w_i = 1 by default.
    """
    scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, RetrievedChunk] = {}
    weights = weights or [1.0] * len(ranked_lists)

    for weight, ranked_list in zip(weights, ranked_lists, strict=True):
        for rank, chunk in enumerate(ranked_list, start=1):
            # Use chunk text hash as unique key
            key = f"{chunk.source_document}_{chunk.page_number}_{hash(chunk.text)}"
            scores[key] += weight / (k + rank)
            if key not in chunk_map:
                chunk_map[key] = chunk

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [
        RetrievedChunk(**{**chunk_map[k].model_dump(), "score": scores[k]}) for k in sorted_keys
    ]


_DEFAULT_TENANT = "default"
_MISSING = object()


def _tenant_key(tenant_id: Optional[str]) -> str:
    # Same namespacing rule as the vector store adapters (``tenant_id or "default"``)
    # so the dense and keyword paths always resolve a caller to the same partition.
    return tenant_id or _DEFAULT_TENANT


def _chunk_field(chunk: RetrievedChunk, key: str) -> Any:
    """Resolve a filter key against a chunk: top-level field first, then metadata."""
    if key in ("source_document", "page_number", "chunk_id"):
        return getattr(chunk, key)
    if key == "content_hash":
        return chunk.chunk_id
    return chunk.metadata.get(key, _MISSING)


def _matches_filters(chunk: RetrievedChunk, filters: Optional[Dict[str, Any]]) -> bool:
    """Exact-match AND semantics, mirroring the Qdrant adapter's ``must`` filter.

    A chunk that lacks a filtered field does not match.
    """
    if not filters:
        return True
    return all(_chunk_field(chunk, k) == v for k, v in filters.items())


class _BM25Partition:
    """Term statistics for one tenant's corpus. Not thread-safe on its own."""

    def __init__(self) -> None:
        self.docs: List[RetrievedChunk] = []
        self.tf: List[Dict[str, float]] = []
        self.doc_len: List[int] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.total_len: int = 0
        self.id_to_pos: Dict[str, int] = {}

    @property
    def avgdl(self) -> float:
        n = len(self.docs)
        return self.total_len / n if n > 0 else 1.0

    def add(self, chunk: RetrievedChunk, tokens: List[str]) -> None:
        freq: Dict[str, float] = defaultdict(float)
        for t in tokens:
            freq[t] += 1

        pos = self.id_to_pos.get(chunk.chunk_id) if chunk.chunk_id else None
        if pos is not None:
            # Upsert: retract the previous version's statistics before replacing it.
            for term in self.tf[pos]:
                self.df[term] -= 1
                if self.df[term] <= 0:
                    del self.df[term]
            self.total_len -= self.doc_len[pos]
            self.docs[pos], self.tf[pos], self.doc_len[pos] = chunk, dict(freq), len(tokens)
        else:
            if chunk.chunk_id:
                self.id_to_pos[chunk.chunk_id] = len(self.docs)
            self.docs.append(chunk)
            self.tf.append(dict(freq))
            self.doc_len.append(len(tokens))

        self.total_len += len(tokens)
        for term in freq:
            self.df[term] += 1


class BM25Index:
    """Lightweight in-memory BM25 index for keyword search, partitioned by tenant.

    Each tenant has its own corpus *and* its own term statistics (IDF, avgdl), so a
    search scoped to one tenant can neither return nor be influenced by another
    tenant's chunks. Calls without ``tenant_id`` use the ``"default"`` partition,
    which keeps the original single-corpus ``build()`` / ``search()`` usage working.

    Limitation: the index lives in process memory only. It is populated by
    ``RAGPipeline.ingest()`` and is lost on restart; it is not rebuilt from the
    vector store (no adapter exposes a scan). Because ``DocumentVersionManager``
    persists to disk, documents ingested before a restart are skipped as
    "unchanged" and will be missing from BM25 until re-ingested with
    ``skip_unchanged=False``. Multi-replica deployments also hold divergent
    indexes. For production, swap with a Tantivy / Elasticsearch / OpenSearch backend.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._partitions: Dict[str, _BM25Partition] = {}
        self._lock = threading.RLock()
        self._token_re = re.compile(r"\b\w+\b", re.I)

    def _tokenise(self, text: str) -> List[str]:
        return [t.lower() for t in self._token_re.findall(text)]

    def build(self, chunks: List[RetrievedChunk], tenant_id: Optional[str] = None) -> None:
        """(Re)build one tenant's partition from scratch, replacing its previous contents."""
        partition = _BM25Partition()
        for chunk in chunks:
            partition.add(chunk, self._tokenise(chunk.text))
        with self._lock:
            self._partitions[_tenant_key(tenant_id)] = partition

    def add(self, chunks: List[RetrievedChunk], tenant_id: Optional[str] = None) -> None:
        """Incrementally add chunks to one tenant's partition.

        Chunks with a ``chunk_id`` already present in that partition replace the
        existing entry (upsert), so re-ingesting a document does not duplicate it.
        """
        tokenised = [(c, self._tokenise(c.text)) for c in chunks]
        with self._lock:
            partition = self._partitions.setdefault(_tenant_key(tenant_id), _BM25Partition())
            for chunk, tokens in tokenised:
                partition.add(chunk, tokens)

    def __len__(self) -> int:
        with self._lock:
            return sum(len(p.docs) for p in self._partitions.values())

    def count(self, tenant_id: Optional[str] = None) -> int:
        """Number of chunks indexed for one tenant."""
        with self._lock:
            partition = self._partitions.get(_tenant_key(tenant_id))
            return len(partition.docs) if partition else 0

    def search(
        self,
        query: str,
        top_k: int = 20,
        tenant_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """BM25 search over a single tenant's partition; returns ranked chunks.

        ``filters`` are exact-match constraints applied before ranking (see
        ``_matches_filters``).
        """
        with self._lock:
            partition = self._partitions.get(_tenant_key(tenant_id))
            if partition is None or not partition.docs:
                return []

            n = len(partition.docs)
            avgdl = partition.avgdl
            q_tokens = self._tokenise(query)
            scores = []

            for i, (chunk, tf) in enumerate(zip(partition.docs, partition.tf, strict=True)):
                if not _matches_filters(chunk, filters):
                    continue
                dl = partition.doc_len[i]
                score = 0.0
                for term in q_tokens:
                    if term not in tf:
                        continue
                    f = tf[term]
                    df = partition.df.get(term, 0)
                    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                    tf_norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
                    score += idf * tf_norm
                # Chunks sharing no query term would otherwise pad the list in
                # arbitrary order and still earn RRF credit downstream.
                if score > 0:
                    scores.append((score, i))

            scores.sort(key=lambda s: s[0], reverse=True)
            return [
                RetrievedChunk(**{**partition.docs[idx].model_dump(), "score": score})
                for score, idx in scores[:top_k]
            ]


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

    @property
    def bm25_index(self) -> Optional[BM25Index]:
        """The keyword index, exposed so the ingest path can populate it."""
        return self._bm25

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

            # BM25 retrieval (async-wrapped sync), scoped to the caller's tenant
            if self._bm25 is not None:
                bm25_task = asyncio.to_thread(
                    self._bm25.search,
                    query,
                    self._cfg.top_k_bm25,
                    tenant_id=tenant_id,
                    filters=filters,
                )
                dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)
                fused = _reciprocal_rank_fusion([dense_results, bm25_results], k=self._cfg.rrf_k)
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
