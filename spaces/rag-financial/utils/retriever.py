"""utils/retriever.py — Embedding, FAISS indexing, and hybrid BM25+dense retrieval.

Mirrors src/rag_system/components/retriever/__init__.py's HybridRetriever:
RRF fusion (k=60, dense x 0.7 + BM25 x 0.3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.pdf_processor import DocumentChunk


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    dense_score: float
    bm25_score: float
    rrf_score: float
    rank: int

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return f"{self.chunk.source_filename}, page {self.chunk.page_number}"


class EmbeddingModel:
    """Lazy-loaded sentence-transformers embedder with in-memory cache."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None
        self._cache: Dict[str, List[float]] = {}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            vecs = model.encode(uncached, normalize_embeddings=True, show_progress_bar=False)
            for text, vec in zip(uncached, vecs):
                self._cache[text] = vec.tolist()
        return np.array([self._cache[t] for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class BM25Index:
    def __init__(self) -> None:
        self._bm25 = None
        self._corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.sub(r"[^a-zA-Z0-9$%.,]", " ", text.lower()).split()

    def build(self, chunks: List[DocumentChunk]) -> None:
        self._corpus = [self._tokenize(c.text) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus)
        except ImportError:
            self._bm25 = None

    def score(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        if not self._corpus:
            return np.array([])
        if self._bm25 is not None:
            scores = np.array(self._bm25.get_scores(tokens))
        else:
            scores = np.array([
                sum(doc.count(t) for t in tokens) / max(len(doc), 1)
                for doc in self._corpus
            ], dtype=np.float32)
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        return scores


def reciprocal_rank_fusion(
    dense_ranks: List[int],
    bm25_ranks: List[int],
    k: int = 60,
    dense_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> List[float]:
    """RRF(d) = sum(weight_i / (k + rank_i(d))). k=60 per Cormack et al. 2009."""
    n = len(dense_ranks)
    scores = []
    for i in range(n):
        rrf = (dense_weight / (k + dense_ranks[i] + 1)) + (bm25_weight / (k + bm25_ranks[i] + 1))
        scores.append(rrf)
    return scores


class VectorIndex:
    """FAISS-backed dense vector index with BM25 hybrid retrieval."""

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None) -> None:
        self._embedder = embedding_model or EmbeddingModel()
        self._chunks: List[DocumentChunk] = []
        self._index = None
        self._bm25 = BM25Index()

    def build(self, chunks: List[DocumentChunk]) -> List[str]:
        if not chunks:
            return ["No chunks to index"]
        steps = []
        self._chunks = chunks
        steps.append(f"Embedding {len(chunks)} chunks with BAAI/bge-small-en-v1.5...")
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)

        try:
            import faiss
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings)
            steps.append(f"Built FAISS IndexFlatIP (dim={dim}, {len(chunks)} vectors)")
        except ImportError:
            self._index = embeddings
            steps.append("Built NumPy cosine index (FAISS not installed)")

        self._bm25.build(chunks)
        steps.append("Built BM25 keyword index for hybrid retrieval")
        steps.append(f"Index ready - {len(chunks)} chunks searchable")
        return steps

    def search(
        self, query: str, top_k: int = 5, chunk_type_filter: Optional[str] = None,
    ) -> Tuple[List[RetrievedChunk], List[str]]:
        steps = []
        if not self._chunks:
            return [], ["Index is empty - ingest a document first"]

        steps.append("Embedding query for dense retrieval...")
        query_vec = self._embedder.embed_query(query).astype(np.float32)

        candidate_k = min(len(self._chunks), top_k * 4)
        if hasattr(self._index, "search"):
            scores, indices = self._index.search(query_vec.reshape(1, -1), candidate_k)
            dense_scores = scores[0].tolist()
            dense_indices = indices[0].tolist()
        else:
            sims = self._index @ query_vec
            ranked = np.argsort(-sims)[:candidate_k]
            dense_indices = ranked.tolist()
            dense_scores = sims[ranked].tolist()

        steps.append(f"Dense retrieval: top-{candidate_k} candidates via cosine similarity")

        bm25_all_scores = self._bm25.score(query)
        if len(bm25_all_scores) > 0:
            bm25_ranked = np.argsort(-bm25_all_scores)[:candidate_k].tolist()
        else:
            bm25_ranked = list(range(min(candidate_k, len(self._chunks))))
        steps.append(f"BM25 keyword retrieval: top-{len(bm25_ranked)} candidates")

        all_candidate_indices = list(dict.fromkeys(dense_indices + bm25_ranked))

        def _dense_rank(idx: int) -> int:
            try:
                return dense_indices.index(idx)
            except ValueError:
                return candidate_k

        def _bm25_rank(idx: int) -> int:
            try:
                return bm25_ranked.index(idx)
            except ValueError:
                return candidate_k

        rrf_scores = []
        for idx in all_candidate_indices:
            rrf = (0.7 / (60 + _dense_rank(idx) + 1)) + (0.3 / (60 + _bm25_rank(idx) + 1))
            rrf_scores.append((idx, rrf))
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        steps.append(f"RRF fusion (k=60, dense x0.7 + BM25 x0.3): merged {len(rrf_scores)} candidates")

        results = []
        for rank, (idx, rrf_score) in enumerate(rrf_scores):
            if idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if chunk_type_filter and chunk.chunk_type != chunk_type_filter:
                continue
            d_score = dense_scores[dense_indices.index(idx)] if idx in dense_indices else 0.0
            b_score = float(bm25_all_scores[idx]) if len(bm25_all_scores) > idx else 0.0
            results.append(RetrievedChunk(
                chunk=chunk, dense_score=float(d_score), bm25_score=b_score,
                rrf_score=rrf_score, rank=rank + 1,
            ))
            if len(results) >= top_k:
                break

        steps.append(f"Retrieved {len(results)} chunks (top-k={top_k})")
        return results, steps
