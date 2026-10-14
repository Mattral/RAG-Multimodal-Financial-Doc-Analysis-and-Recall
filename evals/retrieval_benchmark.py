"""Offline retrieval benchmark on the golden dataset's real source documents.

Builds an in-memory index over the 16 primary documents referenced in
evals/golden_datasets/provenance.json and measures whether each retrieval mode
surfaces an *answer-bearing* chunk: a chunk that contains one of the evidence
strings recorded for that question. No LLM or API key is needed.

It uses the repository's own components: ``BM25Index``, ``LocalEmbedder``
(BAAI/bge-small-en-v1.5 by default) and ``_reciprocal_rank_fusion`` with the
``RetrieverConfig`` defaults (top_k_dense=20, top_k_bm25=20, rrf_k=60). Dense
search is exact cosine kNN in numpy instead of a vector database, and no
reranker is applied. Chunks are fixed-size character windows over plain text
extracted from PDF/HTML, not the unstructured.io parser used at ingest time.

Usage:
    pip install sentence-transformers pypdf
    python -m evals.retrieval_benchmark --chunk-size 1000
    python -m evals.retrieval_benchmark --chunk-size 4000 --output evals/retrieval_4000.json

Metrics (per mode, at k):
    evidence_hit@k   share of questions with an answer-bearing chunk in the top k
    doc_hit@k        share of questions with any chunk from an expected document in the top k
    doc_precision@k  share of top-k chunks that come from an expected document
    mrr@10           mean reciprocal rank of the first answer-bearing chunk (0 if none in top 10)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from evals.sources import (
    DEFAULT_CACHE,
    extract_text,
    fetch,
    load_provenance,
    normalize,
    source_urls,
)
from src.rag_system.components.base import RetrievedChunk
from src.rag_system.components.embedder import LocalEmbedder
from src.rag_system.components.retriever import BM25Index, _reciprocal_rank_fusion
from src.rag_system.config import RetrieverConfig

DATASET = Path(__file__).parent / "golden_datasets" / "financial_qa.jsonl"


def chunk_text(text: str, document: str, size: int, overlap: int) -> List[RetrievedChunk]:
    """Split text into overlapping character windows, snapping ends to whitespace."""
    chunks, start, n = [], 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > 0 else end
        chunks.append(
            RetrievedChunk(
                text=text[start:end],
                score=0.0,
                source_document=document,
                chunk_id=f"{document}#{n}",
            )
        )
        if end >= len(text):
            break
        start, n = max(end - overlap, start + 1), n + 1
    return chunks


class DenseIndex:
    """Exact cosine search over LocalEmbedder vectors (embeddings are L2-normalized)."""

    def __init__(self, embedder: LocalEmbedder, chunks: List[RetrievedChunk]) -> None:
        self._embedder = embedder
        self._chunks = chunks
        self._matrix = None

    async def build(self, cache_dir: Optional[Path] = None, batch_size: int = 64) -> None:
        """Embed all chunks, reusing a cached matrix for the same model + chunk texts."""
        cache_file = None
        if cache_dir is not None:
            digest = hashlib.sha256(self._embedder.name.encode())
            for c in self._chunks:
                digest.update(c.text.encode())
            cache_file = Path(cache_dir) / f"embeddings_{digest.hexdigest()[:16]}.npy"
            if cache_file.exists():
                self._matrix = np.load(cache_file)
                return
        vectors = []
        for i in range(0, len(self._chunks), batch_size):
            batch = [c.text for c in self._chunks[i : i + batch_size]]
            vectors.extend(await self._embedder.embed(batch))
        self._matrix = np.asarray(vectors, dtype=np.float32)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self._matrix)

    async def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        q = np.asarray(await self._embedder.embed_query(query), dtype=np.float32)
        scores = self._matrix @ q
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(**{**self._chunks[i].model_dump(), "score": float(scores[i])})
            for i in order
        ]


def score_ranking(ranked: List[RetrievedChunk], evidence: List[str], docs: set, k: int) -> Dict:
    def bears_answer(chunk: RetrievedChunk) -> bool:
        return any(e in chunk.text for e in evidence)

    top_k = ranked[:k]
    first = next((i for i, c in enumerate(ranked[:10], start=1) if bears_answer(c)), None)
    return {
        "evidence_hit": any(bears_answer(c) for c in top_k),
        "doc_hit": any(c.source_document in docs for c in top_k),
        "doc_precision": sum(c.source_document in docs for c in top_k) / max(len(top_k), 1),
        "rr": 1.0 / first if first else 0.0,
    }


async def run(args: argparse.Namespace) -> Dict:
    provenance = {p["question"]: p for p in load_provenance()}
    rows = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    t0 = time.perf_counter()
    chunks: List[RetrievedChunk] = []
    for doc, url in sorted(source_urls(list(provenance.values())).items()):
        text = extract_text(fetch(url, Path(args.cache_dir)))
        chunks.extend(chunk_text(text, doc, args.chunk_size, args.overlap))
    print(
        f"indexed {len(chunks):,} chunks from {len({c.source_document for c in chunks})} documents"
    )

    longest = max(
        len(normalize(e)) for p in provenance.values() for s in p["sources"] for e in s["evidence"]
    )
    if longest > args.overlap:
        print(
            f"note: longest evidence string is {longest} chars > overlap {args.overlap}; "
            "a few answer-bearing spans may be split across chunks"
        )

    cfg = RetrieverConfig()
    bm25 = BM25Index()
    bm25.build(chunks)
    dense = DenseIndex(LocalEmbedder(args.model), chunks)
    await dense.build(cache_dir=Path(args.cache_dir))
    print(f"index build took {time.perf_counter() - t0:.0f}s")

    per_mode: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        prov = provenance[row["question"]]
        evidence = [normalize(e) for s in prov["sources"] for e in s["evidence"]]
        docs = set(row["source_documents"])
        bm25_hits = bm25.search(row["question"], top_k=cfg.top_k_bm25)
        dense_hits = await dense.search(row["question"], top_k=cfg.top_k_dense)
        rankings = {
            "bm25": bm25_hits,
            "dense": dense_hits,
            "hybrid_rrf": _reciprocal_rank_fusion([dense_hits, bm25_hits], k=cfg.rrf_k),
        }
        for mode, ranked in rankings.items():
            result = score_ranking(ranked, evidence, docs, args.k)
            result["type"] = row["tags"][0]
            per_mode[mode].append(result)

    def summarize(results: List[Dict]) -> Dict:
        n = len(results)
        return {
            "n": n,
            f"evidence_hit@{args.k}": round(sum(r["evidence_hit"] for r in results) / n, 3),
            f"doc_hit@{args.k}": round(sum(r["doc_hit"] for r in results) / n, 3),
            f"doc_precision@{args.k}": round(sum(r["doc_precision"] for r in results) / n, 3),
            "mrr@10": round(sum(r["rr"] for r in results) / n, 3),
        }

    report = {
        "config": {
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "k": args.k,
            "embedding_model": args.model,
            "num_chunks": len(chunks),
            "num_questions": len(rows),
            "rrf_k": cfg.rrf_k,
            "top_k_dense": cfg.top_k_dense,
            "top_k_bm25": cfg.top_k_bm25,
        },
        "overall": {mode: summarize(res) for mode, res in per_mode.items()},
        "by_type": {
            mode: {
                t: summarize([r for r in res if r["type"] == t])
                for t in sorted({r["type"] for r in res})
            }
            for mode, res in per_mode.items()
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline retrieval benchmark")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=None, help="Write the JSON report here")
    args = parser.parse_args()

    report = asyncio.run(run(args))
    k = args.k
    print(
        f"\n{'mode':<12} {'evidence_hit@' + str(k):>15} {'doc_hit@' + str(k):>10} "
        f"{'doc_prec@' + str(k):>11} {'mrr@10':>8}"
    )
    for mode, m in report["overall"].items():
        print(
            f"{mode:<12} {m[f'evidence_hit@{k}']:>15.3f} {m[f'doc_hit@{k}']:>10.3f} "
            f"{m[f'doc_precision@{k}']:>11.3f} {m['mrr@10']:>8.3f}"
        )
    print("\nevidence_hit by question type:")
    for mode, by_type in report["by_type"].items():
        cells = ", ".join(
            f"{t}={m[f'evidence_hit@{k}']:.2f} (n={m['n']})" for t, m in by_type.items()
        )
        print(f"  {mode:<12} {cells}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport written to {args.output}")


if __name__ == "__main__":
    main()
