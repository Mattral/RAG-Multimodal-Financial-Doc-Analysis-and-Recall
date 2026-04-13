# ADR 001: Hybrid Retrieval with Reciprocal Rank Fusion

**Status:** Accepted  
**Date:** 2024-06  
**Deciders:** Core team

## Context

Financial documents contain both semantic content (narrative, risk factors, strategy) and exact-match content (specific numbers, tickers, dates, accounting terms). Pure dense retrieval misses exact numerical matches. Pure BM25 misses semantic paraphrases. We needed a strategy that handles both.

## Decision

Implement hybrid retrieval: dense vector search (top-20) + BM25 keyword search (top-20), fused via Reciprocal Rank Fusion (RRF) with k=60, followed by a cross-encoder reranker returning top-5.

RRF formula: `score(d) = Σ 1/(k + rank_i(d))` across all ranked lists.

## Rationale

- **RRF over weighted linear combination:** RRF requires no tuning of fusion weights, is robust to score distribution differences between dense and sparse retrievers, and consistently outperforms linear combination on BEIR benchmarks.
- **k=60** is the standard default (Cormack et al. 2009) and works well empirically.
- **Cross-encoder reranker after fusion:** The reranker sees the full query-chunk pair and is much more accurate than bi-encoder scores, but too slow to run on all candidates. Over-fetching 40 candidates and reranking to top-5 balances quality and latency.

## Consequences

- **Positive:** Captures both "$23.35 billion" exact matches and "strong revenue growth" semantic matches. Consistently 5-15% better recall@5 vs dense-only in our eval set.
- **Negative:** Two retrieval calls instead of one increases latency by ~30ms. BM25 index is in-process; distributed deployments need an external BM25 service (Elasticsearch/OpenSearch) for consistency.

## Alternatives Considered

- **Dense-only (rejected):** Misses exact financial figures in documents.
- **BM25-only (rejected):** Misses semantic paraphrases and cross-document reasoning.
- **SPLADE (future):** Would replace BM25 with a learned sparse retriever. Higher quality but adds model serving complexity. Tracked in roadmap v2.1.
