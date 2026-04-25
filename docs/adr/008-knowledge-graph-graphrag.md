# ADR 008: Knowledge Graph / GraphRAG Design

**Status:** Deferred (v3.0)
**Date:** 2024-07
**Deciders:** Core team

## Context

Guideline section 7 requires a Knowledge Graph layer for relationship-heavy
financial queries: ownership hierarchies, supplier networks, entity
co-references across multiple filings. Standard vector RAG struggles with
these because the answer requires traversing implicit relationships not
captured in a single chunk.

## Decision

Defer full implementation to v3.0. Provide now:
1. `InMemoryGraphStore` interface in `components/knowledge_graph.py`
2. `EntityExtractor` stub with the correct LLM prompt template
3. `GraphAugmentedRetriever` stub that falls back to standard retrieval
4. Feature flag: `ENABLE_KNOWLEDGE_GRAPH=false` by default

## Rationale

- **Cost vs. value**: Full graph extraction requires one LLM call per chunk
  (roughly 2x ingest cost). Value only materializes for specific
  relationship queries (about 15% of financial analyst queries in our
  golden set).
- **Complexity**: Neo4j as a dependency adds significant ops burden. An
  in-memory graph is sufficient for single-node deployments and dev/test.
- **Interface-first**: Shipping the stub now means the v3.0 implementation
  can swap in a real extractor and store with zero pipeline changes.
- **Priority**: Hybrid RRF plus reranking already handles most "similar
  document" disambiguation, which was the primary motivation for GraphRAG
  in financial documents.

## Future Implementation Plan

1. `EntityExtractor`: GPT-4o with structured output, extracting
   COMPANY / PERSON / METRIC / DATE entities and SUBSIDIARY_OF /
   REPORTED_REVENUE relations from each chunk during ingest.
2. `GraphStore`: in-memory for dev, Neo4j for production with Cypher
   traversal queries.
3. `GraphAugmentedRetriever`: query to NER to graph traversal to extra
   chunks to merge plus rerank.
4. Fallback: if graph traversal returns nothing, fall back to standard
   vector retrieval transparently.

## Consequences

- **Positive**: the interface is already in place; v3.0 work is limited to
  the extractor and store backends, with zero changes required to the
  pipeline orchestrator or retriever interfaces.
- **Negative**: relationship-heavy queries are not graph-powered in v2.0.
  Mitigated by using high `top_k` plus hybrid retrieval for queries with
  explicit entity mentions.
