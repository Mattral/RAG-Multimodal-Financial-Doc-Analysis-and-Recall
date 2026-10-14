# Data Flow

## Ingestion Flow

```
Input PDF / S3 / Azure Blob
         │
         ▼
┌─────────────────────┐
│   Delta Detection   │  Skip unchanged (content hash)
│  VersionManager     │  Bump version if changed
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌─────────────┐
│ Parser │   │ Vision LLM  │
│Unstruct│   │GPT-4o/Gemini│  (for images/charts)
│/Docling│   │/Qwen2-VL    │
└───┬────┘   └──────┬──────┘
    │               │
    └──────┬────────┘
           ▼
┌─────────────────────┐
│  Layout Parser      │  Table+caption pairing
│  Semantic Chunker   │  Multi-page table merge
│  HTML Wrapping      │  <table>, <figure>, <section>
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  PII Redactor       │  Presidio + CUSIP/ISIN/SSN
│  (Presidio)         │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Embedder           │  OpenAI / local, Redis cached
│  (Redis cache)      │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Vector Store       │  DeepLake / pgvector / Qdrant
│  + BM25 Index       │  Per-tenant namespace (BM25: in-memory,
└──────────┬──────────┘  lost on restart — see ADR-001)
           ▼
   Audit Log + Version Registry
```

## Query Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Analyzer     │  Intent, filters, injection check, rewrite
└──────────┬──────────┘
           │ injection → BLOCKED (403)
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌─────────────┐
│ Dense  │   │  BM25       │
│ Search │   │  Keyword    │
└───┬────┘   └──────┬──────┘
    └──────┬─────────┘
           ▼
┌─────────────────────┐
│  RRF Fusion         │  Reciprocal Rank Fusion (k=60)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Cross-Encoder      │  Reranker (local or Cohere)
│  Reranker           │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  LLM Generator      │  Cost-routed: mini→full
│  (+ PoT if numeric) │  Fallback on 429/503
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Guardrails         │  Numeric grounding check
│                     │  Prompt injection (output)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Audit Log          │  SHA-256 tamper hash
│  Cost Tracker       │  Per-tenant accumulation
│  Prometheus Metrics │  Latency, cost, hallucination
└─────────────────────┘
           │
           ▼
  Grounded Answer + Citations + Metrics
```
