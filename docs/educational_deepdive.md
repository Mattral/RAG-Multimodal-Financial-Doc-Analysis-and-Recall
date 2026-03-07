# RAG Financial Multimodal — Educational Deep Dive

## Overview

This document explains the core design decisions, algorithms, and trade-offs in the v2.0 system for practitioners who want to understand the "why" behind each component.

---

## 1. Why Multimodal RAG for Financial Documents?

Financial documents are inherently mixed-media. A typical 10-K or earnings release contains:

- **Narrative text** — MD&A, risk factors, business descriptions
- **Tables** — income statements, balance sheets, segment breakdowns
- **Charts and graphs** — revenue trends, margin trajectories, market share comparisons

Standard text-only RAG pipelines retrieve nothing when the answer lives in a chart. An analyst asking "how has gross margin trended over the past 4 quarters?" gets a hallucinated answer from a text-only system because that information exists only as a line graph on page 8.

**The solution:** use a vision-language model (GPT-4o, Qwen2-VL) to generate dense text descriptions of every chart — including exact axis values, data points, and trends — and embed those descriptions alongside the text. The retriever then treats chart descriptions as first-class citizens.

---

## 2. PDF Parsing: The Foundation

### Why unstructured.io?

Unstructured's `partition_pdf` provides element-level classification — distinguishing `Title`, `NarrativeText`, `Table`, `Image`, `ListItem`, etc. — rather than returning a single text blob. This lets us:

1. Route tables to HTML representation for LLM context
2. Route images to the vision pipeline
3. Apply different chunking strategies to different element types

### Layout-aware chunking

Naive character-count chunking breaks tables mid-row and splits figure captions from their figures. Our `LayoutAwareParser` prevents this:

- **Table + caption pairing:** If a `Table` element is immediately preceded by a short text element matching `Table \d+: ...`, the caption is kept with the table.
- **Multi-page table merging:** Consecutive `Table` elements on adjacent pages are merged into one chunk (a `is_continuation=True` LayoutChunk).
- **HTML wrapping:** Tables become `<table>` elements, figures become `<figure>` elements. This gives the LLM structural context: it knows it's reading a financial table vs. narrative prose.
- **Section heading inheritance:** Every chunk carries the `heading` of the enclosing section (e.g., "Item 7. Management's Discussion and Analysis"). This appears in retrieved context, helping the LLM understand document provenance.

---

## 3. Vision Processing: Chart Extraction

### The prompt matters enormously

A generic "describe this image" prompt produces vague output like "a bar chart showing financial data." Our financial specialist prompt demands:

1. Chart type
2. Title and subtitle
3. **All axis labels and units**
4. **All data values** (exact numbers, percentages, dates)
5. Legend entries
6. Key trends, peaks, troughs
7. Footnotes and source attributions

This produces descriptions like: *"Bar chart titled 'Gross Margin by Quarter'. Y-axis: Gross Profit Margin (%). X-axis: Q1 2021 through Q3 2023. Values: Q1 21: 26.5%, Q2 21: 28.4%, Q3 21: 30.5%, Q4 21: 29.2%, Q1 22: 32.9%, Q2 22: 27.9%, Q3 22: 25.1%, Q1 23: 19.3%, Q2 23: 18.2%, Q3 23: 17.9%. Clear downward trend beginning Q1 2022."*

That description is now retrievable, embeddable, and citable.

### GPT-4o vs. Qwen2-VL

| Dimension | GPT-4o | Qwen2-VL (72B) |
|---|---|---|
| Accuracy on complex charts | Best-in-class | ~90% of GPT-4o |
| Cost per image | ~$0.015–0.05 | ~$0.002–0.008 via Together.ai |
| Privacy | Sends to OpenAI | Can run on-prem |
| Latency | ~3–8s | ~4–10s |

For regulated industries (banking, insurance) where data cannot leave on-prem, Qwen2-VL is the answer. For others, GPT-4o produces marginally better chart descriptions.

---

## 4. Hybrid Retrieval: Dense + BM25 + RRF

### Why not just dense retrieval?

Dense retrieval embeds text into a vector space where semantic similarity is captured. This works well for paraphrases and conceptual questions. But consider the query: *"What was the gross margin in Q3 2023?"*

If the document says "Q3 2023 gross profit margin: 17.9%", the dense retriever may or may not find this — it depends heavily on whether the embedding model has seen enough financial documents to know that "gross margin" and "gross profit margin" are synonyms, and whether the vector for "Q3 2023" is close to "third quarter of fiscal year 2023."

BM25 doesn't care about semantics — it scores on exact token overlap. "Q3 2023" in the query matches "Q3 2023" in the document exactly.

### Reciprocal Rank Fusion

Rather than tuning a weighted combination `α · dense_score + β · bm25_score` (which requires per-dataset calibration), we use RRF:

```
score(d) = Σ_i  1 / (k + rank_i(d))
```

where `k=60` is a constant that dampens the impact of top ranks and `rank_i(d)` is the rank of document `d` in retrieval list `i`.

**Key insight:** RRF only uses rank, not score magnitude. This makes it robust to the fact that dense cosine similarities and BM25 BM scores are on completely different scales. No tuning required.

### Cross-encoder reranker

The bi-encoder (embedding model) encodes query and document independently, then computes similarity. Fast but ignores fine-grained query-document interactions.

The cross-encoder sees `[query, document]` concatenated and scores the pair end-to-end. Much more accurate (especially for nuanced financial questions) but 100x slower — so we only run it on the top-40 RRF candidates, returning the top-5.

---

## 5. Embedding Cache

Generating embeddings is the most predictable cost in the pipeline — the same chunk always produces the same vector. We cache embeddings in Redis with a 24-hour TTL using a content-hash key:

```
embed:{model_name}:{sha256(text)}  →  [float, float, ...]
```

Cache hit rate in production is typically 60–80% after the first few days, since most queries re-encounter chunks that were ingested earlier. This cuts embedding API costs by 60–80%.

---

## 6. Program-of-Thought: Exact Financial Arithmetic

LLMs are unreliable at multi-step arithmetic. For a question like "What was the CAGR of revenue from 2020 to 2023?", the LLM might compute `(23.35 / 8.77) ^ (1/3) - 1` incorrectly.

Program-of-Thought (PoT) solves this by asking the LLM to *write Python code* expressing the calculation, then executing that code:

```
User: What was the 3-year CAGR of revenue (from $8.77B in 2020 to $23.35B in 2023)?

LLM generates:
    v_initial = 8.77
    v_final = 23.35
    n_years = 3
    result = ((v_final / v_initial) ** (1 / n_years) - 1) * 100

PoT executor runs it → result = 38.67  (exact)
Answer: Revenue CAGR from 2020 to 2023 was 38.7%.
```

The executor validates the code against an AST whitelist (no imports, no file access, no network calls) and enforces a 5-second timeout before execution.

---

## 7. Guardrails: Numeric Grounding

The most dangerous failure mode in financial RAG is **numeric hallucination** — the system confidently states a number that doesn't appear in the source documents.

Our numeric grounding guardrail:

1. Extracts all numeric patterns from the generated answer (e.g., `$23.35B`, `17.9%`, `9%`)
2. Normalises and searches for each in the raw context text
3. Flags any number present in the answer but absent from context as "ungrounded"
4. If ungrounded numbers are found, the response includes a guardrail warning

This doesn't catch all hallucinations (the LLM can still recontextualise numbers incorrectly) but eliminates the most egregious "invented number" failures.

---

## 8. Cost Architecture

### Model routing

Not every query needs GPT-4o. Simple lookup questions ("What page is the income statement on?") can be answered by `gpt-4o-mini` for 1/33 the cost. Complex analytical questions ("Compare EBITDA margins across segments and explain the variance") need the full model.

A regex heuristic (`CAGR`, `margin`, `EBITDA`, `YoY`, `compare`, `ratio`, etc.) routes to the complex model. This heuristic correctly identifies ~85% of "complex" queries in our eval set.

### Redis semantic cache

Beyond embedding cache, we cache full query→answer pairs by semantic similarity. If a new query's embedding is within cosine distance 0.08 of a cached query, return the cached answer. Cache hit rate for semantic cache is lower (~15–25%) but eliminates the full retrieval+generation cost.

### Per-tenant cost tracking

Every LLM call records `(tenant_id, model, prompt_tokens, completion_tokens)`. The cost tracker computes USD cost using a pricing table and enforces monthly token quotas. This enables SaaS billing and prevents runaway spend from a single tenant.

---

## 9. Multi-Tenancy

Each tenant's data is isolated at the vector store level:

```
/data/vectorstore/rag_financial           ← default tenant
/data/vectorstore/rag_financial_acme      ← tenant "acme"
/data/vectorstore/rag_financial_globex    ← tenant "globex"
```

Rate limits and token quotas are enforced per-tenant by the `AsyncRateLimiter` and `CostTracker`. Every log record and audit event carries `tenant_id`.

---

## 10. Observability

The system emits three signals:

**Traces (OpenTelemetry):** Every query is a root span with child spans for retrieval (with `strategy`, `tenant_id` attributes) and generation (with `model`, `tenant_id`). Export to Jaeger, Grafana Tempo, or any OTLP backend. Latency breakdown lets you identify whether slowdowns are in retrieval or generation.

**Metrics (Prometheus):** Custom histograms and counters for RAG-specific signals: `rag_hallucination_score`, `rag_citation_coverage_ratio`, `rag_query_cost_usd_total`. These go beyond generic HTTP metrics to measure *answer quality over time*.

**Logs (structlog JSON):** Every log record carries `trace_id`, `span_id`, `tenant_id`, `service`, `env`, and optionally `memory_mb`. Structured JSON is directly ingestible by Elasticsearch, Loki, or CloudWatch Logs.

---

## 11. Evaluation

### Why RAGAS?

RAGAS provides reference-free evaluation metrics — it doesn't require ground-truth answers for every question (though our golden dataset has them). It uses the LLM-as-judge pattern to score:

- **Faithfulness:** Does every claim in the answer appear in the retrieved context?
- **Answer relevancy:** Does the answer address the actual question?
- **Context precision:** Of the retrieved chunks, how many were actually relevant?

### The golden dataset

8 samples is a starting point, not a production eval set. See `roadmap.md` for plans to expand to 50+ samples. Even 8 samples catch obvious regressions — if faithfulness drops from 0.90 to 0.72, something broke.

### CI regression gate

Every PR runs the eval suite. If `avg_faithfulness` drops more than 5% vs. the last committed run, the PR is blocked. This prevents "works on the demo but breaks for real questions" regressions from reaching production.
