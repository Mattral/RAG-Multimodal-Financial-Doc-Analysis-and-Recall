# Tutorial: Knowledge Graph for Relationship-Heavy Financial Analysis

> **Status:** entity extraction and graph-augmented retrieval are both implemented and off by
> default (`ENABLE_KNOWLEDGE_GRAPH=false`). The graph lives in memory. See
> [Limitations](#limitations) before relying on it, and [ADR-008](../adr/008-knowledge-graph-graphrag.md)
> for the design.

The knowledge graph helps when the answer sits in chunks that do not repeat the words of the
question. For example, "Tesla's Q3 2023 results" should also find the cash-flow table, which
mentions neither "Tesla" nor "results". The graph links that table to Tesla through the metrics
extracted from it.

## How it works

**At ingest**, with `ENABLE_KNOWLEDGE_GRAPH=true`:

1. `EntityExtractor` sends each parsed element to the configured LLM (`LLM_CONFIG__PROVIDER`,
   OpenAI or Gemini) in JSON mode. That is **one LLM call per element**.
2. The extractor returns companies, metrics with values, periods, products and so on, plus
   relations such as `Tesla -REPORTED_METRIC-> Gross margin=17.9% -FOR_PERIOD-> Q3 2023`.
3. `InMemoryGraphStore` merges them into one graph per tenant. The same company mentioned in
   different chunks or filings becomes one node, and every entity is linked to the chunks it
   came from.

**At query time**, `GraphAugmentedRetriever` wraps the hybrid retriever:

1. It finds entities named in the query by whole-word match. Companies, metrics and products
   act as *anchors*. Periods such as "Q3 2023" do not, because every company shares them.
2. It expands the anchors to their neighbours up to `RETRIEVER_CONFIG__GRAPH_MAX_HOPS`
   (default 1, e.g. a company's reported metrics), skipping hub nodes.
3. It scores each candidate chunk by how many distinct query concepts it mentions directly
   (company, metric name, period). A chunk reached only through a neighbour gets a half point,
   and ties go to smaller chunks, because big financial statements mention everything.
4. It fuses the top `RETRIEVER_CONFIG__GRAPH_MAX_CHUNKS` (default 10) with the hybrid results
   using Reciprocal Rank Fusion, with the graph list at half weight
   (`RETRIEVER_CONFIG__GRAPH_WEIGHT=0.5`). At equal weight the graph pushed correct hybrid hits
   out of the top 5.

If the query names no anchor entity that exists in the graph, the hybrid results are returned
unchanged.

## Using it

```python
import os
os.environ["ENABLE_KNOWLEDGE_GRAPH"] = "true"

from src.rag_system.pipeline import RAGPipeline

pipeline = await RAGPipeline.create()            # retriever is wrapped automatically
result = await pipeline.ingest(["tesla_10k.pdf"], tenant_id="acme")
print(result["knowledge_graph"])                 # {'entities': ..., 'relations': ..., 'chunks': ...}

answer = await pipeline.query("What was Tesla's gross margin in Q3 2023?", tenant_id="acme")
# Chunks contributed by the graph carry metadata {"retrieval": "graph", "graph_entities": [...]}
```

The pieces can also be used directly:

```python
from src.rag_system.components.knowledge_graph import (
    EntityExtractor, GraphAugmentedRetriever, InMemoryGraphStore,
)

store = InMemoryGraphStore()
entities, relations = await EntityExtractor().extract(chunk.text, chunk.source_document, "acme")
store.add_chunk_extraction(chunk, entities, relations, tenant_id="acme")

retriever = GraphAugmentedRetriever(graph_store=store, base_retriever=hybrid_retriever)
chunks = await retriever.retrieve("Tesla free cash flow", tenant_id="acme")
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ENABLE_KNOWLEDGE_GRAPH` | `false` | Extract at ingest and wrap the retriever |
| `RETRIEVER_CONFIG__GRAPH_MAX_HOPS` | `1` | How far to walk from query entities (0–3) |
| `RETRIEVER_CONFIG__GRAPH_MAX_CHUNKS` | `10` | Graph chunks fused with the hybrid results |
| `RETRIEVER_CONFIG__GRAPH_WEIGHT` | `0.5` | RRF weight of graph results relative to hybrid results |
| `LLM_CONFIG__PROVIDER` / `LLM_CONFIG__MODEL` | `openai` / `gpt-4o-mini` | Extractor model (OpenAI or Gemini) |

`RETRIEVER_CONFIG__STRATEGY=graph_augmented` on its own does nothing. It logs a warning
telling you to set `ENABLE_KNOWLEDGE_GRAPH=true`.

## Does it help?

**A little, and not uniformly.** On the 119-question golden set the share of questions with an
answer-bearing chunk in the top 5 went from 0.706 (hybrid) to **0.723** (hybrid + graph). That
is 7 questions gained and 5 lost. Factual lookups improved (0.68 → 0.75) and table lookups got
worse (0.85 → 0.77). The defaults were chosen on half of the questions and confirmed on the
other half, where the gain was a single question. Treat it as optional, low-risk and unproven.
Details are in the knowledge-graph section of
[BENCHMARK_RESULTS.md](https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall/blob/main/BENCHMARK_RESULTS.md).
To reproduce (one LLM call per chunk, cached and resumable):

```bash
python -m evals.build_graph --chunk-size 4000 --rpm 60 --concurrency 8
python -m evals.retrieval_benchmark --chunk-size 4000 \
  --graph-cache evals/results/graph_extractions_chunk4000.jsonl
```

## Limitations

* **In memory only.** The graph is rebuilt by ingest and lost on restart. With the CLI,
  `rag-financial ingest` and `rag-financial query` are separate processes, so the query sees an
  empty graph and silently falls back to hybrid retrieval. Use it from a long-running process
  (the API server or the SDK) until a persistent store (Neo4j, Kuzu, or a saved graph) is added.
* **Ingest cost.** One LLM call per element. On the benchmark filings that came to roughly
  2k input and 1–2k output tokens per 4,000-character chunk.
* **Name matching.** Query entities are found by whole-word name match, so "Alphabet" does not
  match a graph node named "Google". The extractor is asked to use common company names to
  reduce this.
* **Extraction quality decides everything.** Chunks with no figures (cover pages, inline-XBRL
  metadata) yield few useful entities, and a wrong value in an extracted metric is a wrong node.
