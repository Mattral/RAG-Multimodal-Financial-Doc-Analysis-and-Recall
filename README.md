# RAG Financial Multimodal — Enterprise v2.0

> **World-class multimodal RAG system for financial document analysis.**
> Built to production standards: async, observable, secure, multi-tenant, CI-gated.

[![CI/CD](https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall/actions)
[![Coverage](https://codecov.io/gh/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall/branch/main/graph/badge.svg)](https://codecov.io/gh/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Does

Ingests multi-page financial PDFs (10-K, 10-Q, earnings releases, investor presentations), extracts text, tables, and **charts/graphs via vision LLMs**, indexes everything in a hybrid vector + BM25 store, then answers analyst-grade questions with **grounded citations** — refusing to hallucinate numbers.

```
PDF ──► Parser ──► Layout Grouper ──► PII Redactor
              └──► Vision LLM  ─────────┘
                                        │
                                        ▼
                              Embedder + BM25 Index
                                        │
                               Hybrid Vector Store
                                        │
                    Query ──► RRF Fusion ──► Reranker
                                                 │
                                          LLM Generator
                                          (w/ Guardrails)
                                                 │
                                        Grounded Answer + Citations
```

---

## Architecture at a Glance

| Layer | Component | Technology |
|---|---|---|
| **Parsing** | PDF text + tables | unstructured.io / Docling |
| **Vision** | Chart & graph description | GPT-4o / Qwen2-VL |
| **Layout** | Semantic grouping | Custom layout parser |
| **Embedding** | Dense vectors + cache | OpenAI `text-embedding-3-small` + Redis |
| **Indexing** | Vector store | DeepLake / pgvector / Qdrant |
| **Retrieval** | Hybrid (dense + BM25) | RRF fusion |
| **Reranking** | Cross-encoder | `ms-marco-MiniLM-L-6-v2` / Cohere |
| **Generation** | Cost-routed LLM | GPT-4o-mini → GPT-4o |
| **Guardrails** | Numeric grounding + PII | Presidio + custom AST |
| **Calculations** | Program-of-Thought | Sandboxed Python executor |
| **API** | REST + OpenAPI | FastAPI + uvicorn |
| **Observability** | Traces + metrics | OTel + Prometheus + Grafana |
| **Security** | Auth + audit trail | API key + SHA-256 audit log |
| **Multi-tenancy** | Isolated namespaces | Per-tenant vector partitions + quotas |
| **Deployment** | Container + K8s | Docker + Helm + HPA |

---

## Quick Start (Docker — recommended)

```bash
git clone https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall
cd RAG-Multimodal-Financial-Doc-Analysis-and-Recall
cp .env.example .env          # fill in OPENAI_API_KEY
docker compose up             # API on :8000, metrics on :8001
```

Full observability stack (Prometheus + Grafana + Jaeger):
```bash
docker compose --profile observability up
# Grafana: http://localhost:3000  (admin/admin)
# Jaeger:  http://localhost:16686
```

---

## Quick Start (local)

```bash
# 1. Install
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
sudo apt-get install -y poppler-utils tesseract-ocr   # Linux PDF deps

# 2. Configure
cp .env.example .env && nano .env   # set OPENAI_API_KEY

# 3. Ingest documents
rag-financial ingest tesla_10k.pdf apple_10k.pdf --tenant acme

# 4. Query
rag-financial query "What was Tesla's Q3 2023 gross margin?" --show-sources

# 5. Start API server
rag-financial serve --port 8000
```

---

## SDK Usage

```python
import asyncio
from src.rag_system.sdk import RAGPipeline

async def main():
    pipeline = await RAGPipeline.create(tenant_id="acme")

    # Ingest
    await pipeline.ingest(["tesla_10k.pdf", "apple_10k.pdf"])

    # Query — returns grounded answer with citations
    result = await pipeline.query("What was Q3 revenue?")
    print(result["answer"])
    print(result["sources"])          # document + page citations
    print(result["metrics"])          # latency, cost, chunk count

asyncio.run(main())
```

Synchronous (non-async) contexts:
```python
from src.rag_system.sdk import RAGPipeline

pipeline = RAGPipeline.from_config("config.yaml", tenant_id="acme")
result = pipeline.query_sync("What was the EBITDA margin?")
```

---

## REST API

```bash
# Ingest
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: your-key" -H "X-Tenant-ID: acme" \
  -F "file=@tesla_10k.pdf"

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"query": "What was gross margin in Q3 2023?", "top_k": 5}'

# Health
curl http://localhost:8000/readyz
```

Interactive docs (dev mode): http://localhost:8000/docs

---

## CLI Reference

```
rag-financial ingest [FILES]      Parse, embed, and index financial PDFs
rag-financial query  [QUESTION]   Retrieve and answer from indexed docs
rag-financial evaluate            Run RAGAS quality evals + regression gate
rag-financial serve               Start the FastAPI server
rag-financial health              Check all component health
rag-financial --version           Show version info
```

Key flags:
```bash
rag-financial ingest reports/*.pdf --tenant acme --no-vision
rag-financial query "Revenue trend?" --top-k 10 --show-sources --json
rag-financial evaluate --dataset evals/golden_datasets/financial_qa.jsonl \
                        --fail-on-regression --output report.json
rag-financial serve --host 0.0.0.0 --port 8000 --workers 4
```

---

## Key Features

### Hybrid Retrieval with RRF Fusion
Dense vector search + BM25 keyword search merged via Reciprocal Rank Fusion, then reranked with a cross-encoder. Catches both semantic matches and exact number/term hits — critical for financial data.

### Vision-Language Chart Processing
Every chart, graph, and figure is sent to GPT-4o (or Qwen2-VL for on-prem) with a financial-specialist prompt that extracts axis labels, exact values, legends, and trends. Chart descriptions are embedded alongside text.

### Layout-Aware Chunking
Tables stay with their captions. Multi-page tables are detected and merged. Figures stay paired with surrounding narrative. Sections are wrapped in semantic HTML for richer LLM context.

### Program-of-Thought Calculator
For numerical queries (CAGR, margin change, YoY growth), the LLM generates Python code that is AST-validated and executed in a sandboxed environment with 5-second timeout. Numbers are computed, not hallucinated.

### Financial Guardrails
Every answer is checked: (1) numeric values in the answer must appear verbatim in retrieved context, (2) prompt injection patterns are blocked at query time, (3) PII (SSN, account numbers, CUSIP, ISIN) is redacted from all ingested content.

### Multi-Tenancy
Each tenant gets an isolated vector namespace, per-tenant rate limits, monthly token quotas, and separate audit trails. Tenant ID flows through every component and log record.

### Cost Routing
Simple queries use `gpt-4o-mini`. Queries containing numerical/analytical terms (CAGR, EBITDA, margin, YoY) are automatically routed to `gpt-4o`. Exact cost is tracked per query, per tenant, per model.

### Immutable Audit Log
Every ingest and query is written to an append-only JSONL file with a SHA-256 content hash for tamper detection. GDPR/CCPA delete events are also logged.

---

## Configuration

All settings via environment variables or `.env`:

```bash
# Core
OPENAI_API_KEY=sk-...
ENVIRONMENT=production          # development | staging | production

# Model selection
LLM_CONFIG__MODEL=gpt-4o-mini
LLM_CONFIG__COMPLEX_QUERY_MODEL=gpt-4o
VISION_CONFIG__MODEL=gpt-4o

# Vector store
VECTOR_STORE_CONFIG__PROVIDER=deeplake   # deeplake | pgvector | qdrant | chroma
VECTOR_STORE_CONFIG__ENABLE_HYBRID_SEARCH=true

# Retrieval
RETRIEVER_CONFIG__STRATEGY=hybrid        # dense | hybrid | graph_augmented
RERANKER_CONFIG__PROVIDER=cross_encoder  # cross_encoder | cohere | none

# Caching (Redis)
CACHE_CONFIG__BACKEND=redis
CACHE_CONFIG__REDIS_URL=redis://localhost:6379/0

# Security
RAG_API_MASTER_KEY=your-secret-key
SECURITY_CONFIG__ENABLE_PII_REDACTION=true
SECURITY_CONFIG__ENABLE_GUARDRAILS=true

# Observability
OBSERVABILITY_CONFIG__OTLP_ENDPOINT=http://localhost:4317
OBSERVABILITY_CONFIG__PROMETHEUS_PORT=8001
```

See `.env.example` for the full reference.

---

## Observability

Prometheus metrics exposed at `:8001/metrics`:

| Metric | Description |
|---|---|
| `rag_query_latency_seconds` | End-to-end query latency histogram |
| `rag_retrieval_latency_seconds` | Retrieval stage latency |
| `rag_generation_latency_seconds` | LLM generation latency |
| `rag_query_cost_usd_total` | Cumulative cost by tenant/model |
| `rag_hallucination_score` | Proxy hallucination score histogram |
| `rag_citation_coverage_ratio` | Fraction of claims with citations |
| `rag_cache_hits_total` | Cache hits by type (embedding/semantic) |
| `rag_ingest_documents_total` | Documents ingested by tenant/parser |

OTel traces: every ingest and query is a root span with nested child spans for parsing, retrieval, and generation. Export to Jaeger, Tempo, or any OTLP backend.

Grafana dashboards are provisioned automatically from `grafana/dashboards/`.

---

## Evaluation

```bash
# Run full eval suite with regression gate
rag-financial evaluate \
  --dataset evals/golden_datasets/financial_qa.jsonl \
  --fail-on-regression

# Outputs:
#   Pass Rate: 87.5%
#   Avg Faithfulness: 0.912
#   Avg Numeric Accuracy: 0.884
#   Regression: None ✅
```

Quality thresholds (CI gate):
- Faithfulness ≥ 0.70 per sample
- Numeric accuracy ≥ 0.70 per sample
- Regression detection: if avg faithfulness drops >5% vs last run → CI fails

---

## Deployment

**Docker Compose** (single node):
```bash
docker compose up -d
```

**Kubernetes** (Kustomize):
```bash
kubectl apply -k k8s/overlays/prod/
```

**Helm** (with custom values):
```bash
helm install rag-financial helm/rag-financial/ \
  --set secrets.openaiApiKey=$OPENAI_API_KEY \
  --set config.environment=production \
  --namespace rag-prod --create-namespace
```

The Helm chart includes HPA (2–10 replicas), PodDisruptionBudget, ServiceMonitor for Prometheus Operator, and persistent volumes for vector store and audit logs.

---

## Testing

```bash
# Unit tests (fast, no external deps)
pytest tests/unit/ -v

# Integration tests (in-memory components)
pytest tests/integration/ -v

# Full suite with coverage
pytest --cov=src/rag_system --cov-report=html

# Load test (requires running API)
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

---

## Project Structure

```
src/rag_system/
├── config.py                  # Pydantic v2 config (all settings)
├── pipeline/                  # Orchestrator (DI-wired, tenant-aware)
├── api/                       # FastAPI app, routers, auth middleware
├── sdk/                       # Thin Python SDK for programmatic use
├── cli.py                     # Typer CLI (ingest/query/evaluate/serve)
├── components/
│   ├── base.py                # ABCs / Protocols / shared data models
│   ├── parser/                # PDF parsers (Unstructured, Docling)
│   ├── vision/                # Vision describers (GPT-4o, Qwen2-VL)
│   ├── embedder/              # Embedders + Redis cache
│   ├── vector_store/          # DeepLake, in-memory adapters
│   ├── retriever/             # Hybrid RRF retriever + BM25
│   ├── reranker/              # Cross-encoder, Cohere rerankers
│   ├── generator/             # Multi-provider LLM generator
│   ├── evaluator/             # RAGAS + LLM-judge evaluator
│   ├── guardrails/            # PII redactor + financial guardrails
│   ├── layout_parser.py       # Layout-aware semantic chunker
│   └── pot_executor.py        # Sandboxed PoT calculator
└── utils/
    ├── logger.py              # structlog + OTel trace injection
    ├── exceptions.py          # Rich exception hierarchy
    ├── retry_policy.py        # Full-jitter exponential backoff
    ├── rate_limiter.py        # Per-tenant token bucket + Redis
    ├── telemetry.py           # Prometheus metrics + OTel spans
    ├── audit.py               # Immutable JSONL audit logger
    └── cost_tracker.py        # Per-tenant cost + quota tracking
```

---

## License

MIT License © 2024 — See [LICENSE](LICENSE)
