# RAG Financial Multimodal — Enterprise v2.0

> **Production-grade multimodal RAG for financial document intelligence.**
> Chart understanding · hybrid retrieval · numeric guardrails · multi-tenancy · full observability.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Mattral/RAG-Financial-Multimodal)

---

## System Architecture

![Architecture Pipeline](docs/assets/architecture_pipeline.png)

*Ingestion (top) and query (bottom) pipelines. Every component is pluggable — switch provider by changing one config value.*

---

## Why This System

Financial documents are mixed-media: narrative text, tables, charts, footnotes, cross-references. Standard RAG pipelines fail on charts and hallucinate numbers.

| Problem | Solution |
|---|---|
| Charts contain the most important data but RAG ignores them | GPT-4o / Gemini / Qwen2-VL vision extraction — every chart yields exact axis values |
| Exact figures like `$23.35B` or `TSLA` miss semantic search | Hybrid RRF: dense embeddings + BM25 keyword fused with Reciprocal Rank Fusion |
| LLMs fabricate financial numbers | Numeric grounding guardrail — every stated number verified against source context |
| PII in analyst queries leaks to APIs | Presidio + CUSIP/ISIN/account number redaction before any external call |
| One broken vendor = full outage | Fallback chains — primary → secondary → local for every model-facing layer |

---

## Retrieval Quality

![Retrieval Quality](docs/assets/retrieval_quality.png)

*Hybrid RRF achieves 89% Recall@5 and 84% Precision@5 on our 22-sample financial QA benchmark — 25% better recall than dense-only and 41% better than BM25-only.*

---

## Quickstart

```bash
git clone https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall
cd RAG-Multimodal-Financial-Doc-Analysis-and-Recall
cp .env.example .env          # set OPENAI_API_KEY or GOOGLE_API_KEY
docker compose up -d

curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@tesla_10k.pdf" -F "tenant_id=demo"

curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was gross margin in Q3 2023?", "tenant_id": "demo"}'
```

Or use the CLI:
```bash
pip install -e ".[all]"
rag-financial ingest tesla_10k.pdf --tenant demo
rag-financial query "What was Q3 revenue?" --tenant demo --show-sources
```

---

## Multi-Provider Support

![Provider Matrix](docs/assets/provider_matrix.png)

*Every model-facing layer (text generation, vision, embeddings, vector store) is independently pluggable. Switch via a single `.env` line — zero code changes.*

### Fully open-source / zero-API-cost configuration

```bash
LLM_CONFIG__PROVIDER=local_vllm
LLM_CONFIG__MODEL=meta-llama/Llama-3.1-8B-Instruct
LOCAL_VLLM_GENERATOR_BASE_URL=http://localhost:8090/v1

VISION_CONFIG__PROVIDER=local_vllm
VISION_CONFIG__MODEL=Qwen/Qwen2-VL-7B-Instruct

VECTOR_STORE_CONFIG__EMBEDDING_PROVIDER=local
VECTOR_STORE_CONFIG__EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8090 --host 0.0.0.0
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8080 --host 0.0.0.0
```

---

## Evaluation Quality

![Evaluation Quality](docs/assets/eval_quality_radar.png)

*All five quality metrics exceed the 70% SLO threshold. Evaluated with RAGAS + LLM-as-judge numeric scorer on 22 financial QA samples across Tesla, Apple, Microsoft, Google, NVIDIA, JPMorgan, and Goldman Sachs filings.*

---

## Query Latency

![Latency Benchmarks](docs/assets/latency_benchmarks.png)

*All modes comfortably within the p99 < 8s SLO. Measured at 1000 queries with 50 concurrent users.*

---

## Cost Per Query

![Cost Per Query](docs/assets/cost_per_query.png)

*Smart routing (gpt-4o-mini for simple queries, gpt-4o only for complex ones) combined with Redis embedding cache achieves ~$0.00011/query at 72% cache hit rate.*

---

## SLO Alerting

![SLO Burn Rate](docs/assets/slo_burn_rate.png)

*Multi-window multi-burn-rate alerting from the Google SRE Workbook. Four alert tiers (14.4×, 6×, 3×, 1× burn rate) routing to PagerDuty/OpsGenie.*

---

## Architecture Layers

| Layer | Component | Technology |
|---|---|---|
| **Parsing** | PDF text + tables | Unstructured.io / Docling / Marker |
| **Vision** | Chart + graph extraction | GPT-4o / Gemini 2.0 Flash / Qwen2-VL / Local vLLM (fallback chain) |
| **Layout** | Semantic grouping | Table-caption pairing, multi-page merge, HTML wrapping |
| **Embedding** | Dense vectors + cache | OpenAI / Voyage / Cohere / local BAAI/bge, Redis cached |
| **Indexing** | Vector store | DeepLake / pgvector / Qdrant |
| **Retrieval** | Hybrid (dense + BM25) | Reciprocal Rank Fusion, k=60 |
| **Reranking** | Cross-encoder | ms-marco-MiniLM / Cohere Rerank v3 |
| **Generation** | Cost-routed, multi-provider | OpenAI / Gemini / Anthropic / Local vLLM |
| **Guardrails** | Numeric grounding + PII | Presidio + custom regex + AST-sandboxed PoT calculator |
| **API** | REST + OpenAPI | FastAPI + uvicorn |
| **Observability** | Traces + metrics | OpenTelemetry + Prometheus + Grafana |
| **Security** | Auth + audit trail | API key + SHA-256 tamper-evident audit log |
| **Multi-tenancy** | Isolated namespaces | Per-tenant vector partitions + quotas + rate limits |
| **Deployment** | Container + K8s | Docker + Helm + HPA + NetworkPolicy |

---

## Key Features

### Multimodal ingestion
- **Vision LLM fallback chain**: primary → secondary → local, never silently fails
- **Layout-aware chunker**: tables stay with their captions; multi-page tables merged
- **ColPali visual retrieval**: late-interaction MaxSim scoring on page images (no OCR)
- **Delta detection**: skip unchanged documents on re-ingest; version history for rollback

### Retrieval
- **Hybrid RRF (k=60)**: dense × 0.7 + BM25 × 0.3 — 25% better recall than dense-only
- **Cross-encoder reranking**: ms-marco-MiniLM-L-6-v2 or Cohere Rerank v3
- **Query analyzer**: intent classification → cost routing, entity extraction, query rewriting
- **Semantic cache**: similar queries served from cache (~1800ms saved per hit)
- **Knowledge graph**: LLM-extracted entities/relations (COMPANY, METRIC, REPORTED_REVENUE, etc.)

### Quality assurance
- **Numeric grounding guardrail**: every number in the answer is verified against context
- **Program-of-Thought calculator**: exact arithmetic in a sandboxed Python executor
- **RAGAS evaluation**: faithfulness, answer relevancy, context precision + LLM-as-judge
- **520 tests**: unit, integration, property-based (Hypothesis), API, chaos engineering

### Production operations
- **OpenTelemetry**: distributed traces with ingest/retrieve/generate spans
- **15 Prometheus metrics**: latency, cost, hallucination score, citation coverage, cache hit rate
- **Multi-window SLO alerting**: Google SRE workbook burn-rate pattern, PagerDuty/OpsGenie
- **Kubernetes**: HPA 2–10 replicas, PodDisruptionBudget, NetworkPolicy, IRSA
- **Terraform**: full EKS + RDS(pgvector) + ElastiCache + S3 + KMS infrastructure-as-code

---

## Repository Structure

```
src/rag_system/
├── api/          FastAPI app, routers (ingest/query/documents/tenants/feedback)
├── agentic/      LangGraph multi-step reasoning with self-correction loop
├── cli.py        Typer CLI (ingest, query, evaluate, serve, health)
├── components/
│   ├── base.py            ABCs for all pluggable components
│   ├── parser/             Unstructured, Docling, Marker adapters
│   ├── vision/             GPT-4o, Gemini, Qwen2-VL, LocalVLLM + fallback chain
│   ├── embedder/           OpenAI, Voyage, Cohere, local (BAAI/bge)
│   ├── vector_store/       DeepLake, pgvector, Qdrant
│   ├── retriever/          HybridRetriever (dense + BM25 + RRF)
│   ├── reranker/           CrossEncoder, Cohere, NoOp
│   ├── generator/          OpenAI, Gemini, Anthropic, LocalVLLM
│   ├── evaluator/          RAGAS + LLM-as-judge numeric scorer
│   ├── guardrails/         Numeric grounding, PII redaction, injection detection
│   ├── knowledge_graph.py  Real LLM entity/relation extraction + graph traversal
│   ├── colpali_retriever.py  Real MaxSim late-interaction visual retrieval
│   ├── pot_executor.py     Program-of-Thought sandboxed calculator
│   ├── layout_parser.py    Table-caption pairing, semantic chunking
│   ├── query_analyzer.py   Intent classification, entity extraction
│   ├── version_manager.py  Delta detection, point-in-time retrieval
│   └── connectors/         S3, Azure Blob, GCS
├── config.py     Pydantic v2 BaseSettings, 12 nested sub-configs
├── pipeline/     RAGPipeline orchestrator (dependency injection)
├── sdk/          Python SDK (async + sync wrappers)
└── utils/        Telemetry, cost tracker, audit log, semantic cache, drift detector

terraform/        EKS + RDS(pgvector) + ElastiCache + S3 + IAM + KMS (IaC)
k8s/               Base manifests + Kustomize overlays (dev/prod)
helm/              Production Helm chart
```

---

## Quick Reference

```bash
make setup          # Install deps + copy .env
make dev            # Start full stack with observability
make test           # Run all 520 tests with coverage
make eval           # Run RAGAS evaluation against golden dataset
make lint           # Ruff lint
make typecheck      # mypy
make docs           # Serve MkDocs site
make query Q="What was Q3 revenue?"
```

---

## Documentation

| Document | Description |
|---|---|
| [Architecture Overview](docs/architecture/overview.md) | System design and component interactions |
| [Configuration Reference](docs/configuration.md) | All environment variables |
| [Quickstart](docs/quickstart/docker.md) | Up and running in 10 minutes |
| [10-K Analysis Tutorial](docs/tutorials/10k-analysis.md) | End-to-end walkthrough |
| [Anomaly Detection](docs/tutorials/anomaly-detection.md) | Multi-quarter statistical analysis |
| [Performance & Cost Tuning](docs/performance-cost-tuning.md) | Latency and cost optimisation |
| [Troubleshooting](docs/troubleshooting.md) | Common issues + on-call runbook |
| [Security](docs/security.md) | Auth, PII, audit, compliance |
| [ADR Index](docs/architecture/adr-index.md) | Architecture decisions |
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | Quality, latency, cost numbers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guide |

---

## License

MIT — see [LICENSE](LICENSE). Built with care by [@Mattral](https://github.com/Mattral).
