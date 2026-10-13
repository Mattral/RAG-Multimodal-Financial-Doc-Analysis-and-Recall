# RAG Financial Multimodal — Enterprise v2.0

> **Production-grade multimodal RAG for financial document intelligence.**
> Chart understanding · hybrid retrieval · numeric guardrails · multi-tenancy · full observability.

[![CI/CD](https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall/actions)
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

Measured offline on the 16 real filings behind the golden dataset (119 questions), using this
repo's BM25, local `bge-small` embeddings and RRF fusion with default settings and no reranker.
*Evidence hit@5* means a top-5 chunk contains the exact supporting text for the answer.

| Strategy (4,000-char chunks, default) | Evidence hit@5 | Doc hit@5 |
|---|---|---|
| BM25 only | 0.71 | 0.87 |
| Dense only | 0.61 | 0.99 |
| **Hybrid RRF** | **0.71** | 0.93 |

With 1,000-character chunks, hybrid reaches 0.53 versus 0.42 (BM25) and 0.46 (dense), so fusion
helps there. At the default chunk size it only ties BM25. About 3 in 10 questions get no
answer-bearing chunk in the top 5. Full results, method and caveats are in
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). Earlier versions of this README claimed 89%
Recall@5. That figure could not be reproduced and has been withdrawn.

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

* **Golden dataset:** [119 QA pairs](evals/golden_datasets/README.md) over 16 public filings
  from 10 companies. The mix is 56 factual, 24 calculation, 17 comparative, 13 table and 9 chart
  questions. Every figure is traced to a quoted passage in its source document (209/209 verified
  against the primary documents) and every derived number has an arithmetic check.
  Re-checking found errors in 9 of the original 22 samples, and those have been corrected.
* **Scoring:** RAGAS and an LLM-as-judge numeric scorer (needs an OpenAI key), plus two
  deterministic checks that need no LLM: whether the answer states the expected values, and
  whether it cites the expected source document.
* **Answer quality:** not yet measured on the full set. An 11-question pilot with
  `gemini-3.8-flash` stopped at free-tier limits. It found no wrong figures, but also found
  truncated answers and invented page citations. See
  [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md#3-answer-quality-pilot-with-gemini-38-flash-n--11).

```bash
pytest tests/unit/test_golden_dataset.py      # dataset integrity, offline
python -m evals.retrieval_benchmark           # retrieval, offline
python -m evals.answer_eval --rpm 5           # retrieval + generation, needs an LLM key
```

---

## Latency and Cost

Not yet measured in a reproducible run. Earlier latency, load-test and cost-per-query figures
were withdrawn because no run outputs or scripts reproduce them. `tests/load/locustfile.py`
and `scripts/benchmark.py` can measure them against a running deployment. The SLO targets
(p99 < 8 s, < 1% errors) are configured in `src/rag_system/config.py` and the alerting rules.

---

## SLO Alerting

![SLO Burn Rate](docs/assets/slo_burn_rate.png)

*Multi-window multi-burn-rate alerting from the Google SRE Workbook. Four alert tiers (14.4×, 6×, 3×, 1× burn rate) routing to PagerDuty/OpsGenie.*

---

## Architecture Layers

| Layer | Component | Technology |
|---|---|---|
| **Parsing** | PDF text + tables | Unstructured.io / Docling / Marker |
| **Vision** | Chart + graph extraction | GPT-4o / Gemini 2.5 Flash / Qwen2-VL / Local vLLM (fallback chain) |
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

## Try It Live

**Interactive notebook**: [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) walks through the
full pipeline end-to-end — ingestion, hybrid retrieval, vision/chart queries, Program-of-Thought
calculations, guardrails, and cost tracking — using Tesla's real Q3 2023 Investor Update as the
example document.

A standalone Gradio demo lives in [`spaces/rag-financial/`](spaces/rag-financial/) — deploy it
directly to HuggingFace Spaces or run it locally:

```bash
cd spaces/rag-financial
pip install -r requirements.txt
python app.py
```

Bring your own OpenAI or Google Gemini API key (Gemini has a free tier). The Space mirrors
`src/rag_system/`'s hybrid retrieval + guardrail logic in a self-contained form with full
pipeline transparency — see exactly how each answer was retrieved and generated.

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
spaces/            Standalone HuggingFace Space demos
└── rag-financial/ Gradio app — OpenAI + Gemini, hybrid retrieval, guardrails
```

---

## Quick Reference

```bash
make setup          # Install deps + copy .env
make dev            # Start full stack with observability
make test           # Run the test suite with coverage
make eval           # Run RAGAS evaluation against golden dataset
make lint           # Ruff lint
make typecheck      # mypy
make docs           # Serve MkDocs site
make query Q="What was Q3 revenue?"
```

---

## Limitations

* **Evaluation is partial.** Dataset integrity and offline retrieval are measured. End-to-end
  answer quality, RAGAS scores, latency and cost have not been measured on a reproducible run
  yet (see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)).
* **Retrieval misses often.** About 30% of golden questions get no answer-bearing chunk in the
  top 5 at default settings, and multi-row tables are the weakest case when chunks are small.
* **Small, single-author golden set.** 119 questions on fiscal 2023, Tesla-heavy (40 of 119),
  written by one person from the filings themselves. Their wording favors keyword search.
* **Chart questions are text-checkable.** The 9 chart questions use charts whose values also
  appear in tables, so they do not isolate the vision path.
* **Gemini judge not wired.** The LLM-as-judge and RAGAS scorers call OpenAI. With only a Gemini
  key, just the deterministic metrics are available.
* **Thinking models can truncate answers.** On `gemini-3.8-flash`, answers were cut off
  under the current `maxOutputTokens=2048`, and cited page numbers are unreliable when chunks
  lack page metadata.
* **Retired model names.** Google no longer serves `gemini-2.5-*` to new API keys. Defaults
  that still name them need a current model such as `gemini-3.8-flash`.

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
