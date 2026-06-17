## v2.0 Completed

### Multi-provider architecture (all real, not stubs)
- [x] Text generation: OpenAI, Gemini, Anthropic, Local vLLM
- [x] Vision: GPT-4o, Gemini 2.5 Flash, Qwen2-VL, Local vLLM + fallback chain
- [x] Embeddings: OpenAI, Voyage AI (voyage-finance-2), Cohere, local BAAI/bge
- [x] Vector stores: DeepLake, pgvector (real async), Qdrant (real async, HNSW)
- [x] Cloud connectors: S3, Azure Blob, GCS

### Implementations completed
- [x] ColPali late-interaction visual retrieval (real MaxSim algorithm, index persistence)
- [x] Knowledge graph entity extraction (real LLM call, structured JSON output)
- [x] Semantic query cache (cosine similarity, Redis + memory fallback)
- [x] Numeric grounding guardrail (regex extraction, context cross-check)

### Infrastructure
- [x] Terraform IaC: EKS + RDS + ElastiCache + S3 + KMS (347 lines, brace-balanced)
- [x] Multi-window multi-burn-rate SLO alerting (Google SRE Workbook pattern)
- [x] 7 production charts embedded in README and MkDocs
- [x] HuggingFace Space in spaces/rag-financial/ (Gradio, OpenAI + Gemini)
- [x] 520 test functions across 37 test files

# Roadmap — RAG Financial Multimodal v2.0+

## ✅ v2.0 — Completed

### Architecture
- [x] Pluggable component system (BaseParser, BaseEmbedder, BaseVectorStore, BaseRetriever, BaseReranker, BaseGenerator) via ABCs and DI
- [x] Pydantic v2 centralized config with full env-var override, nested sub-configs, feature flags
- [x] Multi-tenant architecture: per-tenant vector namespaces, rate limits, token quotas, audit trail
- [x] Async-first pipeline orchestrator with structured concurrency

### Ingestion
- [x] Unstructured.io PDF parser with PyPDF2 fallback
- [x] IBM Docling parser for superior table/layout fidelity
- [x] Layout-aware semantic chunker: table-caption pairing, multi-page table merging, heading grouping, HTML wrapping
- [x] GPT-4o vision describer with financial chart prompt (all axes, values, legends, trends)
- [x] Qwen2-VL open-source vision alternative (5-20x cheaper, on-prem capable)
- [x] Presidio PII redaction (PERSON, SSN, IBAN, CUSIP, ISIN, account numbers)

### Retrieval
- [x] Hybrid retrieval: dense vector + in-memory BM25 fused via Reciprocal Rank Fusion
- [x] Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
- [x] Cohere Rerank v3 integration
- [x] Metadata filters for tenant isolation

### Generation & Guardrails
- [x] OpenAI generator with model routing (mini → full for numerical queries)
- [x] Fallback model on 429/503
- [x] Program-of-Thought sandboxed executor: AST whitelist, 5s timeout, financial templates (CAGR, ROI, margin, %)
- [x] Numeric grounding guardrail: every number in answer must appear in context
- [x] Prompt injection detection
- [x] Financial answer system prompt with strict citation rules

### Embeddings & Caching
- [x] OpenAI text-embedding-3-small/large with Redis embedding cache (24h TTL)
- [x] Local sentence-transformers embedder (BAAI/bge-small-en-v1.5)
- [x] Semantic cache stub (threshold-based cache hit detection)

### Vector Store
- [x] DeepLake adapter with per-tenant dataset paths
- [x] In-memory vector store for testing/dev
- [x] GDPR/CCPA delete endpoint

### Observability
- [x] Full OpenTelemetry tracing: ingest span, retrieval span, generation span
- [x] Prometheus custom metrics: latency histograms, cost counters, hallucination score, citation coverage, cache hit rate, token usage
- [x] structlog structured JSON logging with OTel trace/span injection, memory metrics, service context
- [x] Grafana dashboard JSON provisioning
- [x] Cost tracker: per-tenant, per-model USD cost with quota enforcement

### Security & Compliance
- [x] API key authentication middleware (constant-time comparison)
- [x] Immutable JSONL audit log with SHA-256 tamper detection
- [x] GDPR/CCPA deletion audit events

### API & SDK
- [x] FastAPI app with lifespan, CORS, request timing, global exception handler
- [x] Ingest, Query, Health, Tenant routers with Pydantic request/response models
- [x] Thin Python SDK (async + sync wrappers, YAML config loader)
- [x] Typer CLI: ingest, query, evaluate, serve, health commands with Rich progress/tables

### Infrastructure
- [x] Multi-stage Dockerfile (builder + non-root runtime)
- [x] docker-compose with Redis, OTel collector, Prometheus, Grafana, Jaeger
- [x] Kubernetes base manifests: Deployment, Service, HPA, ConfigMap, PVC, ServiceMonitor
- [x] Kustomize overlays (dev/prod)
- [x] Helm chart with HPA, PDB, ServiceMonitor, PVC, secret management

### Testing & Evaluation
- [x] 40+ unit tests: config, cost tracker, PII redactor, guardrails, BM25, RRF fusion, audit logger, in-memory vector store, data models
- [x] Integration tests: ingest→index→query pipeline with mock components
- [x] RAGAS evaluator integration (faithfulness, answer_relevancy, context_precision)
- [x] LLM-as-judge numeric accuracy scorer
- [x] Golden dataset (8 samples across Tesla, Apple, Microsoft 10-Ks)
- [x] CI regression gate: fails PR if avg faithfulness drops >5%
- [x] Locust load test with SLO summary (p99 < 8s, < 1% errors)
- [x] GitHub Actions CI: lint (ruff/black/isort), mypy, unit tests, integration tests, security scan (Trivy + pip-audit), eval gate, Docker build, Helm lint

---

## 🔄 v2.1 — In Progress / Near-term

### Retrieval Improvements
- [ ] **pgvector adapter** — PostgreSQL-native vector store for teams already running Postgres
- [ ] **Qdrant adapter** — for dedicated vector DB deployments
- [ ] **Sparse-dense fusion** — integrate SPLADE or BM25S for better out-of-vocabulary handling
- [ ] **Contextual compression** — LLM-based chunk compression before reranking

### Generation
- [ ] **Anthropic Claude generator** — Claude 3.5 Sonnet as drop-in alternative
- [ ] **Streaming responses** — SSE/WebSocket endpoint for real-time answer streaming
- [ ] **Multi-turn conversation** — stateful session management with conversation history
- [ ] **Self-RAG** — model decides when to retrieve vs answer from parametric knowledge

### Evaluation
- [ ] **Expand golden dataset** to 50+ samples across diverse document types
- [ ] **Human-in-the-loop annotation** tool for golden dataset curation
- [ ] **Per-tenant eval dashboards** — Grafana panels showing per-tenant quality trends
- [ ] **A/B testing harness** — compare two pipeline configs on same golden set

### Infrastructure
- [ ] **Async Redis semantic cache** — cache full query→answer pairs by embedding similarity
- [ ] **Background ingest worker** — Celery/ARQ worker for large document batches
- [ ] **Presigned URL ingest** — accept S3/GCS URLs instead of file uploads
- [ ] **Webhook callbacks** — notify caller on async ingest completion

---

## 🗓 v3.0 — Future

### Vision & Multimodal
- [ ] **ColPali/ColQwen late-interaction** — embed page images directly, no OCR step
- [ ] **Table-to-structured-data** — extract tables as JSON/CSV for direct computation
- [ ] **Financial chart time-series extraction** — structured {date, value} pairs from line charts
- [ ] **Multi-page layout understanding** — cross-page figure references

### Agentic
- [ ] **LangGraph agentic flow** — multi-step agent: search → compute → verify → answer
- [ ] **Tool use** — agent can invoke PoT calculator, web search, screener APIs
- [ ] **Reflection loop** — agent checks its own answer for numeric accuracy before returning

### Knowledge Graph
- [ ] **Entity extraction** — extract company/metric/date triples from filings
- [ ] **Graph-augmented retrieval** — traverse entity relationships during retrieval
- [ ] **Cross-document reasoning** — compare metrics across multiple companies/periods

### Platform
- [ ] **Web UI** — React-based analyst interface with source highlighting
- [ ] **Slack/Teams bot** — analyst queries via chat
- [ ] **Batch export** — scheduled runs that generate structured JSON reports
- [ ] **SOC 2 Type II controls** — formal compliance documentation

---

## SLO Targets

| Metric | Current | v2.1 Target | v3.0 Target |
|---|---|---|---|
| P99 query latency | < 8s | < 5s | < 3s |
| Faithfulness score | ≥ 0.70 | ≥ 0.85 | ≥ 0.92 |
| Numeric accuracy | ≥ 0.70 | ≥ 0.85 | ≥ 0.92 |
| Error rate | < 1% | < 0.5% | < 0.1% |
| Test coverage | ≥ 70% | ≥ 80% | ≥ 90% |
