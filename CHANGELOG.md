# Changelog

All notable changes to RAG Financial Multimodal are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2024-07

### Added — Multi-Provider Architecture
- **Generator providers**: OpenAI (GPT-4o-mini/GPT-4o), Google Gemini (2.0 Flash/1.5 Pro),
  Anthropic (Claude 3.5 Haiku/Sonnet), Local vLLM (Llama-3.1, Qwen2.5, Mistral) — all via
  `LLM_CONFIG__PROVIDER` with zero code changes
- **Embedder providers**: OpenAI, Voyage AI (voyage-finance-2, finance-domain-tuned),
  Cohere (embed-english-v3.0), local BAAI/bge-small-en-v1.5 (zero API cost)
- **Vision providers**: GPT-4o, Gemini 2.5 Flash, Qwen2-VL (Together.ai), local vLLM —
  with automatic fallback chain
- **Vector store**: Qdrant adapter (real async, HNSW m=16 ef_construct=200, per-tenant
  collections) alongside existing DeepLake and pgvector
- **Cloud connectors**: GCS connector (Google Cloud Storage) completing the S3 + Azure + GCS
  enterprise trio

### Added — Real Implementations (replacing stubs)
- **ColPali retriever**: real MaxSim late-interaction scoring `score(Q,D) = Σ max_j(q_i·d_j)`
  with index persistence, graceful fallback when colpali-engine not installed
- **Knowledge graph extraction**: `EntityExtractor` now makes a real LLM call with structured
  JSON output — extracts COMPANY, METRIC, DATE entities and REPORTED_REVENUE, SUBSIDIARY_OF
  relations from each ingested chunk
- **Semantic query cache**: cosine similarity over query embeddings, Redis-backed with
  in-process memory fallback, per-tenant isolation, TTL, eviction (max 500 entries/tenant)

### Added — Infrastructure
- **Terraform**: complete `terraform/main.tf` (347 lines) — EKS, RDS+pgvector, ElastiCache
  Redis, S3 (docs + 7-year audit retention), KMS, IRSA, Secrets Manager
- **SLO alerting**: multi-window multi-burn-rate Prometheus rules (14.4×/6×/3×/1× burn rate,
  Google SRE Workbook pattern) with Alertmanager PagerDuty/OpsGenie routing
- **Grafana**: 2 dashboards — overview (12 panels incl. SLO burn rate) + quality/cost (10 panels)
- **Tenant quota metrics**: `rag_tenant_monthly_tokens_used` / `rag_tenant_monthly_token_quota`
  Prometheus gauges published on every `check_quota()` call
- **HuggingFace Space**: standalone Gradio demo in `spaces/rag-financial/` — OpenAI + Gemini,
  hybrid retrieval, numeric guardrails, full pipeline transparency

### Added — Documentation
- 7 production-quality charts in `docs/assets/`: architecture pipeline, retrieval quality,
  latency benchmarks, evaluation radar, cost per query, SLO burn rate, provider matrix
- README rewritten (250 lines) with all charts embedded
- On-call runbook at `docs/troubleshooting.md#high-error-rate` (anchor matches alert `runbook_url`)
- ADR-008: knowledge graph design decision and v3.0 plan

### Changed
- `pyproject.toml` extras renamed to guideline-specified groups:
  `api`, `enterprise`, `eval`, `graph`, `agentic`, `all`
- `build_vector_store()` factory now routes `qdrant` to `QdrantAdapter` (was fallback to DeepLake)
- `build_embedder()` factory now infers provider from `VECTOR_STORE_CONFIG__EMBEDDING_MODEL`
  name if `VECTOR_STORE_CONFIG__EMBEDDING_PROVIDER` not set explicitly
- `evaluator._llm_numeric_judge()` fixed: added `raise_for_status()`, regex-safe float parsing
  with explicit `except httpx.HTTPStatusError` / `except httpx.TimeoutException` branches —
  previously a 4xx/5xx silently returned 0.5 neutral score instead of surfacing the error

### Fixed
- PR #9: `demo/app.py` example-question button `key=f"ex_{i}"` (was `key=f"ex_{ex[:10]}"`,
  collided when two questions shared a 10-character prefix, crashing the demo on load)
- `connectors/__init__.py`: `class AzureBlobConnector` and `def __init__` were on the same line
- `components.md`: ColPali listed as "stub" — updated to "real implementation"

## [2.0.0] - 2024-07

### Added
- Pluggable ABC architecture: BaseParser, BaseEmbedder, BaseVectorStore, BaseRetriever, BaseReranker, BaseGenerator
- Pydantic v2 centralized config with 12 nested sub-configs and full env-var override
- Hybrid RRF retrieval: dense + BM25 fused with Reciprocal Rank Fusion
- Cross-encoder and Cohere rerankers
- Layout-aware semantic chunker: table-caption pairing, multi-page table merging, HTML wrapping
- GPT-4o and Qwen2-VL vision describers with financial specialist prompt
- Program-of-Thought sandboxed calculator: AST whitelist, 5s timeout, 6 financial templates
- Presidio PII redaction with CUSIP/ISIN/account number extensions
- Numeric grounding guardrail: every answer number verified against context
- Prompt injection detection
- Multi-tenancy: per-tenant vector isolation, rate limits, token quotas
- Per-tenant, per-model cost tracking with monthly quota enforcement
- OpenTelemetry distributed tracing with ingest/retrieval/generation spans
- 15 custom Prometheus metrics (latency, cost, hallucination score, citation coverage)
- Grafana dashboard JSON with 10 panels
- structlog JSON logging with OTel trace/span injection
- FastAPI REST API with auth middleware, OpenAPI docs, health probes
- Python SDK with async and sync wrappers
- Typer CLI: ingest, query, evaluate, serve, health
- Document version manager with delta detection and point-in-time retrieval
- QueryAnalyzer: intent classification, entity extraction, query rewriting
- Enterprise connectors: S3, Azure Blob, local filesystem
- Multi-stage Dockerfile (non-root runtime)
- docker-compose with Redis, OTel, Prometheus, Grafana, Jaeger
- Kubernetes manifests: Deployment, Service, HPA, ConfigMap, PVC, ServiceMonitor
- Kustomize overlays (dev/prod)
- Helm chart with PDB, ServiceMonitor, PVC
- GitHub Actions CI: lint, typecheck, unit tests, integration tests, security scan, eval gate, Docker build, Helm lint
- RAGAS evaluator + LLM-as-judge numeric scorer
- Golden dataset (8 financial QA samples)
- CI regression gate (>5% faithfulness drop blocks PR)
- Locust load test with SLO summary
- 100+ unit and integration tests across 9 test files
- MkDocs Material documentation site
- 4 Architecture Decision Records
- CONTRIBUTING.md, SECURITY.md, CHANGELOG.md

### Changed
- PDF parser upgraded from simple partition to chunked, layout-aware pipeline
- Vision processing now batched with concurrency control and retry
- Config fully migrated from scattered os.getenv() to Pydantic v2 BaseSettings
- Exception hierarchy expanded with HTTP status codes and structured details
- Logger upgraded to structlog with OTel context injection
- Retry policy upgraded to full-jitter AWS pattern with decorator support
- Rate limiter upgraded to per-tenant token buckets + Redis sliding-window option

### Removed
- Hardcoded component instantiation in pipeline
- Global state for configuration
- Binary artifacts (.pkl, .avif) from tracked files

## [1.0.0] - 2024-01

### Added
- Initial release: async PDF ingestion, GPT-4V vision, DeepLake vector store
- Basic CLI and structured logging
