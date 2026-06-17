# Changelog

All notable changes to RAG Financial Multimodal are documented here.
This project follows [Semantic Versioning](https://semver.org/).

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
