# Changelog

All notable changes to RAG Financial Multimodal are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Streaming
- `POST /api/v1/query/stream`: server-sent events `sources` -> `token`... -> `done` (the same payload
  as `POST /query`) or `error`. `POST /api/v1/query` is unchanged
- `RAGPipeline.query_stream()` and SDK `RAGPipeline.query_stream()`. `query()` is now split into
  shared stages (`_begin_query`, `_retrieve_and_compute`, `_finish_query`) with unchanged behaviour
- `BaseGenerator.generate_stream()`, with native token streaming for OpenAI (Chat Completions SSE,
  falling back to the non-streaming fallback-model path on 429/503) and Gemini
  (`streamGenerateContent?alt=sse`, thought parts skipped). Other providers stream the whole
  answer as one piece
- Guardrails still run on the complete answer, after the last token, and are reported in `done`

### Docs
- `roadmap.md` rewritten against the code. The "completed" header no longer contradicts the
  detailed sections. Components that exist but are not wired in (ColPali, LangGraph agentic flow,
  table extractor, Helm ServiceMonitor) are marked as such, and so is CI's non-blocking
  `continue-on-error` gates. SLO targets are separated from measurements

### Knowledge graph
- `GraphAugmentedRetriever` now performs real graph traversal: query entities (companies,
  metrics, products) -> hop-limited neighbourhood -> chunks scored by matched query concepts ->
  weighted RRF fusion with the hybrid results. Falls back to hybrid results unchanged. Off by
  default, behind `ENABLE_KNOWLEDGE_GRAPH`. In-memory graph only
- `InMemoryGraphStore`: canonical tenant-scoped entity ids, entity <-> chunk links, adjacency index
- `EntityExtractor`: OpenAI or Gemini (JSON mode), metric/period-focused prompt, `REPORTED_METRIC`
  and `FOR_PERIOD` relations, prompt truncated at the parser's chunk size (was 3,000 chars)
- Pipeline extracts entities during ingest when the flag is on and wraps the retriever
- New config: `RETRIEVER_CONFIG__GRAPH_MAX_HOPS` (1), `__GRAPH_MAX_CHUNKS` (10), `__GRAPH_WEIGHT` (0.5).
  `_reciprocal_rank_fusion` accepts optional per-list weights
- Measured on the golden set: evidence hit@5 0.706 -> 0.723 (7 gained, 5 lost). See
  BENCHMARK_RESULTS.md section 2b. `evals/build_graph.py` + `retrieval_benchmark --graph-cache`

### Evaluation
- Golden dataset expanded from 22 to 119 QA pairs over 16 public filings (10 companies;
  factual, calculation, comparative, table and chart questions), with `provenance.json`
  recording the source passage behind every figure and an arithmetic check for each derived number
- Corrected 9 of the original 22 samples that had wrong figures or claims (see
  `evals/golden_datasets/README.md`)
- `scripts/verify_golden_dataset.py` checks every evidence string against the primary documents,
  and `tests/unit/test_golden_dataset.py` checks schema, consistency and calculations offline
- Deterministic eval metrics that need no LLM: expected-value recall and source hit rate
- `evals/retrieval_benchmark.py` (offline BM25 / dense / hybrid benchmark on the real filings)
  and `evals/answer_eval.py` (retrieval + generation, rate-limited and resumable)
- Full answer-quality run on all 119 questions with `gemini-3.8-flash`, with deterministic
  scoring plus a number-grounding check (`--regrade` re-scores saved answers offline). No
  fabricated numbers found, and headline figure stated in 68% of answers
- BENCHMARK_RESULTS.md and README rewritten around measured numbers. Earlier quality,
  retrieval, latency and cost figures are no longer cited. The retrieval chart could not have
  come from this repo, and the others are unverified because no run outputs were committed. The
  original charts are kept in BENCHMARK_RESULTS.md section 4 with a provenance note. New charts
  are drawn from `evals/results/` by `scripts/plot_benchmarks.py`

### Fixed
- CI ran zero tests while reporting success. The test job lacked `typer`, `rich` and `pyyaml`, so
  pytest aborted at collection, and the step was `continue-on-error`. Both are fixed, and failing
  tests or coverage below 70% now fail CI
- The pipeline's BM25 index was created empty and never populated, so production "hybrid"
  retrieval was dense-only. `RAGPipeline.ingest()` now adds every element to a tenant-partitioned
  `BM25Index` (own corpus and term statistics per tenant, upsert by `chunk_id`, exact-match
  filters like the vector store), and `HybridRetriever` searches only the caller's partition.
  The index is in memory and lost on restart (see ADR-001). BM25 also no longer pads results
  with zero-score chunks. The offline benchmark headline numbers are unchanged
- Program-of-Thought in `RAGPipeline.query()` passed the text of retrieved chunks to
  `execute_from_llm_response`, so it tried to extract and run code found in documents
  (untrusted input) instead of computing what the question asked. It now asks the LLM to write
  a calculation program from the context (`POT_CODEGEN_SYSTEM_PROMPT`) and runs only that, in
  the existing AST-allowlisted sandbox. Retrieved text is never executed. This adds one LLM call
  to queries routed to PoT
- When PoT succeeded with no caller system prompt, the pipeline replaced the default grounding and
  citation prompt with just the computed value. The note is now appended to the default prompt
- API-key middleware compared SHA-256 hex digests with `!=`, which is not constant-time. It now uses
  `hmac.compare_digest`, and the middleware has its first tests
- `evals/run_evals.py` and `rag-financial evaluate` treated the eval report as a dict and raised
  `TypeError` (a regression from the 2026-09-04 lint commit `95f1c95`)
- `LLM_CONFIG__PROVIDER=gemini` (and `local_vllm`) were rejected by config validation even
  though `build_generator` supports them
- `GOOGLE_API_KEY` in `.env` was ignored. Gemini text and vision now read it through config and
  send it in the `x-goog-api-key` header instead of the URL
- `GeminiGenerator` no longer fails on responses with multiple or missing text parts, skips thought parts, and gives thinking its own token budget
  (`LLM_CONFIG__THINKING_BUDGET_TOKENS`, default 1024). Before this, reasoning could use up
  `maxOutputTokens` and truncate answers. It now logs a warning on `MAX_TOKENS`
- `scripts/seed_golden_dataset.py` no longer overwrites the main golden dataset by default
- Restored `tests/unit/test_cli.py`, which a lint commit had truncated into a syntax error;
  tests no longer read a developer's local `.env`

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
