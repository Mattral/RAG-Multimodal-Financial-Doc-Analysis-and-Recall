# Roadmap — RAG Financial Multimodal

> Checked against the code on 2026-10-14 (`main` after #15 BM25, #16 Program-of-Thought and
> #17 streaming; PR numbers are cited where a fix landed).
>
> `[x]` means the feature exists and is reachable from the pipeline, API or CLI. Items that
> exist as a component but are not wired in, or that work only partly, stay `[ ]` with a
> note saying what is there. Measured results live in
> [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md), not here.

## Implemented

### Architecture & configuration
- [x] Pluggable components via ABCs and dependency injection (parser, embedder, vector store,
      retriever, reranker, generator, evaluator, vision describer)
- [x] Pydantic v2 configuration with env-var overrides, nested sub-configs and feature flags
- [x] Multi-tenancy: per-tenant vector namespaces, per-tenant BM25 partitions (#15), rate
      limits, token quotas, audit trail
- [x] Async pipeline orchestrator

### Ingestion
- [x] Unstructured.io parser with PyPDF fallback; Docling and Marker parser adapters
- [x] Layout-aware chunking (table–caption pairing, multi-page table merging, heading grouping)
- [x] Chart/figure description: GPT-4o, Gemini, Qwen2-VL, local vLLM, with a fallback chain
- [x] Presidio PII redaction with financial identifiers (CUSIP, ISIN, account numbers)
- [x] Cloud connectors: S3, Azure Blob, GCS
- [x] Delta detection and document versioning, including point-in-time lookup (`get_version_at`)

### Retrieval
- [x] Dense vector search: DeepLake, pgvector (asyncpg), Qdrant (async), in-memory
- [x] Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion. The BM25 index is
      populated at ingest and tenant-partitioned (#15). Before #15 it was never populated,
      so the pipeline's hybrid search ran dense-only
- [x] Rerankers: cross-encoder (ms-marco-MiniLM-L-6-v2) and Cohere Rerank
- [x] Metadata filters (vector store and BM25)
- [x] Optional knowledge-graph-augmented retrieval (`ENABLE_KNOWLEDGE_GRAPH`): LLM entity
      extraction at ingest, traversal from query entities, weighted RRF fusion. In memory
      only. Measured effect on the golden set is small (+0.017 evidence hit@5)
- [ ] ColPali visual retrieval in the query path. *The MaxSim retriever with index persistence
      exists in `components/colpali_retriever.py`, but the pipeline never calls it.*

### Generation & guardrails
- [x] Generators: OpenAI, Gemini, Anthropic, local vLLM / Together, selected by
      `LLM_CONFIG__PROVIDER`
- [x] Model routing (simple vs. complex queries); OpenAI fallback model on 429/503
- [x] Gemini thinking-token budget so reasoning cannot truncate answers
- [x] Program-of-Thought: the LLM writes a calculation program over the retrieved figures and
      a restricted arithmetic interpreter evaluates it (#16, #20). Before #16 the pipeline tried
      to execute code found in retrieved document text. Before #20 the sandbox used `exec()`
      and could be escaped through attribute access
- [x] Numeric grounding guardrail, prompt-injection detection, citation-enforcing system prompt
- [x] Streaming answers: `POST /api/v1/query/stream` (SSE) and `query_stream()`, with native
      token streaming for OpenAI and Gemini (#17). Guardrails run after the last token
- [ ] LangGraph agentic flow (analyze → retrieve → verify → calculate → synthesize → guard).
      *Implemented in `src/rag_system/agentic/`, but not reachable from the API, CLI or
      pipeline, and `ENABLE_LANGGRAPH_AGENTIC` is defined but never read.*

### Embeddings & caching
- [x] Embedders: OpenAI (with Redis embedding cache), local BAAI/bge, Voyage, Cohere
- [x] Semantic query cache (embedding similarity, Redis or in-memory), used by the pipeline
- [ ] Structured table output in the pipeline. *`components/table_extractor.py` converts tables
      to JSON / CSV / DataFrames, but nothing in the pipeline calls it.*

### Observability
- [x] OpenTelemetry spans for ingest, retrieval and generation
- [x] 17 Prometheus metrics (latency histograms, cost, tokens, citation coverage, hallucination
      score, cache hits, tenant quota gauges)
- [x] structlog JSON logging with trace/span ids; 2 Grafana dashboards
- [x] Per-tenant, per-model cost tracking with quota enforcement
- [x] Multi-window, multi-burn-rate SLO alert rules; corpus and query-pattern drift detector

### Security & compliance
- [x] API-key authentication. Keys are compared as SHA-256 digests with `hmac.compare_digest`;
      the previous `!=` comparison was not constant-time and has been fixed
- [x] Append-only JSONL audit log with SHA-256 tamper detection
- [x] GDPR/CCPA soft delete with audit events

### API, SDK & CLI
- [x] FastAPI app: ingest, query, query/stream (#17), documents, tenants, feedback, health
- [x] Python SDK (async + sync, YAML config); `query_stream()` (#17)
- [x] Typer CLI: ingest, query, evaluate, serve, health
- [x] Demo apps: `demo/app.py` and the Hugging Face Space in `spaces/rag-financial/` (Gradio)

### Infrastructure
- [x] Multi-stage Dockerfile; docker-compose with Redis, OTel collector, Prometheus,
      Alertmanager, Grafana and Jaeger
- [x] Kubernetes base manifests (incl. ServiceMonitor, NetworkPolicy, HPA) and Kustomize dev/prod
      overlays
- [x] Helm chart with HPA, PDB, PVC, ConfigMap and Secret
- [ ] Helm ServiceMonitor. *`values.yaml` has a `serviceMonitor` key, but no template uses it.*
- [x] Terraform: EKS, RDS, ElastiCache, S3, KMS

### Testing & evaluation
- [x] 868 tests (unit, integration, Hypothesis property tests). Unit + integration coverage
      was 73% when measured before #15–#17. Locust and k6 load-test scripts
- [x] Golden dataset: 119 QA pairs over 16 public filings. Every figure is traced to a quoted
      source passage (`provenance.json`, verified 209/209) and each derived number has an
      arithmetic check
- [x] Deterministic eval metrics (expected-value recall, source hit rate, number grounding)
- [x] Offline retrieval benchmark and answer-quality eval on the real filings
- [x] RAGAS and LLM-as-judge scorers. *They call OpenAI only, and no RAGAS run has been
      published.*
- [x] CI jobs: lint (ruff/black/isort), mypy, tests, Trivy + pip-audit + Semgrep, eval, Docker
      build, Helm lint
- [x] Tests gate merges (#19). Before #19 the test step was `continue-on-error` and its install
      list lacked `typer`/`pyyaml`, so pytest aborted at collection and CI ran zero tests while
      showing green. CI now runs about 790 unit and integration tests with a 70% coverage floor
- [ ] Lint, mypy, Semgrep and the eval gate gating merges. *Still `continue-on-error`. The eval
      job also has no LLM key in CI.*

---

## Next (v2.1)

Highest impact first:

- [ ] **Make the rest of CI blocking.** Tests block since #19. Next is lint, once the existing
      debt is fixed, then the eval gate once it has a key and a budget
- [ ] **Persistence for in-memory indexes.** BM25 and the knowledge graph are lost on restart.
      CLI `ingest` and `query` run as separate processes, so the CLI never benefits from the
      graph
- [ ] **Calculation questions** are the weakest category (the model abstained on 15 of 24).
      Measure the effect of the #16 Program-of-Thought path, and revisit the "never estimate"
      system prompt
- [ ] **Provider-pluggable eval judge** (Gemini), so RAGAS and numeric-accuracy scores can be
      measured without OpenAI
- [ ] **Retired model names.** `gemini-2.5-*` defaults in the code and the HF Space return 404
      for new Google API keys
- [ ] **Load and cost measurement.** Run the Locust suite and publish latency and cost per query
- [ ] Wire ColPali, the LangGraph agentic flow and the table extractor into the pipeline behind
      flags, or remove them
- [ ] Helm ServiceMonitor template
- [ ] Golden set: a second annotator, more sectors and years, and paraphrased questions (the
      current wording favors BM25)
- [ ] Knowledge graph: query-time entity aliases ("Alphabet" vs "Google") and a persistent store
- [ ] Sparse-dense fusion (SPLADE / BM25S)
- [ ] Contextual compression before reranking
- [ ] Multi-turn conversation with session history
- [ ] Self-RAG (decide when to retrieve)
- [ ] Human-in-the-loop annotation tool for the golden set
- [ ] Per-tenant eval dashboards
- [ ] A/B harness comparing two pipeline configs on the golden set
- [ ] Background ingest worker (Celery / ARQ), presigned-URL ingest, webhook callbacks

## Later (v3.0)

- [ ] Structured chart time-series extraction (`{date, value}` pairs from line charts)
- [ ] Cross-page figure and reference resolution
- [ ] Agent tool use (calculator, screener APIs, web search) and a reflection loop in production
- [ ] Cross-document reasoning (compare metrics across companies and periods via the graph)
- [ ] Analyst web UI with source highlighting (beyond the Gradio / demo apps)
- [ ] Slack / Teams bot
- [ ] Scheduled batch export of structured reports
- [ ] SOC 2 Type II controls

---

## SLO targets

These are targets, not measurements. Measured values are in BENCHMARK_RESULTS.md.

| Metric | Target (v2.x) | Measured |
|---|---|---|
| P99 query latency | < 8 s | Not measured under load (generation alone: p50 3.0 s, p95 5.0 s) |
| Faithfulness (RAGAS) | ≥ 0.70 | Not measured |
| Numeric accuracy (LLM judge) | ≥ 0.70 | Not measured. No fabricated numbers in 119 answers (grounding check) |
| Error rate | < 1% | Not measured |
| Test coverage (unit + integration) | ≥ 70% | 73% |
