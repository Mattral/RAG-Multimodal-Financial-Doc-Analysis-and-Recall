# Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `ENVIRONMENT` | `development` | `development\|staging\|production` |
| `LLM_CONFIG__PROVIDER` | `openai` | `openai`\|`gemini`\|`anthropic`\|`local_vllm` — pick any text generation backend |
| `LLM_CONFIG__MODEL` | `gpt-4o-mini` | Default generation model |
| `LOCAL_VLLM_GENERATOR_BASE_URL` | `http://localhost:8090/v1` | Used only when `LLM_CONFIG__PROVIDER=local_vllm` |
| `VECTOR_STORE_CONFIG__EMBEDDING_PROVIDER` | `openai` | `openai`\|`local`\|`voyage`\|`cohere` — `local` requires no API key |
| `LLM_CONFIG__COMPLEX_QUERY_MODEL` | `gpt-4o` | Numerical/complex queries |
| `VISION_CONFIG__PROVIDER` | `openai` | `openai\|qwen2-vl\|pixtral` |
| `VECTOR_STORE_CONFIG__PROVIDER` | `qdrant` | `qdrant\|pgvector\|deeplake\|memory` |
| `VECTOR_STORE_CONFIG__QDRANT_URL` | `http://localhost:6333` | Qdrant server URL; must be http(s). Helm requires `config.qdrantUrl` (the chart does not deploy Qdrant) |
| `VECTOR_STORE_CONFIG__QDRANT_API_KEY` | — | Qdrant API key, if the server requires one |
| `PDF_PARSING_CONFIG__UNSTRUCTURED_STRATEGY` | `hi_res` | `hi_res` (layout model + OCR, needed for tables) \| `fast` (text layer only) \| `auto` |
| `PDF_PARSING_CONFIG__FALLBACK_PARSER` | `pypdf` | Parser used when the primary parser fails on a document: `pypdf` (text layer, one element per page) or `none` (the document fails). Each fallback logs `pdf_parser_fallback` (warning, with the exception) and is recorded in the ingest result's `parsers` map and in `rag_documents_parsed_total{parser,fallback_from}` |
| `DOC_REGISTRY_PATH` | `./data/doc_versions.json` | Document version registry used to skip unchanged documents on re-ingest |
| `RETRIEVER_CONFIG__STRATEGY` | `hybrid` | `dense\|hybrid\|graph_augmented` (`graph_augmented` needs `ENABLE_KNOWLEDGE_GRAPH=true`) |
| `ENABLE_KNOWLEDGE_GRAPH` | `false` | LLM entity extraction at ingest + graph-augmented retrieval (in-memory graph) |
| `RETRIEVER_CONFIG__GRAPH_MAX_HOPS` | `1` | Graph traversal depth from query entities (0–3) |
| `RETRIEVER_CONFIG__GRAPH_MAX_CHUNKS` | `10` | Graph-found chunks fused with hybrid results |
| `RETRIEVER_CONFIG__GRAPH_WEIGHT` | `0.5` | RRF weight of graph-found chunks relative to hybrid results |
| `LLM_CONFIG__THINKING_BUDGET_TOKENS` | `1024` | Reasoning-token budget for Gemini thinking models, added on top of `max_tokens` |
| `LLM_CONFIG__BASE_URL` | — | `openai` provider only: OpenAI-compatible API root (proxy, gateway or local stub). Default `https://api.openai.com/v1` |
| `RERANKER_CONFIG__PROVIDER` | `cross_encoder` | `cross_encoder\|cohere\|none` |
| `CACHE_CONFIG__BACKEND` | `redis` | `redis\|memory` |
| `RAG_API_MASTER_KEY` | — | API authentication key. If unset, API requests get `401` unless `ENVIRONMENT=development` is set explicitly or `AUTH_DISABLED=true` |
| `AUTH_DISABLED` | `false` | Accept requests without a key when `RAG_API_MASTER_KEY` is unset (local use only; logs a warning at startup). Ignored when a key is set |
| `SECURITY_CONFIG__ENABLE_PII_REDACTION` | `true` | Presidio PII redaction |
| `OBSERVABILITY_CONFIG__OTLP_ENDPOINT` | — | OTel gRPC endpoint |
| `MULTI_TENANCY_CONFIG__ENABLED` | `true` | Multi-tenant isolation |
