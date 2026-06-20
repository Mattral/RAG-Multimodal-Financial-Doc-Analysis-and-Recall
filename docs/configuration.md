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
| `VECTOR_STORE_CONFIG__PROVIDER` | `deeplake` | `deeplake\|pgvector\|qdrant` |
| `RETRIEVER_CONFIG__STRATEGY` | `hybrid` | `dense\|hybrid\|graph_augmented` |
| `RERANKER_CONFIG__PROVIDER` | `cross_encoder` | `cross_encoder\|cohere\|none` |
| `CACHE_CONFIG__BACKEND` | `redis` | `redis\|memory` |
| `RAG_API_MASTER_KEY` | — | API authentication key |
| `SECURITY_CONFIG__ENABLE_PII_REDACTION` | `true` | Presidio PII redaction |
| `OBSERVABILITY_CONFIG__OTLP_ENDPOINT` | — | OTel gRPC endpoint |
| `MULTI_TENANCY_CONFIG__ENABLED` | `true` | Multi-tenant isolation |
