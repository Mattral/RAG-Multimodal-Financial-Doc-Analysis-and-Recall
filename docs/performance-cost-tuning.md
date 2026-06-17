# Performance & Cost Tuning

## Latency Targets

| Query Type | Target p50 | Target p99 |
|---|---|---|
| Simple factual | < 1s | < 3s |
| Numerical (PoT) | < 2s | < 5s |
| Comparative (multi-chunk) | < 3s | < 8s |
| Agentic (multi-step) | < 8s | < 20s |

---

## Cost Per Query (Approximate)

| Configuration | Cost/query |
|---|---|
| gpt-4o-mini + text-embedding-3-small | $0.0001–0.0005 |
| gpt-4o + text-embedding-3-small | $0.005–0.02 |
| gpt-4o + vision (1 image) | $0.02–0.10 |
| Local (vLLM + local embeddings) | ~$0 (infra cost) |

---

## Reducing Latency

### 1. Enable embedding cache (biggest win)
```bash
CACHE_CONFIG__BACKEND=redis
CACHE_CONFIG__REDIS_URL=redis://localhost:6379/0
```
Typical cache hit rate: 60–80% after first few days. ~0ms vs ~150ms per embedding call.

### 2. Reduce top-k
```bash
RETRIEVER_CONFIG__TOP_K_DENSE=10      # default: 20
RETRIEVER_CONFIG__TOP_K_FINAL=5       # default: 10
```

### 3. Use smaller reranker
```bash
RERANKER_CONFIG__PROVIDER=none        # skip reranker for lowest latency
RERANKER_CONFIG__PROVIDER=cross_encoder  # ms-marco-MiniLM is fast
# Avoid cohere for latency-sensitive paths (network call)
```

### 4. Local vLLM for vision
Replace GPT-4o vision with Qwen2-VL-7B on local GPU — eliminates network round-trip.
```bash
VISION_CONFIG__PROVIDER=local_vllm
LOCAL_VLLM_BASE_URL=http://your-gpu-server:8080/v1
VISION_CONFIG__MODEL=Qwen/Qwen2-VL-7B-Instruct
```

---

## Reducing Cost

### 1. Model routing (already built-in)
Simple queries use gpt-4o-mini (33× cheaper than gpt-4o).
```bash
LLM_CONFIG__MODEL=gpt-4o-mini
LLM_CONFIG__COMPLEX_QUERY_MODEL=gpt-4o
LLM_CONFIG__ENABLE_MODEL_ROUTING=true
```

### 2. Vision: use Gemini Flash
10–40× cheaper than GPT-4o vision, similar quality.
```bash
VISION_CONFIG__PROVIDER=gemini
# Requires GOOGLE_API_KEY
```

### 3. Skip vision for text-only documents
```bash
rag-financial ingest report.pdf --no-vision
```

### 4. Semantic cache
Repeated/similar queries return cached answers instantly.
```bash
CACHE_CONFIG__SEMANTIC_CACHE_ENABLED=true
CACHE_CONFIG__SEMANTIC_CACHE_THRESHOLD=0.92
```

### 5. Local embeddings
Eliminate embedding API cost entirely.
```bash
# pip install sentence-transformers
VECTOR_STORE_CONFIG__EMBEDDING_MODEL=local
# Uses BAAI/bge-small-en-v1.5 locally
```

---

## Monitoring Cost in Production

```bash
# Real-time cost per tenant (last hour)
curl http://localhost:9090/api/v1/query?query=increase(rag_query_cost_usd_total[1h])

# Per-tenant usage via API
curl http://localhost:8000/api/v1/tenants/acme/usage
```

Set Prometheus alerts for cost burn rate:
```yaml
# In your alerting rules:
- alert: RAGCostBurnRateHigh
  expr: increase(rag_query_cost_usd_total[1h]) > 10
  annotations:
    summary: "RAG cost exceeding $10/hour"
```

---

## Scaling for High Throughput

### Horizontal scaling (K8s)
The API is stateless. Scale replicas:
```bash
kubectl scale deploy/rag-financial --replicas=5
```
HPA auto-scales on CPU (target 70%) or custom metrics (queue depth).

### Redis connection pool
For >100 concurrent users, increase pool:
```python
# In embedder — already using connection pooling
# Ensure Redis maxmemory policy: allkeys-lru
```

### Async batching
Embedding calls are already batched (100 texts per OpenAI call).
For very high ingest throughput, use background workers:
```bash
# Coming in v2.1: Celery/ARQ worker for batch ingest
```
