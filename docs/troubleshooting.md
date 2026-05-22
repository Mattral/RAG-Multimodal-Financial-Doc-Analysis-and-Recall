# Troubleshooting

## Installation Issues

### `poppler-utils` not found
```bash
# Linux
sudo apt-get install -y poppler-utils

# macOS
brew install poppler
```

### `tesseract` not found
```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
```

### `unstructured` ImportError
```bash
pip install "unstructured[all-docs]"
```

---

## API / Auth Issues

### `403 Forbidden` on all endpoints
Ensure `X-API-Key` header matches `RAG_API_MASTER_KEY` in your `.env`. In dev mode with no key set, all requests pass through.

### `503 Service Unavailable` on `/readyz`
Pipeline failed to initialise. Check logs:
```bash
docker compose logs rag-api
# Look for: api_startup_failed
```

---

## Ingestion Issues

### `ingest` returns `num_chunks: 0`
1. Is the PDF text-based (not scanned)? Try: `pdftotext yourfile.pdf -` to check
2. Is `unstructured` installed? `python -c "import unstructured; print('ok')"`
3. Check logs for `unstructured_parse_failed` or `fallback_parse_failed`

### Vision processing fails silently
1. Check `OPENAI_API_KEY` is set: `echo $OPENAI_API_KEY`
2. Check rate limits: look for `429` in logs
3. Disable vision temporarily: `--no-vision` flag

### PDF too large / slow
- Split into chapters before ingesting
- Use `--no-vision` for text-only PDFs
- Increase `BATCH_SIZE` in `.env` if you have more RAM

---

## Query Issues

### Answers are empty / `No context found`
1. Check documents were ingested: `rag-financial health`
2. Verify `VECTOR_STORE_CONFIG__DATASET_PATH` points to same location used during ingest
3. Try `RETRIEVER_CONFIG__STRATEGY=dense` to bypass BM25 for debugging

### Hallucinated numbers in answers
- Guardrails should catch this. Check `guardrails.overall_passed` in response
- If passed but wrong: the number appears in context but was misattributed — add more specific context
- File a bug with the query, answer, and source document

## High Error Rate

> Linked from `RAGAvailabilitySLOFastBurn` / `RAGAvailabilitySLOMediumBurn`
> alerts in `scripts/alerting/slo-burn-rate.yml`. If you were paged here,
> work through these steps in order — they're ordered by likelihood, not
> by severity.

**1. Identify which tenant/model is failing.**
```bash
# Prometheus query — breaks errors down by tenant and model
sum(rate(rag_queries_total{status="error"}[5m])) by (tenant_id, model)
```
A spike concentrated in one tenant usually means a quota/auth issue for
that tenant, not a system-wide outage — check `RAGTenantQuotaNearExhaustion`
firing around the same time.

**2. Check upstream LLM/embedding provider status.**
- OpenAI: https://status.openai.com
- Google (Gemini): https://status.cloud.google.com
- Anthropic: https://status.anthropic.com

If the configured `LLM_CONFIG__PROVIDER` is degraded upstream, errors will
correlate with `429`/`503` in application logs:
```bash
docker compose logs rag-api --since 15m | grep -E "429|503|APIRateLimitError|APITimeoutError"
```

**3. Check for a bad deploy.**
```bash
kubectl rollout history deploy/rag-financial -n rag-prod
```
If the error spike started right after a deploy, roll back:
```bash
kubectl rollout undo deploy/rag-financial -n rag-prod
```

**4. Check Redis (cache + rate limiter) health.**
A Redis outage degrades gracefully for the embedding cache (cache miss, not
an error) but can surface as errors from the rate limiter if it's configured
to fail closed:
```bash
redis-cli -u $CACHE_CONFIG__REDIS_URL ping
```

**5. Check vector store connectivity.**
```bash
rag-financial health
# Look for "vector_store: error: ..." in the output
```

**6. If none of the above explain it: pull a trace.**
Every query is a root OTel span. Find a failing trace in Jaeger
(http://localhost:16686, service=`rag-financial-multimodal`) and inspect
which child span (parse / retrieve / generate) failed and why.

**Mitigations while investigating:**
- Temporarily disable model routing to avoid cascading into the more
  expensive/rate-limited complex model: `LLM_CONFIG__ENABLE_MODEL_ROUTING=false`
- If a single tenant is the source, pause it via the quota endpoint rather
  than degrading service for everyone.

### Very high latency (>10s)
1. Check Redis is running: `redis-cli ping`
2. First query is always slower (model loading) — check p99 after warmup
3. Use `--no-vision` during ingestion if chart extraction is slow
4. Check OpenAI API status: `status.openai.com`

---

## Observability Issues

### Prometheus metrics not showing
1. Check Prometheus port: `curl http://localhost:8001/metrics`
2. Ensure `OBSERVABILITY_CONFIG__PROMETHEUS_PORT=8001` is set
3. Check prometheus.yml scrape config targets

### No traces in Jaeger
1. Set `OBSERVABILITY_CONFIG__OTLP_ENDPOINT=http://localhost:4317`
2. Start OTel collector: `docker compose --profile observability up otel-collector jaeger`

---

## Docker / K8s Issues

### Container OOMKilled
Increase memory limits. For production: 4GB+ RAM recommended.
```yaml
# helm/rag-financial/values.yaml
resources:
  limits:
    memory: "8Gi"
```

### Redis connection refused in container
Check Redis service is healthy:
```bash
docker compose ps redis
docker compose logs redis
```

### Helm chart failing
```bash
helm lint helm/rag-financial/
helm template rag-financial helm/rag-financial/ --debug 2>&1 | head -50
```

---

## Getting Help
- Open a [GitHub issue](.github/ISSUE_TEMPLATE/bug_report.md)
- Check existing issues: `label:bug`
- Join discussions in GitHub Discussions
