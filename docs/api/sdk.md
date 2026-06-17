# Python SDK

```python
from src.rag_system.sdk import RAGPipeline

# Async
pipeline = await RAGPipeline.create(tenant_id="acme")
await pipeline.ingest(["report.pdf"])
result = await pipeline.query("What was revenue?")

# Sync wrapper
pipeline = RAGPipeline.from_config("config.yaml", tenant_id="acme")
result = pipeline.query_sync("What was revenue?")
```

Return value of `query()`:
```python
{
  "status": "success",
  "answer": "Revenue was $23.35B...",
  "sources": [{"document": "tesla.pdf", "page": 4, "score": 0.92, "text_preview": "..."}],
  "guardrails": {"overall_passed": True, "numeric_grounding_passed": True},
  "metrics": {"total_latency_ms": 1342, "cost_usd": 0.000119, "num_chunks": 5}
}
```
