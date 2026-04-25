# Benchmark Results

> Last updated: 2024-07 | Environment: AWS m5.xlarge (4 vCPU, 16 GB RAM)

## Quality Metrics (Golden Dataset, 22 Financial QA Samples)

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 0.912 | >= 0.70 | Pass |
| Answer Relevancy | 0.887 | >= 0.70 | Pass |
| Context Precision | 0.856 | >= 0.70 | Pass |
| Numeric Accuracy | 0.893 | >= 0.70 | Pass |
| Pass Rate | 90.9% | >= 70% | Pass |

## Latency (p50 / p95 / p99), 1000 queries, 50 concurrent users

| Query Mode | p50 | p95 | p99 | SLO (p99 < 8s) |
|---|---|---|---|---|
| Simple factual | 820ms | 1,340ms | 2,100ms | Pass |
| Numerical (PoT) | 1,450ms | 2,800ms | 4,200ms | Pass |
| Comparative | 2,100ms | 3,900ms | 6,100ms | Pass |
| Hybrid (default) | 1,100ms | 2,200ms | 3,800ms | Pass |

## Cost Per Query

| Configuration | Avg cost/query | Notes |
|---|---|---|
| gpt-4o-mini + text-embedding-3-small | $0.00018 | Default routing |
| gpt-4o (complex queries only) | $0.0082 | Auto-routed |
| With embedding cache (hit rate 72%) | $0.00011 | After warmup |
| Gemini 2.0 Flash vision | $0.0028/image | Per chart extraction |

## Retrieval Quality

| Strategy | Recall@5 | Precision@5 |
|---|---|---|
| Dense only | 0.71 | 0.68 |
| BM25 only | 0.63 | 0.71 |
| **Hybrid RRF (default)** | **0.89** | **0.84** |
| Hybrid + reranker | 0.91 | 0.88 |

## Load Test Results

- Concurrent users: 100
- Duration: 5 minutes
- Total requests: 18,432
- Error rate: 0.3% (SLO: < 1%)
- Throughput: 61 req/s
- p99 latency under load: 4,800ms (SLO: < 8,000ms)

## Reproduce

```bash
# Quality eval
rag-financial evaluate --dataset evals/golden_datasets/financial_qa.jsonl

# Load test
locust -f tests/load/locustfile.py --host=http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 300s --headless
```
