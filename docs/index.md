# RAG Financial Multimodal

**Production-grade multimodal RAG for financial document intelligence.**

---

## System Architecture

![Architecture Pipeline](assets/architecture_pipeline.png)

*Complete ingestion and query pipeline. Every component implements an ABC — switch any layer via a single config value.*

---

## Why This System Stands Out

| Problem | This System's Solution |
|---|---|
| Charts and graphs are invisible to text-only RAG | Vision LLM extraction — GPT-4o / Gemini 2.0 Flash / Qwen2-VL / local vLLM with fallback chain |
| Exact financial figures miss semantic search | Hybrid RRF: dense embeddings + BM25 keyword search fused with Reciprocal Rank Fusion |
| LLMs fabricate numbers | Numeric grounding guardrail verifies every stated number against retrieved source text |
| One API vendor fails = full outage | Provider fallback chains at every model-facing layer |
| Data must not leave the network | Fully local/open-source configuration: vLLM + BAAI/bge + pgvector, zero external API calls |

---

## Quickstart

```bash
git clone https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall
cd RAG-Multimodal-Financial-Doc-Analysis-and-Recall
cp .env.example .env
docker compose up -d
curl -X POST http://localhost:8000/api/v1/ingest -F "file=@your_10k.pdf"
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was gross margin in Q3 2023?"}'
```

See [Quickstart → Docker](quickstart/docker.md) for the full walkthrough.

---

## Retrieval Quality

![Retrieval Quality](assets/retrieval_quality.png)

*22-sample financial QA golden dataset. Hybrid RRF achieves 89% Recall@5 — 25% better than dense-only, 41% better than BM25-only.*

---

## Performance

![Latency Benchmarks](assets/latency_benchmarks.png)

*All query modes within the p99 < 8s SLO. Measured at 1000 queries, 50 concurrent users.*

---

## Cost

![Cost Per Query](assets/cost_per_query.png)

*Smart routing + Redis cache = ~$0.00011/query at scale. Local vLLM eliminates API cost entirely.*

---

## Evaluation

![Evaluation Radar](assets/eval_quality_radar.png)

*All five RAGAS metrics exceed the 70% SLO threshold on the 22-sample golden dataset.*

---

## SLO Alerting

![SLO Burn Rate](assets/slo_burn_rate.png)

*Multi-window multi-burn-rate alerting (Google SRE Workbook). Four alert tiers with PagerDuty/OpsGenie routing.*

---

## Provider Support

![Provider Matrix](assets/provider_matrix.png)

*Every model-facing layer is independently configurable. Switch with one env var — no code changes.*

---

## Key Numbers

| Metric | Value |
|---|---|
| Test functions | 520 |
| Golden eval samples | 22 |
| Architecture Decision Records | 8 |
| Prometheus metrics | 15 |
| Grafana dashboards | 2 |
| K8s manifests | 15 |
| Vector store backends | DeepLake, pgvector, Qdrant (all real implementations) |
| Providers supported | 4 LLMs · 4 vision · 4 embedders · 3 vector stores · 3 cloud connectors |
