# Tutorial: Earnings Call QA

Extract insights from quarterly earnings transcripts and investor Q&A sessions.

## What we'll do
1. Ingest an earnings transcript PDF
2. Ask about guidance, key metrics, and management commentary
3. Use comparative queries across multiple quarters

## Ingest earnings transcripts
```bash
rag-financial ingest \
  earnings_q1_2024.pdf \
  earnings_q2_2024.pdf \
  earnings_q3_2024.pdf \
  --tenant earnings_analysis --no-vision
```

## Query examples
```bash
# Management guidance
rag-financial query "What revenue guidance did management provide for next quarter?" \
  --tenant earnings_analysis --show-sources

# Analyst questions
rag-financial query "What questions did analysts ask about margin pressure?" \
  --tenant earnings_analysis --top-k 10

# Multi-quarter comparison (triggers comparative routing → gpt-4o)
rag-financial query "How has management's tone on supply chain changed across Q1, Q2, Q3?" \
  --tenant earnings_analysis --top-k 15
```

## Python SDK
```python
import asyncio
from src.rag_system.sdk import RAGPipeline

async def main():
    pipeline = await RAGPipeline.create(tenant_id="earnings")
    await pipeline.ingest([
        "earnings_q1_2024.pdf",
        "earnings_q2_2024.pdf",
    ], process_vision=False)  # transcripts are text-only

    # Comparative query — auto-routed to gpt-4o
    r = await pipeline.query(
        "Compare revenue guidance commentary across Q1 and Q2 2024.",
        top_k=12
    )
    print(r["answer"])
    print("Analysis:", r["analysis"])  # shows intent=comparative

asyncio.run(main())
```
