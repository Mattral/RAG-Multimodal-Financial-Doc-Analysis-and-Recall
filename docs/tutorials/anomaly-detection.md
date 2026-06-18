# Tutorial: Financial Anomaly Detection

Detect statistical anomalies across multiple quarterly filings using the
agentic RAG pipeline with comparative retrieval and PoT calculations.

## Setup
```bash
rag-financial ingest earnings_q1_2022.pdf earnings_q2_2022.pdf \
  earnings_q3_2022.pdf earnings_q4_2022.pdf \
  earnings_q1_2023.pdf earnings_q2_2023.pdf \
  earnings_q3_2023.pdf earnings_q4_2023.pdf \
  --tenant anomaly_detection
```

## Anomaly detection queries

### Margin anomaly
```bash
rag-financial query \
  "Identify any quarters where gross margin deviated more than 3 percentage \
   points from the 8-quarter average. Cite exact page and table." \
  --tenant anomaly_detection --top-k 16
```

## Python SDK with PoT
```python
import asyncio
from src.rag_system.sdk import RAGPipeline
from src.rag_system.components.pot_executor import PoTExecutor

async def detect_margin_anomaly():
    pipeline = await RAGPipeline.create(tenant_id="anomaly_detection")
    result = await pipeline.query(
        "List the exact gross margin percentage for every quarter Q1 2022 to Q4 2023.",
        top_k=16,
    )
    print(result['answer'])

    executor = PoTExecutor()
    pot_result = await executor.execute_code(
        "margins = [26.5, 28.4, 25.1, 23.8, 19.3, 18.2, 17.9, 16.9]\n"
        "avg = sum(margins) / len(margins)\n"
        "variance = sum((m - avg) ** 2 for m in margins) / len(margins)\n"
        "std = variance ** 0.5\n"
        "anomalies = [(i+1, m) for i, m in enumerate(margins) if abs(m - avg) > 2 * std]\n"
        "result = len(anomalies)"
    )
    print(f'Anomaly quarters: {pot_result.result}')

asyncio.run(detect_margin_anomaly())
```

## Agentic mode
```bash
ENABLE_LANGGRAPH_AGENTIC=true \
rag-financial query \
  "Identify and flag all numeric anomalies in revenue and margin data across all ingested quarters." \
  --tenant anomaly_detection --top-k 20
```
