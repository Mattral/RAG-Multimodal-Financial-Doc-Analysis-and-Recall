# Tutorial: 10-K Analysis

```python
import asyncio
from src.rag_system.sdk import RAGPipeline

async def main():
    pipeline = await RAGPipeline.create(tenant_id="analysis")
    await pipeline.ingest("tesla_10k_2023.pdf", process_vision=True)
    r = await pipeline.query("What was total revenue in fiscal year 2023?")
    print(r["answer"])
    for src in r["sources"]:
        print(f"  -> {src['document']} p.{src['page']}")

asyncio.run(main())
```

## PoT Calculator
```python
from src.rag_system.components.pot_executor import PoTExecutor
executor = PoTExecutor()
result = await executor.execute_template("cagr", v_initial=8.77, v_final=23.35, n_years=3)
print(f"Revenue CAGR: {result.formatted(1)}%")  # 38.7%
```
