# Evaluation

```bash
rag-financial evaluate --dataset evals/golden_datasets/financial_qa.jsonl --fail-on-regression
```

| Metric | Threshold |
|---|---|
| Faithfulness | >= 0.70 |
| Answer Relevancy | >= 0.70 |
| Numeric Accuracy | >= 0.70 |
| P99 Latency | < 8000ms |

CI regression gate: if `avg_faithfulness` drops >5% vs last run, PR is blocked.
