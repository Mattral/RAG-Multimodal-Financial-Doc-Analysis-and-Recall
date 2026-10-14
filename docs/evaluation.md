# Evaluation

## Golden dataset

`evals/golden_datasets/financial_qa.jsonl` contains 119 QA pairs over 16 public filings
(factual, calculation, comparative, table and chart questions). Every figure is traced to a
quoted passage in `provenance.json`. Format, sources and limitations are described in
[`evals/golden_datasets/README.md`](https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall/blob/main/evals/golden_datasets/README.md).

```bash
pytest tests/unit/test_golden_dataset.py                 # schema, consistency, calculations
python scripts/verify_golden_dataset.py                  # evidence vs. primary sources (network)
```

## Full pipeline eval (RAGAS + LLM judge)

```bash
rag-financial evaluate --dataset evals/golden_datasets/financial_qa.jsonl --fail-on-regression
```

| Metric | Threshold |
|---|---|
| Faithfulness | >= 0.70 |
| Answer Relevancy | >= 0.70 |
| Numeric Accuracy | >= 0.70 |
| P99 Latency | < 8000ms |

The report also includes two deterministic metrics that need no LLM:

* **Expected-value recall:** share of `expected_numeric_values` stated in the answer, with units
  normalised so "$26,077 million" matches "26.1 billion".
* **Source hit rate:** share of answers citing a chunk from one of `source_documents`. This
  needs the documents to be ingested under the dataset's file names.

The RAGAS and numeric-judge scorers currently call OpenAI.

CI regression gate: if `avg_faithfulness` drops >5% vs last run, PR is blocked.

## Offline benchmarks

These run against the real source documents and do not need the ingestion stack:

```bash
python -m evals.retrieval_benchmark --chunk-size 4000    # BM25 / dense / hybrid RRF, no API key
python -m evals.answer_eval --rpm 5                       # + the configured generator; resumable
```

Results and caveats are in [BENCHMARK_RESULTS.md](https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall/blob/main/BENCHMARK_RESULTS.md).
