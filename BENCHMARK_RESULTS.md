# Benchmark Results

> Last updated: 2026-10-13. Every number in sections 1–3 was produced by a script in this
> repository, and the raw outputs are in [`evals/results/`](evals/results/). Section 4 lists
> earlier figures that could not be reproduced.

## What has and has not been measured

| Area | Status | Where |
|---|---|---|
| Golden dataset integrity (119 samples) | **Measured.** 209/209 evidence strings found in the source filings; 42/42 calculations reproduce | Section 1 |
| Retrieval quality on the real source documents | **Measured** offline (no API key needed) | Section 2 |
| Answer quality with an LLM | **Pilot only**: 11 of 119 questions before free-tier quota ran out | Section 3 |
| RAGAS faithfulness / answer relevancy / LLM-judged numeric accuracy | **Not measured.** The judge in `RagasEvaluator` currently calls OpenAI only | — |
| Latency under load, throughput, cost per query | **Not reproduced** | Section 4 |

## 1. Golden dataset

`evals/golden_datasets/financial_qa.jsonl` holds 119 QA pairs (22 original, corrected; 97 new)
over 16 public filings and releases from Tesla, Apple, Microsoft, NVIDIA, JPMorgan, Amazon,
Alphabet, Meta, Goldman Sachs and Berkshire Hathaway. By question type: 56 factual,
24 calculation, 17 comparative, 13 table and 9 chart. Each sample's supporting text and
arithmetic are recorded in `provenance.json`. Re-checking the original 22 samples found errors
in 9 of them. See the [dataset README](evals/golden_datasets/README.md).

## 2. Retrieval (offline, real source documents)

`python -m evals.retrieval_benchmark` indexes all 16 documents (2.2M characters) and runs each
golden question through the repository's own `BM25Index`, `LocalEmbedder`
(`BAAI/bge-small-en-v1.5`) and `_reciprocal_rank_fusion`, using the `RetrieverConfig` defaults
(top-20 dense, top-20 BM25, `rrf_k=60`). It applies no reranker.

* **evidence hit@5**: a top-5 chunk contains the exact supporting text for the answer. This is
  strict, because a chunk can contain the answer in different words and still miss.
* **doc hit@5**: a top-5 chunk comes from an expected source document.
* **doc precision@5**: share of the top 5 chunks that come from an expected document.
* **MRR@10**: mean reciprocal rank of the first answer-bearing chunk.

**4,000-character chunks** (the parser's default `max_characters`; 601 chunks):

| Strategy | Evidence hit@5 | Doc hit@5 | Doc precision@5 | MRR@10 |
|---|---|---|---|---|
| BM25 only | 0.706 | 0.874 | 0.566 | 0.502 |
| Dense only | 0.605 | 0.992 | 0.531 | 0.384 |
| **Hybrid RRF (default)** | **0.706** | 0.933 | 0.570 | 0.468 |

**1,000-character chunks** (3,166 chunks):

| Strategy | Evidence hit@5 | Doc hit@5 | Doc precision@5 | MRR@10 |
|---|---|---|---|---|
| BM25 only | 0.420 | 0.824 | 0.534 | 0.269 |
| Dense only | 0.462 | 0.908 | 0.548 | 0.293 |
| **Hybrid RRF (default)** | **0.529** | 0.924 | 0.570 | 0.304 |

Hybrid evidence hit@5 by question type:

| Chunk size | Factual (56) | Calculation (24) | Comparative (17) | Table (13) | Chart (9) |
|---|---|---|---|---|---|
| 4,000 | 0.68 | 0.71 | 0.65 | 0.85 | 0.78 |
| 1,000 | 0.52 | 0.50 | 0.47 | 0.62 | 0.67 |

What this shows:

* With small chunks, hybrid fusion helps: 0.53 versus 0.42 for BM25 and 0.46 for dense.
* With the default 4,000-character chunks, hybrid **ties** BM25 (0.706) and does not beat it.
  `bge-small` only reads its first 512 tokens, about half of a 4,000-character chunk.
* Dense search almost always finds the right *document* (0.99) but often not the right *passage*.
* About 3 in 10 questions have no answer-bearing chunk in the top 5 even in the best setting.
  Retrieval, not generation, is the first bottleneck.
* Earlier versions of this file reported hybrid Recall@5 = 0.89 and Precision@5 = 0.84. Those
  figures are not reproducible (see section 4) and should not be compared with the ones above.

Raw output: [`retrieval_chunk4000.json`](evals/results/retrieval_chunk4000.json),
[`retrieval_chunk1000.json`](evals/results/retrieval_chunk1000.json).

## 3. Answer quality: pilot with `gemini-3.8-flash` (n = 11)

`python -m evals.answer_eval` retrieves the top-5 hybrid chunks (1,000-character setting),
generates an answer with the configured generator (`GeminiGenerator`, `gemini-3.8-flash`,
model routing off), and scores the answer text with no LLM judge. The run stopped after **11 of
119** questions when the Gemini free tier started returning sustained 429s and timeouts. These
are the first 11 questions in the file, mostly Tesla and single-value factual or table lookups.
**They are not representative and should not be quoted as system performance.**

| Metric (11 answers) | Value |
|---|---|
| Headline value stated (first expected value appears in the answer) | 8 / 10 |
| All expected values stated | 1 / 10 |
| Answer cites an expected source document | 8 / 11 |
| Abstained ("not available in the provided documents") | 2 / 11 |
| Answer-bearing chunk was in the context | 6 / 11 |
| Latency p50 / p95 (free tier, includes model thinking time) | 20.3 s / 48.0 s |

Observations from reading all 11 answers:

* No answer contained a wrong figure.
* Both abstentions happened when retrieval missed the supporting text. One was the 2023
  deliveries question, where a 1,000-character chunk split Tesla's annual table.
* **4 of 11 answers were cut off mid-sentence** even though each used fewer than 100 output
  tokens. `GeminiGenerator` sends `maxOutputTokens=2048`, and on thinking models that budget
  probably includes thinking tokens. This needs a fix (a separate thinking budget or a larger
  limit) before a full run.
* **Cited page numbers are invented** ("Page 1", "Page 20"). Benchmark chunks carry no page
  metadata, so the prompt's instruction to cite pages produces pages the context never gave.
* Most answers gave the headline figure only and left out the comparison figures, which is why
  "all expected values" is low. Truncation contributes too.

To finish the run (it resumes where it stopped and never repeats a request):

```bash
LLM_CONFIG__MODEL=gemini-3.8-flash LLM_CONFIG__ENABLE_MODEL_ROUTING=false \
  python -m evals.answer_eval --rpm 5 --answers evals/results/answers_gemini-3.8-flash.jsonl \
  --output evals/results/answer_eval_gemini-3.8-flash.json
```

Raw output: [`answers_gemini-3.8-flash.jsonl`](evals/results/answers_gemini-3.8-flash.jsonl),
[`answer_eval_gemini-3.8-flash_pilot.json`](evals/results/answer_eval_gemini-3.8-flash_pilot.json).

## 4. Previously published figures (not reproduced)

Earlier versions of this file and the README reported the figures below, dated "2024-07" on
an AWS m5.xlarge:

* Quality: faithfulness 0.912, answer relevancy 0.887, context precision 0.856, numeric
  accuracy 0.893, pass rate 90.9%.
* Retrieval: hybrid Recall@5 0.89, Precision@5 0.84, plus dense, BM25 and reranker rows.
* Latency: p99 2.1–6.1 s by query mode.
* Load: 61 req/s at 100 users, 0.3% errors.
* Cost: $0.00011–$0.0082 per query.

They have been withdrawn because nothing in this repository can reproduce them:

* No code computed Recall@k or Precision@k, and no run outputs were committed.
* The evaluation runner indexed its report as a dict (`report["run_id"]`) while
  `GoldenDatasetRunner.run()` returns a dataclass, so `python -m evals.run_evals` and
  `rag-financial evaluate` raised `TypeError` before printing results. That bug is fixed now.
* 9 of the 22 golden samples the figures were computed on had wrong ground truths.

`tests/load/locustfile.py` and `scripts/benchmark.py` can measure latency and cost against a
running deployment. Those numbers will be added here once they are actually run.

## Reproduce

```bash
pip install sentence-transformers pypdf
pytest tests/unit/test_golden_dataset.py                       # dataset integrity (offline)
RAG_EVAL_USER_AGENT="Your Name you@example.com" \
  python scripts/verify_golden_dataset.py                      # evidence vs. primary sources
python -m evals.retrieval_benchmark --chunk-size 4000          # section 2
python -m evals.retrieval_benchmark --chunk-size 1000
python -m evals.answer_eval --rpm 5                            # section 3 (needs an LLM key)
```

## Limitations of these measurements

* **One author's questions.** The questions overlap the filings' wording, which favors BM25 and
  probably flatters retrieval compared with real analyst queries.
* **Not the production ingest path.** The benchmarks chunk plain text extracted with
  `pypdf`/HTML stripping into fixed windows. Production ingest uses unstructured.io
  title-based chunking, vision chart extraction and a vector database, and none of those are
  exercised here. Results may differ in either direction.
* **No reranker** in the retrieval numbers. The cross-encoder reranker that is on by default in
  `RetrieverConfig` would change them.
* **Strict evidence matching.** Evidence hit@k needs the exact recorded text, so it
  undercounts chunks that state the answer differently.
* **The answer eval has no LLM judge.** Expected-value recall checks that numbers appear, not
  that the reasoning is right. The 11-answer pilot is too small to support any conclusion
  about answer quality.
