# Golden financial QA dataset

`financial_qa.jsonl` holds **119 question/answer pairs** about 2023 fiscal-year results of 10
companies, written against 16 public primary documents (SEC filings, earnings releases and
investor decks). Every figure in a ground-truth answer is backed by an evidence string recorded
in `provenance.json`. That string can be found verbatim in the source document, and every derived
number (growth rate, ratio, sum) has an arithmetic check.

| | Count |
|---|---|
| Samples | 119 (22 original, corrected; 97 added in v2) |
| Companies | Tesla 40, JPMorgan 18, NVIDIA 17, Apple 15, Microsoft 15, Alphabet 3, Amazon 3, Meta 3, Berkshire Hathaway 3, Goldman Sachs 2 |
| Source documents | 16 |
| Samples with an `expected_page` | 105 |
| Evidence strings | 209 |
| Recorded calculations | 42 |

## Format

One JSON object per line, loaded into `EvalSample` (`src/rag_system/components/evaluator`):

```json
{
  "question": "Calculate Tesla's full-year 2023 free cash flow from operating cash flow and capital expenditures.",
  "ground_truth": "Tesla's 2023 operating cash flow was $13,256 million and capital expenditures were $8,898 million, so free cash flow was $4,358 million (about $4.4 billion), down 42% from $7,566 million in 2022.",
  "source_documents": ["tesla_q4_2023_update.pdf"],
  "expected_page": 5,
  "expected_numeric_values": ["13.26 billion", "8.9 billion", "4.36 billion", "42%"],
  "tags": ["calculation", "table", "cash_flow", "fcf", "annual", "tesla"]
}
```

* **`tags[0]` is the question type**: `factual`, `calculation`, `table`, `comparative` or `chart`.
  The other tags are topical and may repeat a type (a calculation over a table also carries `table`).
* **`expected_numeric_values`**: every value listed appears in `ground_truth`. A unit test enforces this.
  **The first value is the figure the question asks for** (the *headline*). The rest are
  supporting figures such as prior periods, inputs or growth rates. The runner reports
  *expected-value recall* over all of them, and `evals/answer_eval.py` also reports whether
  the headline was stated. In v2.1, 16 samples were reordered to follow this rule. No values
  changed.
* **`source_documents`**: canonical file names. The *source hit rate* metric only works if you
  ingest the documents under these names.
* **`expected_page`**: the page printed in the filing's footer for SEC 10-K/10-Q documents.
  For PDFs (Tesla decks, Apple statements) it is the 1-based PDF page, and for the JPMorgan
  supplement it is the "Page N" label. It is `null` for unpaginated web press releases and for
  Alphabet's 10-Q, whose HTML has no printed page numbers.

### Question-type mix

| Primary type (`tags[0]`) | All 119 | New 97 | Any tag |
|---|---|---|---|
| factual | 56 | 44 | 56 |
| calculation | 24 | 21 | 25 |
| comparative (across periods, segments or companies) | 17 | 15 | 25 |
| table | 13 | 8 | 44 |
| chart | 9 | 9 | 9 |

Two questions span two companies (`multi_doc`). Five are qualitative (risk factors, drivers,
outlook).

## Source documents

| `source_documents` name | Document |
|---|---|
| `tesla_q3_2023_update.pdf` | [Tesla Q3 2023 Update deck](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q3-2023-Update-3.pdf) |
| `tesla_q4_2023_update.pdf` | [Tesla Q4 2023 Update deck](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2023-Update.pdf) |
| `tesla_10q_q3_2023.pdf` | [Tesla 10-Q, Q3 2023](https://www.sec.gov/Archives/edgar/data/1318605/000162828023034847/tsla-20230930.htm) |
| `tesla_10k_2023.pdf` | [Tesla 10-K, FY2023](https://www.sec.gov/Archives/edgar/data/1318605/000162828024002390/tsla-20231231.htm) |
| `apple_10k_fy2023.pdf` | [Apple 10-K, FY2023](https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm) |
| `apple_fy23_q4_statements.pdf` | [Apple Q4 FY2023 condensed statements](https://www.apple.com/newsroom/pdfs/fy2023-q4/FY23_Q4_Consolidated_Financial_Statements.pdf) |
| `msft_10k_fy2023.pdf` | [Microsoft 10-K, FY2023](https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm) |
| `nvidia_10k_fy2024.pdf` | [NVIDIA 10-K, FY2024](https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm) |
| `nvidia_q2_fy2024_press_release.pdf` | [NVIDIA Q2 FY2024 results release](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2024) |
| `jpmorgan_3q23_earnings_release.pdf` | [JPMorgan 3Q23 earnings release (8-K Ex. 99.1)](https://www.sec.gov/Archives/edgar/data/19617/000001961723000508/a3q23erfexhibit991narrative.htm) |
| `jpmorgan_3q23_financial_supplement.pdf` | [JPMorgan 3Q23 financial supplement (8-K Ex. 99.2)](https://www.sec.gov/Archives/edgar/data/19617/000001961723000508/a3q23erfex992supplement.htm) |
| `amazon_10q_q3_2023.pdf` | [Amazon 10-Q, Q3 2023](https://www.sec.gov/Archives/edgar/data/1018724/000101872423000018/amzn-20230930.htm) |
| `alphabet_10q_q3_2023.pdf` | [Alphabet 10-Q, Q3 2023](https://www.sec.gov/Archives/edgar/data/1652044/000165204423000094/goog-20230930.htm) |
| `meta_q2_2023_press_release.pdf` | [Meta Q2 2023 results release](https://investor.atmeta.com/investor-news/press-release-details/2023/Meta-Reports-Second-Quarter-2023-Results/default.aspx) |
| `goldman_4q23_earnings_release.pdf` | [Goldman Sachs 4Q23 / FY2023 release (8-K Ex. 99.1)](https://www.sec.gov/Archives/edgar/data/886982/000119312524008077/d615098dex991.htm) |
| `berkshire_2023_earnings_release.pdf` | [Berkshire Hathaway FY2023 release (8-K Ex. 99.1)](https://www.sec.gov/Archives/edgar/data/1067983/000119312524045977/d795719dex991.htm) |

The SEC documents are published as HTML. Save or print them to PDF under the names above if
you ingest them through the PDF pipeline.

## How it is verified

* `tests/unit/test_golden_dataset.py` runs offline in CI. It checks the schema (every row loads
  into `EvalSample`), unique questions, the type mix, that each ground truth states all of its
  expected values, that provenance covers every sample, and that all 42 calculations reproduce.
* `scripts/verify_golden_dataset.py` (needs network) downloads all 16 documents, extracts their
  text and confirms each evidence string is present. Last run: **209/209 found**.

```bash
pytest tests/unit/test_golden_dataset.py
RAG_EVAL_USER_AGENT="Your Name you@example.com" python scripts/verify_golden_dataset.py
```

## Corrections to the original 22 samples (v2)

The original samples were rechecked against the filings. Nine contained wrong figures or claims:

| Sample | Was | Source says |
|---|---|---|
| Tesla Q3 2023 free cash flow | $4.4B (OCF $5.1B − capex $0.7B) | **$848M** (OCF $3,308M − capex $2,460M). $4.4B is full-year 2023 FCF |
| Microsoft FY23 net income | $72.4B, **up 20%** from $61.3B | $72.4B, **down 1%** from $72.7B ($61.3B was FY21) |
| Microsoft cloud growth | Intelligent Cloud grew **19%** | Intelligent Cloud grew **17%**. 19% is Server products & cloud services |
| NVIDIA Q2 FY24 revenue | up **122%** YoY | up **101%** YoY ($13,507M vs $6,704M) |
| Amazon AWS Q3 2023 | **$23.4B** | **$23.06B** ($23,059M) |
| Goldman Sachs 2023 net revenues | down **1%** from $47.4B | down **2%** from $47.37B |
| Apple FY23 total debt | **$109.3B** term debt | **$105.1B** term debt + $6.0B commercial paper = $111.1B |
| Berkshire 2023 operating earnings | record "driven by … BNSF railroad results" | Railroad earnings **fell** ($5.95B → $5.09B); growth came from insurance |
| Tesla 2023 margin drivers | included "increased raw material costs" | 10-K cites lower ASP, partly offset by **lower** unit cost and IRA credits |

Several source-document names were also changed to the document actually used for checking
(for example `nvidia_10q_fq2_2024.pdf` became `nvidia_q2_fy2024_press_release.pdf`), and page
numbers were recomputed.

## Limitations

* **Single author, no second annotator.** The questions were written by one person reading the
  documents, so their wording overlaps the source text more than real analyst queries would.
  That overlap favors keyword (BM25) retrieval.
* **Narrow time window.** Coverage is fiscal 2023, with Tesla over-represented (40 of 119). The
  set says little about other sectors, other years or multi-year trend questions.
* **Chart questions are limited.** The 9 chart questions use charts in Tesla's decks whose
  values also appear in the decks' tables. The ground truth was checked against those tables, not
  read off the chart image, so a text-only system can answer some of them without vision.
* **Rounding.** Ground truths quote both exact table values and rounded forms. Expected-value
  recall allows a 1% relative tolerance, which can accept a slightly wrong rounding.
* **No numbers from generation runs ship with this dataset.** See
  [BENCHMARK_RESULTS.md](../../BENCHMARK_RESULTS.md) for what has and has not been measured.
