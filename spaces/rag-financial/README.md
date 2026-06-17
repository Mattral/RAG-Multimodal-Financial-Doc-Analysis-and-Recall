---
title: Multimodal Financial RAG
emoji: 🏦
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: Production RAG for financial documents — charts, hybrid retrieval, numeric guardrails
tags:
  - finance
  - rag
  - nlp
  - document-question-answering
  - gradio
  - multimodal
  - openai
  - gemini
---

# 🏦 Multimodal Financial RAG

**Production-grade document intelligence** — chart understanding · hybrid RRF retrieval · numeric guardrails · source citations

This Space is a faithful demo of the enterprise system at
[github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall](https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall).

## Features

- **👁️ Vision chart extraction** — GPT-4o / Gemini 2.5 Flash describe charts and graphs into searchable text
- **⚡ Hybrid RRF retrieval** — dense (BAAI/bge-small-en-v1.5) + BM25 fused with Reciprocal Rank Fusion (k=60)
- **🔢 Numeric grounding** — every number in the answer cross-checked against source context
- **🔒 PII + injection protection** — SSNs, IBANs, CUSIPs redacted; prompt injection patterns blocked
- **📍 Page-level citations** — every claim attributed to a specific document and page
- **🔬 Full pipeline transparency** — see every retrieval score, RRF weight, generation cost

## Models (v2.0 — current)

| Provider | Text generation | Vision |
|---|---|---|
| Google Gemini | gemini-2.5-flash (default), gemini-2.5-pro | gemini-2.5-flash |
| OpenAI | gpt-4o-mini, gpt-4o | gpt-4o |

## How to Use

1. Enter your API key (Gemini free tier at [aistudio.google.com](https://aistudio.google.com))
2. Upload a 10-K, 10-Q, or earnings release PDF
3. Click **Process Document**
4. Ask a question

## Architecture

```
PDF → pdfplumber (text + tables) + Vision LLM (charts)
    → Semantic chunking (≤800 chars, 100-char overlap, paragraph-boundary split)
    → BAAI/bge-small-en-v1.5 embeddings → FAISS IndexFlatIP
    → BM25Okapi keyword index
    → RRF fusion: 0.7/(60+dense_rank+1) + 0.3/(60+bm25_rank+1)
    → Top-k chunks → GPT-4o-mini / Gemini 2.5 Flash generation
    → Guardrails: injection check → PII redaction → numeric grounding
    → Grounded answer + page-level citations
```

Mirrors the production pipeline in
[`src/rag_system/`](https://github.com/Mattral/RAG-Multimodal-Financial-Doc-Analysis-and-Recall/tree/main/src/rag_system).

## Privacy

API keys are never stored. Uploaded PDFs are processed in-memory and not persisted.

## License

MIT
