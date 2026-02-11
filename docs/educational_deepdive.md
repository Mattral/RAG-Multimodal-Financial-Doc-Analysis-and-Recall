# RAG-Multimodal-Financial-Document-Analysis-and-Recall: Educational Deep Dive

## 🔍 Context & Motivation

Large financial documents such as earnings reports, investor presentations, and regulatory filings are inherently **multimodal**. 
Critical information is distributed across **narrative text, tables, and visual elements such as charts and graphs**. 

While Retrieval-Augmented Generation (RAG) has become a standard approach for grounding large language models in external documents, **most RAG pipelines operate on text alone**. In practice, this leads to incomplete or misleading answers when key insights are encoded visually—for example, trends, growth patterns, or anomalies shown only in charts.

This repository explores a **practical multimodal RAG pipeline** for financial document analysis, where **visual information is explicitly extracted, described, and incorporated into retrieval** alongside textual content.

## 🎯 What This Repository Demonstrates

This project demonstrates an end-to-end workflow for:

- Parsing financial PDF documents into structured components (text, tables, and figures)
- Using a vision-capable LLM to **describe charts and visual trends**
- Storing both textual and visual-derived representations in a vector database
- Enabling a RAG-based chatbot to answer questions that require **visual grounding**, not just text matching

The core hypothesis explored here is that **augmenting retrieval with graph and chart descriptions materially improves answer quality** for financial queries that depend on trends or comparative patterns.

## 🧭 Scope & Design Philosophy

This repository was originally designed as a **clear, inspectable reference implementation**, prioritizing:
- Transparency over architectural complexity
- Readability over optimization
- Practical reproducibility over exhaustive evaluation

## Component Breakdown

### 1. Text/Tables Extraction (Unstructured.io)

Uses the `partition_pdf` function to extract and chunk text and table data:

```python
from unstructured.partition.pdf import partition_pdf

raw_pdf_elements = partition_pdf(
    filename="./TSLA-Q3-2023-Update-3.pdf",
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000
)
```

### 2. Vision Processing (GPT-4V)

Converts PDF pages to images and uses GPT-4V to describe charts:

```python
from pdf2image import convert_from_path

convertor = convert_from_path('./TSLA-Q3-2023-Update-3.pdf')
for idx, image in enumerate(convertor):
    image.save(f"./pages/page-{idx}.png")
```

### 3. Vector Storage (DeepLake)

Stores documents with embeddings in DeepLake:

```python
from llama_index.vector_stores import DeepLakeVectorStore

vector_store = DeepLakeVectorStore(
    dataset_path="hub://genai360/tesla_quarterly_2023",
    runtime={"tensor_db": True},
    overwrite=False
)
```

### 4. Query Engine (LlamaIndex)

Provides RAG-based querying with optional Deep Memory:

```python
query_engine = index.as_query_engine(vector_store_kwargs={"deep_memory": True})
response = query_engine.query("What are the trends in vehicle deliveries?")
```

## Key Insights

- **Multimodal Grounding:** Including chart descriptions in retrieval improves answer quality for trend-dependent questions
- **Deep Memory Impact:** Activeloop's Deep Memory feature achieved ~25% performance improvement in recall@10
- **Cost Considerations:** Processing all pages for vision analysis is expensive; selective flagging of pages with graphs reduces costs

## Limitations & Intended Use

This project should be viewed as:
- A **demonstration** of multimodal RAG concepts
- A **starting point** for further experimentation or system design

It is not intended to:
- Serve as a comprehensive benchmark
- Claim state-of-the-art performance
- Replace domain-specific financial analysis tools
