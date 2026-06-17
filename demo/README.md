# Streamlit Demo UI

Simple web interface for the RAG Financial Multimodal pipeline.

## Run locally

```bash
pip install streamlit
streamlit run demo/app.py
```

Opens at http://localhost:8501

## Features
- Upload PDFs and ingest with live progress
- Ask questions with source citations and page numbers
- Query analysis panel (intent, complexity, filters)
- Latency and cost metrics per query
- Thumbs up/down feedback
- Example questions for quick exploration

## Screenshot
The UI shows:
1. Sidebar — API key, tenant ID, file upload, settings
2. Main area — query input, answer, expandable sources, metrics, feedback

## For production
Replace the in-memory vector store with DeepLake or pgvector by setting
environment variables before launching:
```bash
VECTOR_STORE_CONFIG__PROVIDER=deeplake VECTOR_STORE_CONFIG__DATASET_PATH=./data/vectorstore OPENAI_API_KEY=sk-... streamlit run demo/app.py
```
