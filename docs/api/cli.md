# CLI Reference

## Installation
```bash
pip install -e ".[all]"
# or via Docker:
docker run rag-financial-multimodal:latest rag-financial --help
```

## Commands

### `rag-financial ingest`
Parse, embed, and index financial PDFs.
```bash
rag-financial ingest reports/*.pdf --tenant acme
rag-financial ingest tesla.pdf --no-vision          # skip chart extraction
rag-financial ingest s3://bucket/10k.pdf --tenant acme
```

### `rag-financial query`
Retrieve and answer questions from indexed documents.
```bash
rag-financial query "What was Q3 revenue?" --tenant acme --show-sources
rag-financial query "Calculate CAGR" --top-k 10 --json
rag-financial query "Compare margins" --tenant acme --verbose
```

### `rag-financial evaluate`
Run quality evaluation against the golden dataset.
```bash
rag-financial evaluate
rag-financial evaluate --dataset evals/golden_datasets/financial_qa.jsonl
rag-financial evaluate --fail-on-regression --output report.json
```

### `rag-financial serve`
Start the FastAPI REST server.
```bash
rag-financial serve
rag-financial serve --host 0.0.0.0 --port 8000 --workers 4
rag-financial serve --reload    # dev mode
```

### `rag-financial health`
Check all component health status.
```bash
rag-financial health
```

### `rag-financial --version`
Show current version.
```bash
rag-financial --version
# RAG Financial Multimodal v2.0.0
```
