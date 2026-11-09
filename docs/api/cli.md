# CLI Reference

## Installation
```bash
pip install -e ".[all]"
# or via Docker:
docker run rag-financial-multimodal:latest filing-qa --help
```

## Commands

### `filing-qa ingest`
Parse, embed, and index financial PDFs.
```bash
filing-qa ingest reports/*.pdf --tenant acme
filing-qa ingest tesla.pdf --no-vision          # skip chart extraction
filing-qa ingest s3://bucket/10k.pdf --tenant acme
```

### `filing-qa query`
Retrieve and answer questions from indexed documents.
```bash
filing-qa query "What was Q3 revenue?" --tenant acme --show-sources
filing-qa query "Calculate CAGR" --top-k 10 --json
filing-qa query "Compare margins" --tenant acme --verbose
```

### `filing-qa evaluate`
Run quality evaluation against the golden dataset.
```bash
filing-qa evaluate
filing-qa evaluate --dataset evals/golden_datasets/financial_qa.jsonl
filing-qa evaluate --fail-on-regression --output report.json
```

### `filing-qa serve`
Start the FastAPI REST server.
```bash
filing-qa serve
filing-qa serve --host 0.0.0.0 --port 8000 --workers 4
filing-qa serve --reload    # dev mode
```

### `filing-qa health`
Check all component health status.
```bash
filing-qa health
```

### `filing-qa --version`
Show current version.
```bash
filing-qa --version
# RAG Financial Multimodal v2.0.0
```
