# RAG-Multimodal-Financial-Document-Analysis: Production System

> **Enterprise-grade multimodal Retrieval-Augmented Generation (RAG) system for financial document analysis with async processing, fault tolerance, structured observability, and scalable architecture.**

## Overview

This is a **top-tier production implementation** of a multimodal RAG pipeline for financial document analysis. It transforms the educational concept of combining textual and visual (chart/graph) information from PDFs into a highly available, observable, and resilient enterprise system.

**Key Differentiators:**
- ✅ **Fully Asynchronous:** Non-blocking async/await with structured concurrency
- ✅ **Fault Tolerant:** Exponential backoff retries, rate limiting, and jitter
- ✅ **Observable:** Structured JSON logging with trace IDs and span IDs
- ✅ **Strongly Typed:** 100% strict type hints compatible with `mypy`
- ✅ **Configurable:** Pydantic-based centralized configuration
- ✅ **Tested:** Comprehensive unit and integration tests with CI/CD
- ✅ **Evaluated:** Automated evaluation framework for quality metrics

## Quick Start

```bash
# 1. Install system dependencies and Python venv
bash setup.sh
source .venv/bin/activate

# 2. Set API keys
export OPENAI_API_KEY="sk-..."
export ACTIVELOOP_TOKEN="hub-..."

# 3. Run evaluation
python evals/run_evals.py

# 4. Use in your code (see Usage Examples below)
```

---

## System Architecture

### High-Level Flow

```
PDFs → PDF Parser → [Text Elements]
        ↓
      [Images] → Vision Processor (GPT-4V) → [Graph Descriptions]
        ↓
      All Elements → Vector Indexer (DeepLake) → Indexed Dataset
        ↓
      Query Engine → LLM Response
```

### Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Async Runtime** | asyncio, httpx | Non-blocking I/O |
| **Config** | Pydantic v2 | Validated configuration |
| **PDF Processing** | unstructured.io | Text/table extraction |
| **Vision** | OpenAI GPT-4V | Chart description |
| **Vector Storage** | DeepLake | Embeddings & retrieval |
| **LLM Integration** | LlamaIndex | RAG query engine |
| **Logging** | structlog, JSON | Structured observability |
| **Testing** | pytest, pytest-asyncio | Quality assurance |
| **CI/CD** | GitHub Actions | Automated checks |

---

## Key Features

### 1. Asynchronous Processing
- **Non-blocking I/O** for PDF parsing, API calls, vector operations
- **Concurrent execution** without thread overhead
- **Throughput:** 10-20 documents/minute (configurable)

### 2. Resilience & Fault Tolerance

**Retry Strategy:**
```
Attempt 1 → Wait 1s (±10%)
Attempt 2 → Wait 2s (±10%)
Attempt 3 → Wait 4s (±10%)
Attempt 4 → Wait 8s (±10%)
Attempt 5 → FAIL
```

**Automatic Retries On:**
- HTTP 5xx errors
- Timeout/connection errors
- Rate limits (HTTP 429) with `Retry-After` header

### 3. Structured Logging & Observability
Every log includes:
- `timestamp`: UTC ISO8601
- `level`: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `trace_id`: Correlates requests end-to-end
- `span_id`: Identifies operation within request
- `memory_mb`: Process memory consumption
- `module`, `function`, `line`: Code location
- Custom context fields

### 4. Rate Limiting
- **Token Bucket algorithm** with configurable burst size
- **Default:** 10 requests/second, 20 burst capacity
- **Backpressure:** Automatically waits if quota exceeded
- **Jitter:** Random delays to prevent thundering herd

### 5. Type Safety
- **100% Strict Type Hints:** Full `mypy --strict` compliance
- **Pydantic Validation:** All inputs validated at boundaries
- **IDE Autocomplete:** Full IntelliSense support
- **Error Tracking:** Type mismatches caught at development time

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Ubuntu/Debian or macOS
- OpenAI API key (`OPENAI_API_KEY`)
- Activeloop token (optional, for DeepLake: `ACTIVELOOP_TOKEN`)

###  1-Minute Setup

```bash
# Clone repository
git clone <repo-url>
cd RAG-Multimodal-Financial-Document-Analysis-and-Recall

# Run setup script (installs system deps + Python venv)
bash setup.sh

# Activate venv
source .venv/bin/activate

# Set credentials
export OPENAI_API_KEY="sk-your-key-here"
export ACTIVELOOP_TOKEN="hub-your-token-here"  # optional
```

### Manual Setup (if setup.sh fails)

```bash
# Install system packages
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr

# Create Python venv
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt  # or: poetry install
```

### Verify Installation

```bash
python -c "from src.rag_system.config import get_config; print('✅ Setup complete')"
```

---

## Usage Examples

### Basic: Ingest PDFs and Query

```python
import asyncio
from src.rag_system.pipeline import create_pipeline

async def main():
    # Initialize pipeline
    pipeline = await create_pipeline()
    
    # Ingest documents
    result = await pipeline.ingest_documents(
        pdf_paths=["tesla_q1.pdf", "tesla_q2.pdf"],
        process_vision=True,  # Extract charts with GPT-4V
    )
    
    print(f"Ingested {result['total_elements_processed']} elements")
    
    # Query
    response = await pipeline.query(
        "What are the trends in vehicle deliveries?",
        top_k=5,
        use_deep_memory=True,
    )
    
    print(response['results'])

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced: Custom Configuration

```python
import os
from src.rag_system.config import Config, VisionConfig, RateLimitConfig

# Set environment variables
os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['ENVIRONMENT'] = 'production'
os.environ['DEBUG_MODE'] = 'false'

# Get validated config
config = Config()

print(f"Environment: {config.environment}")
print(f"Vision Model: {config.vision_config.model}")
print(f"Rate Limit: {config.rate_limit_config.requests_per_second} req/s")
```

### Advanced: Custom Retry Policy

```python
from src.rag_system.utils.retry_policy import RetryPolicy

# Custom policy for aggressive retrying
retry_policy = RetryPolicy(
    max_attempts=5,
    base_delay_seconds=2.0,
    max_delay_seconds=300.0,
    backoff_factor=2.0,
    jitter_factor=0.15,
)

# Use with async functions
result = await retry_policy.execute_async(
    my_async_function,
    arg1, arg2,
    kwarg1=value,
)
```

### Advanced: Structured Logging with Context

```python
from src.rag_system.utils.logger import setup_logging, get_logger

# Initialize logging
setup_logging(level="INFO", format_type="json")

# Create logger with tracing
logger = get_logger(
    __name__,
    trace_id="request-12345",
    span_id="operation-67890",
)

# Log with dynamic context
logger.info(
    "Processing document",
    document_path="tesla_q3.pdf",
    chunk_count=250,
    processing_stage="vision",
)
```

---

## Testing & Quality Assurance

### Run All Tests

```bash
# With coverage report
pytest tests/ -v --cov=src/rag_system --cov-report=html --cov-report=term-missing

# Only unit tests (fast)
pytest tests/ -m "not integration" -v

# Specific test file
pytest tests/test_config.py -v
```

### Linting & Type Checking

```bash
# Format code
black src/ tests/
isort src/ tests/

# Style checks
flake8 src/ tests/

# Type checking
mypy src/ --strict

# Static analysis
pylint src/

# Security scan
bandit -r src/ --skip B101
```

### Evaluation Suite

```bash
# Comprehensive system evaluation
python evals/run_evals.py

# Generates evals/eval_report.json with:
# - Configuration validation
# - Component initialization
# - Exception handling
# - Rate limiting
# - Retry policy effectiveness
```

---

## Configuration Reference

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...                          # OpenAI API key

# Optional
ACTIVELOOP_TOKEN=hub-...                       # DeepLake token
ENVIRONMENT=production                         # development|staging|production
DEBUG_MODE=false                               # Enable debug logging

# Advanced (component-specific)
VISION_CONFIG.retry_max_attempts=3
VISION_CONFIG.timeout_seconds=120
RATE_LIMIT_CONFIG.requests_per_second=10.0
RATE_LIMIT_CONFIG.burst_size=20
VECTOR_STORE_CONFIG.enable_deep_memory=true
LOGGING_CONFIG.format=json
LOGGING_CONFIG.level=INFO
```

### Configuration File (.env)

```env
OPENAI_API_KEY=sk-your-key
ACTIVELOOP_TOKEN=hub-your-token
ENVIRONMENT=production
DEBUG_MODE=false

[Vision Config]
retry_max_attempts=3
timeout_seconds=120

[Rate Limit Config]
requests_per_second=10.0
burst_size=20
retry_backoff_factor=2.0

[Logging Config]
level=INFO
format=json
```

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push:

1. **Lint & Type Check** (Python 3.10, 3.11)
   - Black format checks
   - isort import sorting
   - flake8 style validation
   - mypy strict type checking

2. **Unit & Integration Tests**
   - pytest with coverage
   - Report to Codecov

3. **Security Scan**
   - bandit for vulnerabilities

**Status Badge:** ![CI/CD](https://github.com/Mattral/RAG-Multimodal-Financial-Document-Analysis-and-Recall/actions/workflows/ci.yml/badge.svg)

---

## Architecture Decisions

### Why Async/Await?
- **10-100x throughput** vs synchronous code
- **No thread overhead** — single event loop per process
- **Cleaner syntax** than callbacks

### Why Pydantic?
- **Type safety** at runtime with clear error messages
- **IDE autocomplete** and type hints
- **Self-documenting** configuration schema

### Why Structured JSON Logging?
- **Trace correlation** end-to-end via trace IDs
- **Integration** with monitoring systems (DataDog, CloudWatch, NewRelic)
- **Queryable logs** for analysis and debugging

### Why Modular Components?
- **Testability** — each component isolated
- **Reusability** — use components independently
- **Maintainability** — clear boundaries and responsibilities

---

## Production Deployment

### Environment Setup

```bash
# Production config
ENVIRONMENT=production
DEBUG_MODE=false
LOGGING_CONFIG.level=WARNING  # Less verbose
LOGGING_CONFIG.format=json     # For aggregation
VISION_CONFIG.retry_max_attempts=5
RATE_LIMIT_CONFIG.requests_per_second=5  # Conservative
VECTOR_STORE_CONFIG.enable_deep_memory=true
```

### Scaling Strategies

| Dimension | Approach | Config |
|-----------|----------|--------|
| **Throughput** | Increase `batch_size` | 10 → 50 |
| **Latency** | Decrease `batch_size` | 50 → 5 |
| **Concurrency** | Increase `num_workers` | 4 → 16 |
| **API Rate** | Adjust `requests_per_second` | 5 → 20 |

### Monitoring Checklist

- [ ] Structured logs shipped to aggregator (DataDog, ELK, etc.)
- [ ] Trace ID correlation dashboard set up
- [ ] API quota monitoring (OpenAI tokens/requests)
- [ ] Memory/CPU alerts configured
- [ ] Error rate thresholds set
- [ ] SLA metrics tracked

---

## Troubleshooting

### "OPENAI_API_KEY not set"

```bash
export OPENAI_API_KEY="sk-..."
# Or in .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

### "Rate limited by OpenAI"

Adjust in config:
```python
RATE_LIMIT_CONFIG.requests_per_second = 3.0  # Slower
RATE_LIMIT_CONFIG.retry_max_attempts = 5     # More retries
```

### "PDF parsing fails"

Ensure system deps are installed:
```bash
sudo apt-get install -y poppler-utils tesseract-ocr
```

### "mypy type errors"

Run with ignore flag (during development):
```bash
mypy src/ --ignore-missing-imports
```

---

## References & Links

- **Educational Background:** [docs/educational_deepdive.md](docs/educational_deepdive.md)
- **Configuration:** [src/rag_system/config.py](src/rag_system/config.py)
- **Components:** [src/rag_system/components/](src/rag_system/components/)
- **API Docs:** Generated with `pdoc src.rag_system`
- **OpenAI GPT-4V:** https://platform.openai.com/docs/guides/vision
- **DeepLake Docs:** https://docs.activeloop.ai/
- **LlamaIndex:** https://docs.llamaindex.ai/

---

## License

MIT License © 2024
Critical information is distributed across **narrative text, tables, and visual elements such as charts and graphs**. 

