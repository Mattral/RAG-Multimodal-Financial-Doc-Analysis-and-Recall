# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install the locked dependencies into /app/.venv
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Extras for the default configuration: API server, Qdrant vector store, unstructured
# hi_res PDF parsing, and local models for the cross-encoder reranker / local embedder.
# Installed strictly from uv.lock (--frozen); there is no fallback install path.
ARG EXTRAS="--extra api --extra qdrant --extra parsing --extra local-models"

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS}

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${EXTRAS}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — minimal, non-root
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1001 raguser && \
    useradd --uid 1001 --gid raguser --shell /bin/bash --create-home raguser

WORKDIR /app

# PDF processing: poppler (page images), tesseract (OCR for hi_res), libmagic (file
# types), libgl/glib (OpenCV, used by the layout model).
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --chown=raguser:raguser evals/ ./evals/

# Data directories and the model cache (Hugging Face downloads on first use).
RUN mkdir -p /app/data /app/audit_logs /app/logs /app/.cache/huggingface && \
    chown -R raguser:raguser /app/data /app/audit_logs /app/logs /app/.cache

USER raguser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

EXPOSE 8000 8001

ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# One worker: BM25 index, semantic cache and metrics server are per process today.
CMD ["uvicorn", "filing_qa.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
