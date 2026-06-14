# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install all dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libmagic1 \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --only main,api --no-root || \
    pip install --no-cache-dir \
        pydantic>=2.0 pydantic-settings>=2.0 \
        structlog fastapi uvicorn httpx \
        typer rich unstructured pypdf \
        openai numpy pandas tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — minimal, non-root
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1001 raguser && \
    useradd --uid 1001 --gid raguser --shell /bin/bash --create-home raguser

WORKDIR /app

# Copy system libs for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY --chown=raguser:raguser src/ ./src/
COPY --chown=raguser:raguser evals/ ./evals/
COPY --chown=raguser:raguser pyproject.toml ./

# Create data directories
RUN mkdir -p /app/data/vectorstore /app/audit_logs /app/logs && \
    chown -R raguser:raguser /app/data /app/audit_logs /app/logs

USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

EXPOSE 8000 8001

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

CMD ["uvicorn", "src.rag_system.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
