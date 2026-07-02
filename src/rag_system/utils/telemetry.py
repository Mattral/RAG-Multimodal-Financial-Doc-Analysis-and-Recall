"""OpenTelemetry tracing + Prometheus metrics for the RAG pipeline.

Exposes:
- Distributed traces (ingest span, retrieval span, generation span)
- Custom Prometheus counters/histograms for RAG quality & cost
- Grafana-ready metric naming conventions
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Dict, Generator, Optional

# Prometheus
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (RAG-specific)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # Ingestion metrics
    INGEST_DOCS_TOTAL = Counter(
        "rag_ingest_documents_total",
        "Total documents ingested",
        ["tenant_id", "parser", "status"],
    )
    INGEST_CHUNKS_TOTAL = Counter(
        "rag_ingest_chunks_total",
        "Total chunks created during ingestion",
        ["tenant_id", "chunk_type"],
    )
    INGEST_LATENCY = Histogram(
        "rag_ingest_latency_seconds",
        "Document ingestion latency",
        ["tenant_id", "parser"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
    )
    INGEST_COST_USD = Counter(
        "rag_ingest_cost_usd_total",
        "Cumulative ingest cost in USD",
        ["tenant_id", "model"],
    )

    # Query/retrieval metrics
    QUERY_TOTAL = Counter(
        "rag_queries_total",
        "Total queries processed",
        ["tenant_id", "query_mode", "status"],
    )
    QUERY_LATENCY = Histogram(
        "rag_query_latency_seconds",
        "End-to-end query latency",
        ["tenant_id", "query_mode"],
        buckets=[0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 30],
    )
    RETRIEVAL_LATENCY = Histogram(
        "rag_retrieval_latency_seconds",
        "Retrieval stage latency",
        ["tenant_id", "strategy"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1, 2],
    )
    GENERATION_LATENCY = Histogram(
        "rag_generation_latency_seconds",
        "LLM generation latency",
        ["tenant_id", "model"],
        buckets=[0.1, 0.5, 1, 2, 4, 8, 16],
    )
    QUERY_COST_USD = Counter(
        "rag_query_cost_usd_total",
        "Cumulative query cost in USD",
        ["tenant_id", "model"],
    )
    TOKENS_USED = Counter(
        "rag_tokens_total",
        "Total LLM tokens consumed",
        ["tenant_id", "model", "token_type"],  # token_type: prompt|completion
    )
    CITATION_COVERAGE = Histogram(
        "rag_citation_coverage_ratio",
        "Fraction of answer claims backed by citations",
        ["tenant_id"],
        buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],
    )
    HALLUCINATION_SCORE = Histogram(
        "rag_hallucination_score",
        "Proxy hallucination score (0=grounded, 1=hallucinated)",
        ["tenant_id"],
        buckets=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    )
    CACHE_HIT_TOTAL = Counter(
        "rag_cache_hits_total",
        "Cache hits (embedding / semantic / llm)",
        ["cache_type", "tenant_id"],
    )
    RETRIEVAL_CHUNKS_RETURNED = Histogram(
        "rag_retrieval_chunks_returned",
        "Number of chunks returned per query",
        ["tenant_id"],
        buckets=[1, 3, 5, 10, 20, 50],
    )
    ACTIVE_TENANTS = Gauge(
        "rag_active_tenants",
        "Number of currently active tenants",
    )
    TENANT_MONTHLY_TOKENS_USED = Gauge(
        "rag_tenant_monthly_tokens_used",
        "Tokens consumed by a tenant in the current billing month",
        ["tenant_id"],
    )
    TENANT_MONTHLY_TOKEN_QUOTA = Gauge(
        "rag_tenant_monthly_token_quota",
        "Configured monthly token quota for a tenant",
        ["tenant_id"],
    )


# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------

_tracer: Optional[Any] = None


def setup_telemetry(
    service_name: str = "rag-financial-multimodal",
    service_version: str = "2.0.0",
    otlp_endpoint: Optional[str] = None,
    prometheus_port: int = 8001,
    sampling_rate: float = 1.0,
) -> None:
    """Initialise OTel tracing + Prometheus HTTP server."""
    global _tracer

    # --- Prometheus ---
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(prometheus_port)
            logger.info("prometheus_metrics_server_started", port=prometheus_port)
        except Exception as exc:
            logger.warning("prometheus_start_failed", error=str(exc))

    # --- OpenTelemetry ---
    if OTEL_AVAILABLE and otlp_endpoint:
        try:
            resource = Resource.create(
                {
                    ResourceAttributes.SERVICE_NAME: service_name,
                    ResourceAttributes.SERVICE_VERSION: service_version,
                }
            )
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            _tracer = trace.get_tracer(service_name, service_version)
            logger.info("otel_tracing_initialised", endpoint=otlp_endpoint)
        except Exception as exc:
            logger.warning("otel_init_failed", error=str(exc))
    else:
        # Noop tracer
        _tracer = None


def get_tracer() -> Optional[Any]:
    return _tracer


# ---------------------------------------------------------------------------
# Context managers for span-level instrumentation
# ---------------------------------------------------------------------------

@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[None, None, None]:
    """Synchronous span context manager."""
    if _tracer and OTEL_AVAILABLE:
        with _tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v))
            yield
    else:
        yield


@asynccontextmanager
async def async_trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[None, None]:
    """Async span context manager."""
    start = time.perf_counter()
    if _tracer and OTEL_AVAILABLE:
        with _tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v))
            yield
    else:
        yield
    elapsed = time.perf_counter() - start
    logger.debug("span_completed", span_name=name, elapsed_s=round(elapsed, 3))


# ---------------------------------------------------------------------------
# Metric helper functions
# ---------------------------------------------------------------------------

def record_ingest(
    tenant_id: str,
    parser: str,
    status: str,
    num_docs: int = 1,
    num_chunks: int = 0,
    latency_s: float = 0.0,
    cost_usd: float = 0.0,
    model: str = "unknown",
) -> None:
    if not PROMETHEUS_AVAILABLE:
        return
    INGEST_DOCS_TOTAL.labels(tenant_id=tenant_id, parser=parser, status=status).inc(num_docs)
    if num_chunks:
        INGEST_CHUNKS_TOTAL.labels(tenant_id=tenant_id, chunk_type="total").inc(num_chunks)
    if latency_s:
        INGEST_LATENCY.labels(tenant_id=tenant_id, parser=parser).observe(latency_s)
    if cost_usd:
        INGEST_COST_USD.labels(tenant_id=tenant_id, model=model).inc(cost_usd)


def record_query(
    tenant_id: str,
    query_mode: str,
    status: str,
    total_latency_s: float = 0.0,
    retrieval_latency_s: float = 0.0,
    generation_latency_s: float = 0.0,
    cost_usd: float = 0.0,
    model: str = "unknown",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    num_chunks: int = 0,
    citation_coverage: float = 0.0,
    hallucination_score: float = 0.0,
    retrieval_strategy: str = "hybrid",
) -> None:
    if not PROMETHEUS_AVAILABLE:
        return
    QUERY_TOTAL.labels(tenant_id=tenant_id, query_mode=query_mode, status=status).inc()
    if total_latency_s:
        QUERY_LATENCY.labels(tenant_id=tenant_id, query_mode=query_mode).observe(total_latency_s)
    if retrieval_latency_s:
        RETRIEVAL_LATENCY.labels(tenant_id=tenant_id, strategy=retrieval_strategy).observe(retrieval_latency_s)
    if generation_latency_s:
        GENERATION_LATENCY.labels(tenant_id=tenant_id, model=model).observe(generation_latency_s)
    if cost_usd:
        QUERY_COST_USD.labels(tenant_id=tenant_id, model=model).inc(cost_usd)
    if prompt_tokens:
        TOKENS_USED.labels(tenant_id=tenant_id, model=model, token_type="prompt").inc(prompt_tokens)
    if completion_tokens:
        TOKENS_USED.labels(tenant_id=tenant_id, model=model, token_type="completion").inc(completion_tokens)
    if num_chunks:
        RETRIEVAL_CHUNKS_RETURNED.labels(tenant_id=tenant_id).observe(num_chunks)
    if citation_coverage:
        CITATION_COVERAGE.labels(tenant_id=tenant_id).observe(citation_coverage)
    if hallucination_score:
        HALLUCINATION_SCORE.labels(tenant_id=tenant_id).observe(hallucination_score)


def record_cache_hit(cache_type: str, tenant_id: str) -> None:
    if PROMETHEUS_AVAILABLE:
        CACHE_HIT_TOTAL.labels(cache_type=cache_type, tenant_id=tenant_id).inc()


def record_tenant_quota(tenant_id: str, tokens_used: int, monthly_quota: int) -> None:
    """Publish current quota consumption as gauges.

    Called after every cost_tracker.record() so RAGTenantQuotaNearExhaustion
    and cost-burn-rate alerts (scripts/alerting/slo-burn-rate.yml) always see
    fresh values rather than stale/missing series.
    """
    if PROMETHEUS_AVAILABLE:
        TENANT_MONTHLY_TOKENS_USED.labels(tenant_id=tenant_id).set(tokens_used)
        TENANT_MONTHLY_TOKEN_QUOTA.labels(tenant_id=tenant_id).set(monthly_quota)
