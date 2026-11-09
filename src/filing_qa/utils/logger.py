"""Enterprise structured logging via structlog with OTel trace/span injection.

Processors chain:
  add_log_level → add_logger_name → TimeStamper(ISO) → OTelContextProcessor
  → CallsiteParameter → MemoryMetricsProcessor → JSONRenderer (prod)
  | ConsoleRenderer (dev)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

import structlog
from structlog.types import EventDict, WrappedLogger

# ── Custom processors ────────────────────────────────────────────────────────


class OTelContextProcessor:
    """Inject OpenTelemetry trace_id/span_id from active span into log record."""

    def __call__(self, logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.is_valid:
                event_dict["trace_id"] = format(ctx.trace_id, "032x")
                event_dict["span_id"] = format(ctx.span_id, "016x")
        except Exception:
            pass
        return event_dict


class MemoryMetricsProcessor:
    """Inject process memory_mb into log records when enabled."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def __call__(self, logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
        if not self._enabled:
            return event_dict
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            event_dict["memory_mb"] = round(proc.memory_info().rss / 1024 / 1024, 1)
        except Exception:
            pass
        return event_dict


class ServiceContextProcessor:
    """Inject static service metadata into every log record."""

    def __init__(self, service_name: str, service_version: str, environment: str) -> None:
        self._ctx = {
            "service": service_name,
            "version": service_version,
            "env": environment,
        }

    def __call__(self, logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
        event_dict.update(self._ctx)
        return event_dict


# ── Setup ────────────────────────────────────────────────────────────────────


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
    service_name: str = "rag-financial-multimodal",
    service_version: str = "2.0.0",
    environment: str = "development",
    include_memory: bool = False,
) -> None:
    """Configure structlog + stdlib logging for the application.

    Args:
        level: Log level string (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        format_type: "json" for production, "text" for development console.
        log_file: Optional path to write logs to (in addition to stdout).
        service_name: Service name injected into every record.
        service_version: Service version injected into every record.
        environment: Environment name injected into every record.
        include_memory: Whether to log process memory usage.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        OTelContextProcessor(),
        ServiceContextProcessor(service_name, service_version, environment),
        MemoryMetricsProcessor(enabled=include_memory),
        structlog.processors.StackInfoRenderer(),
    ]

    if format_type == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    # Silence noisy third-party loggers
    for noisy in (
        "unstructured",
        "httpx",
        "openai",
        "httpcore",
        "urllib3",
        "multipart",
        "PIL",
        "pdfminer",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger bound to the given name."""
    return structlog.get_logger(name)
