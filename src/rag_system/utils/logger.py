"""Structured logging infrastructure with JSON output and dynamic metadata."""

import logging
import sys
import json
import traceback
import psutil
import os
from typing import Any, Dict, Optional
from datetime import datetime
import uuid

import structlog


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            str: JSON-formatted log line.
        """
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add trace and span IDs if available
        if hasattr(record, "trace_id"):
            log_obj["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_obj["span_id"] = record.span_id

        # Add memory metrics
        try:
            process = psutil.Process(os.getpid())
            log_obj["memory_mb"] = round(process.memory_info().rss / 1024 / 1024, 2)
        except Exception:
            pass

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)

        return json.dumps(log_obj, default=str)


class StructuredLogger:
    """Wrapper for structured logging with dynamic context."""

    def __init__(self, name: str, trace_id: Optional[str] = None, span_id: Optional[str] = None):
        """
        Initialize structured logger.

        Args:
            name: Logger name.
            trace_id: Optional trace ID for correlation.
            span_id: Optional span ID for distributed tracing.
        """
        self.logger = logging.getLogger(name)
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.span_id = span_id or str(uuid.uuid4())[:8]
        self.extra_fields: Dict[str, Any] = {}

    def _make_record(self, level: int, msg: str, *args, **kwargs) -> None:
        """Helper to inject trace/span IDs and extra fields."""
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn="",
            lno=0,
            msg=msg,
            args=args,
            exc_info=kwargs.get("exc_info"),
        )
        record.trace_id = self.trace_id
        record.span_id = self.span_id
        if self.extra_fields:
            record.extra_fields = self.extra_fields
        return record

    def debug(self, msg: str, **context) -> None:
        """Log at DEBUG level with optional context."""
        self.extra_fields = context
        self.logger.debug(msg, extra={"trace_id": self.trace_id, "span_id": self.span_id})

    def info(self, msg: str, **context) -> None:
        """Log at INFO level with optional context."""
        self.extra_fields = context
        self.logger.info(msg, extra={"trace_id": self.trace_id, "span_id": self.span_id})

    def warning(self, msg: str, **context) -> None:
        """Log at WARNING level with optional context."""
        self.extra_fields = context
        self.logger.warning(msg, extra={"trace_id": self.trace_id, "span_id": self.span_id})

    def error(self, msg: str, exc_info: bool = False, **context) -> None:
        """Log at ERROR level with optional context."""
        self.extra_fields = context
        self.logger.error(
            msg, exc_info=exc_info, extra={"trace_id": self.trace_id, "span_id": self.span_id}
        )

    def critical(self, msg: str, exc_info: bool = False, **context) -> None:
        """Log at CRITICAL level with optional context."""
        self.extra_fields = context
        self.logger.critical(
            msg, exc_info=exc_info, extra={"trace_id": self.trace_id, "span_id": self.span_id}
        )


def get_logger(
    name: str, trace_id: Optional[str] = None, span_id: Optional[str] = None
) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__).
        trace_id: Optional trace ID for correlation.
        span_id: Optional span ID for distributed tracing.

    Returns:
        StructuredLogger: A structured logger instance.
    """
    return StructuredLogger(name, trace_id=trace_id, span_id=span_id)


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_type: Format type ('json' or 'text').
        log_file: Optional file path to write logs to.
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress verbose third-party loggers
    logging.getLogger("unstructured").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
