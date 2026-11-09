"""Rich exception hierarchy for the RAG Financial system.

All exceptions carry:
  - machine-readable code (for API error responses)
  - structured details dict (for logging / debugging)
  - optional HTTP status code (for FastAPI exception handlers)
  - tenant_id context where applicable
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class RAGError(Exception):
    """Base exception for all RAG system errors."""

    http_status: int = 500
    default_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        self.message = message
        self.code = code or self.default_code
        self.details: Dict[str, Any] = details or {}
        if tenant_id:
            self.details["tenant_id"] = tenant_id
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for API error responses."""
        return {"error": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, code={self.code!r}, details={self.details})"
        )


# ── Ingestion ─────────────────────────────────────────────────────────────────


class PDFParsingError(RAGError):
    http_status = 422
    default_code = "PDF_PARSING_ERROR"

    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, details=details, **kwargs)


class VisionParsingError(RAGError):
    http_status = 422
    default_code = "VISION_PARSING_ERROR"

    def __init__(
        self,
        message: str,
        image_path: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if image_path:
            details["image_path"] = image_path
        if status_code:
            details["upstream_status_code"] = status_code
        super().__init__(message, details=details, **kwargs)


class DocumentNotFoundError(RAGError):
    http_status = 404
    default_code = "DOCUMENT_NOT_FOUND"


# ── Storage ───────────────────────────────────────────────────────────────────


class VectorStorageError(RAGError):
    http_status = 503
    default_code = "VECTOR_STORAGE_ERROR"

    def __init__(self, message: str, dataset_path: Optional[str] = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if dataset_path:
            details["dataset_path"] = dataset_path
        super().__init__(message, details=details, **kwargs)


class EmbeddingError(RAGError):
    http_status = 503
    default_code = "EMBEDDING_ERROR"


# ── Retrieval / Generation ────────────────────────────────────────────────────


class RetrievalError(RAGError):
    http_status = 503
    default_code = "RETRIEVAL_ERROR"


class GenerationError(RAGError):
    http_status = 503
    default_code = "GENERATION_ERROR"


class NoContextFoundError(RAGError):
    """Raised when no relevant context is retrieved for a query."""

    http_status = 404
    default_code = "NO_CONTEXT_FOUND"


# ── API / Network ─────────────────────────────────────────────────────────────


class APIRateLimitError(RAGError):
    http_status = 429
    default_code = "API_RATE_LIMIT_ERROR"

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        api_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if retry_after:
            details["retry_after_seconds"] = retry_after
        if api_name:
            details["api_name"] = api_name
        super().__init__(message, details=details, **kwargs)


class APITimeoutError(RAGError):
    http_status = 504
    default_code = "API_TIMEOUT_ERROR"

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        api_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if api_name:
            details["api_name"] = api_name
        super().__init__(message, details=details, **kwargs)


# ── Auth / Tenancy ────────────────────────────────────────────────────────────


class AuthenticationError(RAGError):
    http_status = 401
    default_code = "AUTHENTICATION_ERROR"


class AuthorizationError(RAGError):
    http_status = 403
    default_code = "AUTHORIZATION_ERROR"


class QuotaExceededError(RAGError):
    http_status = 429
    default_code = "QUOTA_EXCEEDED"

    def __init__(self, message: str, quota_type: str = "tokens", **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        details["quota_type"] = quota_type
        super().__init__(message, details=details, **kwargs)


class TenantNotFoundError(RAGError):
    http_status = 404
    default_code = "TENANT_NOT_FOUND"


# ── Config / Guardrails ───────────────────────────────────────────────────────


class ConfigurationError(RAGError):
    http_status = 500
    default_code = "CONFIGURATION_ERROR"

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details, **kwargs)


class GuardrailViolationError(RAGError):
    http_status = 400
    default_code = "GUARDRAIL_VIOLATION"

    def __init__(self, message: str, violation_type: str = "unknown", **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        details["violation_type"] = violation_type
        super().__init__(message, details=details, **kwargs)


# ── Retry ─────────────────────────────────────────────────────────────────────


class RetryableError(RAGError):
    http_status = 503
    default_code = "RETRYABLE_ERROR"

    def __init__(
        self,
        message: str,
        retry_count: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if retry_count is not None:
            details["retry_count"] = retry_count
        super().__init__(message, details=details, **kwargs)


class MaxRetriesExceededError(RetryableError):
    default_code = "MAX_RETRIES_EXCEEDED"


# ── Cache ─────────────────────────────────────────────────────────────────────


class CacheError(RAGError):
    http_status = 503
    default_code = "CACHE_ERROR"

    def __init__(self, message: str, cache_key: Optional[str] = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if cache_key:
            details["cache_key"] = cache_key
        super().__init__(message, details=details, **kwargs)


# ── Helper ────────────────────────────────────────────────────────────────────


def is_retryable(exc: Exception) -> bool:
    """Return True if the exception is safe to retry."""
    retryable_types = (RetryableError, APIRateLimitError, APITimeoutError, VectorStorageError)
    return isinstance(exc, retryable_types)
