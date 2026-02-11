"""Custom exception classes for the RAG system."""

from typing import Optional


class RAGException(Exception):
    """Base exception for all RAG system errors."""

    def __init__(self, message: str, code: str = "UNKNOWN_ERROR", details: Optional[dict] = None):
        """
        Initialize RAG exception.

        Args:
            message: Error message.
            code: Error code for categorization.
            details: Optional dict with additional error details.
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error string."""
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed error representation."""
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r}, details={self.details})"


class PDFParsingError(RAGException):
    """Raised when PDF parsing fails."""

    def __init__(self, message: str, file_path: Optional[str] = None, details: Optional[dict] = None):
        """Initialize PDF parsing error."""
        details = details or {}
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, code="PDF_PARSING_ERROR", details=details)


class VisionParsingError(RAGException):
    """Raised when vision model processing fails."""

    def __init__(
        self,
        message: str,
        image_path: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        """Initialize vision parsing error."""
        details = details or {}
        if image_path:
            details["image_path"] = image_path
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, code="VISION_PARSING_ERROR", details=details)


class VectorStorageError(RAGException):
    """Raised when vector storage operations fail."""

    def __init__(self, message: str, dataset_path: Optional[str] = None, details: Optional[dict] = None):
        """Initialize vector storage error."""
        details = details or {}
        if dataset_path:
            details["dataset_path"] = dataset_path
        super().__init__(message, code="VECTOR_STORAGE_ERROR", details=details)


class APIRateLimitError(RAGException):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        api_name: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """Initialize API rate limit error."""
        details = details or {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        if api_name:
            details["api_name"] = api_name
        super().__init__(message, code="API_RATE_LIMIT_ERROR", details=details)


class APITimeoutError(RAGException):
    """Raised when API request times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[int] = None,
        api_name: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """Initialize API timeout error."""
        details = details or {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if api_name:
            details["api_name"] = api_name
        super().__init__(message, code="API_TIMEOUT_ERROR", details=details)


class ConfigurationError(RAGException):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[dict] = None):
        """Initialize configuration error."""
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class RetryableError(RAGException):
    """Base class for retryable errors."""

    def __init__(
        self,
        message: str,
        code: str = "RETRYABLE_ERROR",
        retry_count: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        """Initialize retryable error."""
        details = details or {}
        if retry_count is not None:
            details["retry_count"] = retry_count
        super().__init__(message, code=code, details=details)


class CacheError(RAGException):
    """Raised when cache operations fail."""

    def __init__(self, message: str, cache_key: Optional[str] = None, details: Optional[dict] = None):
        """Initialize cache error."""
        details = details or {}
        if cache_key:
            details["cache_key"] = cache_key
        super().__init__(message, code="CACHE_ERROR", details=details)
