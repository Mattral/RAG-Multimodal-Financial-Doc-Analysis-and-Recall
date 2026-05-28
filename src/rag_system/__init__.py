"""RAG System Package - Multimodal Financial Document Analysis."""

__version__ = "1.0.0"
__author__ = "RAG Team"

from src.rag_system.config import Config, VisionConfig, VectorStoreConfig
from src.rag_system.utils.exceptions import (
    RAGException,
    PDFParsingError,
    VisionParsingError,
    VectorStorageError,
    APIRateLimitError,
)
from src.rag_system.utils.logger import get_logger, setup_logging

__all__ = [
    "Config",
    "VisionConfig",
    "VectorStoreConfig",
    "RAGException",
    "PDFParsingError",
    "VisionParsingError",
    "VectorStorageError",
    "APIRateLimitError",
    "get_logger",
    "setup_logging",
]
