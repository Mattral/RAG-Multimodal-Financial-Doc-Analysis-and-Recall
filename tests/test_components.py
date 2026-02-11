"""Tests for pipeline components."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.rag_system.components.pdf_parser import PDFParser, DocumentElement
from src.rag_system.components.vision_processor import VisionProcessor
from src.rag_system.components.vector_indexer import VectorIndexer


class TestDocumentElement:
    """Test DocumentElement model."""

    def test_creation(self) -> None:
        """Test creating a document element."""
        element = DocumentElement(
            type="text",
            text="Sample text",
            source_document="test.pdf",
        )
        assert element.type == "text"
        assert element.text == "Sample text"
        assert element.source_document == "test.pdf"

    def test_with_metadata(self) -> None:
        """Test creating element with metadata."""
        element = DocumentElement(
            type="table",
            text="Table content",
            source_document="test.pdf",
            metadata={"rows": 5, "columns": 3},
        )
        assert element.metadata["rows"] == 5


@pytest.mark.asyncio
async def test_pdf_parser_initialization() -> None:
    """Test PDF parser initialization."""
    parser = PDFParser()
    assert parser.config is not None
    assert parser.retry_policy is not None


@pytest.mark.asyncio
async def test_vision_processor_initialization() -> None:
    """Test vision processor initialization."""
    processor = VisionProcessor()
    assert processor.config is not None
    assert processor.rate_limiter is not None


@pytest.mark.asyncio
async def test_vector_indexer_initialization() -> None:
    """Test vector indexer initialization."""
    indexer = VectorIndexer()
    assert indexer.config is not None
    assert indexer.vector_store is None  # Not initialized yet
