"""PDF parser compatibility shim — v2.0.

The canonical parser is now in `components/parser/__init__.py`.
This module re-exports from the new location for backward compatibility
with any code that imports from `components.pdf_parser`.
"""

from src.rag_system.components.base import DocumentElement
from src.rag_system.components.parser import DoclingParser, UnstructuredParser, build_parser

# Backward-compatible alias
PDFParser = UnstructuredParser

__all__ = ["PDFParser", "DocumentElement", "UnstructuredParser", "DoclingParser", "build_parser"]
