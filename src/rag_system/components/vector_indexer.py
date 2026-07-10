"""Vector indexer compatibility shim — v2.0.

The canonical vector store is now in `components/vector_store/__init__.py`.
This module re-exports for backward compatibility.
"""

from src.rag_system.components.vector_store import (
    DeepLakeVectorStoreAdapter,
    InMemoryVectorStore,
    build_vector_store,
)

# Backward-compatible alias
VectorIndexer = DeepLakeVectorStoreAdapter

__all__ = [
    "VectorIndexer",
    "DeepLakeVectorStoreAdapter",
    "InMemoryVectorStore",
    "build_vector_store",
]
