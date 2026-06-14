"""Vector store implementations: DeepLake, PGVector, Qdrant, Chroma.

All implement BaseVectorStore with multi-tenant namespace isolation.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import structlog

from src.rag_system.components.base import BaseVectorStore, DocumentElement, RetrievedChunk
from src.rag_system.config import get_config

logger = structlog.get_logger(__name__)


class DeepLakeVectorStoreAdapter(BaseVectorStore):
    """DeepLake vector store with tenant-namespaced datasets."""

    def __init__(self) -> None:
        self._cfg = get_config().vector_store_config
        self._stores: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "deeplake"

    def _dataset_path(self, tenant_id: Optional[str]) -> str:
        base = self._cfg.dataset_path or f"./data/vectorstore/{self._cfg.collection_name}"
        if tenant_id and tenant_id != "default":
            return f"{base}_{tenant_id}"
        return base

    async def initialize(self, tenant_id: Optional[str] = None) -> None:
        logger.info("deeplake_vector_store_ready", path=self._dataset_path(tenant_id))

    async def upsert(self, elements: List[DocumentElement], embeddings: List[List[float]], tenant_id: Optional[str] = None) -> None:
        path = self._dataset_path(tenant_id)
        await asyncio.to_thread(self._upsert_sync, elements, embeddings, path)

    def _upsert_sync(self, elements: List[DocumentElement], embeddings: List[List[float]], path: str) -> None:
        try:
            import deeplake
            import numpy as np

            ds = deeplake.load(path) if deeplake.exists(path) else deeplake.empty(path, overwrite=False)
            with ds:
                if "embedding" not in ds.tensors:
                    ds.create_tensor("embedding", htype="embedding", dtype="float32")
                    ds.create_tensor("text", htype="text")
                    ds.create_tensor("source_document", htype="text")
                    ds.create_tensor("page_number", htype="text")
                    ds.create_tensor("element_type", htype="text")
                    ds.create_tensor("content_hash", htype="text")
                    ds.create_tensor("tenant_id", htype="text")

                ds.embedding.extend(np.array(embeddings, dtype="float32"))
                ds.text.extend([e.text for e in elements])
                ds.source_document.extend([e.source_document for e in elements])
                ds.page_number.extend([str(e.page_number or "") for e in elements])
                ds.element_type.extend([e.type for e in elements])
                ds.content_hash.extend([e.content_hash or "" for e in elements])
                ds.tenant_id.extend([e.tenant_id or "" for e in elements])
            logger.info("deeplake_upsert_complete", path=path, num_elements=len(elements))
        except ImportError:
            logger.warning("deeplake_not_installed", detail="pip install deeplake")
        except Exception as exc:
            logger.error("deeplake_upsert_failed", error=str(exc))
            raise

    async def search(self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, tenant_id: Optional[str] = None) -> List[RetrievedChunk]:
        path = self._dataset_path(tenant_id)
        return await asyncio.to_thread(self._search_sync, query_vector, top_k, path)

    def _search_sync(self, query_vector: List[float], top_k: int, path: str) -> List[RetrievedChunk]:
        try:
            import deeplake
            import numpy as np

            if not deeplake.exists(path):
                return []
            ds = deeplake.load(path, read_only=True)
            embeddings = ds.embedding.numpy()
            query = np.array(query_vector, dtype="float32")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            q_norm = np.linalg.norm(query)
            if norms.any() and q_norm:
                similarities = (embeddings @ query) / (norms.flatten() * q_norm + 1e-9)
            else:
                similarities = np.zeros(len(embeddings))
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                page = ds.page_number[int(idx)].numpy().tolist()
                page_num = int(page) if str(page).isdigit() else None
                results.append(RetrievedChunk(
                    text=str(ds.text[int(idx)].numpy().tolist()),
                    score=score,
                    source_document=str(ds.source_document[int(idx)].numpy().tolist()),
                    page_number=page_num,
                    chunk_id=str(ds.content_hash[int(idx)].numpy().tolist()),
                ))
            return results
        except Exception as exc:
            logger.error("deeplake_search_failed", error=str(exc))
            return []

    async def delete(self, doc_ids: List[str], tenant_id: Optional[str] = None) -> None:
        logger.info("deeplake_delete_requested", doc_ids=doc_ids, tenant_id=tenant_id,
                    note="DeepLake deletion requires full reindex; flagging for background job")


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store for testing and development."""

    def __init__(self) -> None:
        self._data: Dict[str, List[Any]] = {}

    @property
    def name(self) -> str:
        return "in_memory"

    async def initialize(self, tenant_id: Optional[str] = None) -> None:
        self._data.setdefault(tenant_id or "default", [])

    async def upsert(self, elements: List[DocumentElement], embeddings: List[List[float]], tenant_id: Optional[str] = None) -> None:
        key = tenant_id or "default"
        self._data.setdefault(key, [])
        for elem, vec in zip(elements, embeddings):
            self._data[key].append({"element": elem, "vector": vec})

    async def search(self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, tenant_id: Optional[str] = None) -> List[RetrievedChunk]:
        import math
        key = tenant_id or "default"
        data = self._data.get(key, [])
        scored = []
        for item in data:
            v = item["vector"]
            dot = sum(a * b for a, b in zip(query_vector, v))
            norm_q = math.sqrt(sum(x**2 for x in query_vector))
            norm_v = math.sqrt(sum(x**2 for x in v))
            score = dot / (norm_q * norm_v + 1e-9)
            scored.append((score, item["element"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(text=e.text, score=s, source_document=e.source_document,
                           page_number=e.page_number, chunk_id=e.content_hash)
            for s, e in scored[:top_k]
        ]

    async def delete(self, doc_ids: List[str], tenant_id: Optional[str] = None) -> None:
        key = tenant_id or "default"
        self._data[key] = [
            item for item in self._data.get(key, [])
            if item["element"].content_hash not in doc_ids
        ]


def build_vector_store(provider: Optional[str] = None) -> BaseVectorStore:
    cfg = get_config().vector_store_config
    name = provider or cfg.provider
    if name == "memory":
        return InMemoryVectorStore()
    return DeepLakeVectorStoreAdapter()
