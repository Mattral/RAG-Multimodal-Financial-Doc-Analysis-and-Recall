"""Thin Python SDK for the RAG Financial pipeline.

Provides a clean, importable interface for programmatic use:

    from rag_financial import RAGPipeline
    pipeline = await RAGPipeline.from_config("config.yaml", tenant_id="acme")
    await pipeline.ingest(["report.pdf"])
    answer = await pipeline.query("What was Q3 revenue?")
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.rag_system.config import Config, get_config, reset_config
from src.rag_system.pipeline import RAGPipeline as _RAGPipeline


class RAGPipeline:
    """High-level SDK wrapper around the internal RAGPipeline.

    Provides both sync and async interfaces for easy integration.
    """

    def __init__(self, _pipeline: _RAGPipeline, tenant_id: str = "default") -> None:
        self._pipeline = _pipeline
        self._default_tenant = tenant_id

    @classmethod
    async def create(
        cls,
        tenant_id: str = "default",
        config: Optional[Config] = None,
        **component_overrides,
    ) -> "RAGPipeline":
        """Create a fully-wired pipeline (async)."""
        pipeline = await _RAGPipeline.create(config=config, **component_overrides)
        return cls(pipeline, tenant_id=tenant_id)

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        tenant_id: str = "default",
    ) -> "RAGPipeline":
        """Synchronous factory loading config from YAML. Runs event loop internally."""
        cfg_path = Path(config_path)
        if cfg_path.exists():
            with open(cfg_path) as f:
                data = yaml.safe_load(f)
            import os
            for k, v in data.items():
                os.environ.setdefault(k.upper(), str(v))
            reset_config()

        async def _build():
            return await cls.create(tenant_id=tenant_id)

        return asyncio.run(_build())

    async def ingest(
        self,
        file_paths: Union[str, List[str]],
        tenant_id: Optional[str] = None,
        process_vision: bool = True,
    ) -> Dict[str, Any]:
        """Ingest one or more PDF files."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        return await self._pipeline.ingest(
            file_paths=file_paths,
            tenant_id=tenant_id or self._default_tenant,
            process_vision=process_vision,
        )

    async def query(
        self,
        question: str,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query the ingested documents."""
        return await self._pipeline.query(
            query_text=question,
            tenant_id=tenant_id or self._default_tenant,
            top_k=top_k,
            filters=filters,
        )

    def query_sync(self, question: str, **kwargs) -> Dict[str, Any]:
        """Synchronous query wrapper for non-async contexts."""
        return asyncio.run(self.query(question, **kwargs))

    def ingest_sync(self, file_paths: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Synchronous ingest wrapper for non-async contexts."""
        return asyncio.run(self.ingest(file_paths, **kwargs))

    async def health(self) -> Dict[str, Any]:
        """Check pipeline component health."""
        return await self._pipeline.health_check()


__all__ = ["RAGPipeline"]
