"""PDF Parser implementations: Unstructured (default), Docling, Marker.

All implement BaseParser. Switch provider via config without code changes.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.rag_system.components.base import BaseParser, DocumentElement
from src.rag_system.config import get_config
from src.rag_system.utils.telemetry import async_trace_span

logger = structlog.get_logger(__name__)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class UnstructuredParser(BaseParser):
    """Parser backed by unstructured.io — good general-purpose baseline."""

    def __init__(self) -> None:
        self._cfg = get_config().pdf_parsing_config

    @property
    def name(self) -> str:
        return "unstructured"

    async def parse(self, file_path: str, tenant_id: Optional[str] = None) -> List[DocumentElement]:
        async with async_trace_span("parse_document", {"parser": self.name, "file": Path(file_path).name}):
            return await asyncio.to_thread(self._parse_sync, file_path, tenant_id)

    def _parse_sync(self, file_path: str, tenant_id: Optional[str]) -> List[DocumentElement]:
        try:
            from unstructured.partition.pdf import partition_pdf
            from unstructured.chunking.title import chunk_by_title
        except ImportError:
            logger.warning("unstructured_not_installed", detail="pip install 'unstructured[all-docs]'")
            return self._fallback_parse(file_path, tenant_id)

        try:
            raw = partition_pdf(
                filename=file_path,
                infer_table_structure=self._cfg.infer_table_structure,
                extract_images_in_pdf=self._cfg.extract_images,
                include_page_breaks=True,
            )
            chunks = chunk_by_title(
                raw,
                max_characters=self._cfg.max_characters,
                new_after_n_chars=self._cfg.new_after_n_chars,
                combine_text_under_n_chars=self._cfg.combine_text_under_n_chars,
            )
            now = datetime.now(timezone.utc).isoformat()
            source = Path(file_path).name
            elements = []
            for chunk in chunks:
                text = str(chunk).strip()
                if not text:
                    continue
                page = getattr(chunk.metadata, "page_number", None)
                element_type = type(chunk).__name__.lower()
                if "table" in element_type:
                    etype = "table"
                elif "image" in element_type or "figure" in element_type:
                    etype = "image"
                else:
                    etype = "text"
                elements.append(DocumentElement(
                    type=etype,
                    text=text,
                    source_document=source,
                    page_number=page,
                    ingest_timestamp=now,
                    content_hash=_content_hash(text),
                    tenant_id=tenant_id,
                    metadata={"parser": self.name, "raw_type": element_type},
                ))
            logger.info("unstructured_parse_complete", file=source, num_chunks=len(elements))
            return elements
        except Exception as exc:
            logger.error("unstructured_parse_failed", file=file_path, error=str(exc))
            return self._fallback_parse(file_path, tenant_id)

    def _fallback_parse(self, file_path: str, tenant_id: Optional[str]) -> List[DocumentElement]:
        """Minimal fallback: read raw text with PyPDF2."""
        try:
            import pypdf
            source = Path(file_path).name
            now = datetime.now(timezone.utc).isoformat()
            elements = []
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages, 1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        elements.append(DocumentElement(
                            type="text", text=text, source_document=source,
                            page_number=page_num, ingest_timestamp=now,
                            content_hash=_content_hash(text), tenant_id=tenant_id,
                            metadata={"parser": "pypdf_fallback"},
                        ))
            return elements
        except Exception as exc:
            logger.error("fallback_parse_failed", file=file_path, error=str(exc))
            return []

    async def parse_batch(self, file_paths: List[str], tenant_id: Optional[str] = None) -> List[DocumentElement]:
        tasks = [self.parse(fp, tenant_id=tenant_id) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elements: List[DocumentElement] = []
        for fp, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error("batch_parse_file_failed", file=fp, error=str(result))
            else:
                elements.extend(result)
        return elements


class DoclingParser(BaseParser):
    """IBM Docling parser — superior table/layout extraction."""

    @property
    def name(self) -> str:
        return "docling"

    async def parse(self, file_path: str, tenant_id: Optional[str] = None) -> List[DocumentElement]:
        return await asyncio.to_thread(self._parse_sync, file_path, tenant_id)

    def _parse_sync(self, file_path: str, tenant_id: Optional[str]) -> List[DocumentElement]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            logger.warning("docling_not_installed", detail="pip install docling")
            return UnstructuredParser().parse_batch.__func__(
                UnstructuredParser(), [file_path], tenant_id
            )

        converter = DocumentConverter()
        result = converter.convert(file_path)
        now = datetime.now(timezone.utc).isoformat()
        source = Path(file_path).name
        elements = []
        for page in result.pages:
            for element in page.elements:
                text = getattr(element, "text", "").strip()
                if not text:
                    continue
                etype = "table" if "table" in type(element).__name__.lower() else "text"
                elements.append(DocumentElement(
                    type=etype, text=text, source_document=source,
                    page_number=getattr(page, "page_no", None),
                    ingest_timestamp=now, content_hash=_content_hash(text),
                    tenant_id=tenant_id, metadata={"parser": self.name},
                ))
        logger.info("docling_parse_complete", file=source, num_elements=len(elements))
        return elements

    async def parse_batch(self, file_paths: List[str], tenant_id: Optional[str] = None) -> List[DocumentElement]:
        tasks = [self.parse(fp, tenant_id=tenant_id) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elements: List[DocumentElement] = []
        for fp, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error("docling_batch_file_failed", file=fp, error=str(result))
            else:
                elements.extend(result)
        return elements


def build_parser(provider: Optional[str] = None) -> BaseParser:
    """Factory: return the parser specified in config or by argument."""
    cfg = get_config().pdf_parsing_config
    name = provider or cfg.primary_parser
    if name == "docling":
        return DoclingParser()
    return UnstructuredParser()
