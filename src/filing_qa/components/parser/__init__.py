"""PDF Parser implementations: Unstructured (default), Docling, Marker.

All implement BaseParser. Switch provider via config without code changes.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List, Optional

import structlog

from filing_qa.components.base import BaseParser, DocumentElement
from filing_qa.config import get_config
from filing_qa.utils.exceptions import PDFParsingError
from filing_qa.utils.telemetry import async_trace_span, record_document_parsed

logger = structlog.get_logger(__name__)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _group_by_page(elements: List[Any]) -> List[List[Any]]:
    """Consecutive runs of partition elements on the same page; page breaks dropped."""
    groups: List[List[Any]] = []
    current_page: Any = object()
    for element in elements:
        if type(element).__name__ == "PageBreak":
            continue
        page = getattr(getattr(element, "metadata", None), "page_number", None)
        if not groups or page != current_page:
            groups.append([])
            current_page = page
        groups[-1].append(element)
    return groups


class UnstructuredParser(BaseParser):
    """Parser backed by unstructured.io — good general-purpose baseline."""

    def __init__(self) -> None:
        self._cfg = get_config().pdf_parsing_config

    @property
    def name(self) -> str:
        return "unstructured"

    @property
    def variant(self) -> str:
        """Parser and strategy as recorded per document, e.g. "unstructured/hi_res"."""
        return f"unstructured/{self._cfg.unstructured_strategy}"

    async def parse(self, file_path: str, tenant_id: Optional[str] = None) -> List[DocumentElement]:
        async with async_trace_span(
            "parse_document", {"parser": self.name, "file": Path(file_path).name}
        ):
            return await asyncio.to_thread(self._parse_sync, file_path, tenant_id)

    def _parse_sync(self, file_path: str, tenant_id: Optional[str]) -> List[DocumentElement]:
        try:
            from unstructured.chunking.title import chunk_by_title
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            return self._fallback(file_path, tenant_id, exc)

        try:
            raw = partition_pdf(
                filename=file_path,
                strategy=self._cfg.unstructured_strategy,
                infer_table_structure=self._cfg.infer_table_structure,
                extract_images_in_pdf=self._cfg.extract_images,
                include_page_breaks=True,
            )
            # A chunk carries one page number, so chunks must not span pages, or text
            # after a page break is cited to the chunk's first page. chunk_by_title's
            # small-section combining merges across pages even with
            # multipage_sections=False, so each page is chunked on its own.
            chunks = []
            for page_elements in _group_by_page(raw):
                chunks.extend(
                    chunk_by_title(
                        page_elements,
                        multipage_sections=False,
                        max_characters=self._cfg.max_characters,
                        new_after_n_chars=self._cfg.new_after_n_chars,
                        combine_text_under_n_chars=self._cfg.combine_text_under_n_chars,
                    )
                )
            now = datetime.now(UTC).isoformat()
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
                elements.append(
                    DocumentElement(
                        type=etype,
                        text=text,
                        source_document=source,
                        page_number=page,
                        ingest_timestamp=now,
                        content_hash=_content_hash(text),
                        tenant_id=tenant_id,
                        metadata={"parser": self.variant, "raw_type": element_type},
                    )
                )
            logger.info(
                "unstructured_parse_complete",
                file=source,
                parser=self.variant,
                num_chunks=len(elements),
            )
            record_document_parsed(self.variant)
            return elements
        except Exception as exc:
            return self._fallback(file_path, tenant_id, exc)

    def _fallback(
        self, file_path: str, tenant_id: Optional[str], exc: BaseException
    ) -> List[DocumentElement]:
        """Use the configured fallback parser, or fail the document if none is configured."""
        fallback = self._cfg.fallback_parser
        if fallback == "none":
            raise PDFParsingError(
                f"{self.variant} failed on {Path(file_path).name}: {type(exc).__name__}",
                file_path=file_path,
            ) from exc
        logger.warning(
            "pdf_parser_fallback",
            file=Path(file_path).name,
            primary=self.variant,
            fallback=fallback,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        elements = self._fallback_parse(file_path, tenant_id)
        record_document_parsed(fallback, fallback_from=self.variant)
        return elements

    def _fallback_parse(self, file_path: str, tenant_id: Optional[str]) -> List[DocumentElement]:
        """Fallback: the PDF's text layer via pypdf, one element per page (no tables/OCR)."""
        try:
            import pypdf

            source = Path(file_path).name
            now = datetime.now(UTC).isoformat()
            elements = []
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages, 1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        elements.append(
                            DocumentElement(
                                type="text",
                                text=text,
                                source_document=source,
                                page_number=page_num,
                                ingest_timestamp=now,
                                content_hash=_content_hash(text),
                                tenant_id=tenant_id,
                                metadata={
                                    "parser": "pypdf",
                                    "parser_fallback_from": self.variant,
                                },
                            )
                        )
            return elements
        except Exception as exc:
            logger.error("fallback_parse_failed", file=file_path, error=str(exc))
            raise PDFParsingError(
                f"pypdf fallback failed on {Path(file_path).name}: {type(exc).__name__}",
                file_path=file_path,
            ) from exc

    async def parse_batch(
        self, file_paths: List[str], tenant_id: Optional[str] = None
    ) -> List[DocumentElement]:
        tasks = [self.parse(fp, tenant_id=tenant_id) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elements: List[DocumentElement] = []
        failed = []
        for fp, result in zip(file_paths, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("batch_parse_file_failed", file=fp, error=str(result))
                failed.append(Path(fp).name)
            else:
                elements.extend(result)
        if failed:
            # Dropping a document here would report a successful ingest with missing data.
            raise PDFParsingError(f"Could not parse: {', '.join(failed)}")
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
        now = datetime.now(UTC).isoformat()
        source = Path(file_path).name
        elements = []
        for page in result.pages:
            for element in page.elements:
                text = getattr(element, "text", "").strip()
                if not text:
                    continue
                etype = "table" if "table" in type(element).__name__.lower() else "text"
                elements.append(
                    DocumentElement(
                        type=etype,
                        text=text,
                        source_document=source,
                        page_number=getattr(page, "page_no", None),
                        ingest_timestamp=now,
                        content_hash=_content_hash(text),
                        tenant_id=tenant_id,
                        metadata={"parser": self.name},
                    )
                )
        logger.info("docling_parse_complete", file=source, num_elements=len(elements))
        return elements

    async def parse_batch(
        self, file_paths: List[str], tenant_id: Optional[str] = None
    ) -> List[DocumentElement]:
        tasks = [self.parse(fp, tenant_id=tenant_id) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elements: List[DocumentElement] = []
        for fp, result in zip(file_paths, results, strict=True):
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


# Re-import Marker adapter so it's available from the package
try:
    from filing_qa.components.parser.marker_adapter import MarkerParser

    def build_parser(provider: Optional[str] = None) -> BaseParser:
        """Factory: return the parser specified in config or by argument."""
        cfg = get_config().pdf_parsing_config
        name = provider or cfg.primary_parser
        if name == "docling":
            return DoclingParser()
        if name == "marker":
            return MarkerParser()
        return UnstructuredParser()

except ImportError:
    pass  # MarkerParser already has graceful fallback
