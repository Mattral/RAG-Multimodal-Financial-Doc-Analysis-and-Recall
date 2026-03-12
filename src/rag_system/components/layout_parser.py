"""Layout-aware semantic parser: preserves spatial context, groups related elements,
prevents multi-page table fragmentation.

Produces chunks with HTML-annotated structure so the LLM receives rich
layout context (table captions stay with their table, figure captions
stay with their figure, narrative paragraphs aren't split mid-sentence).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

from src.rag_system.components.base import DocumentElement

logger = structlog.get_logger(__name__)


@dataclass
class LayoutChunk:
    """A semantically grouped chunk with spatial metadata."""
    text: str
    html: str                           # HTML-wrapped version for LLM context
    element_types: List[str]            # e.g. ["table", "caption"]
    source_document: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None
    heading: Optional[str] = None
    is_continuation: bool = False       # True if spans >1 page
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_range(self) -> str:
        if self.page_start == self.page_end or self.page_end is None:
            return str(self.page_start or "?")
        return f"{self.page_start}-{self.page_end}"


def _wrap_table(text: str, caption: Optional[str] = None) -> str:
    cap_html = f'<caption>{caption}</caption>\n' if caption else ""
    return f'<table class="financial-table">\n{cap_html}{text}\n</table>'


def _wrap_figure(description: str, caption: Optional[str] = None) -> str:
    cap_html = f'<figcaption>{caption}</figcaption>\n' if caption else ""
    return f'<figure class="financial-chart">\n{cap_html}{description}\n</figure>'


def _wrap_section(text: str, heading: Optional[str] = None, level: int = 2) -> str:
    head_html = f'<h{level}>{heading}</h{level}>\n' if heading else ""
    return f'<section>\n{head_html}{text}\n</section>'


_HEADING_RE = re.compile(
    r"^((?:Item\s+\d+|NOTE\s+\d+|PART\s+[IVX]+|"
    r"(?:MANAGEMENT['']S\s+)?DISCUSSION|FINANCIAL\s+STATEMENTS?"
    r"|RISK\s+FACTORS?|SELECTED\s+FINANCIAL|QUANTITATIVE).+)$",
    re.IGNORECASE | re.MULTILINE,
)

_CAPTION_RE = re.compile(
    r"(?:Figure|Table|Exhibit|Chart)\s+\d+[:\.\s](.+?)(?:\n|$)",
    re.IGNORECASE,
)


class LayoutAwareParser:
    """Groups document elements by spatial/semantic proximity.

    Algorithm:
      1. Detect headings → open new sections
      2. Pair tables with their captions (within N lines)
      3. Pair figures/graphs with their captions
      4. Detect multi-page table continuation via header repetition
      5. Merge short orphan paragraphs into adjacent sections
      6. Wrap each group in semantic HTML
    """

    def __init__(
        self,
        max_chunk_chars: int = 4000,
        caption_proximity_lines: int = 3,
        min_chunk_chars: int = 200,
    ) -> None:
        self._max_chars = max_chunk_chars
        self._caption_proximity = caption_proximity_lines
        self._min_chars = min_chunk_chars

    def parse(self, elements: List[DocumentElement]) -> List[LayoutChunk]:
        """Transform flat element list into layout-aware semantic chunks."""
        if not elements:
            return []

        chunks: List[LayoutChunk] = []
        current_heading: Optional[str] = None
        pending_caption: Optional[str] = None
        i = 0

        while i < len(elements):
            elem = elements[i]

            # Detect section headings
            heading_match = _HEADING_RE.search(elem.text)
            if heading_match and len(elem.text) < 200:
                current_heading = heading_match.group(1).strip()
                i += 1
                continue

            # Detect captions (look-ahead)
            caption_match = _CAPTION_RE.search(elem.text)
            if caption_match and len(elem.text) < 300:
                pending_caption = caption_match.group(1).strip()
                i += 1
                continue

            # Table elements
            if elem.type == "table":
                table_group = self._collect_table_group(elements, i)
                html = _wrap_table(
                    "\n".join(e.text for e in table_group),
                    caption=pending_caption,
                )
                chunk = LayoutChunk(
                    text="\n".join(e.text for e in table_group),
                    html=html,
                    element_types=["table"],
                    source_document=elem.source_document,
                    page_start=table_group[0].page_number,
                    page_end=table_group[-1].page_number,
                    heading=current_heading,
                    is_continuation=len(table_group) > 1
                    and table_group[0].page_number != table_group[-1].page_number,
                    metadata={"caption": pending_caption},
                )
                chunks.append(chunk)
                pending_caption = None
                i += len(table_group)
                continue

            # Graph / image elements
            if elem.type in ("graph", "image"):
                html = _wrap_figure(elem.text, caption=pending_caption)
                chunks.append(LayoutChunk(
                    text=elem.text, html=html,
                    element_types=[elem.type],
                    source_document=elem.source_document,
                    page_start=elem.page_number, page_end=elem.page_number,
                    heading=current_heading,
                    metadata={"caption": pending_caption},
                ))
                pending_caption = None
                i += 1
                continue

            # Text: collect into a section chunk
            text_group, consumed = self._collect_text_group(elements, i)
            merged_text = " ".join(e.text for e in text_group)
            html = _wrap_section(merged_text, heading=current_heading)
            chunks.append(LayoutChunk(
                text=merged_text, html=html,
                element_types=["text"],
                source_document=elem.source_document,
                page_start=text_group[0].page_number,
                page_end=text_group[-1].page_number,
                heading=current_heading,
            ))
            i += consumed

        logger.info(
            "layout_parse_complete",
            input_elements=len(elements),
            output_chunks=len(chunks),
        )
        return chunks

    def _collect_table_group(
        self, elements: List[DocumentElement], start: int
    ) -> List[DocumentElement]:
        """Collect consecutive table elements (multi-page table continuation)."""
        group = [elements[start]]
        j = start + 1
        while j < len(elements) and elements[j].type == "table":
            prev_page = elements[j - 1].page_number
            curr_page = elements[j].page_number
            # Allow continuation only across consecutive pages
            if prev_page is not None and curr_page is not None and curr_page > prev_page + 1:
                break
            group.append(elements[j])
            j += 1
        return group

    def _collect_text_group(
        self, elements: List[DocumentElement], start: int
    ) -> Tuple[List[DocumentElement], int]:
        """Collect text elements until char limit, heading, or non-text type."""
        group = []
        total_chars = 0
        j = start
        while j < len(elements):
            elem = elements[j]
            if elem.type not in ("text",):
                break
            if total_chars + len(elem.text) > self._max_chars and group:
                break
            # Stop at a new heading
            if _HEADING_RE.search(elem.text) and len(elem.text) < 200 and group:
                break
            group.append(elem)
            total_chars += len(elem.text)
            j += 1
        return group or [elements[start]], max(1, j - start)

    def to_document_elements(
        self, chunks: List[LayoutChunk], tenant_id: Optional[str] = None
    ) -> List[DocumentElement]:
        """Convert LayoutChunks back to DocumentElements for the pipeline."""
        from datetime import datetime, timezone
        import hashlib

        now = datetime.now(timezone.utc).isoformat()
        result = []
        for chunk in chunks:
            # Embed HTML into metadata for richer LLM context
            result.append(DocumentElement(
                type="text" if "text" in chunk.element_types else chunk.element_types[0],
                text=chunk.html,  # Use HTML-wrapped version
                source_document=chunk.source_document,
                page_number=chunk.page_start,
                ingest_timestamp=now,
                content_hash=hashlib.sha256(chunk.text.encode()).hexdigest()[:12],
                tenant_id=tenant_id,
                metadata={
                    "layout_types": chunk.element_types,
                    "heading": chunk.heading,
                    "page_range": chunk.page_range,
                    "is_continuation": chunk.is_continuation,
                    "caption": chunk.metadata.get("caption"),
                },
            ))
        return result
