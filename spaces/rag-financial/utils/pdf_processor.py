"""utils/pdf_processor.py — PDF ingestion pipeline for the HF Space.

Standalone module mirroring src/rag_system/components/layout_parser.py logic:
PDF -> page images + raw text -> semantic chunk -> ready for embedding.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DocumentChunk:
    text: str
    page_number: int
    chunk_index: int
    source_filename: str
    chunk_type: str = "text"
    table_data: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        content = f"{self.source_filename}:{self.page_number}:{self.chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class IngestResult:
    filename: str
    num_pages: int
    num_chunks: int
    num_tables: int
    num_charts: int
    chunks: List[DocumentChunk]
    page_images: List[object]
    processing_steps: List[str]


def extract_text_and_tables(pdf_path: str) -> Tuple[List[dict], List[str]]:
    steps = []
    pages = []
    try:
        import pdfplumber
        steps.append("Opened PDF with pdfplumber")
        with pdfplumber.open(pdf_path) as pdf:
            steps.append(f"Found {len(pdf.pages)} pages")
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                pages.append({"page_num": i, "text": text.strip(), "tables": tables})
            steps.append(f"Extracted text from {len(pages)} pages")
            total_tables = sum(len(p["tables"]) for p in pages)
            if total_tables:
                steps.append(f"Found {total_tables} tables across all pages")
    except ImportError:
        steps.append("pdfplumber not available - trying PyPDF2 fallback")
        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    pages.append({"page_num": i, "text": text.strip(), "tables": []})
            steps.append(f"Extracted text (no table detection) from {len(pages)} pages")
        except Exception as e:
            steps.append(f"PDF extraction failed: {str(e)[:80]}")
    except Exception as e:
        steps.append(f"Extraction error: {str(e)[:80]}")
    return pages, steps


def render_page_images(pdf_path: str, max_pages: int = 8) -> Tuple[List[object], List[str]]:
    steps = []
    images = []
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=max_pages)
        steps.append(f"Rendered {len(images)} page images at 150 DPI for vision analysis")
    except ImportError:
        steps.append("pdf2image not available - skipping page rendering")
    except Exception as e:
        steps.append(f"Page rendering failed: {str(e)[:80]}")
    return images, steps


def semantic_chunk_text(
    text: str,
    page_num: int,
    source: str,
    chunk_start_index: int = 0,
    max_chars: int = 800,
    overlap_chars: int = 100,
) -> List[DocumentChunk]:
    if not text.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[DocumentChunk] = []
    buffer = ""
    chunk_idx = chunk_start_index
    for para in paragraphs:
        if len(buffer) + len(para) < max_chars:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
        else:
            if buffer:
                chunks.append(DocumentChunk(
                    text=buffer, page_number=page_num, chunk_index=chunk_idx,
                    source_filename=source, chunk_type="text",
                ))
                chunk_idx += 1
                buffer = buffer[-overlap_chars:] + "\n\n" + para if len(buffer) > overlap_chars else para
            else:
                buffer = para
    if buffer.strip():
        chunks.append(DocumentChunk(
            text=buffer.strip(), page_number=page_num, chunk_index=chunk_idx,
            source_filename=source, chunk_type="text",
        ))
    return chunks


def format_table_as_text(table: List[List[str]]) -> str:
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(c or "").strip() for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def chunk_tables(
    tables: List[List[List[str]]],
    page_num: int,
    source: str,
    chunk_start_index: int = 0,
) -> List[DocumentChunk]:
    chunks = []
    for t_idx, table in enumerate(tables):
        table_text = format_table_as_text(table)
        if table_text.strip():
            chunks.append(DocumentChunk(
                text=f"[TABLE on page {page_num}]\n{table_text}",
                page_number=page_num, chunk_index=chunk_start_index + t_idx,
                source_filename=source, chunk_type="table",
                table_data=table_text,
            ))
    return chunks


def ingest_pdf(pdf_path: str, process_vision: bool = True, vision_fn=None) -> IngestResult:
    filename = Path(pdf_path).name
    steps: List[str] = [f"Processing: {filename}"]
    all_chunks: List[DocumentChunk] = []
    page_images = []

    pages, extract_steps = extract_text_and_tables(pdf_path)
    steps.extend(extract_steps)

    chunk_counter = 0
    num_tables = 0
    for page in pages:
        text_chunks = semantic_chunk_text(page["text"], page["page_num"], filename, chunk_counter)
        all_chunks.extend(text_chunks)
        chunk_counter += len(text_chunks)
        table_chunks = chunk_tables(page["tables"], page["page_num"], filename, chunk_counter)
        all_chunks.extend(table_chunks)
        num_tables += len(table_chunks)
        chunk_counter += len(table_chunks)

    steps.append(f"Chunked into {len(all_chunks)} semantic chunks ({num_tables} tables)")

    num_charts = 0
    if process_vision and vision_fn is not None:
        imgs, img_steps = render_page_images(pdf_path)
        steps.extend(img_steps)
        page_images = imgs
        steps.append("Running vision LLM on page images...")
        for img_idx, img in enumerate(imgs[:6]):
            try:
                description = vision_fn(img)
                if description and len(description) > 30:
                    all_chunks.append(DocumentChunk(
                        text=f"[VISUAL CONTENT - page {img_idx + 1}]\n{description}",
                        page_number=img_idx + 1, chunk_index=chunk_counter,
                        source_filename=filename, chunk_type="chart_description",
                    ))
                    chunk_counter += 1
                    num_charts += 1
            except Exception as e:
                steps.append(f"Vision failed for page {img_idx + 1}: {str(e)[:60]}")
        if num_charts:
            steps.append(f"Generated {num_charts} visual descriptions from page images")
    elif process_vision:
        steps.append("Vision skipped - no vision model provided")

    steps.append(f"Ingestion complete: {len(all_chunks)} total chunks ready for retrieval")

    return IngestResult(
        filename=filename, num_pages=len(pages), num_chunks=len(all_chunks),
        num_tables=num_tables, num_charts=num_charts, chunks=all_chunks,
        page_images=page_images, processing_steps=steps,
    )
